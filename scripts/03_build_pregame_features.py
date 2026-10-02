"""
03 - Point-in-time (leakage-proof) pre-game feature builder.

THE ONE RULE: a feature for a game played on calendar date D may only use
information time-stamped strictly BEFORE D.

Anything that happens on or after D is "post-kickoff" information, even if it
was an earlier game on the same day, the game's own score, box score, or starting
QB. That rule is applied the same way to every input source:

  source                       timestamp used                    how it's enforced
  ---------------------------  --------------------------------  ---------------------------------
  scores / box scores          gameday of that game              as-of join, strict "<"
  starting QB (actual)         gameday of that game              only the PREVIOUS game's starter is
                                                                 used (the "expected starter")
  injury reports               date_modified of each report row  rows with no timestamp or a
                                                                 timestamp >= D are dropped
  Elo                          updated after each game           rating recorded before the update

The only per-game fields used for game D itself are schedule facts published
before kickoff: date, week, teams, site/neutral, rest days, division game, roof,
surface, head coach, kickoff weather (temp/wind), and game type.

Deliberately NOT used (see docs/data_dictionary.md):
  * EPA or any nflfastR model output: the expected-points model was fit on
    seasons after many of the games it scores.
  * players master table: its status/team/experience fields are current, not historical.
  * ANY roster file (season or weekly). Season rosters are end-of-season snapshots,
    and the pre-2017 weekly rosters were rebuilt after the fact: player statuses are
    back-filled, so a player who went on IR in week 10 shows as inactive from week 1.
    "Active roster size" built from them predicted margin at r = 0.21 in 2002-2015 and
    about 0 from 2016 on. That's a leak a timestamp test can't catch, so these
    sources are excluded entirely.
  * Vegas lines: written to a separate benchmark file, never to the feature file.
  * actual starting QB of game D.

Outputs (data/processed/):
  nfl_model_features_1999_2024.{parquet,csv}   one row per game: features only (no outcomes)
  nfl_model_labels_1999_2024.{parquet,csv}     one row per game: home_win, margin, total ...
  nfl_market_benchmark_1999_2024.{parquet,csv} one row per game: closing spread/total/moneyline
  nfl_team_features_pregame_1999_2024.parquet  one row per team-game, same features, long format

scripts/04_verify_no_leakage.py re-runs build() with every piece of post-D information
corrupted or deleted and checks that no feature on or before D changes.
"""
import glob
import os
import numpy as np
import pandas as pd

RAW = "data/raw"
OUT = "data/processed"

# Every historical/alternate code -> one franchise code, so a franchise keeps its
# history (Elo, streaks, QB, rosters) when it relocates.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA", "ARZ": "ARI", "BLT": "BAL",
             "CLV": "CLE", "HST": "HOU", "SL": "LA"}

DIV_2002_ON = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SF", "SEA"],
}
DIV_PRE_2002 = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ", "IND"], "AFC Central": ["BAL", "CIN", "CLE", "PIT", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LV", "LAC", "SEA"], "NFC East": ["DAL", "NYG", "PHI", "WAS", "ARI"],
    "NFC Central": ["CHI", "DET", "GB", "MIN", "TB"], "NFC West": ["LA", "SF", "ATL", "CAR", "NO"],
}
TEAM_DIV = {}
for _s in range(1999, 2030):
    for _d, _ts in (DIV_2002_ON if _s >= 2002 else DIV_PRE_2002).items():
        for _t in _ts:
            TEAM_DIV[(_s, _t)] = _d

# Elo constants are FIXED published values (538's NFL Elo), not tuned on this data.
ELO_K, ELO_HFA, ELO_START, ELO_REVERT = 20.0, 65.0, 1500.0, 1 / 3

POS_GROUP = {"QB": "QB", "RB": "SKILL", "FB": "SKILL", "WR": "SKILL", "TE": "SKILL",
             "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
             "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB"}


def fr(s):
    return s.replace(FRANCHISE)


# ----------------------------------------------------------------------------- inputs
def load_inputs(root="."):
    games = pd.read_parquet(f"{root}/{OUT}/nfl_games_1999_2024.parquet")
    ps = pd.read_parquet(f"{root}/{OUT}/nfl_player_offense_stats_weekly_1999_2024.parquet",
                         columns=["player_id", "team", "game_id", "attempts", "completions", "passing_yards",
                                  "passing_tds", "interceptions", "sacks", "sack_yards", "carries",
                                  "rushing_yards", "rushing_tds", "sack_fumbles_lost", "rushing_fumbles_lost",
                                  "receiving_fumbles_lost"])
    inj = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/{RAW}/injuries/injuries_*.parquet"))],
                    ignore_index=True)[["season", "week", "team", "gsis_id", "position", "report_status", "date_modified"]]
    return {"games": games, "ps": ps, "injuries": inj}


# ----------------------------------------------------------------------------- helpers
def _prior(df, col, by, window=None):
    """Sum of `col` over the group's PREVIOUS rows (current row excluded). df must be time-sorted."""
    g = df.groupby(by, sort=False)[col]
    if window is None:
        return g.cumsum() - df[col]
    return g.transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())


def _ratio(num, den):
    return np.where(den > 0, num / den.where(den > 0, 1), np.nan)


def _asof(left, right, by, left_on, right_on, cols):
    """Strict backward as-of join: right rows with right_on < left_on only."""
    l = left[[by, left_on]].reset_index().sort_values(left_on)
    r = right[[by, right_on] + cols].sort_values(right_on)
    m = pd.merge_asof(l, r, left_on=left_on, right_on=right_on, by=by,
                      allow_exact_matches=False, direction="backward")
    return m.set_index("index").reindex(left.index)[cols]


# ----------------------------------------------------------------------------- schedule / team-game table
def prep_schedule(games):
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["home_fr"], g["away_fr"] = fr(g["home_team"]), fr(g["away_team"])
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def team_game_table(g):
    cols = ["game_id", "season", "week", "game_type", "gameday"]
    h = g[cols + ["home_fr", "away_fr", "home_rest", "home_coach", "home_qb_id", "home_score", "away_score"]].copy()
    h.columns = cols + ["team", "opp", "rest_days", "coach", "qb_actual", "pts_for", "pts_against"]
    h["is_home"] = 1
    a = g[cols + ["away_fr", "home_fr", "away_rest", "away_coach", "away_qb_id", "away_score", "home_score"]].copy()
    a.columns = h.columns[:-1]
    a["is_home"] = 0
    tg = pd.concat([h, a], ignore_index=True)
    tg = tg.sort_values(["team", "gameday"]).reset_index(drop=True)
    assert not tg.duplicated(["team", "gameday"]).any(), "a team plays twice on one date"
    return tg


def box_scores(ps, g):
    p = ps.copy()
    p["team"] = fr(p["team"])
    for c in ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]:
        p[c] = p[c].fillna(0)
    p["fumbles_lost"] = p.sack_fumbles_lost + p.rushing_fumbles_lost + p.receiving_fumbles_lost
    b = p.groupby(["game_id", "team"], as_index=False)[
        ["attempts", "completions", "passing_yards", "passing_tds", "interceptions", "sacks", "sack_yards",
         "carries", "rushing_yards", "rushing_tds", "fumbles_lost"]].sum()
    b["anya_num"] = b.passing_yards - b.sack_yards + 20 * b.passing_tds - 45 * b.interceptions
    b["anya_den"] = b.attempts + b.sacks
    b["turnovers"] = b.interceptions + b.fumbles_lost
    return b[b.game_id.isin(g.game_id)]


# ----------------------------------------------------------------------------- Elo
def elo_ratings(g):
    elo, last_season, pre = {}, {}, {}
    for r in g.itertuples(index=False):  # g is sorted by date
        for t in (r.home_fr, r.away_fr):
            if t not in elo:
                elo[t] = ELO_START
            elif last_season[t] != r.season:
                elo[t] = ELO_START + (elo[t] - ELO_START) * (1 - ELO_REVERT)
            last_season[t] = r.season
        rh, ra = elo[r.home_fr], elo[r.away_fr]
        pre[(r.game_id, r.home_fr)], pre[(r.game_id, r.away_fr)] = rh, ra
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        hfa = 0.0 if r.location == "Neutral" else ELO_HFA
        diff = rh + hfa - ra
        exp_h = 1 / (1 + 10 ** (-diff / 400))
        m = r.home_score - r.away_score
        act = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        w_diff = diff if m > 0 else -diff
        mult = np.log(abs(m) + 1) * 2.2 / (w_diff * 0.001 + 2.2) if m != 0 else 1.0
        elo[r.home_fr] = rh + ELO_K * mult * (act - exp_h)
        elo[r.away_fr] = ra - ELO_K * mult * (act - exp_h)
    return pre


# ----------------------------------------------------------------------------- main feature build
def build(inputs):
    g = prep_schedule(inputs["games"])
    tg = team_game_table(g)
    key = tg.team

    # post-game facts of each row (only ever used via _prior / shift / strict as-of)
    # Unplayed games (blank scores, e.g. the upcoming week) are kept in the schedule but
    # contribute nothing: played=0, zero points/wins, and NaN where a "last result" is read.
    played = tg.pts_for.notna() & tg.pts_against.notna()
    tg["played"] = played.astype(float)
    tg["win"] = (tg.pts_for > tg.pts_against).astype(float)
    tg["loss"] = (tg.pts_for < tg.pts_against).astype(float)
    tg["tie"] = ((tg.pts_for == tg.pts_against) & played).astype(float)
    tg["pt_diff_played"] = (tg.pts_for - tg.pts_against).where(played)
    tg["pts_for"], tg["pts_against"] = tg.pts_for.where(played, 0.0), tg.pts_against.where(played, 0.0)
    tg["pt_diff"] = tg.pts_for - tg.pts_against
    box = box_scores(inputs["ps"], g)
    tg = tg.merge(box.add_prefix("o_").rename(columns={"o_game_id": "game_id", "o_team": "team"}),
                  on=["game_id", "team"], how="left")
    tg = tg.merge(box.add_prefix("d_").rename(columns={"d_game_id": "game_id", "d_team": "opp"}),
                  on=["game_id", "opp"], how="left")
    tg = tg.sort_values(["team", "gameday"]).reset_index(drop=True)
    tg["has_box"] = tg.o_attempts.notna().astype(float)
    tg["takeaways"] = tg.d_turnovers
    tg["to_margin"] = tg.d_turnovers - tg.o_turnovers
    stat_cols = ["pts_for", "pts_against", "pt_diff", "win", "loss", "tie", "played"]
    box_cols = ["o_anya_num", "o_anya_den", "d_anya_num", "d_anya_den", "o_carries", "o_rushing_yards",
                "d_carries", "d_rushing_yards", "o_turnovers", "takeaways", "to_margin", "o_sacks", "d_sacks",
                "o_passing_yards", "d_passing_yards", "has_box"]
    for c in box_cols:
        tg[c] = tg[c].fillna(0)

    F = pd.DataFrame(index=tg.index)
    ts = [tg.team, tg.season]

    # ---- season-to-date (resets each season)
    for c in stat_cols + box_cols:
        tg[f"_s_{c}"] = _prior(tg, c, ts)
    n, nb = tg._s_played, tg._s_has_box
    F["games_played"] = n
    F["wins"], F["losses"], F["ties"] = tg._s_win, tg._s_loss, tg._s_tie
    F["win_pct"] = _ratio(tg._s_win + 0.5 * tg._s_tie, n)
    for c in ["pts_for", "pts_against", "pt_diff"]:
        F[f"season_avg_{c}"] = _ratio(tg[f"_s_{c}"], n)
    F["season_off_anya"] = _ratio(tg._s_o_anya_num, tg._s_o_anya_den)
    F["season_def_anya_allowed"] = _ratio(tg._s_d_anya_num, tg._s_d_anya_den)
    F["season_off_ypc"] = _ratio(tg._s_o_rushing_yards, tg._s_o_carries)
    F["season_def_ypc_allowed"] = _ratio(tg._s_d_rushing_yards, tg._s_d_carries)
    for c, nm in [("o_passing_yards", "season_off_pass_ypg"), ("d_passing_yards", "season_def_pass_ypg_allowed"),
                  ("o_rushing_yards", "season_off_rush_ypg"), ("d_rushing_yards", "season_def_rush_ypg_allowed"),
                  ("o_turnovers", "season_giveaways_pg"), ("takeaways", "season_takeaways_pg"),
                  ("to_margin", "season_to_margin_pg"), ("o_sacks", "season_sacks_taken_pg"),
                  ("d_sacks", "season_sacks_made_pg")]:
        F[nm] = _ratio(tg[f"_s_{c}"], nb)

    # ---- last 3 games (within season)
    for c in ["pt_diff", "pts_for", "pts_against", "win", "played", "o_anya_num", "o_anya_den",
              "d_anya_num", "d_anya_den", "to_margin", "has_box"]:
        tg[f"_l3_{c}"] = _prior(tg, c, ts, window=3)
    F["last3_avg_pt_diff"] = _ratio(tg._l3_pt_diff, tg._l3_played)
    F["last3_avg_pts_for"] = _ratio(tg._l3_pts_for, tg._l3_played)
    F["last3_avg_pts_against"] = _ratio(tg._l3_pts_against, tg._l3_played)
    F["last3_win_pct"] = _ratio(tg._l3_win, tg._l3_played)
    F["last3_off_anya"] = _ratio(tg._l3_o_anya_num, tg._l3_o_anya_den)
    F["last3_def_anya_allowed"] = _ratio(tg._l3_d_anya_num, tg._l3_d_anya_den)
    F["last3_to_margin_pg"] = _ratio(tg._l3_to_margin, tg._l3_has_box)

    # ---- exponentially weighted, across seasons (so week 1 has signal)
    def ewm_prior(col):
        return tg.groupby("team", sort=False)[col].transform(lambda s: s.shift(1).ewm(halflife=8).mean())
    F["ewm_pt_diff"] = ewm_prior("pt_diff_played")
    tg["_o_anya_num_b"] = tg.o_anya_num.where(tg.has_box == 1)
    tg["_o_anya_den_b"] = tg.o_anya_den.where(tg.has_box == 1)
    tg["_d_anya_num_b"] = tg.d_anya_num.where(tg.has_box == 1)
    tg["_d_anya_den_b"] = tg.d_anya_den.where(tg.has_box == 1)
    F["ewm_off_anya"] = ewm_prior("_o_anya_num_b") / ewm_prior("_o_anya_den_b")
    F["ewm_def_anya_allowed"] = ewm_prior("_d_anya_num_b") / ewm_prior("_d_anya_den_b")

    # ---- previous season (entirely in the past)
    ps_agg = tg.groupby(["team", "season"], as_index=False)[["win", "tie", "played", "pt_diff", "o_anya_num",
                                                               "o_anya_den", "d_anya_num", "d_anya_den"]].sum()
    ps_agg["prev_season_win_pct"] = _ratio(ps_agg.win + 0.5 * ps_agg.tie, ps_agg.played)
    ps_agg["prev_season_pt_diff_pg"] = _ratio(ps_agg.pt_diff, ps_agg.played)
    ps_agg["prev_season_off_anya"] = _ratio(ps_agg.o_anya_num, ps_agg.o_anya_den)
    ps_agg["prev_season_def_anya_allowed"] = _ratio(ps_agg.d_anya_num, ps_agg.d_anya_den)
    ps_agg["season"] += 1
    pcols = [c for c in ps_agg.columns if c.startswith("prev_season_")]
    F[pcols] = tg[["team", "season"]].merge(ps_agg[["team", "season"] + pcols], how="left").set_index(tg.index)[pcols]

    # ---- Elo + strength of schedule
    pre = elo_ratings(g)
    F["elo"] = [pre[(a, b)] for a, b in zip(tg.game_id, tg.team)]
    tg["opp_elo"] = [pre[(a, b)] for a, b in zip(tg.game_id, tg.opp)]
    tg["_opp_elo_played"] = tg.opp_elo * tg.played
    F["season_avg_opp_elo"] = _ratio(_prior(tg, "_opp_elo_played", ts), n)

    # ---- streak (signed, across seasons; resets on a tie)
    streak = np.zeros(len(tg))
    cur, prev_team = 0, None
    for i, (t, w, l, pl) in enumerate(zip(tg.team.values, tg.win.values, tg.loss.values, tg.played.values)):
        if t != prev_team:
            cur, prev_team = 0, t
        streak[i] = cur
        if not pl:
            continue
        cur = (cur + 1 if cur > 0 else 1) if w == 1 else (cur - 1 if cur < 0 else -1) if l == 1 else 0
    F["streak"] = streak

    # ---- schedule facts known before kickoff
    F["rest_days"] = tg.rest_days
    F["off_bye"] = (tg.rest_days >= 13).astype(float)
    F["short_week"] = (tg.rest_days <= 5).astype(float)

    # ---- head to head (last meeting, either site)
    h2h = tg.sort_values(["team", "opp", "gameday"])
    grp = h2h.groupby(["team", "opp"], sort=False)
    # last PLAYED meeting (skips unplayed ones)
    h2h = h2h.assign(_m=h2h.pt_diff_played, _w=h2h.win.where(h2h.played == 1),
                     _d=h2h.gameday.where(h2h.played == 1))
    grp = h2h.groupby(["team", "opp"], sort=False)
    F["h2h_last_margin"] = grp._m.transform(lambda s: s.shift(1).ffill())
    F["h2h_last_won"] = grp._w.transform(lambda s: s.shift(1).ffill())
    F["h2h_days_since"] = (h2h.gameday - grp._d.transform(lambda s: s.shift(1).ffill())).dt.days
    F["h2h_prior_meetings"] = (grp.played.cumsum() - h2h.played).astype(int)

    # ---- head coach (listed coach of this game is a pre-game fact; tenure counts prior games)
    prev_coach = tg.groupby("team").coach.shift(1)
    new_stint = (tg.coach != prev_coach).astype(int)
    stint = new_stint.groupby(tg.team).cumsum()
    F["coach_games_with_team"] = tg.groupby([tg.team, stint]).cumcount()
    # 1 if the current coach's stint with this team began this season
    F["coach_new_this_season"] = (tg.groupby([tg.team, stint]).season.transform("min") == tg.season).astype(float)
    F.loc[tg.season == tg.season.min(), "coach_new_this_season"] = np.nan  # no history before first season

    # ---- QB: expected starter = whoever started this team's previous game
    tg["exp_qb"] = tg.groupby("team").qb_actual.shift(1)
    qb = inputs["ps"][["player_id", "game_id", "attempts", "passing_yards", "sack_yards", "sacks",
                       "passing_tds", "interceptions"]].copy()
    qb = qb[qb.attempts > 0].merge(g[["game_id", "gameday", "season"]], on="game_id")
    qb["num"] = qb.passing_yards - qb.sack_yards + 20 * qb.passing_tds - 45 * qb.interceptions
    qb["den"] = qb.attempts + qb.sacks
    qb = qb.sort_values(["player_id", "gameday"]).reset_index(drop=True)
    gq = qb.groupby("player_id")
    qb["c_num"], qb["c_den"], qb["c_games"] = gq.num.cumsum(), gq.den.cumsum(), gq.cumcount() + 1
    qb["r_num"] = gq.num.transform(lambda s: s.rolling(8, min_periods=1).sum())
    qb["r_den"] = gq.den.transform(lambda s: s.rolling(8, min_periods=1).sum())
    qb["s_num"] = qb.groupby(["player_id", "season"]).num.cumsum()
    qb["s_den"] = qb.groupby(["player_id", "season"]).den.cumsum()
    qb = qb.rename(columns={"player_id": "exp_qb", "season": "qb_season"})
    tg_q = tg[tg.exp_qb.notna()]
    qa = _asof(tg_q, qb, "exp_qb", "gameday", "gameday",
               ["c_num", "c_den", "c_games", "r_num", "r_den", "s_num", "s_den", "qb_season"])
    qa = qa.reindex(tg.index)
    F["qb_career_anya"] = _ratio(qa.c_num, qa.c_den)
    F["qb_career_dropbacks"] = qa.c_den.fillna(0).where(tg.exp_qb.notna())
    F["qb_career_games"] = qa.c_games.fillna(0).where(tg.exp_qb.notna())
    F["qb_last8_anya"] = _ratio(qa.r_num, qa.r_den)
    same = qa.qb_season == tg.season
    F["qb_season_anya"] = np.where(same, _ratio(qa.s_num, qa.s_den), np.nan)
    starts = tg[["qb_actual", "gameday"]].rename(columns={"qb_actual": "exp_qb"}).sort_values(["exp_qb", "gameday"])
    starts["c_starts"] = starts.groupby("exp_qb").cumcount() + 1
    st = _asof(tg_q, starts, "exp_qb", "gameday", "gameday", ["c_starts"]).reindex(tg.index)
    F["qb_career_starts"] = st.c_starts.fillna(0).where(tg.exp_qb.notna())
    prim = tg.groupby(["team", "season"]).qb_actual.agg(lambda s: s.value_counts().index[0]).rename("prev_primary")
    prim = prim.reset_index()
    prim["season"] += 1
    pp = tg[["team", "season"]].merge(prim, how="left").set_index(tg.index).prev_primary
    F["qb_new_vs_last_season"] = np.where(pp.isna() | tg.exp_qb.isna(), np.nan, (tg.exp_qb != pp).astype(float))
    F["qb_changed_last_game"] = np.where(tg.groupby("team").qb_actual.shift(2).isna(), np.nan,
                                         (tg.exp_qb != tg.groupby("team").qb_actual.shift(2)).astype(float))

    # ---- division standing (regular season, as of strictly before this date)
    F = F.join(division_standing(tg))

    # ---- injury report (only rows time-stamped strictly before game date)
    F = F.join(injury_features(inputs["injuries"], tg, g))

    ids = tg[["game_id", "season", "week", "gameday", "team", "opp", "is_home"]]
    team_feats = pd.concat([ids, F], axis=1).sort_values(["gameday", "game_id", "is_home"],
                                                         ascending=[True, True, False]).reset_index(drop=True)
    return assemble(g, team_feats)


def division_standing(tg):
    reg = tg[tg.game_type == "REG"].copy()
    reg["div"] = [TEAM_DIV.get((s, t), "UNK") for s, t in zip(reg.season, reg.team)]
    reg = reg.sort_values(["team", "gameday"])
    grp = reg.groupby(["team", "season"])
    reg["cw"], reg["cl"], reg["ct"] = grp.win.cumsum(), grp.loss.cumsum(), grp.tie.cumsum()
    out = tg[["team", "season", "gameday"]].copy()
    out["div"] = [TEAM_DIV.get((s, t), "UNK") for s, t in zip(out.season, out.team)]
    assert (out["div"] != "UNK").all(), "team missing from division table"
    # every row x every division member (incl. self)
    members = pd.DataFrame([(s, d, t) for (s, t), d in TEAM_DIV.items()], columns=["season", "div", "member"])
    x = out.reset_index().merge(members, on=["season", "div"])
    rec = reg[["team", "season", "gameday", "cw", "cl", "ct"]].rename(columns={"team": "member", "season": "rs"})
    x["key"] = x.member + "_" + x.season.astype(str)
    rec["key"] = rec.member + "_" + rec.rs.astype(str)
    x = x.sort_values("gameday")
    m = pd.merge_asof(x, rec.sort_values("gameday")[["key", "gameday", "cw", "cl", "ct"]], on="gameday", by="key",
                      allow_exact_matches=False)
    m[["cw", "cl", "ct"]] = m[["cw", "cl", "ct"]].fillna(0)
    gp = m.cw + m.cl + m.ct
    m["pct"] = np.where(gp > 0, (m.cw + 0.5 * m.ct) / gp.where(gp > 0, 1), 0.5)
    lead = m.groupby("index").pct.max()
    own = m[m.member == m.team].set_index("index").pct
    rank = m.assign(r=m.groupby("index").pct.rank(ascending=False, method="min"))
    rank = rank[rank.member == rank.team].set_index("index").r
    return pd.DataFrame({"div_pct_behind_leader": lead - own, "div_rank": rank}).reindex(tg.index)


def injury_features(inj, tg, g):
    cols = ["inj_n_out", "inj_n_doubtful", "inj_n_questionable", "inj_exp_qb_out_or_doubtful"] + \
           [f"inj_out_or_doubtful_{p}" for p in ["SKILL", "OL", "DL", "LB", "DB"]]
    i = inj.dropna(subset=["season", "week", "team", "gsis_id"]).copy()
    i["season"], i["week"] = i.season.astype(int), i.week.astype(int)
    i["team"] = fr(i.team)
    i["ts"] = pd.to_datetime(i.date_modified, utc=True).dt.tz_convert(None)
    key = tg[["season", "week", "team", "gameday"]].drop_duplicates()
    i = i.merge(key, on=["season", "week", "team"], how="inner")
    i = i[i.ts.notna() & (i.ts < i.gameday)]                      # <-- the leakage guard
    i = i.sort_values("ts").groupby(["season", "week", "team", "gsis_id"], as_index=False).last()
    i["grp"] = i.position.map(POS_GROUP).fillna("OTHER")
    i["od"] = i.report_status.isin(["Out", "Doubtful"]).astype(int)
    agg = i.groupby(["season", "week", "team"]).agg(
        inj_n_out=("report_status", lambda s: (s == "Out").sum()),
        inj_n_doubtful=("report_status", lambda s: (s == "Doubtful").sum()),
        inj_n_questionable=("report_status", lambda s: (s == "Questionable").sum()))
    for p in ["SKILL", "OL", "DL", "LB", "DB"]:
        agg[f"inj_out_or_doubtful_{p}"] = i[i.grp == p].groupby(["season", "week", "team"]).od.sum()
    agg = agg.fillna(0).reset_index()
    qbo = i[i.od == 1][["season", "week", "team", "gsis_id"]].rename(columns={"gsis_id": "exp_qb"})
    qbo["inj_exp_qb_out_or_doubtful"] = 1.0
    out = tg[["season", "week", "team", "exp_qb"]].merge(agg, how="left", on=["season", "week", "team"]) \
        .merge(qbo.drop_duplicates(), how="left", on=["season", "week", "team", "exp_qb"]).set_index(tg.index)
    # league-weeks with time-stamped reports: absent team = 0 injuries; other weeks = unknown (NaN)
    covered = set(zip(i.season, i.week))
    has = pd.Series([(s, w) in covered for s, w in zip(tg.season, tg.week)], index=tg.index)
    for c in cols:
        out[c] = out[c].fillna(0).where(has)
    return out[cols]


def assemble(g, team_feats):
    feat_cols = [c for c in team_feats.columns if c not in ("game_id", "season", "week", "gameday", "team", "opp", "is_home")]
    h = team_feats[team_feats.is_home == 1].set_index("game_id")[feat_cols].add_prefix("home_")
    a = team_feats[team_feats.is_home == 0].set_index("game_id")[feat_cols].add_prefix("away_")
    d = pd.DataFrame({f"diff_{c}": h[f"home_{c}"] - a[f"away_{c}"] for c in feat_cols}, index=h.index)
    surface = g.surface.fillna("").str.strip().str.lower()
    ctx = pd.DataFrame({
        "game_id": g.game_id, "season": g.season, "week": g.week, "gameday": g.gameday,
        "game_type": g.game_type, "home_team": g.home_team, "away_team": g.away_team,
        "is_playoff": (g.game_type != "REG").astype(int),
        "neutral_site": (g.location == "Neutral").astype(int),
        "div_game": g.div_game, "roof": g.roof,
        "is_indoor": g.roof.isin(["dome", "closed"]).astype(int),
        "is_grass": np.where(surface == "", np.nan, surface.isin(["grass", "dessograss"]).astype(float)),
        "temp": g.temp, "wind": g.wind,
        "is_primetime": ((g.weekday.isin(["Monday", "Thursday"])) |
                         ((g.weekday == "Sunday") & (g.gametime.fillna("") >= "20:00"))).astype(int),
    }).set_index("game_id")
    features = ctx.join(h).join(a).join(d).reset_index()
    features = features.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    lab = g[["game_id", "season", "gameday", "home_score", "away_score"]].copy()
    lab["margin"] = lab.home_score - lab.away_score
    lab["total_points"] = lab.home_score + lab.away_score
    lab["is_tie"] = (lab.margin == 0).astype(int)
    lab["home_win"] = np.where(lab.margin == 0, np.nan, (lab.margin > 0).astype(float))
    lab["overtime"] = g.overtime
    lab = lab.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    bench = g[["game_id", "season", "gameday", "spread_line", "total_line", "home_moneyline", "away_moneyline"]].copy()
    ph = np.where(bench.home_moneyline < 0, -bench.home_moneyline / (-bench.home_moneyline + 100), 100 / (bench.home_moneyline + 100))
    pa = np.where(bench.away_moneyline < 0, -bench.away_moneyline / (-bench.away_moneyline + 100), 100 / (bench.away_moneyline + 100))
    bench["home_implied_prob_novig"] = ph / (ph + pa)
    bench = bench.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return {"features": features, "labels": lab, "benchmark": bench, "team_features": team_feats}


FEATURE_FILE = "nfl_model_features_1999_2024"
LABEL_FILE = "nfl_model_labels_1999_2024"
BENCH_FILE = "nfl_market_benchmark_1999_2024"
TEAM_FILE = "nfl_team_features_pregame_1999_2024"


def main():
    out = build(load_inputs())
    for name, df in [(FEATURE_FILE, out["features"]), (LABEL_FILE, out["labels"]), (BENCH_FILE, out["benchmark"])]:
        df.to_parquet(f"{OUT}/{name}.parquet", index=False)
        df.to_csv(f"{OUT}/{name}.csv", index=False)
    out["team_features"].to_parquet(f"{OUT}/{TEAM_FILE}.parquet", index=False)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
