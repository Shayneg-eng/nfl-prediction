"""
NFL game predictor: the saved best model (v1 ensemble), usable any week.

Usage (run from the project root or anywhere):
    python modeling/predict.py                      # next unplayed week, current data
    python modeling/predict.py --refresh            # download latest schedule/stats/injuries first
    python modeling/predict.py --season 2026 --week 5
    python modeling/predict.py --retrain            # refit the model on every completed game
    python modeling/predict.py --backtest 2025      # score a finished season (out-of-sample if the
                                                    # model was trained before it)

What it does:
  1. Loads every season of data: 1999-2024 from data/processed/, 2025+ straight from
     data/raw/ (nflverse renamed its player-stats release after 2024; mapped here).
  2. Builds features with the SAME leakage-verified builder (scripts/03) for the requested
     games, using only completed games before each game's date.
  3. Averages the frozen v1 ensemble (results/selection.json) to give P(home win) and
     predicted margin, and writes predictions/<season>_week<NN>.csv.

Any future week can be predicted "as of today". Games whose teams still have unplayed
earlier games are flagged as early predictions; re-run after those games for sharper
numbers. Run --refresh each week after games finish; for the injury features, run it after
Friday's final injury reports.
"""
import argparse
import datetime as dt
import glob
import importlib.util
import json
import os
import sys
import urllib.request
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW, PROC = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed")
MODEL_PATH = os.path.join(HERE, "results", "production_models.joblib")
PRED_DIR = os.path.join(ROOT, "predictions")
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
# when each raw file was downloaded; used as the "known by" time for injury reports that
# nflverse no longer timestamps (2025+)
DOWNLOAD_LOG = os.path.join(RAW, "download_log.json")
LAST_OLD_FORMAT_SEASON = 2024

sys.path.insert(0, HERE)
import common  # noqa: E402

_spec = importlib.util.spec_from_file_location("builder", os.path.join(ROOT, "scripts", "03_build_pregame_features.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


# ------------------------------------------------------------------ data
def current_season(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def refresh(seasons=None):
    """Download the latest schedule plus new-format player stats and injuries (2025+)."""
    seasons = seasons or range(LAST_OLD_FORMAT_SEASON + 1, current_season() + 1)
    os.makedirs(os.path.join(RAW, "player_stats_weekly"), exist_ok=True)
    jobs = [(f"{BASE}/schedules/games.parquet", os.path.join(RAW, "games.parquet"))]
    for y in seasons:
        jobs.append((f"{BASE}/stats_player/stats_player_week_{y}.parquet",
                     os.path.join(RAW, "player_stats_weekly", f"stats_player_week_{y}.parquet")))
        jobs.append((f"{BASE}/injuries/injuries_{y}.parquet", os.path.join(RAW, "injuries", f"injuries_{y}.parquet")))
    log = _read_download_log()
    for url, path in jobs:
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            log[os.path.basename(path)] = dt.datetime.now(dt.timezone.utc).isoformat()
            print("downloaded", os.path.relpath(path, ROOT))
        except Exception as e:  # e.g. injuries not published yet for a brand-new season
            if os.path.exists(tmp):
                os.remove(tmp)
            print("skipped", url, "->", e)
    with open(DOWNLOAD_LOG, "w") as fh:
        json.dump(log, fh, indent=2)


def _read_download_log():
    try:
        return json.load(open(DOWNLOAD_LOG))
    except (FileNotFoundError, ValueError):
        return {}


def load_inputs():
    """All seasons, in the exact shape scripts/03 build() expects."""
    old = B.load_inputs(ROOT)
    games = pd.read_parquet(os.path.join(RAW, "games.parquet"))   # full schedule incl. future games
    new = []
    for f in sorted(glob.glob(os.path.join(RAW, "player_stats_weekly", "stats_player_week_*.parquet"))):
        n = pd.read_parquet(f)
        n = n.rename(columns={"passing_interceptions": "interceptions", "sacks_suffered": "sacks"})
        n["sack_yards"] = -n["sack_yards_lost"]        # new format stores sack yards as negative
        new.append(n[old["ps"].columns])
    ps = pd.concat([old["ps"]] + new, ignore_index=True).drop_duplicates(["player_id", "game_id"], keep="first")
    # Injury reports: 2025+ files have no per-row timestamp. The strict rule is "only use a
    # report if it's provably public before game day", so stamp those rows with the time the
    # file was downloaded. Reports downloaded before a game count for it; games already played
    # when we downloaded get no injury data (unknown), exactly like the 2009 season.
    inj = old["injuries"].copy()
    if "date_modified" not in inj or inj.date_modified.isna().any():
        log = _read_download_log()
        stamp = {int(k[9:13]): v for k, v in log.items() if k.startswith("injuries_")}
        miss = inj.date_modified.isna() & inj.season.isin(list(stamp))
        inj.loc[miss, "date_modified"] = pd.to_datetime(inj.loc[miss, "season"].astype(int).map(stamp), utc=True)
    return {"games": games, "ps": ps, "injuries": inj}


def features_for(inputs, target_ids):
    """Leakage-safe features for target games, as of now: every completed game counts,
    unplayed games (including the targets) are kept in the schedule but contribute nothing.
    Returns the features and the target games whose teams still have an unplayed earlier game
    (those are 'early' predictions: they will sharpen once those games are played)."""
    g = inputs["games"].copy()
    g["gameday_dt"] = pd.to_datetime(g.gameday)
    played = g.home_score.notna() & g.away_score.notna()
    tgt = g[g.game_id.isin(target_ids)]
    early = []
    for r in tgt.itertuples():
        earlier = g[~played & (g.gameday_dt < r.gameday_dt) &
                    (g.home_team.isin([r.home_team, r.away_team]) | g.away_team.isin([r.home_team, r.away_team]))]
        if len(earlier):
            early.append(r.game_id)
    sched = g[played | (g.gameday_dt <= tgt.gameday_dt.max())].drop(columns="gameday_dt")
    ps = inputs["ps"][inputs["ps"].game_id.isin(g.game_id[played])]
    out = B.build({"games": sched, "ps": ps, "injuries": inputs["injuries"]})
    f = out["features"]
    f = f[f.game_id.isin(target_ids)].copy()
    f["home_adv"] = 1 - f.neutral_site
    return f, early


def expected_qbs(games, f):
    """Name of each team's expected starter = starter of its previous completed game."""
    g = games[games.home_score.notna()].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    long = pd.concat([g[["gameday", "home_team", "home_qb_name"]].set_axis(["gameday", "team", "qb"], axis=1),
                      g[["gameday", "away_team", "away_qb_name"]].set_axis(["gameday", "team", "qb"], axis=1)])
    long["team"] = B.fr(long.team)
    last = {}
    for r in f.itertuples():
        for t in (r.home_team, r.away_team):
            s = long[(long.team == B.fr(pd.Series([t]))[0]) & (long.gameday < r.gameday)]
            last[(r.game_id, t)] = s.sort_values("gameday").qb.iloc[-1] if len(s) else "?"
    return last


# ------------------------------------------------------------------ model
def predict_frame(bundle, f):
    ms, sel = bundle["models"], bundle["selection"]
    p = np.mean([ms[n].predict_proba(f) for n in sel["winner_ensemble"]], axis=0)
    pm = np.mean([ms[n].predict_margin(f) for n in sel["margin_ensemble"]], axis=0)
    return p, pm


def retrain(inputs):
    """Refit the frozen v1 ensemble (same configs) on every completed game."""
    g = inputs["games"]
    done = g[g.home_score.notna() & g.away_score.notna()]
    out = B.build({"games": done, "ps": inputs["ps"], "injuries": inputs["injuries"]})
    d = out["features"].merge(out["labels"][["game_id", "home_win", "margin", "is_tie"]], on="game_id")
    d = d[d.is_tie == 0].reset_index(drop=True)
    d["home_adv"] = 1 - d.neutral_site
    sel = json.load(open(os.path.join(HERE, "results", "selection.json")))
    C = common.candidates(common.feature_cols(d))
    models = {}
    for n in sorted(set(sel["winner_ensemble"]) | set(sel["margin_ensemble"])):
        m = C[n][1]().fit(d)
        if hasattr(m, "make_est"):
            m.make_est = None
        models[n] = m
    last = pd.to_datetime(done.gameday).max().date()
    bundle = {"models": models, "selection": sel, "trained_through": str(last), "n_games": int(len(d))}
    joblib.dump(bundle, MODEL_PATH)
    print(f"retrained on {len(d):,} completed games through {last}; saved {os.path.relpath(MODEL_PATH, ROOT)}")
    return bundle


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--refresh", action="store_true", help="download latest data first")
    ap.add_argument("--retrain", action="store_true", help="refit the model on all completed games")
    ap.add_argument("--backtest", type=int, metavar="SEASON", help="score every game of a finished season")
    a = ap.parse_args()

    if a.refresh:
        refresh()
    inputs = load_inputs()
    bundle = retrain(inputs) if a.retrain else joblib.load(MODEL_PATH)
    g = inputs["games"]

    if a.backtest:
        ids = g[(g.season == a.backtest) & g.home_score.notna()].game_id
        # rebuild with the season's results hidden from later games is unnecessary: build()
        # only ever uses games strictly before each date, so one pass is leakage-safe.
        done = g[g.home_score.notna()]
        out = B.build({"games": done, "ps": inputs["ps"], "injuries": inputs["injuries"]})
        f = out["features"][out["features"].game_id.isin(ids)].copy()
        f["home_adv"] = 1 - f.neutral_site
        p, pm = predict_frame(bundle, f)
        lab = out["labels"].set_index("game_id").loc[f.game_id]
        keep = lab.is_tie.values == 0
        y = lab.home_win.values[keep]
        vg = g.set_index("game_id").loc[f.game_id]
        r = common.win_metrics(y, p[keep])
        vacc = np.mean(((vg.spread_line.values > 0) == (lab.margin.values > 0))[keep & (vg.spread_line.values != 0)])
        print(f"Backtest {a.backtest}: {r['n']} games | model accuracy {r['accuracy']:.1%}, log loss {r['log_loss']:.4f} | "
              f"Vegas spread accuracy {vacc:.1%} | margin MAE model {np.mean(np.abs(lab.margin - pm)):.2f}, "
              f"Vegas {np.mean(np.abs(lab.margin - vg.spread_line.values)):.2f}")
        trained_year = int(str(bundle.get("trained_through"))[:4])
        if trained_year >= a.backtest:
            print("  note: the saved model was trained on data through", bundle.get("trained_through"),
                  "-> this season was seen in training, so this is NOT out-of-sample")
        else:
            print(f"  out-of-sample: model trained through {bundle.get('trained_through')}")
        return

    if a.season and a.week:
        tgt = g[(g.season == a.season) & (g.week == a.week)]
    else:
        up = g[g.home_score.isna()].copy()
        up["gd"] = pd.to_datetime(up.gameday)
        if up.empty:
            print("No unplayed games in the schedule. Try --refresh.")
            return
        first = up.sort_values("gd").iloc[0]
        tgt = g[(g.season == first.season) & (g.week == first.week)]
    tgt = tgt[tgt.home_score.isna()]
    if tgt.empty:
        print("Those games are already played. Use --backtest SEASON to score finished games.")
        return

    f, early = features_for(inputs, set(tgt.game_id))
    p, pm = predict_frame(bundle, f)
    qbs = expected_qbs(g, f)
    sched = g.set_index("game_id")
    rows = []
    for i, r in enumerate(f.itertuples()):
        s = sched.loc[r.game_id]
        pick, other = (r.home_team, r.away_team) if p[i] >= 0.5 else (r.away_team, r.home_team)
        spread = s.spread_line
        rows.append({
            "date": pd.to_datetime(r.gameday).date(), "matchup": f"{r.away_team} @ {r.home_team}",
            "pick": pick, "win_prob": round(max(p[i], 1 - p[i]), 3),
            "p_home_win": round(p[i], 3),
            "pred_margin": f"{pick} by {abs(pm[i]):.1f}",
            "vegas_line": ("" if pd.isna(spread) else (f"{r.home_team} -{spread:g}" if spread > 0 else
                                                       f"{r.away_team} -{-spread:g}" if spread < 0 else "PK")),
            "expected_qbs": f"{qbs[(r.game_id, r.away_team)]} @ {qbs[(r.game_id, r.home_team)]}",
            "early_prediction": r.game_id in early,
            "game_id": r.game_id,
        })
    res = pd.DataFrame(rows).sort_values(["date", "win_prob"], ascending=[True, False])
    season, week = int(tgt.season.iloc[0]), int(tgt.week.iloc[0])
    os.makedirs(PRED_DIR, exist_ok=True)
    path = os.path.join(PRED_DIR, f"{season}_week{week:02d}.csv")
    res.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(f"\n{season} week {week} | model trained through {bundle.get('trained_through')}\n")
    print(res.drop(columns=["game_id", "p_home_win"]).to_string(index=False))
    if early:
        print(f"\n{len(early)} game(s) flagged early_prediction: a team still has an unplayed earlier game, "
              "so re-run after those games for sharper numbers.")
    print(f"\nsaved {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
