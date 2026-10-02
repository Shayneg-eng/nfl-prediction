"""
04 - Leakage verification. Exits non-zero if any check fails.

Main test (future-corruption invariance):
  For many cutoff dates D, build a copy of every input in which everything a
  forecaster could NOT have known before D is scrambled:
    * games on/after D:   scores, result, total, overtime, actual starting QBs, betting
                          lines -> random; half get blank (NaN) scores, as if unplayed
    * games after D:      every non-key schedule field (coach, rest, roof, surface,
                          weather, div flag, time) -> random
    * player box scores:  games on/after D -> random numbers, half the rows dropped
    * injury reports:     rows time-stamped on/after D, or with no timestamp -> random status/position
    * weekly rosters:     (only if a future builder uses them again) weeks whose last game
                          is on/after D -> random players/draft slots/status
  Then re-run the real 03 build() on that corrupted copy. Every feature for every
  game on or before D must be byte-for-byte identical to the clean build. If any
  feature changes, it used information from D or later -> leakage.

Positive control: the same harness is run against a deliberately leaky builder
(adds the game's own margin as a feature), and it MUST catch it. That shows the
test can detect leakage when it's there.

Plus static checks: no outcome/betting columns in the feature file, one row per
game, features and labels cover the same games, no two games for a team on one date.
Plus a smoke alarm: a simple logistic regression on a time split must stay under
70% accuracy (NFL ceiling is about 66-67%). Clearing it proves nothing on its own; failing it means something leaks.

Writes docs/leakage_verification_report.md.
"""
import importlib.util
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("builder", os.path.join(HERE, "03_build_pregame_features.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

N_RANDOM_CUTOFFS = int(os.environ.get("N_RANDOM_CUTOFFS", "20"))
FORBIDDEN_EXACT = {"home_score", "away_score", "result", "total", "overtime", "margin", "total_points",
                   "home_win", "is_tie", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
                   "qb_actual", "exp_qb", "referee", "win", "loss", "tie", "pts_for", "pts_against"}
# "roster": weekly/season roster files were rebuilt after the fact (see 03 docstring)
FORBIDDEN_SUBSTR = ["spread", "moneyline", "odds", "total_line", "implied", "_epa", "score", "roster"]
OUTCOME_COLS = ["home_score", "away_score", "result", "total", "overtime", "home_qb_id", "away_qb_id",
                "home_qb_name", "away_qb_name", "spread_line", "total_line", "home_moneyline", "away_moneyline",
                "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"]
SCHEDULE_KEEP = {"game_id", "season", "week", "gameday", "home_team", "away_team", "game_type", "location"}


def corrupt(inp, D, rng):
    g = inp["games"].copy()
    gd = pd.to_datetime(g.gameday)
    on_after, after = gd >= D, gd > D
    qb_pool = pd.concat([g.home_qb_id, g.away_qb_id]).dropna().unique()
    n = on_after.sum()
    for c in ["home_score", "away_score"]:
        g.loc[on_after, c] = rng.integers(0, 55, n).astype(float)
    # half of games on/after D get blank scores, as if unplayed (the predict.py path for upcoming games)
    g.loc[on_after & (rng.random(len(g)) < 0.5), ["home_score", "away_score"]] = np.nan
    g.loc[on_after, "result"] = g.loc[on_after, "home_score"] - g.loc[on_after, "away_score"]
    g.loc[on_after, "total"] = rng.integers(0, 100, n)
    g.loc[on_after, "overtime"] = rng.integers(0, 2, n)
    for c in ["home_qb_id", "away_qb_id"]:
        g.loc[on_after, c] = rng.choice(qb_pool, n)
    for c in ["spread_line", "total_line", "home_moneyline", "away_moneyline"]:
        g.loc[on_after, c] = rng.normal(0, 200, n)
    na = after.sum()
    for c in g.columns:
        if c in SCHEDULE_KEEP or c in OUTCOME_COLS:
            continue
        if pd.api.types.is_numeric_dtype(g[c]):
            g.loc[after, c] = rng.integers(0, 30, na)
        else:
            g.loc[after, c] = rng.choice(g[c].dropna().unique(), na)

    ps = inp["ps"].copy()
    fut = ps.game_id.isin(set(g.game_id[on_after]))
    num = [c for c in ps.columns if c not in ("player_id", "team", "game_id")]
    ps.loc[fut, num] = rng.integers(0, 40, (fut.sum(), len(num)))
    ps = ps[~(fut & (rng.random(len(ps)) < 0.5))]

    inj = inp["injuries"].copy()
    ts = pd.to_datetime(inj.date_modified, utc=True).dt.tz_convert(None)
    bad = ts.isna() | (ts >= D)
    inj.loc[bad, "report_status"] = rng.choice(["Out", "Doubtful", "Questionable", None], bad.sum())
    inj.loc[bad, "position"] = rng.choice(["QB", "WR", "T", "DE", "CB"], bad.sum())

    out = {"games": g, "ps": ps, "injuries": inj}
    if "rosters" not in inp:
        return out
    ros = inp["rosters"].copy()
    wk_end = inp["games"].assign(gameday=pd.to_datetime(inp["games"].gameday)).groupby(["season", "week"]).gameday.max()
    end = pd.MultiIndex.from_frame(ros[["season", "week"]]).map(wk_end.to_dict().get)
    end = pd.to_datetime(pd.Series(end, index=ros.index))
    rb = end.isna() | (end >= D)
    ids = ros.gsis_id.dropna().unique()
    ros.loc[rb, "gsis_id"] = rng.choice(ids, rb.sum())
    ros.loc[rb, "status"] = rng.choice(["ACT", "RES", "CUT"], rb.sum())
    ros.loc[rb, "draft_number"] = rng.integers(1, 260, rb.sum())
    ros.loc[rb, "years_exp"] = rng.integers(0, 15, rb.sum())
    out["rosters"] = ros
    return out


def diff_frames(a, b, key):
    a = a.set_index(key).sort_index()
    b = b.set_index(key).sort_index()
    assert a.index.equals(b.index), "row sets differ"
    bad = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            same = (np.isnan(xv) & np.isnan(yv)) | np.isclose(xv, yv, rtol=0, atol=1e-9)
        else:
            same = (x.isna() & y.isna()) | (x.astype(str) == y.astype(str))
        if not np.all(same):
            bad.append((c, int((~np.asarray(same)).sum())))
    return bad


def invariance_check(build_fn, inp, clean, D, rng):
    dirty = build_fn(corrupt(inp, D, rng))
    f_clean = clean["features"][clean["features"].gameday <= D]
    f_dirty = dirty["features"][dirty["features"].game_id.isin(f_clean.game_id)]
    t_clean = clean["team_features"][clean["team_features"].gameday <= D]
    t_dirty = dirty["team_features"][dirty["team_features"].gameday <= D]
    return (diff_frames(f_clean, f_dirty, "game_id") +
            diff_frames(t_clean, t_dirty, ["game_id", "team"])), len(f_clean)


def leaky_build(inp):
    out = B.build(inp)
    g = inp["games"][["game_id", "home_score", "away_score"]]
    f = out["features"].merge(g, on="game_id", how="left")
    f["leaky_feature"] = f.home_score - f.away_score
    out["features"] = f.drop(columns=["home_score", "away_score"])
    return out


def main():
    os.chdir(os.path.join(HERE, ".."))
    rep, fails = [], []
    t0 = time.time()
    inp = B.load_inputs()
    clean = B.build(inp)
    feats, labels = clean["features"], clean["labels"]

    # ---- static checks
    cols = set(feats.columns) | set(clean["team_features"].columns)
    hit = sorted(c for c in cols if c in FORBIDDEN_EXACT or
                 any(s in c for s in FORBIDDEN_SUBSTR) or
                 c.replace("home_", "", 1).replace("away_", "", 1).replace("diff_", "", 1) in FORBIDDEN_EXACT)
    static = {
        "no outcome / betting / EPA / roster columns in feature files": not hit,
        "one row per game in features": feats.game_id.is_unique,
        "features and labels cover identical games": set(feats.game_id) == set(labels.game_id),
        "no team plays twice on one date": not clean["team_features"].duplicated(["team", "gameday"]).any(),
        "exactly two team rows per game": (clean["team_features"].groupby("game_id").size() == 2).all(),
    }
    rep.append("## Static checks\n\n| Check | Result |\n|---|---|")
    for k, v in static.items():
        rep.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
        if not v:
            fails.append(k)
    if hit:
        rep.append(f"\nForbidden columns found: {hit}")

    # ---- cutoffs: random dates + hand-picked edge cases
    rng = np.random.default_rng(20260922)
    days = pd.Series(sorted(feats.gameday.unique()))
    picks = list(rng.choice(days, N_RANDOM_CUTOFFS, replace=False))
    first_of_season = feats.groupby("season").gameday.min()
    specials = {
        "first game day of 2000 (season rollover)": first_of_season[2000],
        "first game day of 2010 (injury-report era)": first_of_season[2010],
        "first game day of 2024": first_of_season[2024],
        "Thanksgiving 2012 (multiple games, same day)": pd.Timestamp("2012-11-22"),
        "2020-12-02 (COVID-moved Wednesday game)": pd.Timestamp("2020-12-02"),
        "Super Bowl LVIII (2024-02-11)": pd.Timestamp("2024-02-11"),
        "last date in data": days.iloc[-1],
    }
    cut = [(f"random {pd.Timestamp(d).date()}", pd.Timestamp(d)) for d in sorted(picks)] + list(specials.items())

    rep.append("\n## Future-corruption invariance test\n")
    rep.append("For each cutoff D, all post-D information is scrambled and the pipeline is rebuilt. "
               "Every feature of every game on or before D has to match the clean build exactly.\n")
    rep.append("| Cutoff | Games compared | Changed features | Result |\n|---|---|---|---|")
    for label, D in cut:
        assert (feats.gameday == D).any(), f"cutoff {D} is not a game date"
        bad, ncmp = invariance_check(B.build, inp, clean, D, rng)
        ok = not bad
        rep.append(f"| {label} | {ncmp:,} | {bad[:5] if bad else 0} | {'PASS' if ok else 'FAIL'} |")
        print(label, "PASS" if ok else f"FAIL {bad[:5]}", flush=True)
        if not ok:
            fails.append(f"invariance @ {label}")

    # ---- positive control
    D = pd.Timestamp("2015-10-18")
    bad, _ = invariance_check(leaky_build, inp, leaky_build(inp), D, rng)
    caught = any(c == "leaky_feature" for c, _ in bad)
    rep.append("\n## Positive control\n")
    rep.append(f"A builder that deliberately adds each game's own final margin as a feature was run through the same test "
               f"(cutoff {D.date()}). Test flagged: {bad}. **{'PASS: leakage detected' if caught else 'FAIL: test missed deliberate leakage'}**")
    if not caught:
        fails.append("positive control")

    # ---- smoke alarm
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        df = feats.merge(labels[["game_id", "home_win"]], on="game_id").dropna(subset=["home_win"])
        X = df[[c for c in df.columns if c.startswith("diff_")] + ["neutral_site", "is_playoff", "div_game"]]
        tr, te = df.season <= 2018, df.season >= 2019
        mdl = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        mdl.fit(X[tr], df.home_win[tr])
        acc = (mdl.predict(X[te]) == df.home_win[te]).mean()
        ok = acc < 0.70
        rep.append(f"\n## Smoke alarm\n\nLogistic regression, trained on 1999–2018 and tested on 2019–2024: "
                   f"**{acc:.1%}** accuracy on {te.sum():,} games (alarm threshold 70%). {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append("smoke alarm")
    except ImportError:
        rep.append("\n## Smoke alarm\n\nskipped (scikit-learn not installed)")

    status = "ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"
    head = [f"# Leakage verification report", "",
            f"Generated by `scripts/04_verify_no_leakage.py` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
            f"Runtime {time.time() - t0:.0f}s. **{status}**", ""]
    with open("docs/leakage_verification_report.md", "w") as f:
        f.write("\n".join(head + rep) + "\n")
    print(status)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
