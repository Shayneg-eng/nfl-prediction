"""Experiment: use the Vegas closing spread as an input and learn corrections to it
(hype, recency overreaction, favorite/underdog bias, etc.).
Walk-forward validation 2015-2022 only. Test seasons 2023-24 are NOT loaded.

The closing spread is public before kickoff, so it's a legitimate input, not leakage.
Targets:
  resid = margin - spread_line   (how much the home team beat the market's expectation)
  cover = resid > 0               (home covered; pushes excluded when scoring)
Win prob = P(N(spread + predicted resid, sigma) > 0), sigma estimated on training data.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from common import *
warnings.filterwarnings("ignore")

d = load(include_test=False)
b = pd.read_parquet(f"{PROC}/nfl_market_benchmark_1999_2024.parquet")[["game_id", "spread_line", "home_implied_prob_novig"]]
d = d.merge(b, on="game_id").dropna(subset=["spread_line"]).reset_index(drop=True)
d["resid"] = d.margin - d.spread_line
d["abs_spread"] = d.spread_line.abs()
d["home_fav"] = (d.spread_line > 0).astype(int)
# "hype" / recency features: recent form relative to the team's longer-run level
d["diff_form_vs_season"] = d.diff_last3_avg_pt_diff - d.diff_season_avg_pt_diff
d["diff_form_vs_ewm"] = d.diff_last3_avg_pt_diff - d.diff_ewm_pt_diff
d["model_gap"] = np.nan  # filled per fold: v1-style model margin minus spread

BIAS = ["spread_line", "abs_spread", "home_fav", "home_adv", "is_primetime", "div_game", "is_playoff",
        "diff_streak", "diff_last3_avg_pt_diff", "diff_form_vs_season", "diff_form_vs_ewm",
        "diff_h2h_last_margin", "diff_off_bye", "diff_short_week", "diff_rest_days",
        "diff_qb_changed_last_game", "diff_inj_exp_qb_out_or_doubtful", "diff_coach_new_this_season",
        "diff_prev_season_win_pct", "diff_win_pct", "temp", "wind", "is_indoor"]
cols_all = feature_cols(load(include_test=False).iloc[:5])  # v1 feature list


def fit_resid(tr, kind, feats):
    X, y = tr[feats], tr.resid.values
    if kind == "ridge":
        m = ridge(3000)().fit(X, y)
    else:
        m = HistGradientBoostingRegressor(learning_rate=0.02, max_depth=2, max_iter=200, min_samples_leaf=100,
                                          l2_regularization=5.0, random_state=0).fit(X, y)
    return m


rows, oof = [], []
for s in VAL_SEASONS:
    tr, va = d[d.season < s].copy(), d[d.season == s].copy()
    # v1 model (no Vegas) margin, as a "second opinion" feature: fit on train only
    v1 = MarginModel(ridge(3000), CURATED).fit(tr)
    inner_last = sorted(tr.season.unique())[-3:]
    # model_gap for training rows must be out-of-sample too: use an inner model on earlier seasons
    v1_inner = MarginModel(ridge(3000), CURATED).fit(tr[~tr.season.isin(inner_last)])
    tr["model_gap"] = v1_inner.predict_margin(tr) - tr.spread_line
    va["model_gap"] = v1.predict_margin(va) - va.spread_line
    trg = tr[tr.season.isin(inner_last)]  # gap feature only trustworthy on inner-holdout seasons
    sig = float(np.std(tr.resid))
    o = va[["game_id", "season", "home_win", "margin", "spread_line", "resid"]].copy()
    o["p_vegas_spread"] = norm.cdf(va.spread_line / sig)
    for name, kind, feats, train_df in [
        ("resid_ridge_bias", "ridge", BIAS, tr),
        ("resid_hgb_bias", "hgb", BIAS, tr),
        ("resid_ridge_bias+modelgap", "ridge", BIAS + ["model_gap"], trg),
        ("resid_ridge_modelgap_only", "ridge", ["model_gap", "spread_line"], trg),
    ]:
        m = fit_resid(train_df, kind, feats)
        r = m.predict(va[feats])
        o[f"r_{name}"] = r
        o[f"p_{name}"] = norm.cdf((va.spread_line + r) / sig)
    # stacking: logistic on [spread, v1 margin]
    st = LogisticRegression(C=1.0).fit(np.c_[trg.spread_line, trg.spread_line + trg.model_gap], trg.home_win)
    o["p_stack_logit"] = st.predict_proba(np.c_[va.spread_line, va.spread_line + va.model_gap])[:, 1]
    o["p_v1"] = v1.predict_proba(va)
    oof.append(o)
oof = pd.concat(oof)
os.makedirs("results/market_residual", exist_ok=True)
oof.to_parquet("results/market_residual/oof_2015_2022.parquet", index=False)

y = oof.home_win.values
nonpush = oof.resid != 0
cover = (oof.resid > 0).values
out = []
for c in [c for c in oof.columns if c.startswith("p_")]:
    r = {"model": c[2:], **win_metrics(y, oof[c].values)}
    rc = f"r_{c[2:]}"
    if rc in oof:
        pick = oof[rc].values > 0
        r["ats_hit_rate"] = float(np.mean(pick[nonpush] == cover[nonpush]))
        big = (np.abs(oof[rc].values) > 1.5) & nonpush.values
        r["ats_hit_when_|edge|>1.5"] = float(np.mean(pick[big] == cover[big])) if big.sum() else None
        r["n_edge>1.5"] = int(big.sum())
        r["resid_mae"] = float(np.mean(np.abs(oof.resid - oof[rc])))
    out.append(r)
res = pd.DataFrame(out)
res["resid_mae_baseline(0)"] = float(np.mean(np.abs(oof.resid)))
print(res.round(4).to_string(index=False))
res.round(4).to_csv("results/market_residual/validation_summary.csv", index=False)
# vegas moneyline for reference where available
mm = oof.merge(b[["game_id", "home_implied_prob_novig"]], on="game_id").dropna(subset=["home_implied_prob_novig"])
print("vegas moneyline:", {k: round(v, 4) for k, v in win_metrics(mm.home_win.values, mm.home_implied_prob_novig.values).items()})
