"""One-time final evaluation on 2023-2024 (models trained on 1999-2022 only).
The model choice is read from results/selection.json, frozen from the 2015-2022 validation results."""
import json, warnings
import numpy as np
import pandas as pd
from common import *
warnings.filterwarnings("ignore")

sel = json.load(open("results/selection.json"))
d = load(include_test=True)
cols = feature_cols(d)
C = candidates(cols)
train, test = d[d.season < min(TEST_SEASONS)], d[d.season.isin(TEST_SEASONS)]
assert train.season.max() < min(TEST_SEASONS)

names = sorted(set(sel["winner_ensemble"]) | set(sel["margin_ensemble"]) | {"elo_logit"})
fit = {n: C[n][1]().fit(train) for n in names}
P = {n: m.predict_proba(test) for n, m in fit.items()}
M = {n: fit[n].predict_margin(test) for n in sel["margin_ensemble"]}
p = np.mean([P[n] for n in sel["winner_ensemble"]], axis=0)
pm = np.mean([M[n] for n in sel["margin_ensemble"]], axis=0)

bench = pd.read_parquet(f"{PROC}/nfl_market_benchmark_1999_2024.parquet")
t = test[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_win", "margin", "neutral_site"]].copy()
t = t.merge(bench[["game_id", "spread_line", "home_implied_prob_novig"]], on="game_id", how="left")
t["p_home_win"], t["pred_margin"], t["p_elo"] = p, pm, P["elo_logit"]

y = t.home_win.values
res = {"Our model (ensemble)": win_metrics(y, p),
       "Elo only": win_metrics(y, t.p_elo.values),
       "Always pick home": win_metrics(y, np.full(len(y), train.home_win.mean())),
       "Vegas (moneyline, no-vig)": win_metrics(y, t.home_implied_prob_novig.values)}
res["Always pick home"]["accuracy"] = float(np.mean(y == 1))
mm = {"Our model (ensemble)": margin_metrics(t.margin.values, pm),
      "Vegas closing spread": margin_metrics(t.margin.values, t.spread_line.values),
      "Predict 0": margin_metrics(t.margin.values, np.zeros(len(t)))}
t["conf"] = np.abs(p - 0.5) + 0.5
t["correct"] = (p > 0.5) == (y == 1)
t["vegas_correct"] = (t.home_implied_prob_novig > 0.5) == (y == 1)
bins = pd.cut(t.conf, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0])
buckets = t.groupby(bins, observed=True).agg(games=("correct", "size"), our_accuracy=("correct", "mean"),
                                             avg_confidence=("conf", "mean"), vegas_accuracy=("vegas_correct", "mean"))
agree = (p > 0.5) == (t.home_implied_prob_novig > 0.5)
by_season = t.groupby("season").agg(ours=("correct", "mean"), vegas=("vegas_correct", "mean"), games=("correct", "size"))
t.drop(columns=["conf"]).to_csv("results/test_predictions_2023_2024.csv", index=False)

out = {"test_seasons": TEST_SEASONS, "n_games": int(len(t)), "winner": res, "margin": mm,
       "pick_agreement_with_vegas": float(agree.mean()),
       "when_disagree_with_vegas": {"n": int((~agree).sum()), "ours_correct": float(t.correct[~agree].mean()),
                                    "vegas_correct": float(t.vegas_correct[~agree].mean())},
       "by_season": by_season.round(4).reset_index().to_dict("records"),
       "confidence_buckets": buckets.round(4).reset_index().astype({"conf": str}).to_dict("records")}
json.dump(out, open("results/test_results_2023_2024.json", "w"), indent=2)
print(json.dumps(out, indent=1))

