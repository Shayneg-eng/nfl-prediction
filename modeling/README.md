# NFL game models: winner + margin (v1, 2026-09-22)

The goal is to predict **who wins** (the priority) and **by how much**, using only the leakage-verified pre-game features in `data/processed/nfl_model_features_1999_2024`.

## ▶ How to use the saved model

The best model (v1 ensemble) is saved in `results/production_models.joblib`. `predict.py` is the tool for using it. Run it from the project folder:

| Command | What it does |
|---|---|
| `python modeling/predict.py --refresh` | **The weekly command.** Downloads the latest schedule, stats and injury reports, then predicts the next unplayed week. Writes `predictions/<season>_week<NN>.csv`. |
| `python modeling/predict.py --season 2026 --week 5` | Predict a specific week. Games whose teams still have unplayed earlier games are flagged `early_prediction`. |
| `python modeling/predict.py --retrain` | Refit the model (same frozen configuration) on every completed game. A few times a season is plenty. |
| `python modeling/predict.py --backtest 2025` | Score a finished season. It says whether that season was out-of-sample for the saved model. |

Each row gives the pick, win probability, predicted margin, the Vegas line for comparison, and the **expected starting QBs**: the model assumes each team's previous starter plays. If a starter is out, discount that game's prediction.

**Best time to run:** Friday or Saturday, after the final injury reports. For Thursday games, run Wednesday night.

**Needs:** Python with pandas, numpy, scikit-learn, scipy, joblib and pyarrow, plus internet access for `--refresh`.

**How it stays leak-free:** features come from the same verified builder (`scripts/03`), which only uses information from before each game's date. Unplayed games stay in the schedule but contribute nothing; this path is covered by `scripts/04` (half the post-cutoff games in its test have blank scores). 2025+ injury reports carry no timestamps from nflverse, so each report is stamped with the time we downloaded it. It only counts for games after that moment.

**Track record:**

| Period | Model | Vegas |
|---|---|---|
| Validation 2015–22 | 64.7% | 65.4% |
| Test 2023–24 | 64.7% | 69.1% |
| Out-of-sample 2025 (trained ≤2024) | 63.4% (log loss 0.629, margin MAE 10.02) | 65.8% (MAE 9.67) |

`03_fit_production.py` only covers the 1999–2024 research tables and is superseded by `predict.py --retrain`.

## Protocol (fixed before any results were seen)

| Stage | Seasons | Purpose |
|---|---|---|
| Walk-forward validation | 2015–2022 (2,164 games) | Each season is predicted by models trained only on earlier seasons. All model/hyperparameter choices were made here, by **log loss** (steadier than accuracy at this sample size). |
| Final test | 2023–2024 (570 games) | Scored **once**, after the choice was frozen in `results/selection.json`. `01_validate.py` can't load these seasons. |
| Production | 1999–2024 | Same frozen ensembles refit on everything, for predicting future games. |

- **Both orientations:** every training game is also fed in with home and away swapped, and predictions are averaged over both orientations. The only home effect the model can learn comes from the explicit `home_adv` flag (0 at neutral sites).
- **Ties:** the 14 ties are dropped, since there's no winner to predict.
- **No leakage from preprocessing:** imputers and scalers are fit inside each training fold.

## What was tried (validation, 2015–22)

| Model family | Best validation accuracy | Best log loss |
|---|---|---|
| Elo only (logistic on Elo difference + home) | 64.0% | 0.6357 |
| Logistic regression, all 70 difference features | 64.1% | 0.6324 |
| Logistic regression, 22 curated features | 64.4% | 0.6318 |
| Ridge margin → win probability, curated | 64.7% | 0.6314 |
| Gradient boosting classifier | 64.0% | 0.6343 |
| Gradient boosting margin regressor | 64.4% | 0.6330 |
| **Ensemble (selected)** | **64.7%** | **0.6308** |
| *Vegas closing line (reference)* | *65.4%* | — |

Heavy regularization won everywhere. That's typical for NFL data: about 270 games a season is a small, noisy sample. Tuning the Elo constants (on 2002–14 only) was also tested; it gained at most 0.0008 in log loss, so they stay fixed.

**Selected winner model:** the average win probability of 4 models:

- ridge margin model, all features (α=3000)
- ridge margin model, curated features (α=3000)
- logistic regression, curated features (C=0.001)
- gradient-boosted margin model (learning rate 0.03, depth 3, 300 trees)

Margin models become win probabilities through a normal distribution, with σ estimated on held-out training seasons.

**Selected margin model:** the average of the three margin regressors above.

## Final test: 2023–2024 (570 games, scored once)

### Who wins

| Predictor | Accuracy | Log loss | Brier |
|---|---|---|---|
| Always pick home | 55.6% | 0.687 | 0.247 |
| Elo only | 63.9% | 0.635 | 0.222 |
| **Our model** | **64.7%** (±3.9 pts, 95% CI) | **0.631** | **0.220** |
| Vegas moneyline (no-vig) | 69.1% | 0.608 | 0.210 |

By season: 62.5% in 2023, 67.0% in 2024.

### By how much

| Predictor | MAE | RMSE |
|---|---|---|
| Predict 0 | 11.30 | 14.65 |
| **Our model** | **10.19** | **13.21** |
| Vegas closing spread | 9.84 | 12.95 |

### Confidence is honest (calibration)

| Model confidence | Games | Our accuracy | Vegas accuracy |
|---|---|---|---|
| 50–55% | 122 | 48% | 67% |
| 55–60% | 123 | 63% | 66% |
| 60–65% | 113 | 67% | 67% |
| 65–70% | 80 | 71% | 69% |
| 70–75% | 60 | 73% | 75% |
| 75%+ | 72 | 76% | 76% |

When the model says 70%+, it's right about 75% of the time and matches Vegas. Its losses to Vegas are almost all in toss-up games (<55% confidence), where it's no better than a coin flip.

### How to read this

- **It beats Elo, but only modestly:** +0.9 pts accuracy and better log loss. Head-to-head on the games where they disagree (29 vs 24) isn't significant (McNemar p = 0.58).
- **It trails Vegas, and that gap is real.** They pick the same winner 85% of the time. In the 85 games where they disagree, Vegas is right 55 times and the model 30 (p = 0.009). Some of the gap is Vegas having an unusually good two years: 69.1% vs its 65.4% over 2015–22.
- **Why Vegas has an edge:** the market sees things this dataset can't. Confirmed starters and late injury news, depth charts, weather forecasts, and matchup specifics.
- **What drives the picks** (curated logistic, standardized coefficients): home advantage, Elo gap, weighted point differential (recent seasons + this season), season point differential, expected QB out/doubtful, division standing, and QB recent ANY/A.

## Files

| File | What |
|---|---|
| `common.py` | Data loading, orientation-swapping augmentation, model definitions, metrics |
| `01_validate.py` | Walk-forward validation → `results/validation_oof/`, `results/validation_summary_2015_2022.csv` |
| `02_final_test.py` | One-time test → `results/test_results_2023_2024.json`, `results/test_predictions_2023_2024.csv` |
| `03_fit_production.py` | Refit on all seasons → `results/production_models.joblib`; has a `predict()` helper |
| `results/selection.json` | The frozen model choice |

Run from inside `modeling/`: `python3 01_validate.py`, then `02_final_test.py`, then `03_fit_production.py`.

## Most promising next steps (for winner accuracy)

1. **Confirmed starting QB** at prediction time (announced before kickoff). The biggest known information gap vs Vegas; the model currently assumes last week's starter.
2. **An opponent-adjusted team rating** (SRS-style, fit point-in-time), as an alternative to plain Elo.
3. **2025 season as a second test set.** The schedule and scores are already in `data/raw/games.parquet`; player stats need refreshing upstream first.

## Experiment: Vegas line as an input (2026-09-23, `04_market_residual_experiment.py`)

**Question:** if the closing spread is an input, can our features correct human biases in it (hype, recency, favorite/underdog)? Walk-forward validation 2015–22 only; test seasons not touched.

| Model | Winner accuracy | Log loss | Against-the-spread hit rate |
|---|---|---|---|
| Vegas spread alone | 65.3% | 0.6147 | — |
| Vegas + ridge bias correction | 65.5% | 0.6142 | 50.5% |
| Vegas + boosted bias correction | 65.6% | 0.6155 | 48.8% |
| Vegas + bias + v1 model's opinion | 65.6% | 0.6156 | 50.0% (44% on its "confident" picks) |
| Stack of Vegas + v1 | 65.2% | 0.6155 | — |
| v1 (no Vegas) | 64.7% | 0.6314 | — |

- **The corrections added noise, not signal.** Error on the Vegas residual got slightly worse than predicting zero (9.855 vs 9.839). The accuracy differences (±0.3 pts) are within noise (±1 pt).
- **The residual model's high-conviction picks did *worse* than a coin flip.** It fits noise, and the sample punishes that.
- **Classic betting "rules"** (home dogs, fade hot teams, primetime dogs, divisional dogs, big dogs, playoff dogs, off-bye teams) were each checked for how often they covered: found on 1999–2014, then re-checked on 2015–22. All land at 47–54%, none is statistically significant in either era, and none reliably clears the 52.4% needed to profit at standard −110 odds. Divisional underdogs (51.9% / 52.3%) and fading teams on a 3+ win streak (51.0% / 52.4%) lean the same way in both eras, but are indistinguishable from noise.

**Conclusion:** with data available at the closing line, the market is efficient. What's left over is almost pure randomness. A real edge would need information the market doesn't price by close (player-level injury impact, earlier lines before news is priced in), not public-data bias correction.

## Experiment: attention-based models (2026-09-23, `05_attention_experiment.py`)

**Question:** do attention models find complex relationships the v1 ensemble misses? Walk-forward validation 2015–22; test seasons not touched. Run in the cloud workspace (needs PyTorch, which isn't installed on the local machine).

- **FT-Transformer:** each of the 71 v1 features becomes a token, and attention runs across features (2 layers, 32 dimensions, 3 seeds).
- **History attention:** a transformer reads each team's last 16 games strictly before the game date (result, efficiency, opponent strength at the time, days ago, home/away) and learns which past games matter. It's combined with pre-game facts (Elo, QB, rest, injuries), and the architecture is antisymmetric in home/away.

| Model | Accuracy | Log loss | Δ log loss vs v1 | Margin MAE |
|---|---|---|---|---|
| v1 ensemble | 64.7% | 0.6308 | — | 10.09 |
| Elo only | 64.0% | 0.6357 | +0.0049 ± 0.0025 | — |
| FT-Transformer | 64.3% | 0.6339 | +0.0031 ± 0.0026 (worse) | 10.15 |
| History attention | 64.0% | 0.6303 | −0.0005 ± 0.0025 (tie) | 10.12 |
| v1 + history attention | 64.4% | 0.6288 | −0.0020 ± 0.0013 | 10.06 |
| v1 + history + FT | 64.6% | 0.6287 | −0.0021 ± 0.0013 | 10.05 |

- **Feature attention doesn't help.** This matches the gradient-boosting result: the v1 features have little hidden interaction structure. Their signal is mostly additive.
- **History attention matches v1.** It learned from raw game logs about as well as the hand-built averages. That's a useful sign for future player-level sequence models.
- **Blending gives a small, not-yet-significant gain** (1.5 standard errors) in log loss and margin, and no gain in accuracy. Its predictions correlate 0.94 with v1's, so it's mostly re-learning the same information.
- **Decision:** not adopted into production. The gain is inside the noise, and it would add a heavy PyTorch dependency plus about 7× the training time. Worth revisiting once the history sequences carry new information (play-by-play efficiency, player participation), which is where sequence models should shine.
