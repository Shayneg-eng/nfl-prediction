# Data audit — pre-modeling (2026-09-22)

> **Status: superseded.** Every bug and leakage risk below was fixed in the rebuilt pipeline (scripts 03/04). The files named here as the "main modeling table" no longer exist. Use `nfl_model_features_1999_2024` + `nfl_model_labels_1999_2024` (see `data_dictionary.md`). A later finding: the weekly rosters are also not point-in-time, so all roster features were dropped.

Profiled every processed table and read the full `scripts/` pipeline before training anything. Summary: the data is rich and mostly well built, but **five pipeline bugs** need fixing before modeling, plus two leakage risks to handle deliberately.

## What we have

| File | Grain | Rows | Role for modeling |
|---|---|---|---|
| `nfl_pregame_matchup_features` | 1 row per game | 6,991 × 166 | **Main modeling table** (home vs away features + target) |
| `nfl_team_game_log_with_pregame_features` | 1 row per team-game | 13,982 × 88 | Source for rebuilding features |
| `nfl_games_1999_2024` | 1 row per game | 6,991 × 46 | Scores, Vegas lines, weather, QB/coach/ref |
| player offense / defense weekly | player-game | 134k / 240k | Box scores that roll up into team features |
| season / weekly rosters, players, injuries | various | — | Enrichment sources |

- **Seasons:** 1999–2024, 6,695 regular-season + 296 playoff games. `game_id` is unique, no missing scores.
- **Out-of-sample bonus:** `data/raw/games.parquet` already has the **full 2025 season (285 games) and 32 games of 2026**. Player stats stop at 2024, so box-score features can't be built for those yet — but Elo/record/schedule features can.

## Targets

| Target | Definition | Facts |
|---|---|---|
| Winner | `home_win` = home_score > away_score | Home team wins 56.5% (excl. ties); falls from 58.1% (1999–04) to 53.8% (2020–24). 14 ties are coded 0 — drop or handle them. |
| Margin | `result` = home_score − away_score | Mean +2.4, SD 14.6, range −49 to +59. Very lumpy: 3 pts = 15.1% of games, 7 = 9.0%, 6 = 6.1%, 10 = 5.6%. |

86 games are at **neutral sites** (home "wins" 43.5%) — the `location` column is in the games file but not the matchup file; add it.

## Benchmarks to beat (non-tie games)

| Predictor | Accuracy | AUC | Margin RMSE |
|---|---|---|---|
| Always pick home | 56.5% | — | — |
| Elo (`elo_diff` + 65 home bump) | 64.2% | 0.684 | — |
| Vegas closing spread | **66.5%** | **0.715** | **13.24** (MAE 10.29) |

Realistic ceiling is ~65–67% accuracy; anything well above that on a fair holdout means leakage.

## Strongest raw signals (home − away difference vs. margin)

| Feature | corr w/ margin | AUC |
|---|---|---|
| pregame_elo | 0.374 | 0.684 |
| pregame_wins / win_pct | 0.28–0.31 | 0.647 |
| roster_continuity_pct ⚠️ | 0.285 | 0.646 |
| pregame_avg_points_for | 0.276 | 0.643 |
| pregame_avg_off_epa | 0.256 | 0.635 |
| qb_pregame_avg_epa_per_att | 0.235 | 0.631 |

Injury counts are near zero (|r| < 0.02) and fumble features are noise.

## 🐛 Bugs to fix before modeling

1. **Relocated teams lose all box-score stats.** Player stats use current codes (LV/LAC/LA); script 03 joins them on raw historical codes (OAK/SD/STL). Result: **100% of OAK/SD/STL team-games (869 rows) have null offense/defense features and turnover margin forced to 0.** This is most of the 12.2% null rate in the box-score columns.
2. **Elo resets to 1500 on relocation.** LV 2020, LAC 2017, LA 2016 all restart at 1500 (their prior-season Elo was 1392 / 1405 / 1467).
3. **QB "last 3" features are really "last 1".** Script 04 merges QB stats with games on (season, week), creating ~3 duplicate rows per game (one for each game day that week). Verified: `qb_pregame_last3_*` equals the *previous single game* 63% of the time and the true last-3 only 2%; `qb_pregame_starts_played` is inflated ~2.75×. No current-game leakage, just wrong numbers.
4. **Division standings are broken for relocated teams.** `team_division()` only knows LV/LAC/LA, so OAK/SD/STL fall into an "UNKNOWN" division and `games_back` is computed against each other.
5. **Season-roster team codes vary by era** (ARZ, BLT, CLV, HST, SL in 2002–2015), so ~15% of roster-continuity values are null in those years.

Also: `surface` has `"grass "` (trailing space) and `""` values; scripts 04/05 aren't idempotent (re-running adds duplicate columns), despite what ORGANIZATION.md says.

## ⚠️ Leakage risks

- **`roster_continuity_pct` and `avg_roster_draft_capital` use the end-of-season roster** (snapshot from week 17–19), applied to every game that season, including week 1. Losing teams churn their rosters mid-season, so this can quietly encode the season's outcome. Its correlation (0.285) is suspiciously close to Elo's. **Drop it, or rebuild it from weekly rosters as of the week before each game.**
- **Vegas lines** (`spread_line`, moneylines) are the benchmark, not features. Keep them out of the model.
- **Starting QB identity** comes from who actually started. Usually known before kickoff, so acceptable, but late scratches are a small leak.

## Structural gaps

- **Week 1 has no rolling features** (412 games, 5.9%): averages reset each season. Elo, QB history, roster, and coaching tenure still cover it. Consider carrying prior-season stats into the early weeks.
- **Injuries:** 2009+ only (pre-2009 = 0, not NaN). They also ignore the "Probable" status. Low signal anyway.
- **Moneylines:** missing 1999–2005. **Weather:** null for all dome/closed/open-roof games (expected), plus 43% of outdoor 2022 games.
- `off_epa` is passing EPA only. `off_rec_yards` ≈ `off_pass_yards` (redundant).

## Recommended modeling setup

1. Fix bugs 1–4 (normalize team codes at the start of 03/05 and dedupe the QB merge), then rebuild.
2. Build **difference features** (home − away) plus a home/neutral flag; drop names, IDs, Vegas lines, and `away_team_check`.
3. **Split by time, never at random:** train 1999–2019 (or a rolling window), validate 2020–22, test 2023–24. Then 2025 as a true out-of-sample check using the features that don't need box scores.
4. Models: logistic regression and gradient boosting for the winner (scored by log-loss and accuracy); ridge and gradient boosting for the margin (scored by RMSE and MAE). Compare everything to the Vegas and Elo rows above.
