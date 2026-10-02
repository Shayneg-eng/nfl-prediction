# NFL Dataset: Data Dictionary

Source: [nflverse](https://github.com/nflverse/nflverse-data), pulled from their public GitHub release assets on 2026-09-22. Every processed file covers the **1999–2024** seasons (player box scores stop after 2024 upstream).

## The leakage rule

> A feature for a game played on calendar date **D** may only use information time-stamped **strictly before D**.

`scripts/03_build_pregame_features.py` enforces the rule, and `scripts/04_verify_no_leakage.py` proves it held. The verification step rebuilds every feature after scrambling all information from on or after D, at 27 cutoff dates. It confirms that nothing changes for any game on or before D, and it confirms it can catch a deliberately planted leak. Results are in `docs/leakage_verification_report.md`. **If you change 03, re-run 04.**

## Files for modeling

Features, labels, and the betting-market benchmark live in **separate files** on purpose. You can't train on an outcome or a Vegas line by accident; you have to join them in explicitly. Join on `game_id`.

| File | Grain | Rows × cols | Contents |
|---|---|---|---|
| `nfl_model_features_1999_2024` (csv + parquet) | 1 row per game | 6,991 × 214 | IDs, game context, and per-team features as `home_*`, `away_*`, and `diff_*` (= home − away). **No outcomes.** |
| `nfl_model_labels_1999_2024` (csv + parquet) | 1 row per game | 6,991 × 10 | `home_win` (NaN for the 14 ties), `margin` (home − away), `total_points`, `is_tie`, scores, `overtime` |
| `nfl_market_benchmark_1999_2024` (csv + parquet) | 1 row per game | 6,991 × 8 | Closing `spread_line` (home perspective, + = home favored), `total_line`, moneylines, vig-free `home_implied_prob_novig`. Moneylines are missing for 1999–2005. **Benchmark only, never a feature.** |
| `nfl_team_features_pregame_1999_2024` (parquet) | 1 row per team-game | 13,982 × 73 | The same per-team features in long format (team, opp, is_home) |

### Context columns (one per game)

`season, week, gameday, game_type, home_team, away_team` (IDs, historical abbreviations), `is_playoff, neutral_site, div_game, roof, is_indoor, is_grass, temp, wind, is_primetime`. `temp` and `wind` are kickoff conditions and are null for indoor games.

### Per-team features (each appears as `home_`, `away_`, and `diff_`)

| Group | Columns | Notes |
|---|---|---|
| Record | `games_played, wins, losses, ties, win_pct` | Season to date; reset each season |
| Season averages | `season_avg_pts_for / _pts_against / _pt_diff`, `season_off_anya`, `season_def_anya_allowed`, `season_off_ypc`, `season_def_ypc_allowed`, `season_off/def_pass_ypg(_allowed)`, `season_off/def_rush_ypg(_allowed)`, `season_giveaways_pg`, `season_takeaways_pg`, `season_to_margin_pg`, `season_sacks_taken_pg`, `season_sacks_made_pg` | ANY/A = (pass yds − sack yds + 20·TD − 45·INT) / (att + sacks). "Allowed" and takeaway stats come from the opponent's box score in the same game. |
| Last 3 games | `last3_avg_pt_diff / _pts_for / _pts_against`, `last3_win_pct`, `last3_off_anya`, `last3_def_anya_allowed`, `last3_to_margin_pg` | Within the season |
| Across seasons | `ewm_pt_diff`, `ewm_off_anya`, `ewm_def_anya_allowed` | Exponentially weighted (half-life 8 games) across all prior games, so week 1 has signal |
| Previous season | `prev_season_win_pct`, `prev_season_pt_diff_pg`, `prev_season_off_anya`, `prev_season_def_anya_allowed` | Null in 1999 |
| Strength | `elo`, `season_avg_opp_elo` | 538-style Elo with fixed published constants (K=20, home-field advantage 65, 1/3 regression to the mean between seasons; no home bonus at neutral sites). Carries through relocations. |
| Form / schedule | `streak` (signed), `rest_days`, `off_bye`, `short_week` | |
| Head to head | `h2h_last_margin`, `h2h_last_won`, `h2h_days_since`, `h2h_prior_meetings` | Last meeting at either site |
| Coach | `coach_games_with_team`, `coach_new_this_season` | Tenure is counted from 1999, so long-tenured coaches are undercounted early in the data |
| QB | `qb_career_anya`, `qb_last8_anya`, `qb_season_anya`, `qb_career_starts`, `qb_career_games`, `qb_career_dropbacks`, `qb_new_vs_last_season`, `qb_changed_last_game` | QB = **expected starter**: whoever started the team's previous game. The actual starter of game D is treated as post-kickoff information. |
| Standings | `div_rank`, `div_pct_behind_leader` | Regular-season division standings as of the day before |
| Injury report | `inj_n_out`, `inj_n_doubtful`, `inj_n_questionable`, `inj_out_or_doubtful_{SKILL,OL,DL,LB,DB}`, `inj_exp_qb_out_or_doubtful` | Only report rows time-stamped before game day. **NaN before 2010** (2009 reports have no timestamps). |

Expected nulls: week 1 has no season-to-date or last-3 values (use `ewm_*`, `prev_season_*`, `elo`, and QB features); 1999 has no previous-season values; injuries are null for 1999–2009.

## Deliberately excluded (and why)

| Excluded | Why |
|---|---|
| EPA / any nflfastR model output | The expected-points model was fit on seasons after many of the games it scores, so it's indirect future information. ANY/A is used instead. |
| All roster files (season **and** weekly) | Season rosters are end-of-season snapshots. Pre-2017 weekly rosters were **rebuilt after the fact**: statuses are back-filled, so a player placed on IR in week 10 shows inactive from week 1. Only 5–14% of players change status in-season before 2017, vs 28–62% after. A roster-size feature built from them correlated r = 0.21 with margin in 2002–15 and about 0 after 2016: a leak no timestamp check can catch. |
| `nfl_players_master` fields | Status, team, and experience describe players as of today, not as of the game |
| Actual starting QB of the game | Not known with certainty until kickoff |
| Vegas lines | Market benchmark, kept in its own file |
| Injury reports without a timestamp, or stamped on/after game day | Can't prove they were public before kickoff |

## Source tables (inputs to the pipeline, not modeling files)

| File | Rows | Notes |
|---|---|---|
| `nfl_games_1999_2024` | 6,991 | One row per game: schedule, scores, Vegas lines, weather, QBs, coaches. **Contains outcomes.** |
| `nfl_player_offense_stats_weekly_1999_2024` | 134,470 | Per player-game box score with `game_id` (uses current franchise codes) |
| `nfl_player_defense_stats_weekly_1999_2024` | 239,955 | Per player-game defensive box score (not used by 03) |
| `nfl_players_master` | 24,830 | Player bio / ID crosswalk (current-state fields: not for features) |
| `nfl_season_rosters_1999_2024`, `nfl_weekly_rosters_*` | 63k / 860k | Kept for reference only. Not point-in-time; see above. |

Team codes: game and label files use the abbreviation in use at the time (OAK/SD/STL). Internally the pipeline maps every alternate code (OAK→LV, SD→LAC, STL→LA, and roster-era ARZ/BLT/CLV/HST/SL) to one franchise code.

## Using the data without introducing leakage at modeling time

The files are leak-free, but a modeling setup can still leak:

- **Split by time, never at random.** For example, train ≤ 2019, validate 2020–22, test 2023–24.
- Fit imputers, scalers, and feature selection on the training split only (use an sklearn `Pipeline`).
- Tune hyperparameters on the validation split, not the test split.
- If you use the long team file, keep both rows of a game in the same split.

## Anonymized LLM-prediction experiment

Start from `nfl_model_features_1999_2024`. Drop `game_id`, `gameday`, and the team names. Randomize which side is "Team A" per row: swap the `home_`/`away_` prefixes, negate the `diff_` columns, and flip `neutral_site` handling accordingly. Join the label from the labels file relabeled to Team A. The experiment script belongs in `experiments/`.
