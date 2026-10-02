# NFL Game Prediction

An NFL game-outcome prediction pipeline built on [nflverse](https://github.com/nflverse) data,
with a strong emphasis on **avoiding data leakage** and benchmarking honestly against the
Vegas market.

## What makes it careful

- **Strict point-in-time separation.** Source tables (which contain outcomes) are never trained
  on directly. Only **pre-game features** go into the model; outcomes live in separate label
  tables joined by `game_id`. See [`ORGANIZATION.md`](ORGANIZATION.md) for the full data contract.
- **Leakage is verified, not assumed.** `scripts/04_verify_no_leakage.py` and
  `docs/leakage_verification_report.md` check that no post-game information leaks into features.
- **The market is the benchmark.** Models are compared against Vegas lines
  (`nfl_market_benchmark_*`), because beating a naive baseline is easy and beating the market is
  the real test.

## Pipeline

```
scripts/01_download_raw_data.sh      # pull raw nflverse data (→ data/raw/)
scripts/02_build_core_tables.py      # derive clean source tables (→ data/processed/)
scripts/03_build_pregame_features.py # build pre-game-only feature + label tables
scripts/04_verify_no_leakage.py      # assert features contain no future information

modeling/01_validate.py              # cross-validation
modeling/02_final_test.py            # held-out test
modeling/03_fit_production.py         # fit the production model
modeling/04_market_residual_experiment.py  # model the residual vs. the Vegas line
modeling/05_attention_experiment.py  # attention-based variant
modeling/predict.py                  # generate predictions
```

See `modeling/README.md` for modeling details and `docs/` for the data dictionary and audits.

## Data

The `data/` directory (raw + processed, ~128 MB) is **not committed**. Rebuild it with
`scripts/01_download_raw_data.sh` — see [`DATA.md`](DATA.md).

## Run it

```bash
python -m pip install pandas numpy scikit-learn pyarrow
bash scripts/01_download_raw_data.sh
python scripts/02_build_core_tables.py
python scripts/03_build_pregame_features.py
python modeling/01_validate.py
```
