#!/usr/bin/env bash
# Downloads raw source files from nflverse-data (public GitHub release assets)
# into data/raw/. Re-run this to refresh once nflverse updates player_stats
# past the 2024 season cutoff (see docs/data_dictionary.md).
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW/injuries" "$RAW/season_rosters" "$RAW/weekly_rosters"

BASE=https://github.com/nflverse/nflverse-data/releases/download

curl -sSL -o "$RAW/games.parquet"               "$BASE/schedules/games.parquet"
curl -sSL -o "$RAW/players.parquet"              "$BASE/players/players.parquet"
curl -sSL -o "$RAW/player_stats_offense.parquet" "$BASE/player_stats/player_stats.parquet"
curl -sSL -o "$RAW/player_stats_defense.parquet" "$BASE/player_stats/player_stats_def.parquet"

for y in $(seq 2002 2024); do
  curl -sSL -o "$RAW/weekly_rosters/roster_weekly_$y.parquet" "$BASE/weekly_rosters/roster_weekly_$y.parquet"
done

for y in $(seq 1999 2024); do
  curl -sSL -o "$RAW/season_rosters/roster_$y.parquet" "$BASE/rosters/roster_$y.parquet"
done

for y in $(seq 2009 2024); do
  curl -sSL -o "$RAW/injuries/injuries_$y.parquet" "$BASE/injuries/injuries_$y.parquet"
done

# 2025+: nflverse moved weekly player stats to a new release ("stats_player", renamed
# columns, sack yards negative). These feed modeling/predict.py, which maps them to the old
# format; the 1999-2024 research tables in data/processed/ don't use them.
# modeling/predict.py --refresh re-downloads these plus games.parquet and logs download times.
CUR=$(( $(date +%m) >= 8 ? $(date +%Y) : $(date +%Y) - 1 ))
mkdir -p "$RAW/player_stats_weekly"
for y in $(seq 2025 "$CUR"); do
  curl -sSL -o "$RAW/player_stats_weekly/stats_player_week_$y.parquet" "$BASE/stats_player/stats_player_week_$y.parquet"
  curl -sSL -o "$RAW/injuries/injuries_$y.parquet" "$BASE/injuries/injuries_$y.parquet"
done

echo "Done. Run scripts/02_build_core_tables.py next."
