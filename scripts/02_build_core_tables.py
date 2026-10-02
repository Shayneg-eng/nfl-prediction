"""
Builds the core dataset tables from data/raw/ into data/processed/, trimmed
to seasons 1999-2024 (the last season with complete player-level box scores
upstream — see docs/data_dictionary.md "Known limitations").

Handles the one real gotcha in the raw data: player_stats uses each
franchise's CURRENT abbreviation for all of history (e.g. Raiders are "LV"
even in 1999 game logs), while games.parquet uses the historical
abbreviation for the season it happened (OAK/SD/STL). Both are normalized
to the current code purely for joining; the games table keeps the
historical abbreviation for display.

Run after scripts/01_download_raw_data.sh. Produces:
  nfl_games_1999_2024.{csv,parquet}
  nfl_players_master.{csv,parquet}
  nfl_season_rosters_1999_2024.parquet
  nfl_weekly_rosters_2002_2024.parquet
  nfl_player_offense_stats_weekly_1999_2024.parquet
  nfl_player_defense_stats_weekly_1999_2024.parquet
"""
import glob
import pandas as pd

RAW = "data/raw"
OUT = "data/processed"
CUTOFF = 2024

TEAM_NORM = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def norm(col):
    return col.map(lambda t: TEAM_NORM.get(t, t))


def _fix_mixed_type_columns(df):
    """The raw roster files store a couple of numeric-looking columns
    (jersey_number, draft_number) as object dtype with a mix of int/float/str
    across seasons, which the local fastparquet engine can't serialize.
    Coerce them to a single numeric dtype (NaN where missing/non-numeric)."""
    for col in ("jersey_number", "draft_number"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    games = pd.read_parquet(f"{RAW}/games.parquet")
    games = games[games.season <= CUTOFF].copy()
    games.to_csv(f"{OUT}/nfl_games_1999_2024.csv", index=False)
    games.to_parquet(f"{OUT}/nfl_games_1999_2024.parquet", index=False)
    print("games:", games.shape)

    players = pd.read_parquet(f"{RAW}/players.parquet")
    players.to_csv(f"{OUT}/nfl_players_master.csv", index=False)
    players.to_parquet(f"{OUT}/nfl_players_master.parquet", index=False)
    print("players:", players.shape)

    sr = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/season_rosters/roster_*.parquet"))],
        ignore_index=True,
    )
    sr = sr[sr.season <= CUTOFF]
    sr = _fix_mixed_type_columns(sr)
    sr.to_parquet(f"{OUT}/nfl_season_rosters_1999_2024.parquet", index=False)
    print("season rosters:", sr.shape)

    wr = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/weekly_rosters/roster_weekly_*.parquet"))],
        ignore_index=True,
    )
    wr = wr[wr.season <= CUTOFF]
    wr = _fix_mixed_type_columns(wr)
    # split into three season-range files so no single file exceeds ~20MB
    # (keeps every file small enough to transfer to a connected local folder)
    for lo, hi in [(2002, 2010), (2011, 2017), (2018, 2024)]:
        chunk = wr[(wr.season >= lo) & (wr.season <= hi)]
        chunk.to_parquet(f"{OUT}/nfl_weekly_rosters_{lo}_{hi}.parquet", index=False)
        print(f"weekly rosters {lo}-{hi}:", chunk.shape)

    # game_id lookup, keyed on the CURRENT franchise code so it joins cleanly
    # against player_stats (which is coded that way for all history)
    home = games[["game_id", "season", "week", "home_team", "away_team", "gameday"]].copy()
    home["team_norm"] = norm(home["home_team"])
    home = home.rename(columns={"away_team": "opponent_team_hist"})[
        ["game_id", "season", "week", "team_norm", "opponent_team_hist", "gameday"]
    ]

    away = games[["game_id", "season", "week", "away_team", "home_team", "gameday"]].copy()
    away["team_norm"] = norm(away["away_team"])
    away = away.rename(columns={"home_team": "opponent_team_hist"})[
        ["game_id", "season", "week", "team_norm", "opponent_team_hist", "gameday"]
    ]
    team_game_lookup = pd.concat([home, away], ignore_index=True)

    ps_off = pd.read_parquet(f"{RAW}/player_stats_offense.parquet").rename(columns={"recent_team": "team"})
    ps_def = pd.read_parquet(f"{RAW}/player_stats_defense.parquet")
    ps_off["team_norm"] = ps_off["team"]
    ps_def["team_norm"] = ps_def["team"]

    ps_off = ps_off.merge(
        team_game_lookup[["game_id", "season", "week", "team_norm", "gameday"]],
        on=["season", "week", "team_norm"], how="left",
    ).drop(columns=["team_norm"])
    ps_def = ps_def.merge(
        team_game_lookup[["game_id", "season", "week", "team_norm", "gameday"]],
        on=["season", "week", "team_norm"], how="left",
    ).drop(columns=["team_norm"])

    ps_off = ps_off[ps_off.season <= CUTOFF]
    ps_def = ps_def[ps_def.season <= CUTOFF]

    print("player offense:", ps_off.shape, "game_id match rate:", ps_off.game_id.notna().mean())
    print("player defense:", ps_def.shape, "game_id match rate:", ps_def.game_id.notna().mean())

    ps_off.to_parquet(f"{OUT}/nfl_player_offense_stats_weekly_1999_2024.parquet", index=False)
    ps_def.to_parquet(f"{OUT}/nfl_player_defense_stats_weekly_1999_2024.parquet", index=False)


if __name__ == "__main__":
    main()
