"""
Game-context features usable across ALL player-prop models: Vegas total/spread (as a
game-script/pace proxy — higher total = more expected plays = more opportunity across
the board), weather, rest, and home/away. None of these are rolling — they're known
pre-game directly from the schedule, same as in the team-level game-winner model, but
never applied to player props until now.

Also builds a gsis_id <-> pfr_id crosswalk (via nfl_data_py's import_ids) to unlock
snap share as a feature — previously skipped because snap_counts uses pfr_player_id
while weekly_stats uses gsis player_id, two different ID systems.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def build_game_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team-week: that team's own perspective on total/spread/weather/rest."""
    home = schedules[["game_id", "season", "week", "home_team", "away_team", "total_line",
                       "spread_line", "temp", "wind", "roof", "home_rest", "away_rest"]].copy()
    home["team"] = home["home_team"]
    home["is_home"] = 1
    home["team_implied_total"] = (home["total_line"] / 2) - (home["spread_line"] / 2)
    home["rest_days"] = home["home_rest"]

    away = schedules[["game_id", "season", "week", "home_team", "away_team", "total_line",
                       "spread_line", "temp", "wind", "roof", "home_rest", "away_rest"]].copy()
    away["team"] = away["away_team"]
    away["is_home"] = 0
    away["team_implied_total"] = (away["total_line"] / 2) + (away["spread_line"] / 2)
    away["rest_days"] = away["away_rest"]

    context = pd.concat([home, away], ignore_index=True)

    # For an unplayed game, schedules.csv's temp/wind are NaN (nflverse only records
    # ACTUAL observed conditions, filled in after the game). Prefer a real forecast
    # (data/pull_weather_forecast.py) when one's been pulled for that game -- only
    # falls through to the median/0 placeholder below when neither a played-game
    # value nor a forecast exists yet (game too far out, or forecast not pulled).
    forecast_path = os.path.join(RAW_DIR, "weather_forecast.csv")
    if os.path.exists(forecast_path):
        forecast = pd.read_csv(forecast_path, usecols=["game_id", "temp_forecast", "wind_forecast"])
        context = context.merge(forecast, on="game_id", how="left")
        context["temp"] = context["temp"].fillna(context["temp_forecast"])
        context["wind"] = context["wind"].fillna(context["wind_forecast"])
        context = context.drop(columns=["temp_forecast", "wind_forecast"])

    context["temp"] = context["temp"].fillna(context["temp"].median())
    context["wind"] = context["wind"].fillna(0)
    context["is_dome"] = context["roof"].isin(["dome", "closed"]).astype(int)

    return context[["team", "season", "week", "is_home", "team_implied_total",
                     "rest_days", "temp", "wind", "is_dome"]]


def build_gsis_to_pfr_crosswalk() -> pd.DataFrame:
    import nfl_data_py as nfl
    ids = nfl.import_ids()
    return ids[["gsis_id", "pfr_id"]].dropna().rename(columns={"gsis_id": "player_id"})


def add_snap_share(df: pd.DataFrame, id_col: str = "player_id") -> pd.DataFrame:
    """Adds rolling snap share (offense_pct) to a player-week dataframe that already has
    player_id (gsis), season, week."""
    crosswalk = build_gsis_to_pfr_crosswalk()
    snaps = pd.read_csv(os.path.join(RAW_DIR, "snap_counts.csv"), low_memory=False)
    snaps = snaps.drop_duplicates(subset=["pfr_player_id", "season", "week"])
    snaps = snaps.merge(crosswalk.rename(columns={"pfr_id": "pfr_player_id"}), on="pfr_player_id", how="inner")
    snaps = snaps.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    snaps["offense_pct_rolling"] = (
        snaps.groupby(["player_id", "season"])["offense_pct"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    return df.merge(
        snaps[["player_id", "season", "week", "offense_pct_rolling"]],
        on=[id_col, "season", "week"], how="left",
    )


if __name__ == "__main__":
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    context = build_game_context(schedules)
    print(context.shape)
    print(context.head())

    crosswalk = build_gsis_to_pfr_crosswalk()
    print(f"\nID crosswalk: {len(crosswalk)} players mapped gsis<->pfr")
