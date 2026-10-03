"""
Player-prop feature engineering for passing yards (QB-focused). Same no-leakage
discipline as receiving/rushing: strictly-prior rolling averages only.

Same honest limitation as the other prop models: backtested against the player's own
trailing average as a proxy line, since no historical Underdog line archive exists yet.
"""
import os

import pandas as pd

from feature_engineering import cross_season_rolling, load_ngs, merge_ngs, qualify

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_ATTEMPTS_TO_QUALIFY = 10  # filters out garbage-time/emergency QB appearances

# See feature_engineering.cross_season_rolling's docstring for why these are bounded
# and cross a season boundary rather than resetting there.
_ROLLING_WINDOW = 16
_TEAM_WINDOW = 8


def build_passing_yards_dataset(min_week: int = 1) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    ngs_passing = load_ngs("passing")  # drops week-0 season totals (look-ahead)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    qb = weekly[weekly["position"] == "QB"].copy()

    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 "attempts", "passing_yards", "passing_tds", "interceptions"]
    qb = qb[keep_cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Deliberately NOT pre-filtered to attempts >= MIN_ATTEMPTS_TO_QUALIFY here -- same
    # bug as player_prop_features.py's receiving-yards dataset (fixed 2026-08-23):
    # filtering low-attempt games out before this rolling average meant a QB's own
    # baseline silently excluded their mop-up/emergency games, not just this week's
    # label. Applied below instead, after the rolling features are computed.
    for col in ["passing_yards", "attempts", "passing_tds", "interceptions"]:
        qb[f"{col}_rolling"] = cross_season_rolling(qb, "player_id", col, window=_ROLLING_WINDOW)
        qb[f"{col}_last3"] = cross_season_rolling(qb, "player_id", col, window=3)

    # NGS passing: CPOE and avg intended air yards — skill + aggression signals
    ngs = ngs_passing.rename(columns={"player_gsis_id": "player_id"})
    ngs = ngs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["completion_percentage_above_expectation", "avg_intended_air_yards",
                "aggressiveness", "avg_time_to_throw"]:
        ngs[f"{col}_rolling"] = cross_season_rolling(ngs, "player_id", col, window=_ROLLING_WINDOW)
    ngs_cols = ["player_id", "season", "week"] + [
        f"{c}_rolling" for c in ["completion_percentage_above_expectation", "avg_intended_air_yards",
                                   "aggressiveness", "avg_time_to_throw"]
    ]
    qb = merge_ngs(qb, ngs[ngs_cols])

    # Opponent's pass defense strength
    from feature_engineering import build_team_week_offense, build_team_week_defense
    defense = build_team_week_defense(build_team_week_offense(weekly), schedules)
    defense = defense.sort_values(["team", "season", "week"]).reset_index(drop=True)
    defense["def_epa_allowed_rolling"] = cross_season_rolling(defense, "team", "def_epa_allowed", window=_TEAM_WINDOW)

    home = schedules[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "recent_team", "away_team": "opponent"})
    away = schedules[["season", "week", "home_team", "away_team"]].rename(
        columns={"away_team": "recent_team", "home_team": "opponent"})
    matchups = pd.concat([home, away], ignore_index=True).drop_duplicates()

    qb = qb.merge(matchups, on=["recent_team", "season", "week"], how="left")
    qb = qb.merge(
        defense[["team", "season", "week", "def_epa_allowed_rolling"]].rename(columns={"team": "opponent"}),
        on=["opponent", "season", "week"], how="left",
    )

    qb = qb[qb["week"] >= min_week].reset_index(drop=True)

    # Applied here, after every rolling/merge step above, not on the raw weekly log --
    # see the matching comment where qb is first built.
    qb = qualify(qb, "attempts", "attempts_rolling", MIN_ATTEMPTS_TO_QUALIFY)

    qb["proxy_line"] = qb["passing_yards_rolling"]
    qb["over_proxy_line"] = (qb["passing_yards"] > qb["proxy_line"]).astype(int)

    return qb


if __name__ == "__main__":
    df = build_passing_yards_dataset()
    print(df.shape)
    print(df[["player_display_name", "season", "week", "passing_yards", "passing_yards_rolling",
              "completion_percentage_above_expectation_rolling", "over_proxy_line"]].dropna().head(10))
