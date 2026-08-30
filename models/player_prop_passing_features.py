"""
Player-prop feature engineering for passing yards (QB-focused). Same no-leakage
discipline as receiving/rushing: strictly-prior rolling averages only.

Same honest limitation as the other prop models: backtested against the player's own
trailing average as a proxy line, since no historical Underdog line archive exists yet.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_ATTEMPTS_TO_QUALIFY = 10  # filters out garbage-time/emergency QB appearances


def build_passing_yards_dataset(min_week: int = 4) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    ngs_passing = pd.read_csv(os.path.join(RAW_DIR, "ngs_passing.csv"))
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
        qb[f"{col}_rolling"] = (
            qb.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        qb[f"{col}_last3"] = (
            qb.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    # NGS passing: CPOE and avg intended air yards — skill + aggression signals
    ngs = ngs_passing.rename(columns={"player_gsis_id": "player_id"})
    ngs = ngs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["completion_percentage_above_expectation", "avg_intended_air_yards",
                "aggressiveness", "avg_time_to_throw"]:
        ngs[f"{col}_rolling"] = (
            ngs.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    ngs_cols = ["player_id", "season", "week"] + [
        f"{c}_rolling" for c in ["completion_percentage_above_expectation", "avg_intended_air_yards",
                                   "aggressiveness", "avg_time_to_throw"]
    ]
    qb = qb.merge(ngs[ngs_cols], on=["player_id", "season", "week"], how="left")

    # Opponent's pass defense strength
    from feature_engineering import build_team_week_offense, build_team_week_defense
    defense = build_team_week_defense(build_team_week_offense(weekly), schedules)
    defense = defense.sort_values(["team", "season", "week"]).reset_index(drop=True)
    defense["def_epa_allowed_rolling"] = (
        defense.groupby(["team", "season"])["def_epa_allowed"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

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
    qb = qb[qb["attempts"] >= MIN_ATTEMPTS_TO_QUALIFY].reset_index(drop=True)

    qb["proxy_line"] = qb["passing_yards_rolling"]
    qb["over_proxy_line"] = (qb["passing_yards"] > qb["proxy_line"]).astype(int)

    return qb


if __name__ == "__main__":
    df = build_passing_yards_dataset()
    print(df.shape)
    print(df[["player_display_name", "season", "week", "passing_yards", "passing_yards_rolling",
              "completion_percentage_above_expectation_rolling", "over_proxy_line"]].dropna().head(10))
