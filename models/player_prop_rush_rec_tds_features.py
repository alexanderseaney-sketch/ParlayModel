"""
Player-prop feature engineering for combined rush + receiving touchdowns
(RB/WR/TE "anytime TD"-style prop). Same no-leakage discipline as the other
prop models: every feature is a strictly-prior rolling average
(shift(1).expanding() / shift(1).rolling(3)).

Unlike the yardage props, this one does NOT need a rolling-average proxy line.
Underdog's real posted line for rush_rec_tds is a constant 0.5 across
essentially every player (confirmed against the live pulled props: all 398
rows), so the proxy line is set to that same constant and the target is
simply "did they score at least one rushing or receiving TD" -- this proxy
matches the real market almost exactly rather than approximating it.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_TOUCHES_TO_QUALIFY = 3  # carries + targets, filters out garbage-time/inactive players
PROXY_LINE = 0.5


def build_rush_rec_tds_dataset(min_week: int = 4) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    skill = weekly[weekly["position"].isin(["RB", "WR", "TE"])].copy()
    skill["touches"] = skill["carries"] + skill["targets"]
    skill = skill[skill["touches"] >= MIN_TOUCHES_TO_QUALIFY]

    skill["rush_rec_tds"] = skill["rushing_tds"] + skill["receiving_tds"]

    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 "carries", "targets", "rushing_yards", "receiving_yards", "rush_rec_tds"]
    skill = skill[keep_cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for col in ["rush_rec_tds", "carries", "targets"]:
        skill[f"{col}_rolling"] = (
            skill.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        skill[f"{col}_last3"] = (
            skill.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    for col in ["rushing_yards", "receiving_yards"]:
        skill[f"{col}_rolling"] = (
            skill.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )

    # Opponent defense strength -- how many EPA/play they typically allow
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

    skill = skill.merge(matchups, on=["recent_team", "season", "week"], how="left")
    skill = skill.merge(
        defense[["team", "season", "week", "def_epa_allowed_rolling"]].rename(columns={"team": "opponent"}),
        on=["opponent", "season", "week"], how="left",
    )

    skill = skill[skill["week"] >= min_week].reset_index(drop=True)

    skill["proxy_line"] = PROXY_LINE
    skill["over_proxy_line"] = (skill["rush_rec_tds"] >= 1).astype(int)

    return skill


if __name__ == "__main__":
    df = build_rush_rec_tds_dataset()
    print(df.shape)
    print(df["over_proxy_line"].value_counts(normalize=True))
    print(df[["player_display_name", "season", "week", "rush_rec_tds", "rush_rec_tds_rolling",
              "carries_rolling", "targets_rolling", "over_proxy_line"]].dropna().head(10))
