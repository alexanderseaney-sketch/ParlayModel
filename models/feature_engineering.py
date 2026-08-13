"""
Builds pre-game team-week features from the raw nflverse pulls, then joins home/away
features onto each scheduled game.

Every feature here is computed using ONLY data from games strictly before the game being
predicted (rolling season-to-date average, shifted by one game) — this is the same
no-leakage discipline as the Elo backtest, just applied to a richer feature set.
"""
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def build_team_week_offense(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregates player-week stats up to team-week offensive production."""
    for col in ["passing_epa", "rushing_epa", "receiving_epa"]:
        weekly_stats[col] = weekly_stats[col].fillna(0)

    team_week = weekly_stats.groupby(["recent_team", "season", "week"]).agg(
        off_epa=("passing_epa", lambda s: s.sum()),
    ).reset_index()

    # total EPA = sum of passing + rushing + receiving EPA across all players that team-week
    epa_sum = weekly_stats.groupby(["recent_team", "season", "week"])[
        ["passing_epa", "rushing_epa", "receiving_epa"]
    ].sum().sum(axis=1).reset_index(name="off_epa_total")

    team_week = team_week.drop(columns=["off_epa"]).merge(
        epa_sum, on=["recent_team", "season", "week"]
    )
    return team_week.rename(columns={"recent_team": "team"})


def build_team_week_defense(offense: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """A team's defensive EPA allowed = the opponent's offensive EPA that week."""
    games = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team"]
    ].copy()

    home_side = games.rename(columns={"home_team": "team", "away_team": "opponent"})
    away_side = games.rename(columns={"away_team": "team", "home_team": "opponent"})
    matchups = pd.concat([home_side, away_side], ignore_index=True)

    matchups = matchups.merge(
        offense.rename(columns={"team": "opponent", "off_epa_total": "def_epa_allowed"}),
        on=["opponent", "season", "week"], how="left",
    )
    return matchups[["team", "season", "week", "def_epa_allowed"]]


def build_team_week_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    """Counts players listed as Out/Doubtful/Questionable per team-week — a crude but
    real proxy for team health going into a game."""
    injuries = injuries.drop_duplicates(subset=["gsis_id", "season", "week"])
    concerning = injuries[injuries["report_status"].isin(["Out", "Doubtful", "Questionable"])]
    counts = concerning.groupby(["team", "season", "week"]).size().reset_index(name="injury_count")
    return counts


def build_team_week_ngs(ngs_passing: pd.DataFrame, ngs_rushing: pd.DataFrame, ngs_receiving: pd.DataFrame) -> pd.DataFrame:
    """Team-week averages of the Next Gen Stats that most plausibly carry predictive
    signal beyond box-score EPA: CPOE (passing efficiency vs. expectation) and average
    separation (receiving — open receivers = easier offense)."""
    passing = ngs_passing.groupby(["team_abbr", "season", "week"]).agg(
        cpoe=("completion_percentage_above_expectation", "mean"),
    ).reset_index().rename(columns={"team_abbr": "team"})

    receiving = ngs_receiving.groupby(["team_abbr", "season", "week"]).agg(
        avg_separation=("avg_separation", "mean"),
    ).reset_index().rename(columns={"team_abbr": "team"})

    merged = passing.merge(receiving, on=["team", "season", "week"], how="outer")
    return merged


def add_rolling_pregame_features(team_week: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """For each feature, computes the team's season-to-date average using only STRICTLY
    PRIOR weeks (shift(1) before the expanding mean) — this is what makes it usable as a
    pre-game prediction feature without leaking the current week's own result into itself."""
    team_week = team_week.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in feature_cols:
        team_week[f"{col}_rolling"] = (
            team_week.groupby(["team", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    return team_week


def build_game_features(min_week: int = 3) -> pd.DataFrame:
    """Full pipeline: builds the game-level dataset with pre-game rolling features for
    both home and away teams. min_week=3 drops the first two weeks of each season, where
    rolling averages are built from too little data to mean much."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    weekly_stats = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    ngs_passing = pd.read_csv(os.path.join(RAW_DIR, "ngs_passing.csv"))
    ngs_rushing = pd.read_csv(os.path.join(RAW_DIR, "ngs_rushing.csv"))
    ngs_receiving = pd.read_csv(os.path.join(RAW_DIR, "ngs_receiving.csv"))

    offense = build_team_week_offense(weekly_stats)
    defense = build_team_week_defense(offense, schedules)
    inj = build_team_week_injuries(injuries)
    ngs = build_team_week_ngs(ngs_passing, ngs_rushing, ngs_receiving)

    team_week = offense.merge(defense, on=["team", "season", "week"], how="outer")
    team_week = team_week.merge(inj, on=["team", "season", "week"], how="left")
    team_week = team_week.merge(ngs, on=["team", "season", "week"], how="left")
    team_week["injury_count"] = team_week["injury_count"].fillna(0)

    feature_cols = ["off_epa_total", "def_epa_allowed", "injury_count", "cpoe", "avg_separation"]
    team_week = add_rolling_pregame_features(team_week, feature_cols)

    rolling_cols = [f"{c}_rolling" for c in feature_cols]
    team_week_pregame = team_week[["team", "season", "week"] + rolling_cols]

    games = schedules[schedules["game_type"] == "REG"].copy()
    games = games.dropna(subset=["home_score", "away_score"])
    games = games[games["week"] >= min_week]

    games = games.merge(
        team_week_pregame.rename(columns={c: f"home_{c}" for c in rolling_cols} | {"team": "home_team"}),
        on=["home_team", "season", "week"], how="left",
    )
    games = games.merge(
        team_week_pregame.rename(columns={c: f"away_{c}" for c in rolling_cols} | {"team": "away_team"}),
        on=["away_team", "season", "week"], how="left",
    )

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    for c in rolling_cols:
        home_col, away_col = f"home_{c}", f"away_{c}"
        games[f"diff_{c}"] = games[home_col] - games[away_col]

    return games


if __name__ == "__main__":
    df = build_game_features()
    print(df.shape)
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    print("Feature columns:", diff_cols)
    print("Null counts:\n", df[diff_cols].isna().sum())
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "game_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
