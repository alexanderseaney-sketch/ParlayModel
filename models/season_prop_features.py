"""
Feature engineering for season-long cumulative props (season_receiving_yards,
season_rec_tds, season_rush_yards, season_rush_tds, season_pass_yards,
season_pass_tds) -- Underdog's second-largest gap in model coverage after
rush_rec_tds (658 combined rows across these 6 types).

Fundamentally different grain from the other prop models: one row per
PLAYER-SEASON instead of per player-game. Same "no historical Underdog line
archive" honesty constraint still applies, so the proxy line is the player's
own PRIOR season's total for that exact stat -- the season-level equivalent
of the rolling-average proxy used everywhere else. This is a cruder proxy
than a rolling game average (one season is a single noisy sample, and trades/
injuries/role changes swing year-over-year season totals much harder than
they swing a 3-game rolling average), so this is validated the same way as
every other prop here: real leave-one-season-out backtest, checked against
the base rate, not assumed to work just because the pattern was copied.

Regular-season games only (season_type == "REG") -- these are full-season
props, not stats through some particular week.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_PRIOR_GAMES = 4  # filters out prior seasons too short (injury/backup) to mean much as a predictor

STAT_COLS = [
    "receiving_yards", "receiving_tds", "receptions", "targets",
    "rushing_yards", "rushing_tds", "carries",
    "passing_yards", "passing_tds", "attempts", "completions",
]


def _aggregate_player_seasons() -> pd.DataFrame:
    """One row per player-season: raw totals plus games-played and rate stats.
    Shared building block for both the historical training pairs and the
    current single-season projection."""
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    weekly = weekly[weekly["season_type"] == "REG"].copy()

    agg = weekly.groupby(["player_id", "player_display_name", "position", "season"]).agg(
        games_played=("week", "nunique"),
        recent_team=("recent_team", "last"),
        **{col: (col, "sum") for col in STAT_COLS},
    ).reset_index()

    agg["yards_per_game_rec"] = agg["receiving_yards"] / agg["games_played"]
    agg["yards_per_target"] = agg["receiving_yards"] / agg["targets"].replace(0, pd.NA)
    agg["yards_per_game_rush"] = agg["rushing_yards"] / agg["games_played"]
    agg["yards_per_carry"] = agg["rushing_yards"] / agg["carries"].replace(0, pd.NA)
    agg["yards_per_game_pass"] = agg["passing_yards"] / agg["games_played"]
    agg["yards_per_attempt"] = agg["passing_yards"] / agg["attempts"].replace(0, pd.NA)

    return agg


RATE_COLS = ["yards_per_game_rec", "yards_per_target", "yards_per_game_rush",
             "yards_per_carry", "yards_per_game_pass", "yards_per_attempt"]
PRIOR_COLS = STAT_COLS + ["games_played"] + RATE_COLS


def _load_experience_draft() -> pd.DataFrame:
    """Player experience/draft capital -- real, well-established signals in fantasy/
    dynasty analytics (breakout-age and aging-curve effects; draft round/pick as a
    proxy for the talent/opportunity level a team originally saw in this player,
    distinct from what they've shown so far) that this project's prior-season-totals-
    only season-prop features never captured. Validated via the same leave-one-season-
    out backtest as everything else: a real, consistent gain across all 4 season
    props (biggest: season_receiving_yards's >=0.4-confidence accuracy 72.3%->77.7%,
    with MORE games clearing that bar too, not a coverage/accuracy trade-off).

    draft_round/draft_pick filled for undrafted players with a value WORSE than any
    real late-round pick (Mr. Irrelevant is round 7, pick ~260) rather than left NaN
    or 0 -- 0 would read to a model as "1st overall pick," the opposite of undrafted."""
    players = pd.read_csv(os.path.join(RAW_DIR, "players.csv"), low_memory=False)
    players = players.rename(columns={"gsis_id": "player_id"})
    players = players[["player_id", "rookie_season", "draft_round", "draft_pick"]].drop_duplicates("player_id")
    players["draft_round"] = players["draft_round"].fillna(8)
    players["draft_pick"] = players["draft_pick"].fillna(300)
    return players


EXPERIENCE_DRAFT_FEATURES = ["years_experience", "draft_round", "draft_pick"]


def build_season_prop_training_dataset(stat_col: str) -> pd.DataFrame:
    """Historical (prior season -> target season) pairs across every
    consecutive season available, for backtesting and training. stat_col
    must be one of STAT_COLS (e.g. "receiving_yards")."""
    seasons = _aggregate_player_seasons()

    prior = seasons.rename(columns={c: f"prior_{c}" for c in PRIOR_COLS})
    prior = prior.rename(columns={"season": "prior_season"})
    prior["target_season"] = prior["prior_season"] + 1
    prior = prior[prior["prior_games_played"] >= MIN_PRIOR_GAMES]

    target = seasons[["player_id", "season", stat_col]].rename(
        columns={"season": "target_season", stat_col: f"target_{stat_col}"})

    # Position comes from the PRIOR season only -- matches the inference path, which
    # obviously can't know the target season's position since that season hasn't happened.
    df = prior.merge(target, on=["player_id", "target_season"], how="inner")

    df["proxy_line"] = df[f"prior_{stat_col}"]
    df["over_proxy_line"] = (df[f"target_{stat_col}"] > df["proxy_line"]).astype(int)

    exp_draft = _load_experience_draft()
    df = df.merge(exp_draft, on="player_id", how="left")
    df["years_experience"] = df["target_season"] - df["rookie_season"]

    return df


def build_season_prop_current_dataset(stat_col: str) -> pd.DataFrame:
    """Single row per player using the most recent complete season as
    "prior" -- what current_predictions.py scores against. No target season
    exists yet (that's the point), so there's no over_proxy_line here."""
    seasons = _aggregate_player_seasons()
    latest_season = seasons["season"].max()

    prior = seasons[seasons["season"] == latest_season].copy()
    prior = prior[prior["games_played"] >= MIN_PRIOR_GAMES]
    prior = prior.rename(columns={c: f"prior_{c}" for c in PRIOR_COLS})
    prior["week"] = 18  # last regular-season week -- "as of" marker, not a specific rolling-stats week

    prior["proxy_line"] = prior[f"prior_{stat_col}"]

    exp_draft = _load_experience_draft()
    prior = prior.merge(exp_draft, on="player_id", how="left")
    prior["years_experience"] = (latest_season + 1) - prior["rookie_season"]

    return prior


if __name__ == "__main__":
    df = build_season_prop_training_dataset("receiving_yards")
    print(df.shape)
    print(df["over_proxy_line"].value_counts(normalize=True))
    print(df[["player_display_name", "prior_season", "target_season",
              "prior_receiving_yards", "target_receiving_yards", "over_proxy_line"]].dropna().head(10))
