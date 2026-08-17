"""
Shared infrastructure for period-scoped props (period_first_touchdown_scored,
period_1_*, period_1_2_*) -- deliberately built last, since these are a
different SHAPE of question than everything else here. Every other model
answers "will this stat beat a threshold"; these answer "will this happen
within a slice of the game" (a specific quarter/half) or "will this specific
player be the one who does it first, out of everyone on the field."

Requires the full pbp.csv schema (qtr, order_sequence, td_player_id,
rusher/receiver/passer_player_id, per-play yards) -- the same full pull
brought in for the sacks model.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

PBP_PERIOD_COLS = [
    "game_id", "season", "week", "qtr", "order_sequence", "posteam", "defteam",
    "touchdown", "rush_touchdown", "pass_touchdown", "td_player_id",
    "rusher_player_id", "receiver_player_id", "passer_player_id",
    "rushing_yards", "receiving_yards", "passing_yards",
]


def _load_pbp_period_cols() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DIR, "pbp.csv"), usecols=PBP_PERIOD_COLS, low_memory=False)


def build_first_td_scorer_table() -> pd.DataFrame:
    """One row per game: the game_id and the player_id who scored the first
    touchdown of that game (whichever team), ordered by real play sequence."""
    pbp = _load_pbp_period_cols()
    tds = pbp[pbp["touchdown"] == 1].dropna(subset=["td_player_id"])
    first_td = (
        tds.sort_values(["game_id", "order_sequence"])
        .groupby("game_id").first().reset_index()
    )
    return first_td[["game_id", "season", "week", "td_player_id"]].rename(
        columns={"td_player_id": "first_td_player_id"})


def build_period_totals(period: str) -> pd.DataFrame:
    """Per-player, per-game rush/rec/pass yards and TDs scoped to a slice of the
    game. period: "q1" (1st quarter only) or "h1" (1st half, quarters 1-2)."""
    pbp = _load_pbp_period_cols()
    qtrs = [1] if period == "q1" else [1, 2]
    pbp = pbp[pbp["qtr"].isin(qtrs)]

    rush = pbp.dropna(subset=["rusher_player_id"]).groupby(
        ["rusher_player_id", "game_id", "season", "week"]
    ).agg(
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})

    rec = pbp.dropna(subset=["receiver_player_id"]).groupby(
        ["receiver_player_id", "game_id", "season", "week"]
    ).agg(
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})

    passing = pbp.dropna(subset=["passer_player_id"]).groupby(
        ["passer_player_id", "game_id", "season", "week"]
    ).agg(
        passing_yards=("passing_yards", "sum"),
        passing_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"passer_player_id": "player_id"})

    totals = rush.merge(rec, on=["player_id", "game_id", "season", "week"], how="outer")
    totals = totals.merge(passing, on=["player_id", "game_id", "season", "week"], how="outer")
    for col in ["rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds",
                "passing_yards", "passing_tds"]:
        totals[col] = totals[col].fillna(0)
    totals["rush_rec_tds"] = totals["rushing_tds"] + totals["receiving_tds"]

    return totals


def game_id_lookup() -> pd.DataFrame:
    """recent_team/season/week -> game_id, for joining player-week feature rows
    (weekly_stats grain) to pbp's game_id grain."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    home = schedules[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "recent_team"})
    away = schedules[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "recent_team"})
    return pd.concat([home, away], ignore_index=True)
