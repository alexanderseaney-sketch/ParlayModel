"""
Feature engineering for period_1_rush_rec_tds (1st quarter) and
period_1_2_rush_rec_tds (1st half) -- quarter/half-scoped versions of
rush_rec_tds. Real Underdog line is a constant 0.5 for both (confirmed
against live pulled props), same anytime-TD structure as the full-game
version, so the same constant-proxy convention applies.

Reuses rush_rec_tds's own rolling features as predictors (own full-game
scoring rate, touches, opponent defense) -- a player's role/opportunity
should carry similar signal for "scores within the first quarter/half" as
it does for "scores at some point in the game." Target comes from
period_features.build_period_totals(), which sums rushing/receiving TDs
from pbp.csv scoped to the relevant quarters.
"""
import pandas as pd

from player_prop_rush_rec_tds_features import build_rush_rec_tds_dataset
from period_features import build_period_totals, game_id_lookup

PROXY_LINE = 0.5


def build_period_rush_rec_tds_dataset(period: str, min_week: int = 4) -> pd.DataFrame:
    """period: "q1" or "h1" (see period_features.build_period_totals)."""
    df = build_rush_rec_tds_dataset(min_week=min_week)

    games = game_id_lookup()
    df = df.merge(games, on=["recent_team", "season", "week"], how="left")
    df = df.dropna(subset=["game_id"])

    totals = build_period_totals(period)
    df = df.merge(
        totals[["player_id", "game_id", "rush_rec_tds"]].rename(columns={"rush_rec_tds": "period_rush_rec_tds"}),
        on=["player_id", "game_id"], how="left",
    )
    df["period_rush_rec_tds"] = df["period_rush_rec_tds"].fillna(0.0)

    df["proxy_line"] = PROXY_LINE
    df["over_proxy_line"] = (df["period_rush_rec_tds"] >= 1).astype(int)

    return df


if __name__ == "__main__":
    for period in ["q1", "h1"]:
        df = build_period_rush_rec_tds_dataset(period)
        print(period, df.shape, df["over_proxy_line"].mean())
