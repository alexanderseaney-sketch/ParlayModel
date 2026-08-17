"""
Feature engineering for period_first_touchdown_scored -- "will THIS player
score the game's first touchdown," out of everyone on the field. Real
Underdog line for this is a constant 0.5 (confirmed against live pulled
props), same anytime-TD-style market structure as rush_rec_tds, so the same
constant proxy_line=0.5 convention applies -- no line-mismatch problem like
the season props hit.

Different from rush_rec_tds in one important way: only ONE player in the
entire game can be positive here, so the base rate is roughly 1-in-30 to
1-in-40 (however many skill players see the field), not "however many
players happen to score at least one TD." Reuses rush_rec_tds's own rolling
features as the predictor set -- a player's share of their team's scoring
opportunity should carry over reasonably well from "scores a TD at all" to
"scores the game's FIRST TD."
"""
import os

import pandas as pd

from player_prop_rush_rec_tds_features import build_rush_rec_tds_dataset
from period_features import build_first_td_scorer_table, game_id_lookup

PROXY_LINE = 0.5


def build_first_td_scorer_dataset(min_week: int = 4) -> pd.DataFrame:
    df = build_rush_rec_tds_dataset(min_week=min_week)

    games = game_id_lookup()
    df = df.merge(games, on=["recent_team", "season", "week"], how="left")
    df = df.dropna(subset=["game_id"])

    first_td = build_first_td_scorer_table()
    df = df.merge(first_td[["game_id", "first_td_player_id"]], on="game_id", how="left")

    df["proxy_line"] = PROXY_LINE
    df["over_proxy_line"] = (df["player_id"] == df["first_td_player_id"]).astype(int)

    return df


if __name__ == "__main__":
    df = build_first_td_scorer_dataset()
    print(df.shape)
    print(df["over_proxy_line"].value_counts(normalize=True))
    print(df[df["over_proxy_line"] == 1][
        ["player_display_name", "season", "week", "rush_rec_tds_rolling", "carries_rolling", "targets_rolling"]
    ].head(10))
