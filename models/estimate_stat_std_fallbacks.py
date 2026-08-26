"""
Computes real, empirical league-average week-to-week standard deviations per
position/stat combination, used as a fallback in dashboard/utils.py's
estimate_player_stat_std() only when a specific player doesn't have enough of their
own real games (<4) to trust their own variance estimate.

Real WITHIN-PLAYER variance, averaged across players -- not the std of all rows
pooled together, which would wrongly conflate real week-to-week fluctuation with
between-player skill differences (a league pooled directly would overstate spread,
since it includes "Justin Jefferson vs. a WR3" variance, not just "Justin Jefferson
week to week" variance, which is what this is actually meant to estimate).

Run this and paste the printed dict into utils.py's _FALLBACK_STAT_STD whenever the
underlying weekly_stats.csv data is meaningfully refreshed -- not on every pull,
since league-average variance is a slow-moving number, not something that needs to
track week-to-week noise itself.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

COMBOS = [
    ("WR", "receiving_yards"), ("TE", "receiving_yards"), ("RB", "receiving_yards"),
    ("RB", "rushing_yards"), ("QB", "passing_yards"), ("QB", "rushing_yards"),
    ("WR", "receptions"), ("TE", "receptions"), ("RB", "receptions"),
]


def compute_fallback_stds(recent_seasons: int = 2, min_games: int = 4) -> dict:
    ws = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    recent = ws[ws["season"] >= ws["season"].max() - recent_seasons + 1]

    results = {}
    for pos, stat in COMBOS:
        sub = recent[(recent["position"] == pos) & (recent[stat].notna())]
        per_player = sub.groupby("player_id")[stat].agg(["std", "count"])
        per_player = per_player[per_player["count"] >= min_games]
        results[(pos, stat)] = (round(per_player["std"].mean(), 1), len(per_player))
    return results


if __name__ == "__main__":
    results = compute_fallback_stds()
    print("_FALLBACK_STAT_STD = {")
    for (pos, stat), (std, n) in results.items():
        print(f'    ("{pos}", "{stat}"): {std},   # n={n} players')
    print("}")
