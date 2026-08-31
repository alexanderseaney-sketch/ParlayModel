"""
Fantasy points allowed to each position by each defense -> data/raw/fantasy_matchups.csv.
The "who should I target / avoid this week" table behind the Fantasy page's Matchups tab
and Start/Sit reasoning.

Uses nflverse's own per-game fantasy_points (standard) and fantasy_points_ppr columns
from weekly_stats.csv -- half-PPR is exactly their midpoint (PPR - standard = receptions,
so standard + 0.5*rec = (standard + ppr)/2). No pbp needed, so unlike the other
build_*.py this one CAN run in the 6h refresh workflow.

Window: the most recent regular season with >= 4 played weeks, plus the current one if
it has any games yet -- matchup strength shifts season to season with personnel, so
older years would just add noise.

Output: one row per (def_team, position) with games, fantasy points allowed per game
(ppr / half / standard), and a 1-32 rank per scoring (1 = toughest defense).

Usage:
    python models/build_fantasy_matchups.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
from fantasy_scoring import fantasy_points_from_weekly  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
WEEKLY_PATH = os.path.join(RAW_DIR, "weekly_stats.csv")
OUT_PATH = os.path.join(RAW_DIR, "fantasy_matchups.csv")

POSITIONS = ["QB", "RB", "WR", "TE"]


def main():
    w = pd.read_csv(WEEKLY_PATH, low_memory=False)
    w = w[(w["season_type"] == "REG") & w["position"].isin(POSITIONS)].copy()

    played = w.groupby("season")["week"].nunique()
    seasons = sorted(s for s in played.index if played[s] >= 4)
    window = seasons[-1:] if seasons else []
    if seasons and seasons[-1] + 1 in set(w["season"]):
        window.append(seasons[-1] + 1)
    w = w[w["season"].isin(window)]
    if w.empty:
        raise SystemExit("weekly_stats.csv has no usable regular-season data yet.")

    w["fp_ppr"] = fantasy_points_from_weekly(w, "ppr")
    w["fp_half"] = fantasy_points_from_weekly(w, "half")
    w["fp_std"] = fantasy_points_from_weekly(w, "std")

    # a defense's games played -- distinct (season, week) it appears in as opponent_team
    team_games = (w.groupby("opponent_team")[["season", "week"]]
                  .apply(lambda g: g.drop_duplicates().shape[0]).rename("team_games"))

    # points a DEFENSE gave up = sum over the players it faced (opponent_team is the
    # defense from the offensive player's row).
    rows = (w.groupby(["opponent_team", "position"])
            .agg(fp_ppr_pg=("fp_ppr", "sum"), fp_half_pg=("fp_half", "sum"),
                 fp_std_pg=("fp_std", "sum"))
            .reset_index().rename(columns={"opponent_team": "def_team"}))
    rows = rows.merge(team_games, left_on="def_team", right_index=True)

    for col in ("fp_ppr_pg", "fp_half_pg", "fp_std_pg"):
        rows[col] = (rows[col] / rows["team_games"]).round(2)

    # rank within position, 1 = fewest points allowed = toughest matchup
    for scoring, col in (("ppr", "fp_ppr_pg"), ("half", "fp_half_pg"), ("std", "fp_std_pg")):
        rows[f"rank_{scoring}"] = rows.groupby("position")[col].rank(method="min").astype(int)

    rows["derived_from_seasons"] = ", ".join(str(int(s)) for s in window)
    rows = rows.sort_values(["position", "rank_ppr"])
    rows.to_csv(OUT_PATH, index=False)
    print(f"{len(rows)} (def_team, position) rows, seasons {window} -> {OUT_PATH}")
    print(rows[rows.position == "RB"][["def_team", "team_games", "fp_ppr_pg", "rank_ppr"]]
          .head(8).to_string(index=False))


if __name__ == "__main__":
    main()
