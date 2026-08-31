"""
Head-coach game history, derived from pbp.csv's per-game home_coach/away_coach
columns -> data/raw/coach_history.csv. One row per TEAM-GAME: season, week,
team, coach, opponent, opponent's coach, points for/against, result. Feeds the
Compare page's Coaches tab (W-L by season/team, PPG for/against, head-to-head
between any two coaches).

Same pattern as build_team_scheme_tendencies.py: derived from the full pbp.csv
(which the 6h refresh workflow does NOT pull -- --skip-pbp), so this is a
manual rebuild after a pbp pull; the committed CSV persists between rebuilds.
Covers whatever seasons pbp.csv holds (2014-2025 as of 2026-08-30). Regular
season and playoffs both included, flagged by season_type.

Usage:
    python models/build_coach_history.py
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PBP_PATH = os.path.join(RAW_DIR, "pbp.csv")
OUT_PATH = os.path.join(RAW_DIR, "coach_history.csv")


def main():
    use = ["game_id", "season", "week", "season_type", "home_team", "away_team",
           "home_coach", "away_coach", "home_score", "away_score"]
    df = pd.concat(
        pd.read_csv(PBP_PATH, usecols=lambda c: c in use, chunksize=200_000, low_memory=False),
        ignore_index=True,
    )
    # home_score/away_score on a pbp row are the FINAL score (constant per game),
    # so one row per game_id carries everything needed.
    games = df.drop_duplicates("game_id").dropna(subset=["home_coach", "away_coach"])
    print(f"{len(games):,} games with coach data, seasons "
          f"{int(games.season.min())}-{int(games.season.max())}")

    home = games.rename(columns={
        "home_team": "team", "home_coach": "coach", "home_score": "points_for",
        "away_team": "opponent", "away_coach": "opp_coach", "away_score": "points_against",
    })
    home["is_home"] = True
    away = games.rename(columns={
        "away_team": "team", "away_coach": "coach", "away_score": "points_for",
        "home_team": "opponent", "home_coach": "opp_coach", "home_score": "points_against",
    })
    away["is_home"] = False

    cols = ["game_id", "season", "week", "season_type", "team", "coach",
            "opponent", "opp_coach", "points_for", "points_against", "is_home"]
    out = pd.concat([home[cols], away[cols]], ignore_index=True)
    out["won"] = (out.points_for > out.points_against).astype(int)
    out["tied"] = (out.points_for == out.points_against).astype(int)
    out = out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)
    n_coaches = out.coach.nunique()
    print(f"{len(out):,} team-game rows, {n_coaches} distinct head coaches -> {OUT_PATH}")


if __name__ == "__main__":
    main()
