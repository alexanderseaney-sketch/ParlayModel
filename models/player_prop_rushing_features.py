"""
Player-prop feature engineering for rushing yards. Same no-leakage discipline as
the receiving-yards model: every feature is a strictly-prior rolling average
(shift(1).expanding()).

Same honest limitation as the receiving model: backtested against the player's own
trailing average as a proxy line, since no historical Underdog line archive exists yet.

Includes QB in the base filter (not just RB) for the same reason receiving_yards
includes RB: real Underdog rushing_yds props are heavily QB-driven (mobile
quarterbacks), and excluding them by position was a real coverage bug, not a
data limitation -- confirmed via a live-props audit (2026-08-17), ~20 of 22
unmatched rushing_yds legs were QBs with full NFL history. Same NGS gap as
receiving_yards/RB, though: ngs_rushing.csv has zero QB rows, so the production
QB model (train_rushing_yards_qb_props.py) uses non-NGS features only, same
pattern as train_receiving_yards_rb_props.py.
"""
import os

import pandas as pd

from feature_engineering import cross_season_rolling, load_ngs, merge_ngs, qualify

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_CARRIES_TO_QUALIFY = 3

# See feature_engineering.cross_season_rolling's docstring for why these are bounded
# and cross a season boundary rather than resetting there.
_ROLLING_WINDOW = 16
_TEAM_WINDOW = 8


def build_rushing_yards_dataset(min_week: int = 1) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    ngs_rushing = load_ngs("rushing")  # drops week-0 season totals (look-ahead)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    rb = weekly[weekly["position"].isin(["RB", "QB"])].copy()

    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 "carries", "rushing_yards", "targets", "receiving_yards"]
    rb = rb[keep_cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Deliberately NOT pre-filtered to carries >= MIN_CARRIES_TO_QUALIFY here -- same
    # bug as player_prop_features.py's receiving-yards dataset (fixed alongside this,
    # 2026-08-23): filtering low-carry games out before this rolling average meant a
    # player's own baseline silently excluded their cold games, not just this week's
    # label, inflating the proxy for anyone with an irregular rushing role. Applied
    # below instead, after the rolling features are computed.
    for col in ["rushing_yards", "carries", "targets", "receiving_yards"]:
        rb[f"{col}_rolling"] = cross_season_rolling(rb, "player_id", col, window=_ROLLING_WINDOW)
        rb[f"{col}_last3"] = cross_season_rolling(rb, "player_id", col, window=3)

    # NGS rushing: rush yards over expected is the cleanest efficiency signal here —
    # directly analogous to CPOE for the receiving model
    ngs = ngs_rushing.rename(columns={"player_gsis_id": "player_id"})
    ngs = ngs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["rush_yards_over_expected_per_att", "percent_attempts_gte_eight_defenders", "efficiency"]:
        ngs[f"{col}_rolling"] = cross_season_rolling(ngs, "player_id", col, window=_ROLLING_WINDOW)
    ngs_cols = ["player_id", "season", "week"] + [f"{c}_rolling" for c in ["rush_yards_over_expected_per_att", "percent_attempts_gte_eight_defenders", "efficiency"]]
    rb = merge_ngs(rb, ngs[ngs_cols])

    # Opponent's rushing defense strength — how many yards do they typically allow
    from feature_engineering import build_team_week_offense, build_team_week_defense
    defense = build_team_week_defense(build_team_week_offense(weekly), schedules)
    defense = defense.sort_values(["team", "season", "week"]).reset_index(drop=True)
    defense["def_epa_allowed_rolling"] = cross_season_rolling(defense, "team", "def_epa_allowed", window=_TEAM_WINDOW)

    home = schedules[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "recent_team", "away_team": "opponent"})
    away = schedules[["season", "week", "home_team", "away_team"]].rename(
        columns={"away_team": "recent_team", "home_team": "opponent"})
    matchups = pd.concat([home, away], ignore_index=True).drop_duplicates()

    rb = rb.merge(matchups, on=["recent_team", "season", "week"], how="left")
    rb = rb.merge(
        defense[["team", "season", "week", "def_epa_allowed_rolling"]].rename(columns={"team": "opponent"}),
        on=["opponent", "season", "week"], how="left",
    )

    rb = rb[rb["week"] >= min_week].reset_index(drop=True)

    # Applied here, after every rolling/merge step above, not on the raw weekly log --
    # see the matching comment where rb is first built.
    rb = qualify(rb, "carries", "carries_rolling", MIN_CARRIES_TO_QUALIFY)

    rb["proxy_line"] = rb["rushing_yards_rolling"]
    rb["over_proxy_line"] = (rb["rushing_yards"] > rb["proxy_line"]).astype(int)

    return rb


if __name__ == "__main__":
    df = build_rushing_yards_dataset()
    print(df.shape)
    print(df[["player_display_name", "season", "week", "rushing_yards", "rushing_yards_rolling",
              "rush_yards_over_expected_per_att_rolling", "over_proxy_line"]].dropna().head(10))
