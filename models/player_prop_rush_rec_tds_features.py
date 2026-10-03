"""
Player-prop feature engineering for combined rush + receiving touchdowns
("anytime TD"-style prop). Same no-leakage discipline as the other prop
models: every feature is a strictly-prior rolling average
(shift(1).expanding() / shift(1).rolling(3)).

Unlike the yardage props, this one does NOT need a rolling-average proxy line.
Underdog's real posted line for rush_rec_tds is a constant 0.5 across
essentially every player (confirmed against the live pulled props: all 398
rows), so the proxy line is set to that same constant and the target is
simply "did they score at least one rushing or receiving TD" -- this proxy
matches the real market almost exactly rather than approximating it.

Includes QB in the base filter (not just RB/WR/TE) -- unlike
receiving_yards/rushing_yards, nothing here is NGS-derived (weekly_stats +
team-level EPA + pbp-derived red zone usage, all position-agnostic), so
there's no data-availability reason to exclude QBs. Found via a live-props
audit (2026-08-17): rush_rec_tds and everything that reuses this dataset
(period_first_touchdown_scored, period_1/1_2 rush_rec_tds) was excluding
every mobile QB -- Josh Allen, Lamar Jackson, Jalen Hurts, etc. -- 71
unmatched legs combined, all with real NFL rushing history. QBs are trained
as a SEPARATE model though (see train_rush_rec_tds_qb_props.py) rather than
folded into the RB/WR/TE one -- QB touchdown-scoring mechanics (goal-line
sneaks, scramble TDs) are different enough from RB/WR/TE that mixing
populations risked diluting the existing, already-validated model rather
than improving it.
"""
import os

import pandas as pd

from feature_engineering import cross_season_rolling, qualify

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_TOUCHES_TO_QUALIFY = 3  # carries + targets, filters out garbage-time/inactive players
PROXY_LINE = 0.5

# See feature_engineering.cross_season_rolling's docstring for why these are bounded
# and cross a season boundary rather than resetting there.
_ROLLING_WINDOW = 16
_TEAM_WINDOW = 8


def build_rush_rec_tds_dataset(min_week: int = 1) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    skill = weekly[weekly["position"].isin(["RB", "WR", "TE", "QB"])].copy()
    skill["rush_rec_tds"] = skill["rushing_tds"] + skill["receiving_tds"]

    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 "carries", "targets", "rushing_yards", "receiving_yards", "rush_rec_tds"]
    skill = skill[keep_cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Deliberately NOT pre-filtered to touches (carries + targets) >= MIN_TOUCHES_TO_QUALIFY
    # here -- same bug as player_prop_features.py's receiving-yards dataset (fixed
    # 2026-08-23): filtering low-touch games out before this rolling average meant a
    # player's own baseline silently excluded their cold/inactive games, not just this
    # week's label. Applied below instead, after the rolling features are computed.
    for col in ["rush_rec_tds", "carries", "targets"]:
        skill[f"{col}_rolling"] = cross_season_rolling(skill, "player_id", col, window=_ROLLING_WINDOW)
        skill[f"{col}_last3"] = cross_season_rolling(skill, "player_id", col, window=3)

    for col in ["rushing_yards", "receiving_yards"]:
        skill[f"{col}_rolling"] = cross_season_rolling(skill, "player_id", col, window=_ROLLING_WINDOW)

    # Opponent defense strength -- how many EPA/play they typically allow
    from feature_engineering import build_team_week_offense, build_team_week_defense
    defense = build_team_week_defense(build_team_week_offense(weekly), schedules)
    defense = defense.sort_values(["team", "season", "week"]).reset_index(drop=True)
    defense["def_epa_allowed_rolling"] = cross_season_rolling(defense, "team", "def_epa_allowed", window=_TEAM_WINDOW)

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

    # Red-zone usage share: the single strongest signal for "is this player their
    # team's actual goal-line option" -- tested head-to-head, +0.06 to +0.10 AUC
    # across every per-game TD prop that reuses this dataset (rush_rec_tds,
    # period_first_touchdown_scored, period_1/1_2 rush_rec_tds). A player with no
    # red-zone touches in any prior game is a real 0, not missing data, so this
    # fills rather than drops -- dropping was tested first and looked like a
    # regression, but it was actually just discarding the (highly informative)
    # zero-usage rows before the model ever saw them.
    from red_zone_features import build_red_zone_usage
    red_zone = build_red_zone_usage(min_week=1)
    skill = skill.merge(red_zone, on=["player_id", "recent_team", "season", "week"], how="left")
    skill["red_zone_touches_rolling"] = skill["red_zone_touches_rolling"].fillna(0.0)
    skill["red_zone_share_rolling"] = skill["red_zone_share_rolling"].fillna(0.0)

    skill = skill[skill["week"] >= min_week].reset_index(drop=True)

    # Applied here, after every rolling/merge step above, not on the raw weekly log --
    # see the matching comment where skill is first built.
    # Pre-game role, not same-game touches (post-game info) -- see
    # feature_engineering.PREGAME_QUALIFIER.
    skill["touches"] = skill["carries"] + skill["targets"]
    skill["touches_rolling"] = skill["carries_rolling"] + skill["targets_rolling"]
    skill = qualify(skill, "touches", "touches_rolling", MIN_TOUCHES_TO_QUALIFY)

    skill["proxy_line"] = PROXY_LINE
    skill["over_proxy_line"] = (skill["rush_rec_tds"] >= 1).astype(int)

    return skill


if __name__ == "__main__":
    df = build_rush_rec_tds_dataset()
    print(df.shape)
    print(df["over_proxy_line"].value_counts(normalize=True))
    print(df[["player_display_name", "season", "week", "rush_rec_tds", "rush_rec_tds_rolling",
              "carries_rolling", "targets_rolling", "over_proxy_line"]].dropna().head(10))
