"""
Player-prop feature engineering for receptions (WR/TE). Reuses the same underlying data
as receiving yards, just targets a different stat column.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_TARGETS_TO_QUALIFY = 2


def build_receptions_dataset(min_week: int = 4) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    ngs_receiving = pd.read_csv(os.path.join(RAW_DIR, "ngs_receiving.csv"))
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    wr_te = weekly[weekly["position"].isin(["WR", "TE", "RB"])].copy()

    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 "targets", "receptions", "target_share", "receiving_air_yards"]
    wr_te = wr_te[keep_cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Deliberately NOT pre-filtered to targets >= MIN_TARGETS_TO_QUALIFY here -- same
    # bug as player_prop_features.py's receiving-yards dataset (fixed 2026-08-23):
    # filtering low-target games out before this rolling average meant a player's own
    # baseline silently excluded their cold games, not just this week's label. Applied
    # below instead, after the rolling features are computed.
    for col in ["receptions", "targets", "target_share", "receiving_air_yards"]:
        wr_te[f"{col}_rolling"] = (
            wr_te.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        wr_te[f"{col}_last3"] = (
            wr_te.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )
    # Catch rate — completions per target, a real skill/role signal separate from volume.
    # replace(0, nan), not (0, pd.NA): dividing by a pd.NA-bearing Series yields an
    # object-dtype column that XGBoost's DMatrix rejects outright. targets_rolling can
    # now actually be 0 here -- since the MIN_TARGETS_TO_QUALIFY filter moved below the
    # rolling step, a player whose whole trailing window was 0-target games reaches this
    # line -- so the guard has to produce a real float NaN, which XGBoost handles natively.
    wr_te["catch_rate_rolling"] = (
        wr_te["receptions_rolling"] / wr_te["targets_rolling"].replace(0, float("nan"))
    )

    ngs = ngs_receiving.rename(columns={"player_gsis_id": "player_id"})
    ngs = ngs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["avg_separation", "avg_cushion", "catch_percentage"]:
        ngs[f"{col}_rolling"] = (
            ngs.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    ngs_cols = ["player_id", "season", "week"] + [f"{c}_rolling" for c in ["avg_separation", "avg_cushion", "catch_percentage"]]
    wr_te = wr_te.merge(ngs[ngs_cols], on=["player_id", "season", "week"], how="left")

    from feature_engineering import build_team_week_offense, build_team_week_defense
    defense = build_team_week_defense(build_team_week_offense(weekly), schedules)
    defense = defense.sort_values(["team", "season", "week"]).reset_index(drop=True)
    defense["def_epa_allowed_rolling"] = (
        defense.groupby(["team", "season"])["def_epa_allowed"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    home = schedules[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "recent_team", "away_team": "opponent"})
    away = schedules[["season", "week", "home_team", "away_team"]].rename(
        columns={"away_team": "recent_team", "home_team": "opponent"})
    matchups = pd.concat([home, away], ignore_index=True).drop_duplicates()

    wr_te = wr_te.merge(matchups, on=["recent_team", "season", "week"], how="left")
    wr_te = wr_te.merge(
        defense[["team", "season", "week", "def_epa_allowed_rolling"]].rename(columns={"team": "opponent"}),
        on=["opponent", "season", "week"], how="left",
    )

    wr_te = wr_te[wr_te["week"] >= min_week].reset_index(drop=True)

    # Applied here, after every rolling/merge step above, not on the raw weekly log --
    # see the matching comment where wr_te is first built.
    wr_te = wr_te[wr_te["targets"] >= MIN_TARGETS_TO_QUALIFY].reset_index(drop=True)

    wr_te["proxy_line"] = wr_te["receptions_rolling"]
    wr_te["over_proxy_line"] = (wr_te["receptions"] > wr_te["proxy_line"]).astype(int)

    return wr_te


if __name__ == "__main__":
    df = build_receptions_dataset()
    print(df.shape)
    print(df[["player_display_name", "receptions", "receptions_rolling", "over_proxy_line"]].dropna().head())
