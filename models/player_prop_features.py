"""
Player-prop feature engineering — receiving yards, as the flagship prop type (the same
approach generalizes to rushing/passing/receptions, noted at the end).

Important honest limitation: we don't have historical Underdog prop lines to backtest
against (no historical archive exists). As a backtestable proxy for "the line," we use
the player's own trailing rolling average — i.e., testing whether the model can predict
whether a player will go OVER or UNDER their own recent-form baseline. This is a
reasonable stand-in (real prop lines are usually set close to recent form + matchup
adjustment) but it is NOT the same as backtesting against real historical Underdog
lines, which don't exist to test against. Treat accuracy numbers here as "can the model
beat recent-form momentum," not "can the model beat Underdog specifically."

All features are pre-game / no-leakage: rolling averages use shift(1).expanding(),
same discipline as the team-level model.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_TARGETS_TO_QUALIFY = 2  # filters out garbage-time/emergency snaps from the training signal


def build_receiving_yards_dataset(min_week: int = 4) -> pd.DataFrame:
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    ngs_receiving = pd.read_csv(os.path.join(RAW_DIR, "ngs_receiving.csv"))
    snaps = pd.read_csv(os.path.join(RAW_DIR, "snap_counts.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    wr_te = weekly[weekly["position"].isin(["WR", "TE", "RB"])].copy()
    wr_te = wr_te[wr_te["targets"] >= MIN_TARGETS_TO_QUALIFY]

    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 "targets", "receiving_yards", "target_share", "receiving_air_yards"]
    wr_te = wr_te[keep_cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Rolling pre-game features from the player's own history (strictly prior weeks)
    for col in ["receiving_yards", "targets", "target_share", "receiving_air_yards"]:
        wr_te[f"{col}_rolling"] = (
            wr_te.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        # Recent form: trailing 3-game average, more responsive than season-to-date
        wr_te[f"{col}_last3"] = (
            wr_te.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    # NGS rolling (avg separation, YAC over expected — same no-leakage pattern)
    ngs = ngs_receiving.rename(columns={"player_gsis_id": "player_id", "yards": "ngs_yards"})
    ngs = ngs.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["avg_separation", "avg_cushion", "avg_yac_above_expectation"]:
        ngs[f"{col}_rolling"] = (
            ngs.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    ngs_cols = ["player_id", "season", "week"] + [f"{c}_rolling" for c in ["avg_separation", "avg_cushion", "avg_yac_above_expectation"]]
    wr_te = wr_te.merge(ngs[ngs_cols], on=["player_id", "season", "week"], how="left")

    # Snap share rolling (opportunity signal)
    snaps_r = snaps.rename(columns={"pfr_player_id": "player_id_pfr"})
    # snap_counts uses pfr_player_id, weekly_stats uses gsis player_id — different ID systems.
    # Without a clean crosswalk in this pull, skip snap share for now (documented gap, not silently wrong).

    # Opponent defensive strength vs. the pass, using team-level defensive EPA we already built
    from feature_engineering import build_team_week_offense, build_team_week_defense
    offense = build_team_week_offense(weekly)
    defense = build_team_week_defense(offense, schedules)
    defense = defense.sort_values(["team", "season", "week"]).reset_index(drop=True)
    defense["def_epa_allowed_rolling"] = (
        defense.groupby(["team", "season"])["def_epa_allowed"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # Get each player's opponent that week from schedules
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

    # Team-level scheme signal: this player's own team's pass rate over expected —
    # a pass-heavy offense means more targets exist to go around, regardless of the player's own history
    from scheme_features import build_team_week_scheme
    pbp = pd.read_csv(os.path.join(RAW_DIR, "pbp.csv"), low_memory=False)
    scheme = build_team_week_scheme(pbp)
    wr_te = wr_te.merge(
        scheme[["team", "season", "week", "pass_oe_rolling"]].rename(columns={"team": "recent_team", "pass_oe_rolling": "team_pass_oe_rolling"}),
        on=["recent_team", "season", "week"], how="left",
    )

    # The backtestable proxy target: did the player beat their own season-to-date average?
    wr_te["proxy_line"] = wr_te["receiving_yards_rolling"]
    wr_te["over_proxy_line"] = (wr_te["receiving_yards"] > wr_te["proxy_line"]).astype(int)

    return wr_te


if __name__ == "__main__":
    df = build_receiving_yards_dataset()
    print(df.shape)
    print(df[["player_display_name", "season", "week", "receiving_yards", "receiving_yards_rolling",
              "receiving_yards_last3", "def_epa_allowed_rolling", "over_proxy_line"]].dropna().head(10))
    print("\nNull counts:")
    print(df.isna().sum()[df.isna().sum() > 0])
