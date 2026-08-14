"""
Real matchup-specific defensive splits from play-by-play — more granular than the
overall def_epa_allowed used so far, which mixes pass and run defense together.

Built:
- Pass defense vs run defense, split cleanly (not blended)
- Defense's EPA/yards allowed BY POSITION GROUP (WR/TE/RB) — the real "player matchup"
  signal: does this specific defense struggle against a player's specific position
- Sack rate allowed on pass plays — pressure signal that should hurt passing yards
- Run-stuff rate — run plays defense holds to 0 or negative yards, a "run funnel" signal
"""
import os

import pandas as pd

from pbp_features import filter_garbage_time

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def build_split_pass_rush_defense(pbp: pd.DataFrame) -> pd.DataFrame:
    clean = filter_garbage_time(pbp)

    pass_plays = clean[clean["pass"] == 1]
    rush_plays = clean[clean["rush"] == 1]

    pass_def = pass_plays.groupby(["defteam", "season", "week"]).agg(
        def_pass_epa_allowed=("epa", "mean"),
        def_pass_success_rate_allowed=("success", "mean"),
        def_sack_rate=("sack", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    rush_def = rush_plays.groupby(["defteam", "season", "week"]).agg(
        def_rush_epa_allowed=("epa", "mean"),
        def_rush_success_rate_allowed=("success", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})
    rush_def["def_rush_stuff_rate"] = (
        rush_plays.groupby(["defteam", "season", "week"])["yards_gained"]
        .apply(lambda s: (s <= 0).mean()).values
    )

    return pass_def.merge(rush_def, on=["team", "season", "week"], how="outer")


def build_def_epa_allowed_by_position(pbp: pd.DataFrame, weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Defense's EPA allowed specifically against each receiver position group (WR/TE/RB)."""
    clean = filter_garbage_time(pbp)
    receiving_plays = clean.dropna(subset=["receiver_player_id"])

    # Position lookup: player_id -> position, from weekly_stats (take most common position per player)
    positions = weekly_stats.groupby("player_id")["position"].agg(lambda s: s.mode()[0] if len(s.mode()) else None)

    receiving_plays = receiving_plays.copy()
    receiving_plays["receiver_position"] = receiving_plays["receiver_player_id"].map(positions)

    by_position = receiving_plays.dropna(subset=["receiver_position"]).groupby(
        ["defteam", "season", "week", "receiver_position"]
    )["epa"].mean().reset_index().rename(columns={"defteam": "team", "epa": "def_epa_allowed_by_pos"})

    # Pivot so each position becomes its own rolling-ready column
    pivoted = by_position.pivot_table(
        index=["team", "season", "week"], columns="receiver_position", values="def_epa_allowed_by_pos"
    ).reset_index()
    pivoted.columns = [f"def_epa_allowed_vs_{c}" if c in ("WR", "TE", "RB") else c for c in pivoted.columns]
    return pivoted


def add_rolling(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in cols:
        if col in df.columns:
            df[f"{col}_rolling"] = (
                df.groupby(["team", "season"])[col]
                .apply(lambda s: s.shift(1).expanding().mean())
                .reset_index(level=[0, 1], drop=True)
            )
    return df


def build_all_matchup_features(pbp: pd.DataFrame, weekly_stats: pd.DataFrame) -> pd.DataFrame:
    split_def = build_split_pass_rush_defense(pbp)
    by_position = build_def_epa_allowed_by_position(pbp, weekly_stats)

    merged = split_def.merge(by_position, on=["team", "season", "week"], how="outer")

    rolling_cols = ["def_pass_epa_allowed", "def_pass_success_rate_allowed", "def_sack_rate",
                     "def_rush_epa_allowed", "def_rush_success_rate_allowed", "def_rush_stuff_rate",
                     "def_epa_allowed_vs_WR", "def_epa_allowed_vs_TE", "def_epa_allowed_vs_RB"]
    merged = add_rolling(merged, rolling_cols)
    return merged


if __name__ == "__main__":
    from pbp_features import load_pbp
    pbp = load_pbp()
    weekly_stats = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    df = build_all_matchup_features(pbp, weekly_stats)
    print(df.shape)
    rolling_cols = [c for c in df.columns if c.endswith("_rolling")]
    print(df[["team", "season", "week"] + rolling_cols].dropna(subset=rolling_cols, how="all").head(5))
    print("\nNull rates:")
    print(df[rolling_cols].isna().mean())
