"""
Builds richer team-week efficiency features from real play-by-play data — EPA/play and
success rate instead of the volume-based EPA totals used before, with garbage time
filtered out (the standard practice in public EPA models: a blowout's meaningless late
plays otherwise distort a team's true efficiency numbers).

Garbage time definition (fairly standard): score differential of 16+ points in the 4th
quarter, or 21+ at any point in the second half. Excluded from all efficiency calcs.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Only the columns actually used across pbp_features/scheme_features/matchup_features —
# a full pbp.csv load (397 columns) uses ~3.6GB RAM by itself, close to this sandbox's
# entire memory budget. Loading only what's needed cuts that dramatically.
PBP_USECOLS = [
    "season", "week", "posteam", "defteam", "play_type", "epa", "success",
    "pass", "rush", "yards_gained", "qtr", "score_differential",
    "shotgun", "no_huddle", "down", "xpass", "pass_oe", "defenders_in_box",
    "receiver_player_id", "rusher_player_id", "sack",
]


def load_pbp() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DIR, "pbp.csv"), usecols=PBP_USECOLS, low_memory=False)


def filter_garbage_time(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp = pbp.dropna(subset=["epa", "posteam", "defteam"])

    is_garbage = (
        ((pbp["qtr"] == 4) & (pbp["score_differential"].abs() >= 16))
        | ((pbp["qtr"].isin([3, 4])) & (pbp["score_differential"].abs() >= 21))
    )
    return pbp[~is_garbage]


def build_team_week_pbp_offense(pbp: pd.DataFrame) -> pd.DataFrame:
    clean = filter_garbage_time(pbp)

    off = clean.groupby(["posteam", "season", "week"]).agg(
        off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"),
        n_plays=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team"})

    pass_plays = clean[clean["pass"] == 1].groupby(["posteam", "season", "week"])["epa"].mean().reset_index(name="off_pass_epa_per_play")
    rush_plays = clean[clean["rush"] == 1].groupby(["posteam", "season", "week"])["epa"].mean().reset_index(name="off_rush_epa_per_play")

    explosive = clean.groupby(["posteam", "season", "week"]).apply(
        lambda g: (g["yards_gained"] >= 20).mean(), include_groups=False
    ).reset_index(name="off_explosive_rate")

    off = off.merge(pass_plays.rename(columns={"posteam": "team"}), on=["team", "season", "week"], how="left")
    off = off.merge(rush_plays.rename(columns={"posteam": "team"}), on=["team", "season", "week"], how="left")
    off = off.merge(explosive.rename(columns={"posteam": "team"}), on=["team", "season", "week"], how="left")

    return off


def build_team_week_pbp_defense(pbp: pd.DataFrame) -> pd.DataFrame:
    """Defense's own per-play numbers (EPA/success rate ALLOWED), computed directly from
    plays where this team was on defense — more accurate than inferring from opponent's
    offensive totals, since it's the same garbage-time-filtered play set."""
    clean = filter_garbage_time(pbp)

    defn = clean.groupby(["defteam", "season", "week"]).agg(
        def_epa_per_play_allowed=("epa", "mean"),
        def_success_rate_allowed=("success", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    return defn


def add_rolling(team_week: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    team_week = team_week.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in cols:
        team_week[f"{col}_rolling"] = (
            team_week.groupby(["team", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    return team_week


def build_pbp_team_week_features(pbp: pd.DataFrame = None) -> pd.DataFrame:
    if pbp is None:
        pbp = load_pbp()

    off = build_team_week_pbp_offense(pbp)
    defn = build_team_week_pbp_defense(pbp)

    team_week = off.merge(defn, on=["team", "season", "week"], how="outer")

    cols = ["off_epa_per_play", "off_success_rate", "off_pass_epa_per_play",
            "off_rush_epa_per_play", "off_explosive_rate",
            "def_epa_per_play_allowed", "def_success_rate_allowed"]
    team_week = add_rolling(team_week, cols)
    return team_week


if __name__ == "__main__":
    df = build_pbp_team_week_features()
    print(df.shape)
    rolling_cols = [c for c in df.columns if c.endswith("_rolling")]
    print(df[["team", "season", "week"] + rolling_cols].dropna().head(5))
    print("\nNull counts:")
    print(df[rolling_cols].isna().sum())
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pbp_team_week_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
