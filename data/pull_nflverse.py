"""
Pulls NFL data from nflverse (via nfl_data_py): schedules, play-by-play, Next Gen Stats,
weekly player stats, injuries, and snap counts. Each pull is validated on the way in
(row counts, null checks, duplicate checks) so bad data gets caught immediately instead
of silently feeding into the model later.

Usage:
    python data/pull_nflverse.py --years 2019 2020 2021 2022 2023 2024
    python data/pull_nflverse.py --years 2023 2024 --skip-pbp   # pbp is large/slow
"""
import argparse
import os
import sys

import nfl_data_py as nfl
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def validate(df: pd.DataFrame, name: str, key_cols: list[str], warn_null_cols: list[str] | None = None) -> None:
    """Basic sanity checks on a pulled dataset. Raises on hard failures, warns on soft ones."""
    issues = []

    if df.empty:
        raise ValueError(f"[{name}] pulled 0 rows — aborting, this would silently break everything downstream.")

    missing_key_cols = [c for c in key_cols if c not in df.columns]
    if missing_key_cols:
        raise ValueError(f"[{name}] missing expected key columns: {missing_key_cols}. Source schema may have changed.")

    dupes = df.duplicated(subset=key_cols).sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate rows on key {key_cols}")

    for col in key_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            issues.append(f"{n_null} nulls in key column '{col}'")

    if warn_null_cols:
        for col in warn_null_cols:
            if col in df.columns:
                pct_null = df[col].isna().mean() * 100
                if pct_null > 5:
                    issues.append(f"'{col}' is {pct_null:.1f}% null")

    print(f"[{name}] {len(df)} rows, {len(df.columns)} columns.")
    if issues:
        print(f"[{name}] VALIDATION WARNINGS:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"[{name}] validation passed clean.")


def save(df: pd.DataFrame, filename: str, key_cols: list[str]) -> None:
    """Merges into any existing file rather than overwriting it. Fixed 2026-08-15 after
    a real near-miss: re-running this with a different --years range (e.g. pulling
    2019-2023 to backfill history) used to silently wipe out whatever years an earlier
    pull had already saved, since every pull_* function targets the same fixed
    filename. That matters a lot more once this runs weekly during the season instead
    of as a one-off backfill -- a weekly refresh must not nuke prior weeks every time
    it runs. New rows win on key collisions (e.g. a corrected stat or an updated injury
    report), old rows outside the new pull's range are preserved."""
    out_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, low_memory=False)
        combined = pd.concat([df, existing], ignore_index=True)
        df = combined.drop_duplicates(subset=key_cols, keep="first")
    df.to_csv(out_path, index=False)
    print(f"   saved -> {out_path} ({len(df)} rows total)\n")


def pull_schedules(years):
    """Game-level schedule + results, including closing spread/total lines."""
    df = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv")
    df = df[df["season"].isin(years)]
    key_cols = ["game_id"]
    validate(df, "schedules", key_cols=key_cols, warn_null_cols=["home_score", "spread_line"])
    save(df, "schedules.csv", key_cols)
    return df


def pull_pbp(years):
    """Play-by-play data — source for EPA, success rate, and other advanced stats."""
    df = nfl.import_pbp_data(years, downcast=True)
    key_cols = ["game_id", "play_id"]
    validate(df, "play_by_play", key_cols=key_cols, warn_null_cols=["epa", "posteam"])
    save(df, "pbp.csv", key_cols)
    return df


def pull_weekly_stats(years):
    """Weekly player-level box score stats."""
    df = nfl.import_weekly_data(years)
    key_cols = ["player_id", "season", "week"]
    validate(df, "weekly_stats", key_cols=key_cols)
    save(df, "weekly_stats.csv", key_cols)
    return df


def pull_ngs(years):
    """Next Gen Stats (AWS-powered tracking data): passing, rushing, receiving."""
    frames = {}
    key_cols = ["player_gsis_id", "season", "week"]
    for stat_type in ("passing", "rushing", "receiving"):
        df = nfl.import_ngs_data(stat_type, years)
        validate(df, f"ngs_{stat_type}", key_cols=key_cols)
        save(df, f"ngs_{stat_type}.csv", key_cols)
        frames[stat_type] = df
    return frames


def pull_injuries(years):
    df = nfl.import_injuries(years)
    key_cols = ["gsis_id", "season", "week"]
    validate(df, "injuries", key_cols=key_cols, warn_null_cols=["report_status"])
    save(df, "injuries.csv", key_cols)
    return df


def pull_snap_counts(years):
    df = nfl.import_snap_counts(years)
    key_cols = ["pfr_player_id", "season", "week"]
    validate(df, "snap_counts", key_cols=key_cols)
    save(df, "snap_counts.csv", key_cols)
    return df


def main():
    parser = argparse.ArgumentParser(description="Pull and validate nflverse data")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--skip-pbp", action="store_true", help="Skip play-by-play pull (large/slow)")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    failures = []
    pulls = [
        ("schedules", lambda: pull_schedules(args.years)),
        ("weekly_stats", lambda: pull_weekly_stats(args.years)),
        ("ngs", lambda: pull_ngs(args.years)),
        ("injuries", lambda: pull_injuries(args.years)),
        ("snap_counts", lambda: pull_snap_counts(args.years)),
    ]
    if not args.skip_pbp:
        pulls.insert(1, ("pbp", lambda: pull_pbp(args.years)))

    for name, fn in pulls:
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e}\n")
            failures.append(name)

    if failures:
        print(f"Completed with failures: {failures}")
        sys.exit(1)
    print("All pulls completed.")


if __name__ == "__main__":
    main()
