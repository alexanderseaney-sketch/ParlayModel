"""
Pulls historical NFL data from nflverse (via nfl_data_py) for use in model features.

Usage:
    python data/pull_nflverse.py --years 2019 2020 2021 2022 2023 2024
"""
import argparse
import os

import nfl_data_py as nfl

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def pull_schedules(years):
    """Game-level schedule + results (final scores, spreads, totals as listed)."""
    df = nfl.import_schedules(years)
    out_path = os.path.join(RAW_DIR, "schedules.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


def pull_pbp(years):
    """Play-by-play data — source for EPA, success rate, and other advanced stats."""
    df = nfl.import_pbp_data(years, downcast=True)
    out_path = os.path.join(RAW_DIR, "pbp.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Pull nflverse data")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--skip-pbp", action="store_true", help="Skip play-by-play pull (large/slow)")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    pull_schedules(args.years)
    if not args.skip_pbp:
        pull_pbp(args.years)


if __name__ == "__main__":
    main()
