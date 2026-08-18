"""
Derives line movement from data/raw/underdog_history/'s timestamped snapshots --
how much a specific prop's posted line has shifted since it was first seen, a classic
sharp-money signal in betting analytics (a line moving against the public side often
means real information, e.g. an injury designation or a beat-writer report, hit before
it's widely known). Not a new pull: pull_underdog.py has saved a full timestamped
snapshot on every run since 2026-08-15 already; this just reads what's already there.

Deliberately NOT wired into any trained model yet -- the archive only spans a couple
of days as of 2026-08-18, nowhere near enough history to validate whether movement
actually predicts outcomes (this project's own discipline throughout has been: test
before trusting a new feature, same as red-zone share was validated and scheme/tempo
features were found not to help). Surfaced as a dashboard signal for now, the same way
Footballguys' live status tags are used before they have their own validated feature
weight -- genuinely useful to a human glancing at it, not yet something a model should
be trained against.
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HISTORY_DIR = os.path.join(RAW_DIR, "underdog_history")

KEY_COLS = ["full_name", "stat_name", "choice"]


def compute_line_movement() -> pd.DataFrame:
    """One row per (player, stat, choice) that's appeared in at least one snapshot --
    earliest and latest observed line, and the movement between them. Snapshots sort
    correctly by filename alone (the timestamp is embedded in ISO-ish order:
    YYYYMMDDTHHMMSSZ), no need to parse pulled_at just to get chronological order."""
    snapshot_paths = sorted(glob.glob(os.path.join(HISTORY_DIR, "underdog_props_*.csv")))
    if len(snapshot_paths) < 2:
        return pd.DataFrame(columns=KEY_COLS + [
            "earliest_line", "latest_line", "line_movement", "line_movement_pct",
            "first_seen_at", "last_seen_at", "n_snapshots"])

    frames = []
    for path in snapshot_paths:
        snap = pd.read_csv(path, usecols=KEY_COLS + ["stat_value", "pulled_at"], low_memory=False)
        snap = snap.dropna(subset=KEY_COLS + ["stat_value"])
        frames.append(snap)
    history = pd.concat(frames, ignore_index=True)

    # A player can be re-pulled multiple times within what's meant to be "one"
    # snapshot run in edge cases (retries, partial re-runs) -- collapsing to one row
    # per (key, pulled_at) first keeps a single bad duplicate from skewing first/last.
    history = history.drop_duplicates(subset=KEY_COLS + ["pulled_at"])
    history = history.sort_values("pulled_at")

    grouped = history.groupby(KEY_COLS)
    result = grouped.agg(
        earliest_line=("stat_value", "first"),
        latest_line=("stat_value", "last"),
        first_seen_at=("pulled_at", "first"),
        last_seen_at=("pulled_at", "last"),
        n_snapshots=("stat_value", "count"),
    ).reset_index()

    result["line_movement"] = result["latest_line"] - result["earliest_line"]
    result["line_movement_pct"] = (result["line_movement"] / result["earliest_line"].abs()).where(
        result["earliest_line"] != 0)

    return result


if __name__ == "__main__":
    movement = compute_line_movement()
    print(f"{len(movement)} (player, stat, choice) combos with snapshot history")
    moved = movement[movement["line_movement"] != 0].sort_values(
        "line_movement_pct", key=lambda s: s.abs(), ascending=False)
    print(f"{len(moved)} with any real movement")
    print(moved.head(15).to_string())
