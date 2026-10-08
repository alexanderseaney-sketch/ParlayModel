"""
Slim, committed history of Underdog lines (added 2026-10-08) -- what the hosted app
reads for line moves (Injury Tracker, Player Pages, Home).

data/raw/underdog_history/ holds full snapshots but is git-ignored (~12 MB each), so it
only exists on the machine that pulled them. This file records a row ONLY when a
player's line for a stat changes (or first appears / disappears), from the OVER side
of single-game props, so it stays small enough to commit every 6 hours:

    full_name, stat_name, line, seen_at (UTC ISO), event ("open" | "move" | "pulled")

Run after data/pull_underdog.py (the refresh workflow does). `--seed` rebuilds it from
every local snapshot in underdog_history/ first.
"""
import argparse
import glob
import os
import sys

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(RAW_DIR, "underdog_line_history.csv")
COLS = ["full_name", "stat_name", "line", "seen_at", "event"]
KEEP_DAYS = 200  # about a season; trimmed on every run


def _board(df: pd.DataFrame) -> pd.DataFrame:
    """One line per (player, stat) for single-game props, from a raw Underdog pull."""
    d = df[df["choice"].astype(str).str.lower().isin(["over", "higher"])]
    if "match_type" in d.columns:
        d = d[d["match_type"].fillna("Game") == "Game"]
    d = d[~d["stat_name"].astype(str).str.startswith(("period_", "season_"))]
    d = d.dropna(subset=["full_name", "stat_name", "stat_value"])
    return d.drop_duplicates(["full_name", "stat_name"])[["full_name", "stat_name", "stat_value"]] \
        .rename(columns={"stat_value": "line"})


def _apply(history: pd.DataFrame, board: pd.DataFrame, seen_at: str) -> pd.DataFrame:
    """Append rows for lines that opened, moved or were pulled since the last record."""
    if history.empty:
        last = pd.DataFrame(columns=["full_name", "stat_name", "line", "event"])
    else:
        last = history.sort_values("seen_at").groupby(["full_name", "stat_name"]).tail(1)
    prev = last[["full_name", "stat_name", "line", "event"]].rename(columns={"line": "line_prev", "event": "event_prev"})
    merged = board.merge(prev, on=["full_name", "stat_name"], how="outer", indicator=True)
    new_rows = []
    opened = merged[(merged["_merge"] == "left_only") | ((merged["_merge"] == "both") & (merged["event_prev"] == "pulled"))]
    new_rows.append(opened.assign(event="open")[["full_name", "stat_name", "line"]].assign(event="open"))
    moved = merged[(merged["_merge"] == "both") & (merged["event_prev"] != "pulled") & (merged["line"] != merged["line_prev"])]
    new_rows.append(moved[["full_name", "stat_name", "line"]].assign(event="move"))
    pulled = merged[(merged["_merge"] == "right_only") & (merged["event_prev"] != "pulled")]
    new_rows.append(pulled[["full_name", "stat_name", "line_prev"]].rename(columns={"line_prev": "line"}).assign(event="pulled"))
    add = pd.concat(new_rows, ignore_index=True).assign(seen_at=seen_at)[COLS]
    return pd.concat([history, add], ignore_index=True) if not add.empty else history


def _trim(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
    ts = pd.to_datetime(history["seen_at"], utc=True, errors="coerce")
    return history[ts >= cutoff]


def load() -> pd.DataFrame:
    if not os.path.exists(OUT):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="rebuild from local underdog_history snapshots first")
    args = ap.parse_args()
    history = pd.DataFrame(columns=COLS) if args.seed else load()
    sources = []
    if args.seed:
        sources += sorted(glob.glob(os.path.join(RAW_DIR, "underdog_history", "underdog_props_*.csv")))
    sources.append(os.path.join(RAW_DIR, "underdog_props.csv"))
    seen = set(history["seen_at"].astype(str)) if not history.empty else set()
    for path in sources:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in (
            "full_name", "stat_name", "stat_value", "choice", "match_type", "pulled_at"))
        if df.empty or "pulled_at" not in df.columns:
            continue
        seen_at = pd.to_datetime(df["pulled_at"].iloc[0], utc=True).isoformat()
        if seen_at in seen:
            continue
        history = _apply(history, _board(df), seen_at)
        seen.add(seen_at)
    history = _trim(history).sort_values(["seen_at", "full_name", "stat_name"])
    history.to_csv(OUT, index=False)
    print(f"{len(history)} line-history rows ({history['event'].value_counts().to_dict() if len(history) else {}}) -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
