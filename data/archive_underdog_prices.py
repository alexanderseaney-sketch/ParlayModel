"""
Slim, committed history of Underdog's PRICES for each pick (added 2026-11-22), so the
weekly regrade can judge picks against what Champions really pays.

underdog_line_history.csv keeps only the line. Underdog also prices every side on its
own (american_price, and the payout_multiplier Champions shows on the slip: 0.75x on a
heavy favourite, 1.00x on a plain line, 1.18x on an underdog side). Those prices decide
the break-even of every pick, and their skew tells us which way Underdog's own book
leans even when both sides pay 1.00x.

One row each time a (player, stat, line) first appears, a multiplier changes, or (while
both sides pay 1.00x) the de-vigged chance moves half a point or more. A line's price at
any moment is its last row before then. Quarter / half and first / last TD scorer props
are left out to keep the file small (they are near-certain-only for picks anyway).

    full_name, stat_name, line, over_price, under_price, over_mult, under_mult, seen_at (UTC)

Run after data/pull_underdog.py (the refresh workflow does). `--seed-git` rebuilds it
from every committed data/raw/underdog_props.csv in git history first (needs a full,
unshallow clone: `git fetch --filter=blob:none --unshallow origin main`).
"""
import argparse
import io
import os
import subprocess
import sys

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RAW_DIR, "underdog_price_history.csv")
KEY = ["full_name", "stat_name", "line"]
PRICE_COLS = ["over_price", "under_price", "over_mult", "under_mult"]
COLS = KEY + PRICE_COLS + ["seen_at"]
KEEP_DAYS = 200
MIN_MOVE = 0.005  # de-vigged chance move worth a new row
READ_COLS = ("full_name", "stat_name", "stat_value", "choice", "match_type", "american_price",
             "payout_multiplier", "pulled_at")


def implied(american) -> pd.Series:
    """American price -> implied chance (with Underdog's margin)."""
    a = pd.to_numeric(pd.Series(american), errors="coerce").reset_index(drop=True)
    return (100 / (a + 100)).where(a >= 0, -a / (-a + 100))


def fair_over(over_price, under_price) -> pd.Series:
    """De-vigged chance of the Over from the two american prices (NaN if a side is missing)."""
    io, iu = implied(over_price).to_numpy(), implied(under_price).to_numpy()
    return pd.Series(io / (io + iu))


def _board(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, stat, line) with both sides' prices, single-game props only."""
    d = df.copy()
    if "match_type" in d.columns:
        d = d[d["match_type"].fillna("Game") == "Game"]
    st = d["stat_name"].astype(str)
    d = d[~st.str.startswith(("season_", "period_")) & ~st.str.endswith("_touchdown_scored")]
    d = d.dropna(subset=["full_name", "stat_name", "stat_value"])
    side = d["choice"].astype(str).str.lower().replace({"higher": "over", "lower": "under"})
    d = d.assign(_side=side, line=d["stat_value"])
    out = None
    for s in ("over", "under"):
        x = d[d["_side"] == s].drop_duplicates(KEY)[KEY + ["american_price", "payout_multiplier"]] \
            .rename(columns={"american_price": f"{s}_price", "payout_multiplier": f"{s}_mult"})
        out = x if out is None else out.merge(x, on=KEY, how="outer")
    return out if out is not None else pd.DataFrame(columns=KEY + PRICE_COLS)


def _apply(history: pd.DataFrame, board: pd.DataFrame, seen_at: str) -> pd.DataFrame:
    last = history if history.empty else history.sort_values("seen_at").groupby(KEY).tail(1)
    prev = last[KEY + PRICE_COLS].rename(columns={c: c + "_prev" for c in PRICE_COLS})
    m = board.merge(prev, on=KEY, how="outer", indicator=True)
    both = m["_merge"] == "both"
    changed = (m["_merge"] == "left_only")
    for c in ("over_mult", "under_mult"):
        a, b = pd.to_numeric(m[c], errors="coerce"), pd.to_numeric(m[c + "_prev"], errors="coerce")
        changed |= both & ~((a == b) | (a.isna() & b.isna()))
    fo = fair_over(m["over_price"], m["under_price"]).set_axis(m.index)
    fo_prev = fair_over(m["over_price_prev"], m["under_price_prev"]).set_axis(m.index)
    # The lean only matters while both sides pay 1.00x (models/price_rules.flat_price_lean);
    # otherwise the multipliers alone carry what the picks need.
    flat = (pd.to_numeric(m["over_mult"], errors="coerce") == 1) & (pd.to_numeric(m["under_mult"], errors="coerce") == 1)
    changed |= both & flat & ((fo - fo_prev).abs() >= MIN_MOVE).fillna(fo.notna() != fo_prev.notna())
    add = m[changed][KEY + PRICE_COLS].assign(seen_at=seen_at)[COLS]
    return pd.concat([history, add], ignore_index=True) if not add.empty else history


def _trim(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
    return history[pd.to_datetime(history["seen_at"], utc=True, errors="coerce") >= cutoff]


def load() -> pd.DataFrame:
    return pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)


def _git_snapshots():
    """Every committed version of underdog_props.csv, oldest first."""
    log = subprocess.check_output(["git", "-C", REPO, "log", "--reverse", "--format=%H", "HEAD", "--",
                                   "data/raw/underdog_props.csv"], text=True).split()
    for sha in log:
        raw = subprocess.run(["git", "-C", REPO, "show", f"{sha}:data/raw/underdog_props.csv"],
                             capture_output=True).stdout
        try:
            yield pd.read_csv(io.BytesIO(raw), low_memory=False, usecols=lambda c: c in READ_COLS)
        except Exception:
            continue


def _add(history, df, seen):
    if df.empty or "pulled_at" not in df.columns or "american_price" not in df.columns:
        return history
    seen_at = pd.to_datetime(df["pulled_at"].iloc[0], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    if seen_at in seen:
        return history
    seen.add(seen_at)
    return _apply(history, _board(df), seen_at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-git", action="store_true", help="rebuild from every committed snapshot first")
    args = ap.parse_args()
    history = pd.DataFrame(columns=COLS) if args.seed_git else load()
    seen = set(history["seen_at"].astype(str)) if not history.empty else set()
    if args.seed_git:
        for df in _git_snapshots():
            history = _add(history, df, seen)
    path = os.path.join(RAW_DIR, "underdog_props.csv")
    if os.path.exists(path):
        history = _add(history, pd.read_csv(path, low_memory=False, usecols=lambda c: c in READ_COLS), seen)
    history = _trim(history).sort_values(["seen_at", "full_name", "stat_name", "line"])
    history.to_csv(OUT, index=False)
    print(f"{len(history)} price-history rows -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
