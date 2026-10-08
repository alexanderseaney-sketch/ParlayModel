"""
Bet-log entry helpers shared by the Bet Log (app.py), Bankroll and Home pages: group
picks into entries, read an entry's payout multiple, map stored results to tracker
statuses. Moved out of app.py unchanged (2026-10 redesign) so other pages can reuse them.
"""
import re

import pandas as pd


def _with_entries(bets: pd.DataFrame) -> pd.DataFrame:
    """Bet rows with an entry_id on every row. Rows logged before entries existed (no
    entry_id) are treated as their own single-pick entry."""
    b = bets.copy().reset_index(drop=True)
    for col in ("entry_id", "entry_payout"):
        if col not in b.columns:
            b[col] = pd.NA
    missing = b["entry_id"].isna() | (b["entry_id"].astype(str).str.strip().isin(["", "nan"]))
    # Older rows: picks saved together share a date + note (e.g. the assistant's
    # "Parlay #3, 12.9x"), so group on that; a row with no note stays a single.
    notes = b["notes"].fillna("").astype(str).str.strip() if "notes" in b.columns else pd.Series("", index=b.index)
    grouped = missing & notes.ne("")
    b.loc[grouped, "entry_id"] = "legacy|" + b.loc[grouped, "date"].astype(str) + "|" + notes[grouped]
    single = missing & ~grouped
    b.loc[single, "entry_id"] = "row" + b.index[single].astype(str)
    b["result"] = b["result"].fillna("pending").replace("", "pending")
    return b


def _entry_payout(legs: pd.DataFrame) -> float | None:
    """The entry's payout multiple: entry_payout if logged, else (single picks logged
    the old way) that pick's multiplier_or_odds."""
    vals = legs["entry_payout"].dropna() if "entry_payout" in legs.columns else pd.Series(dtype=object)
    vals = vals[vals.astype(str).str.strip().ne("")]
    if not vals.empty:
        return _payout_multiple(vals.iloc[0])
    if len(legs) == 1 and "multiplier_or_odds" in legs.columns:
        return _payout_multiple(legs["multiplier_or_odds"].iloc[0])
    m = re.search(r"(\d+(?:\.\d+)?)\s*x\b", str(legs["notes"].iloc[0]) if "notes" in legs.columns else "")
    return float(m.group(1)) if m else None


def _payout_multiple(val) -> float | None:
    try:
        out = float(str(val).lower().replace("x", "").strip())
        return out if out > 0 else None
    except (TypeError, ValueError):
        return None


_RESULT_TO_STATUS = {"won": "hit", "lost": "miss", "push": "push", "pending": "pending"}
_STATUS_TO_RESULT = {"hit": "won", "miss": "lost", "push": "push", "void": "push"}
