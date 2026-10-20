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


# Underdog's legacy fixed table (models/parlay_calculator.STANDARD_PAYOUTS), used only to
# shrink an entry's payout after a void when the legs' own multipliers weren't logged.
_STANDARD_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5, 7: 65.0, 8: 120.0}


def settle_entry(legs: pd.DataFrame) -> tuple[str, float | None]:
    """(status, payout multiple actually paid) for one entry, under Underdog's rules:
    any lost leg loses the entry; a pushed or voided leg (stored as "push": DNP, inactive,
    landed exactly on the line) is removed and the entry pays at the reduced multiplier
    for the legs left; an entry left with fewer than two live legs is refunded (1.0x).
    status is won | lost | push (refunded) | pending."""
    res = legs["result"].fillna("pending").astype(str).str.lower().str.strip()
    full = _entry_payout(legs)
    if (res == "lost").any():
        return "lost", 0.0
    if not (res.isin(["won", "push"])).all():
        return "pending", full
    live = res == "won"
    if live.sum() == 0 or (len(legs) > 1 and live.sum() < 2):
        return "push", 1.0
    if live.all():
        return "won", full
    return "won", _reduced_payout(legs, live, full)


def _reduced_payout(legs: pd.DataFrame, live: pd.Series, full: float | None) -> float | None:
    """Full payout with the voided legs taken out. Each leg's own multiplier (logged
    since the My parlay screen) divides out exactly; otherwise scale by the fixed table."""
    if full is None:
        return None
    voided = (pd.to_numeric(legs["multiplier_or_odds"], errors="coerce")[~live.values]
              if "multiplier_or_odds" in legs.columns else pd.Series(dtype=float))
    if len(voided) and voided.notna().all() and (voided > 1).all():
        return round(full / float(voided.prod()), 2)
    n, k = len(legs), int(live.sum())
    if n in _STANDARD_PAYOUTS and k in _STANDARD_PAYOUTS:
        return round(full * _STANDARD_PAYOUTS[k] / _STANDARD_PAYOUTS[n], 2)
    return None
