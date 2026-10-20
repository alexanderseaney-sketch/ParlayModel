"""
Automatic bet settlement (added 2026-10-20): grades every pending leg in the bet log
from the final box score, so My Bets settles entries without anyone tapping Won/Lost.

Sources, both free: the ESPN box scores live_tracker uses (no key), and as a backup when
ESPN can't be reached, nflverse's weekly player stats (data/raw/weekly_stats.csv, pulled
by the 6-hourly refresh). A leg is written back only once its game is final:
  hit -> "won", miss -> "lost", push (exactly on the line) or void (player inactive / not
  in the box score) -> "push", which bet_entries.settle_entry removes from the entry and
  re-prices it at the reduced multiplier, the way Underdog does.
Legs still to be played, still live, or on a stat type the box score can't grade stay
"pending". Which stat types can be graded is live_tracker.STAT_FUNCS (Underdog stat_name
-> ESPN box-score value): add a stat there and it is tracked live AND auto-graded. The
nflverse backup has its own map, WEEKLY_FUNCS below (it can't void: a player missing from
nflverse may just be a name mismatch, so that leg waits for ESPN or a manual settle).

The final stat value is stored next to the result in an "actual" column.
"""
import os

import pandas as pd
import streamlit as st

from live_tracker import STAT_FUNCS, bet_week, track_leg
from utils import RAW_DIR, normalize_name

_STATUS_TO_RESULT = {"hit": "won", "miss": "lost", "push": "push", "void": "push"}


def _sum(*cols):
    return lambda w: sum(w[c].fillna(0) for c in cols)


# Underdog stat_name -> f(nflverse weekly_stats rows).
WEEKLY_FUNCS = {
    "passing_yds": lambda w: w["passing_yards"],
    "passing_tds": lambda w: w["passing_tds"],
    "passing_ints": lambda w: w["interceptions"],
    "passing_comps": lambda w: w["completions"],
    "passing_att": lambda w: w["attempts"],
    "passing_and_rushing_yds": _sum("passing_yards", "rushing_yards"),
    "rushing_yds": lambda w: w["rushing_yards"],
    "rushing_att": lambda w: w["carries"],
    "receiving_rec": lambda w: w["receptions"],
    "receptions": lambda w: w["receptions"],
    "receiving_yds": lambda w: w["receiving_yards"],
    "receiving_tgts": lambda w: w["targets"],
    "rush_rec_yds": _sum("rushing_yards", "receiving_yards"),
    "rush_rec_tds": _sum("rushing_tds", "receiving_tds"),
    "fumbles_lost": _sum("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
}


def can_grade(stat) -> bool:
    return str(stat or "") in STAT_FUNCS or str(stat or "") in WEEKLY_FUNCS


@st.cache_data(ttl=3600, show_spinner=False)
def _weekly(mtime: float) -> pd.DataFrame:
    w = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    w = w[w["season_type"] == "REG"] if "season_type" in w.columns else w
    return w.assign(_k=w["player_display_name"].map(normalize_name))


def _weekly_value(row: pd.Series) -> float | None:
    """The leg's final stat from nflverse, or None when nflverse can't settle it (stat
    not mapped, week not posted yet, player not found or ambiguous)."""
    f = WEEKLY_FUNCS.get(str(row.get("stat") or ""))
    path = os.path.join(RAW_DIR, "weekly_stats.csv")
    if f is None or not os.path.exists(path):
        return None
    wk = bet_week(bet_date_for_week(row))
    if wk is None:
        return None
    w = _weekly(os.path.getmtime(path))
    hit = w[(w["season"] == wk[0]) & (w["week"] == wk[1]) & (w["_k"] == normalize_name(row["player"]))]
    if len(hit) != 1:
        return None
    v = f(hit).iloc[0]
    return float(v) if pd.notna(v) else None


def bet_date_for_week(row: pd.Series):
    """The date to pick the bet's NFL week from. Rows saved before 2026-10-20 stamped
    `date` in the server's UTC clock, so a Monday-night bet saved after 8pm ET reads as
    Tuesday -- which would point at NEXT week's game. Such a row (Tuesday, logged before
    08:00 that same UTC day) is moved back to Monday."""
    d = pd.to_datetime(row.get("date"), errors="coerce")
    logged = pd.to_datetime(row.get("logged_at"), errors="coerce")
    if (pd.notna(d) and pd.notna(logged) and d.dayofweek == 1
            and logged.normalize() == d.normalize() and logged.hour < 8):
        return (d - pd.Timedelta(days=1)).date().isoformat()
    return row.get("date")


def grade_leg(row: pd.Series) -> tuple[str, float | None, dict]:
    """(result to store -- won|lost|push|pending, final stat value, tracker output)."""
    t = track_leg(row["player"], row["stat"], row["choice"], row["line"], bet_date_for_week(row))
    if t["state"] == "final" and t["status"] in _STATUS_TO_RESULT:
        return _STATUS_TO_RESULT[t["status"]], t.get("value"), t
    if t["state"] in ("unknown", "untracked"):  # ESPN unreachable / player or stat not found
        value, line = _weekly_value(row), pd.to_numeric(row["line"], errors="coerce")
        if value is not None and pd.notna(line):
            over = str(row["choice"]).lower() in ("over", "higher")
            result = "push" if value == line else ("won" if (value > line) == over else "lost")
            return result, value, t
    return "pending", None, t


def grade_pending(log: pd.DataFrame) -> tuple[pd.DataFrame, int, dict]:
    """Grade every pending leg. Returns (updated log, legs newly settled, tracker output
    for the legs still pending keyed by (logged_at, player, stat) -- for the page to show
    live progress)."""
    out = log.copy()
    if "actual" not in out.columns:
        out["actual"] = pd.NA
    out["actual"] = out["actual"].astype(object)
    res = out["result"].fillna("pending").astype(str).str.lower().str.strip()
    settled, live = 0, {}
    for idx in out.index[res.isin(["pending", ""])]:
        row = out.loc[idx]
        result, value, t = grade_leg(row)
        if result == "pending":
            live[(str(row["logged_at"]), str(row["player"]), str(row["stat"]))] = t
            continue
        out.at[idx, "result"] = result
        out.at[idx, "actual"] = value
        settled += 1
    return out, settled, live
