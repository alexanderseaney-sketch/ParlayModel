"""
Where each Underdog line opened this week, and the "line already moved to the Under"
hold-back (added 2026-10-23).

Graded on real closing lines, 2026 weeks 1 and 4-7 (backtesting/line_history_graded.csv,
Good+ picks on volume stats): when the model likes the Under but Underdog has already
cut the line 2%+ since it opened that week, the Under hit 48.3% (n=406), against 56.0%
for the rest of the Under picks. Lower in every graded week. The market moved toward the
Under for a reason the model can't see (role news, weather, injuries), and the value is
gone by the time we'd bet it. Over picks after the line went up showed no such drop
(54.4%, n=114), so only Unders are held back.

Opening line = the first line data/raw/underdog_line_history.csv logged for that player
and stat after the previous NFL week's last game day, so last week's line never counts
as this week's open.
"""
import numpy as np
import pandas as pd

# Volume stats only: TD, interception, sack and yes / no props move in price, not line.
STATS = {
    "receiving_yds", "rushing_yds", "passing_yds", "rush_rec_yds", "passing_and_rushing_yds",
    "receiving_rec", "receiving_tgts", "rushing_att", "passing_att", "passing_comps",
    "receiving_long", "rushing_long", "passing_long",
    "tackles", "assists", "tackles_and_assists", "fantasy_points", "kicking_points",
}
DROP = 0.02  # line cut by at least this share of its opening value


def _week_bounds(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["game_type"] == "REG"] if "game_type" in schedules.columns else schedules
    wk = reg.groupby(["season", "week"])["gameday"].agg(["min", "max"]).reset_index()
    wk["min"] = pd.to_datetime(wk["min"])
    wk["max"] = pd.to_datetime(wk["max"])
    wk = wk.sort_values("min").reset_index(drop=True)
    # Lines for a week are posted after the previous week's last game.
    wk["start"] = (wk["max"].shift(1) + pd.Timedelta(days=1)).fillna(wk["min"] - pd.Timedelta(days=30))
    wk["end"] = wk["max"] + pd.Timedelta(days=1)
    return wk


def _history(history: pd.DataFrame, normalize) -> pd.DataFrame:
    h = history[history["event"] != "pulled"].copy()
    h["t"] = (pd.to_datetime(h["seen_at"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None))
    h["_k"] = h["full_name"].apply(normalize)
    return h.sort_values("t")


def week_open_lines(history: pd.DataFrame, schedules: pd.DataFrame, season: int, week: int,
                    normalize) -> pd.DataFrame:
    """(_k, stat_name, open_line) for one week."""
    wk = _week_bounds(schedules)
    w = wk[(wk["season"] == season) & (wk["week"] == week)]
    if w.empty:
        return pd.DataFrame(columns=["_k", "stat_name", "open_line"])
    w = w.iloc[0]
    h = _history(history, normalize)
    h = h[(h["t"] >= w["start"]) & (h["t"] < w["end"])]
    return h.drop_duplicates(["_k", "stat_name"])[["_k", "stat_name", "line"]] \
        .rename(columns={"line": "open_line"})


def current_open_lines(history: pd.DataFrame, schedules: pd.DataFrame, now: pd.Timestamp,
                       normalize) -> pd.DataFrame:
    """(_k, stat_name, open_line) for the week the board at `now` (US/Eastern, naive) is for."""
    wk = _week_bounds(schedules)
    up = wk[wk["end"] > now]
    if up.empty:
        return pd.DataFrame(columns=["_k", "stat_name", "open_line"])
    w = up.iloc[0]
    return week_open_lines(history, schedules, int(w["season"]), int(w["week"]), normalize)


def moved_to_under(stat_name, line, open_line, picks_under) -> np.ndarray:
    """True where the pick is an Under on a volume stat whose line was cut DROP or more
    since it opened."""
    stat = pd.Series(stat_name).astype(str).to_numpy()
    line = np.asarray(line, float)
    open_line = np.asarray(open_line, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cut = (open_line - line) / np.where(open_line > 0, open_line, np.nan)
    return np.isin(stat, list(STATS)) & np.asarray(picks_under, bool) & (np.nan_to_num(cut) >= DROP)
