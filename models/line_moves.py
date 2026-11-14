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
# "Line moved your way" (2026-11-14): Underdog raises a line when Over money and news push
# it up, and on real closing lines those raises overshoot. Model Under picks on a line
# raised 10%+ since it opened hit 60.4% (n=889, 2026 wks 1, 4-9) against 55% for Unders
# on lines that didn't move, and beat them in every week from week 4 on. First seen on
# wks 1, 4-7 (62.8%, n=333) and it held on wks 8 and 9 (61%, 64%), which it had not seen.
# Taking the Under after a raise with no model behind it hit 58%; where the model liked
# the Over instead it hit 49%, so the model still has to agree. pick_rules counts these
# Unders as proven from 55% model confidence instead of 60%.
RISE = 0.10


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
    h = history.copy()
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
    h = h[h["t"] < w["end"]]
    # A team that played on Sunday gets next week's lines late Sunday or on Monday, before
    # the previous week's Monday game. Those rows sit before w["start"], so the old window
    # missed them and took a later, already-moved line as the open (2026-11-12: Antonio
    # Williams catches opened 4.5 on 11/09, were cut to 3.5, and the Under was not held
    # back). A line counts as this week's once it comes after the player's last-week line
    # was pulled; players with no pulled row fall back to the old start.
    prev = wk[wk["min"] < w["min"]]
    prev_start = prev["min"].iloc[-1] if len(prev) else w["start"]
    h = h[h["t"] >= prev_start]
    pulled = h[(h["event"] == "pulled") & (h["t"] < w["start"])]
    last_pull = pulled.groupby(["_k", "stat_name"])["t"].max().rename("_pull").reset_index()
    h = h[h["event"] != "pulled"].merge(last_pull, on=["_k", "stat_name"], how="left")
    h = h[h["t"] > h["_pull"].fillna(w["start"] - pd.Timedelta(seconds=1))]
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


def moved_your_way(stat_name, line, open_line, picks_under) -> np.ndarray:
    """True where the pick is an Under on a volume stat whose line was raised RISE or more
    since it opened."""
    stat = pd.Series(stat_name).astype(str).to_numpy()
    line = np.asarray(line, float)
    open_line = np.asarray(open_line, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rise = (line - open_line) / np.where(open_line > 0, open_line, np.nan)
    return np.isin(stat, list(STATS)) & np.asarray(picks_under, bool) & (np.nan_to_num(rise) >= RISE)
