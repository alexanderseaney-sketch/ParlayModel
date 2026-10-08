"""
Shared data for the Research pages added in the 2026-10 redesign (Home, Game Center,
Injury Tracker, Player Pages, Usage Trends, Matchup Heatmap, Bankroll). Loaders only --
no Streamlit layout -- so each page module stays presentation.

Everything reads the project's existing pulls: schedules.csv, weather_forecast.csv,
injuries.csv, nfl_rosters.csv, weekly_stats.csv, snap_counts.csv, the live Underdog
board (utils.load_underdog_props) and data/raw/underdog_line_history.csv (written by
data/archive_underdog_lines.py, committed so the hosted app has line history too).
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from utils import RAW_DIR, load_csv_if_exists, normalize_name, load_underdog_props

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
SKILL_POS = ["QB", "RB", "WR", "TE"]
# Main Underdog stat per position -- the line the Injury Tracker / Player Page track.
PRIMARY_STAT = {"QB": "passing_yds", "RB": "rushing_yds", "WR": "receiving_yds", "TE": "receiving_yds"}
STAT_WEEKLY_COL = {"passing_yds": "passing_yards", "rushing_yds": "rushing_yards", "receiving_yds": "receiving_yards",
                   "receiving_rec": "receptions"}
SEVERITY = {"—": 0, "Q": 1, "D": 2, "O": 3, "IR": 4}
STATUS_ABBR = {"Questionable": "Q", "Doubtful": "D", "Out": "O"}
PRACTICE_ABBR = {"Did Not Participate In Practice": "DNP", "Limited Participation in Practice": "LP",
                 "Full Participation in Practice": "FP"}


def _mtime(name: str) -> float:
    p = os.path.join(RAW_DIR, name)
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


# ------------------------------------------------------------------ games
def current_week() -> tuple[int, int] | None:
    """(season, week) of the next week with an unplayed REG game."""
    s = load_csv_if_exists("schedules.csv")
    if s is None:
        return None
    s = s[s["game_type"] == "REG"]
    unplayed = s[s["home_score"].isna()]
    if unplayed.empty:
        return None
    first = unplayed.sort_values(["season", "week"]).iloc[0]
    return int(first["season"]), int(first["week"])


def _kick_label(row) -> str:
    try:
        dt = pd.to_datetime(f"{row['gameday']} {row['gametime']}")
        hour = dt.strftime("%I:%M %p").lstrip("0")
        return f"{dt.strftime('%a').upper()} {hour}"
    except (ValueError, TypeError):
        return str(row.get("gameday", ""))


@st.cache_data(show_spinner=False)
def _week_games(sched_m: float, wx_m: float) -> pd.DataFrame:
    s = load_csv_if_exists("schedules.csv")
    cw = current_week()
    if s is None or cw is None:
        return pd.DataFrame()
    g = s[(s["season"] == cw[0]) & (s["week"] == cw[1]) & (s["game_type"] == "REG")].copy()
    g = g.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    wx = load_csv_if_exists("weather_forecast.csv")
    if wx is not None and not wx.empty:
        g = g.merge(wx[["game_id", "temp_forecast", "wind_forecast", "precip_prob_forecast"]], on="game_id", how="left")
    else:
        g["temp_forecast"] = g["wind_forecast"] = g["precip_prob_forecast"] = np.nan
    g["kick_label"] = g.apply(_kick_label, axis=1)
    g["matchup"] = g["away_team"] + " @ " + g["home_team"]
    # spread_line is the home margin nflverse expects (positive = home favored).
    sp = pd.to_numeric(g["spread_line"], errors="coerce")
    tot = pd.to_numeric(g["total_line"], errors="coerce")
    g["home_implied"] = (tot + sp) / 2
    g["away_implied"] = (tot - sp) / 2

    def spread_txt(r):
        v = r["spread_line"]
        if pd.isna(v):
            return "—"
        if v == 0:
            return "PK"
        fav = r["home_team"] if v > 0 else r["away_team"]
        return f"{fav} -{abs(v):g}"

    def weather_txt(r):
        if str(r.get("roof")) in ("dome", "closed"):
            return "Indoors"
        if pd.notna(r.get("temp_forecast")):
            bits = [f"{r['temp_forecast']:.0f}°F"]
            if pd.notna(r.get("wind_forecast")):
                bits.append(f"wind {r['wind_forecast']:.0f} mph")
            if pd.notna(r.get("precip_prob_forecast")) and r["precip_prob_forecast"] >= 30:
                bits.append(f"{r['precip_prob_forecast']:.0f}% rain")
            return " · ".join(bits)
        return "Retractable roof" if str(r.get("roof")) == "open" else "Forecast not out yet"

    g["spread_txt"] = g.apply(spread_txt, axis=1)
    g["weather_txt"] = g.apply(weather_txt, axis=1)
    return g


def week_games() -> pd.DataFrame:
    """This week's REG games with kickoff label, spread/total text, implied points, weather."""
    return _week_games(_mtime("schedules.csv"), _mtime("weather_forecast.csv"))


# ------------------------------------------------------------------ line history
@st.cache_data(show_spinner=False)
def _line_history(m: float) -> pd.DataFrame:
    h = load_csv_if_exists("underdog_line_history.csv")
    if h is None or h.empty:
        return pd.DataFrame(columns=["full_name", "stat_name", "line", "seen_at", "event", "_k"])
    h = h.copy()
    h["seen_at"] = pd.to_datetime(h["seen_at"], utc=True, errors="coerce")
    h["_k"] = h["full_name"].map(normalize_name)
    return h


def line_history() -> pd.DataFrame:
    return _line_history(_mtime("underdog_line_history.csv"))


def live_over_lines() -> pd.DataFrame:
    """Current single-game OVER lines from the live board: _k, stat_name, line, over_price, under_price."""
    props = load_underdog_props()
    if props is None or props.empty:
        return pd.DataFrame(columns=["_k", "full_name", "stat_name", "line", "over_price", "under_price"])
    p = props[~props["stat_name"].astype(str).str.startswith(("period_", "season_"))].copy()
    if "match_type" in p.columns:
        p = p[p["match_type"].fillna("Game") == "Game"]
    p["_k"] = p["full_name"].map(normalize_name)
    side = p["choice"].astype(str).str.lower()
    over = p[side.isin(["over", "higher"])][["_k", "full_name", "stat_name", "stat_value", "decimal_price"]]
    under = p[side.isin(["under", "lower"])][["_k", "stat_name", "stat_value", "decimal_price"]]
    out = over.merge(under, on=["_k", "stat_name", "stat_value"], how="left", suffixes=("", "_u"))
    return out.rename(columns={"stat_value": "line", "decimal_price": "over_price", "decimal_price_u": "under_price"}) \
        .drop_duplicates(["_k", "stat_name"])


def week_start_utc() -> pd.Timestamp | None:
    """When this week's board opened: the day after the previous week's last game."""
    s = load_csv_if_exists("schedules.csv")
    cw = current_week()
    if s is None or cw is None:
        return None
    prev = s[(s["season"] == cw[0]) & (s["week"] == cw[1] - 1)]
    if prev.empty:
        return None
    return pd.Timestamp(pd.to_datetime(prev["gameday"]).max() + pd.Timedelta(days=1), tz="UTC")


# ------------------------------------------------------------------ injuries
@st.cache_data(ttl=300, show_spinner=False)
def _injury_report(inj_m: float, ros_m: float, hist_m: float) -> pd.DataFrame:
    inj = load_csv_if_exists("injuries.csv")
    cw = current_week()
    if inj is None or cw is None:
        return pd.DataFrame()
    inj = inj[(inj["season"] == cw[0]) & inj["position"].isin(SKILL_POS)].copy()
    now = inj[inj["week"] == cw[1]]
    prev = inj[inj["week"] == cw[1] - 1]

    def status(df):
        return df.set_index("gsis_id")["report_status"].map(STATUS_ABBR).fillna("—")

    now_status, prev_status = status(now), status(prev)
    # Game designations (Q/D/O) are posted the day before a team's game; earlier in the
    # week its report only has practice participation. Until a team has posted, its
    # players are "TBD" rather than read as cleared -- otherwise every Q/O from last
    # week looks like an upgrade on a Wednesday.
    posted_teams = set(now.loc[now["report_status"].notna(), "team"])
    reporting_teams = set(now["team"])
    rows = now.drop_duplicates("gsis_id").copy()
    rows["now"] = np.where(rows["team"].isin(posted_teams),
                           rows["gsis_id"].map(now_status).fillna("—"), "TBD")
    rows["prev"] = rows["gsis_id"].map(prev_status).fillna("—")
    # On last week's report with a game status but absent from this week's practice
    # report = practising fully (cleared), when the team has filed this week at all.
    cleared = prev[~prev["gsis_id"].isin(now["gsis_id"])].drop_duplicates("gsis_id").copy()
    cleared = cleared[cleared["gsis_id"].map(prev_status).isin(["Q", "D", "O"])]
    cleared["prev"] = cleared["gsis_id"].map(prev_status)
    cleared["now"] = np.where(cleared["team"].isin(reporting_teams), "—", "TBD")
    cleared["practice_status"] = np.nan
    rows = pd.concat([rows, cleared], ignore_index=True)

    # Reserve/Injured on the current club roster overrides to IR.
    ros = load_csv_if_exists("nfl_rosters.csv")
    if ros is not None:
        ir = set(ros[ros["roster_status"].astype(str).str.startswith("Reserve/Injured")]["player"].map(normalize_name))
        rows.loc[rows["full_name"].map(normalize_name).isin(ir), "now"] = "IR"

    rows["injury"] = rows["report_primary_injury"].fillna(rows["practice_primary_injury"]).fillna("—")
    rows["practice"] = rows["practice_status"].map(PRACTICE_ABBR)
    rows["sev_prev"] = rows["prev"].map(SEVERITY)
    rows["sev_now"] = rows["now"].map(SEVERITY).fillna(rows["sev_prev"])  # TBD = unchanged
    rows["dir"] = np.where(rows["sev_now"] > rows["sev_prev"], "down",
                           np.where(rows["sev_now"] < rows["sev_prev"], "up", "same"))

    # Line move: this week's first recorded line for the primary stat vs the live line now.
    rows["_k"] = rows["full_name"].map(normalize_name)
    rows["stat"] = rows["position"].map(PRIMARY_STAT)
    hist, start = line_history(), week_start_utc()
    live = live_over_lines()
    opens, nows = {}, {}
    if not hist.empty and start is not None:
        wk = hist[hist["seen_at"] >= start]
        first = wk[wk["event"] != "pulled"].sort_values("seen_at").drop_duplicates(["_k", "stat_name"])
        opens = {(k, s): l for k, s, l in zip(first["_k"], first["stat_name"], first["line"])}
    if not live.empty:
        nows = {(k, s): l for k, s, l in zip(live["_k"], live["stat_name"], live["line"])}
    rows["line_open"] = [opens.get((k, s)) for k, s in zip(rows["_k"], rows["stat"])]
    rows["line_now"] = [nows.get((k, s)) for k, s in zip(rows["_k"], rows["stat"])]
    rows["line_pulled"] = rows["line_open"].notna() & rows["line_now"].isna()
    rows["line_move"] = pd.to_numeric(rows["line_now"], errors="coerce") - pd.to_numeric(rows["line_open"], errors="coerce")
    rows["abs_move"] = rows["line_move"].abs().fillna(0) + rows["line_pulled"].astype(float) * 999
    return rows[["full_name", "team", "position", "injury", "prev", "now", "dir", "practice", "stat",
                 "line_open", "line_now", "line_move", "line_pulled", "abs_move", "sev_now"]] \
        .sort_values(["abs_move", "sev_now"], ascending=False).reset_index(drop=True)


def injury_report() -> pd.DataFrame:
    return _injury_report(_mtime("injuries.csv"), _mtime("nfl_rosters.csv"), _mtime("underdog_line_history.csv"))


# ------------------------------------------------------------------ priced props
@st.cache_data(ttl=60, show_spinner=False)
def priced_candidates(prob_source: str = "Blend") -> pd.DataFrame:
    """The +EV Finder's priced, comparable single-game Underdog options (both sides),
    with `prob` = blended (default) or model probability for that side vs the real line.
    Shared by Game Center, Player Pages and Home; refreshed with the live board."""
    from ev_finder import build_candidates, _apply_prob_source
    c, _ = build_candidates()
    if c.empty:
        return c
    c = _apply_prob_source(c, prob_source)
    c["_k"] = c["player"].map(normalize_name)
    return c
