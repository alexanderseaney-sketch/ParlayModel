"""
Last-5-games form for a pick, like the mini bar charts on Underdog and Sleeper prop
cards: one bar per game the player actually played (a row in nflverse's
weekly_stats.csv), the line drawn across, and how many of those games went the way
the pick needs.

New stat types plug in by adding one entry to STAT_FORMULAS: the Underdog stat_name
mapped to the weekly_stats.csv column(s) that add up to it. Stats with no per-game
column in nflverse (first-half / first-quarter splits, longest reception, fantasy
points with Underdog's scoring) are left out and simply show no chart.
"""
import html

import pandas as pd
import streamlit as st

from research_data import _mtime
from utils import load_csv_if_exists, normalize_name

N_GAMES = 5

# Underdog stat_name -> weekly_stats.csv columns summed per game.
STAT_FORMULAS: dict[str, tuple[str, ...]] = {
    "passing_yds": ("passing_yards",),
    "passing_tds": ("passing_tds",),
    "passing_ints": ("interceptions",),
    "passing_comps": ("completions",),
    "passing_att": ("attempts",),
    "rushing_yds": ("rushing_yards",),
    "rushing_att": ("carries",),
    "receiving_yds": ("receiving_yards",),
    "receiving_rec": ("receptions",),
    "receptions": ("receptions",),
    "receiving_tgts": ("targets",),
    "rush_rec_tds": ("rushing_tds", "receiving_tds"),
    "rush_rec_yds": ("rushing_yards", "receiving_yards"),
    "passing_and_rushing_yds": ("passing_yards", "rushing_yards"),
    "fumbles_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
}

@st.cache_data(show_spinner=False)
def _weekly(mtime: float) -> pd.DataFrame:
    """The same weekly_stats.csv game logs the player card (player_page.py) shows: regular
    season, last two seasons, with a normalized name key. mtime only busts the cache."""
    w = load_csv_if_exists("weekly_stats.csv")
    if w is None:
        return pd.DataFrame()
    w = w[(w["season"] >= w["season"].max() - 1) & (w["season_type"].fillna("REG") == "REG")]
    w = w.assign(_k=w["player_display_name"].map(normalize_name))
    return w.sort_values(["season", "week"])


def recent_games(player: str, stat_name: str, team: str | None = None, n: int = N_GAMES) -> list[dict]:
    """The player's last n games for this stat, oldest first: [{"value", "opp", "season", "week"}].
    Empty when the stat has no per-game column or the player isn't in the data."""
    cols = STAT_FORMULAS.get(stat_name)
    if not cols:
        return []
    w = _weekly(_mtime("weekly_stats.csv"))
    if w.empty or any(c not in w.columns for c in cols):
        return []
    rows = w[w["_k"] == normalize_name(player)]
    if rows.empty:
        return []
    if rows["player_id"].nunique() > 1:
        # Two players with the same name: prefer the one on this team, else the most recent.
        latest = rows.groupby("player_id").tail(1)
        on_team = latest[latest["recent_team"] == team] if team else latest.iloc[0:0]
        pid = (on_team if not on_team.empty else latest).iloc[-1]["player_id"]
        rows = rows[rows["player_id"] == pid]
    rows = rows.tail(n)
    vals = rows[list(cols)].fillna(0).sum(axis=1)
    return [{"value": float(v), "opp": o if isinstance(o, str) else "", "season": int(s), "week": int(wk)}
            for v, o, s, wk in zip(vals, rows["opponent_team"], rows["season"], rows["week"])]


def hit(value: float, line: float, choice: str) -> bool | None:
    """True/False for whether this game would have won the pick, None for a push."""
    if value == line:
        return None
    over = str(choice).lower() in ("over", "higher")
    return value > line if over else value < line


CSS = """
<style>
.rf {display: flex; align-items: flex-end; gap: 10px; margin-top: 6px;}
.rf-bars {position: relative; display: flex; align-items: flex-end; gap: 4px;}
.rf-col {display: flex; flex-direction: column; align-items: center; width: 22px;}
.rf-v {font-size: 10px; line-height: 12px; margin-top: 1px; color: var(--pm-n300, #cfd3e5);
  font-family: 'JetBrains Mono', monospace;}
.rf-bar {width: 18px; border-radius: 3px 3px 0 0; min-height: 2px;}
.rf-hit {background: #3fb97a;}
.rf-miss {background: #d9596a;}
.rf-push {background: var(--pm-n600, #75798c);}
.rf-line {position: absolute; left: -2px; right: -2px; border-top: 1.5px dashed var(--pm-n300, #cfd3e5);
  opacity: .8; pointer-events: none;}
.rf-opp {font-size: 9px; line-height: 12px; color: var(--pm-n500, #9397ab); margin-top: 2px;}
.rf-count {font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', monospace; line-height: 1.1;}
.rf-count small {display: block; font-size: 10px; font-weight: 400; color: var(--pm-n500, #9397ab);
  font-family: Inter, sans-serif;}
.rf-good {color: #7ee2a8;}
.rf-bad {color: #f08a96;}
.rf-mid {color: var(--pm-n300, #cfd3e5);}
</style>
"""

_BAR_PX = 44  # height of the tallest bar


def chart_html(games: list[dict], line: float, choice: str) -> str:
    """A small bar chart of the games, with the line across and an "x/5" hit count."""
    if not games:
        return ""
    top = max(max(g["value"] for g in games), float(line), 1.0) * 1.1
    cols, hits = [], 0
    for g in games:
        h = hit(g["value"], line, choice)
        hits += h is True
        cls = "rf-push" if h is None else ("rf-hit" if h else "rf-miss")
        px = max(2, round(g["value"] / top * _BAR_PX))
        cols.append(
            f'<div class="rf-col" title="{g["season"]} week {g["week"]} vs {html.escape(g["opp"])}: {g["value"]:g}">'
            f'<div class="rf-bar {cls}" style="height:{px}px"></div><div class="rf-v">{g["value"]:g}</div>'
            f'<div class="rf-opp">{html.escape(g["opp"])}</div></div>')
    # Bars sit on a baseline 27px (value + opponent labels) above the bottom of .rf-bars.
    line_px = 27 + round(float(line) / top * _BAR_PX)
    n = len(games)
    tone = "rf-good" if hits / n >= 0.6 else ("rf-bad" if hits / n <= 0.4 else "rf-mid")
    return (f'<div class="rf"><div class="rf-bars" style="height:{_BAR_PX + 27}px">{"".join(cols)}'
            f'<div class="rf-line" style="bottom:{line_px}px"></div></div>'
            f'<div class="rf-count {tone}">{hits}/{n}<small>last {n}</small></div></div>')


def css():
    st.markdown(CSS, unsafe_allow_html=True)
