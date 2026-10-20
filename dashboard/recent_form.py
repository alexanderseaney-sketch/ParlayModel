"""
Last-5-games form for a pick, like the mini bar charts on Underdog and Sleeper prop
cards: one bar per game the player actually played, the line drawn across, and how
many of those games went the way the pick needs.

The games come from data/raw/prop_game_logs.csv via ev_finder._last5_by_player, the
same last-5 the Picks page's "Against his last 5" tag uses. That file has one column per
Underdog stat_name, so a new prop type shows a chart as soon as its column is pulled.
"""
import html
import os

import pandas as pd
import streamlit as st

from utils import RAW_DIR, normalize_name

N_GAMES = 5


def recent_games(player: str, stat_name: str) -> list[dict]:
    """The player's last 5 games for this stat, oldest first:
    [{"value", "opp", "season", "week"}]. Empty when there's no column for the stat or
    the player isn't in the logs."""
    from ev_finder import _last5_by_player
    path = os.path.join(RAW_DIR, "prop_game_logs.csv")
    g = _last5_by_player(os.path.getmtime(path) if os.path.exists(path) else 0.0).get(normalize_name(player))
    if g is None or stat_name not in g.columns:
        return []
    vals = pd.to_numeric(g[stat_name], errors="coerce")
    opps = g["opponent_team"] if "opponent_team" in g.columns else [""] * len(g)
    return [{"value": float(v), "opp": o if isinstance(o, str) else "", "season": int(s), "week": int(wk)}
            for v, o, s, wk in zip(vals, opps, g["season"], g["week"]) if pd.notna(v)]


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
