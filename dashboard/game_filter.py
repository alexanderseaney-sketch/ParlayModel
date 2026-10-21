"""
"Which game" for every pick list: the game each team plays next (from schedules.csv),
labelled "BAL @ BUF · 11/01 10:00 AM" in Pacific time, plus the shared game filter
the Picks views and My parlay use.

The filter's choice lives in st.session_state["game_sel"], not in the widget's own key:
Streamlit forgets a widget's state when its view isn't drawn, and the choice should
survive switching between Top picks, +EV Finder and the Underdog board.
"""
import os

import pandas as pd
import streamlit as st

from utils import RAW_DIR, load_csv_if_exists, normalize_name

ALL = "All games"
TZ = "US/Pacific"
_STORE = "game_sel"


@st.cache_data(ttl=600, show_spinner=False)
def _next_games(mtime: float, hour: str) -> dict[str, dict]:
    """team -> {"key", "label", "kickoff"} for that team's next game (or the one on now)."""
    s = load_csv_if_exists("schedules.csv")
    if s is None or s.empty:
        return {}
    s = s.dropna(subset=["gameday", "away_team", "home_team"]).copy()
    # nflverse gameday/gametime are US Eastern.
    s["kickoff"] = (pd.to_datetime(s["gameday"] + " " + s["gametime"].fillna("13:00"), errors="coerce")
                    .dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT").dt.tz_convert(TZ))
    # A game stays "next" until ~4 hours after kickoff, so in-progress games still filter.
    s = s[s["kickoff"] >= pd.Timestamp(hour, tz="UTC") - pd.Timedelta(hours=4)].sort_values("kickoff")
    out = {}
    for r in s.itertuples():
        g = {"key": r.game_id, "kickoff": r.kickoff,
             "label": f"{r.away_team} @ {r.home_team} · {r.kickoff.strftime('%m/%d %I:%M %p')}"}
        out.setdefault(r.away_team, g)
        out.setdefault(r.home_team, g)
    return out


def next_games() -> dict[str, dict]:
    path = os.path.join(RAW_DIR, "schedules.csv")
    hour = pd.Timestamp.now(tz="UTC").floor("h").isoformat()
    return _next_games(os.path.getmtime(path) if os.path.exists(path) else 0.0, hour)


def game_key(team) -> str | None:
    g = next_games().get(team) if isinstance(team, str) else None
    return g["key"] if g else None


def game_label(team) -> str:
    g = next_games().get(team) if isinstance(team, str) else None
    return g["label"] if g else ""


@st.cache_data(ttl=600, show_spinner=False)
def _roster_teams(mtime: float) -> dict[str, str]:
    df = load_csv_if_exists("nfl_rosters.csv")
    if df is None:
        return {}
    df = df.assign(_k=df["player"].apply(normalize_name))
    df = df.sort_values("roster_status", key=lambda s: ~s.fillna("").str.startswith("Active"))
    return df.drop_duplicates("_k").set_index("_k")["team_abbr"].to_dict()


def teams_for_names(names: pd.Series) -> pd.Series:
    """Current team per player name (the Underdog board has no team abbreviation column)."""
    path = os.path.join(RAW_DIR, "nfl_rosters.csv")
    teams = _roster_teams(os.path.getmtime(path) if os.path.exists(path) else 0.0)
    return names.astype(str).map(normalize_name).map(teams)


def game_filter(teams: pd.Series, key: str) -> str | None:
    """Draw the "Game" dropdown for a pick list (games ordered by kickoff) and return the chosen game's key, or None for all games."""
    games = next_games()
    keys = teams.map(lambda t: games[t]["key"] if isinstance(t, str) and t in games else None)
    by_key = {g["key"]: g for g in games.values()}
    options = [ALL] + sorted(keys.dropna().unique(), key=lambda k: (by_key[k]["kickoff"], by_key[k]["label"]))
    saved = st.session_state.get(_STORE, ALL)
    if saved not in options:
        saved = ALL
    wkey = f"_game_filter_{key}"
    st.session_state[wkey] = saved

    def _save():
        st.session_state[_STORE] = st.session_state[wkey]

    st.selectbox("Game", options, key=wkey, on_change=_save,
                 format_func=lambda k: ALL if k == ALL else by_key[k]["label"])
    return None if saved == ALL else saved


def selected() -> str | None:
    """The game chosen on the Picks tab, if any (My parlay uses it as its default game)."""
    sel = st.session_state.get(_STORE, ALL)
    return None if sel == ALL else sel
