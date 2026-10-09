"""
Live bet tracking (added 2026-10-04): each logged leg's current stat vs its line during
games, and whether each entry is still alive.

Source: ESPN's public site API (unofficial, no key) -- the scoreboard for game
state/clock, and each game's summary box score for player stats. nflverse only
updates after games end, so it can't do this. Cached 60 s (shared by every viewer).

Underdog stat_name -> box-score value is STAT_FUNCS below. Not tracked live (shown as
such, never guessed): quarter/half props (period_*), first/last TD scorer,
passing_long and fantasy_points.
"""
import math
import re
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

from utils import normalize_name, load_csv_if_exists

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
NFLVERSE_TO_ESPN = {"LA": "LAR", "WAS": "WSH"}
LIVE_TTL = 60


def _num(x) -> float | None:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _made(x) -> float | None:
    """'2/3' -> 2."""
    m = re.match(r"\s*(\d+)\s*/", str(x))
    return float(m.group(1)) if m else None


def _att(x) -> float | None:
    m = re.match(r"\s*\d+\s*/\s*(\d+)", str(x))
    return float(m.group(1)) if m else None


def _sum(*vals):
    present = [v for v in vals if v is not None]
    return sum(present) if present else None


# Underdog stat_name -> f(stats dict) using ESPN box-score categories/labels.
STAT_FUNCS = {
    "passing_yds": lambda s: s.get("passing.YDS"),
    "passing_tds": lambda s: s.get("passing.TD"),
    "passing_ints": lambda s: s.get("passing.INT"),
    "passing_comps": lambda s: s.get("passing.CMP"),
    "passing_att": lambda s: s.get("passing.ATT"),
    "passing_and_rushing_yds": lambda s: _sum(s.get("passing.YDS"), s.get("rushing.YDS")),
    "rushing_yds": lambda s: s.get("rushing.YDS"),
    "rushing_att": lambda s: s.get("rushing.CAR"),
    "rushing_long": lambda s: s.get("rushing.LONG"),
    "receiving_rec": lambda s: s.get("receiving.REC"),
    "receptions": lambda s: s.get("receiving.REC"),
    "receiving_yds": lambda s: s.get("receiving.YDS"),
    "receiving_tgts": lambda s: s.get("receiving.TGTS"),
    "receiving_long": lambda s: s.get("receiving.LONG"),
    "rush_rec_yds": lambda s: _sum(s.get("rushing.YDS"), s.get("receiving.YDS")),
    "rush_rec_tds": lambda s: _sum(s.get("rushing.TD"), s.get("receiving.TD")),
    "fumbles_lost": lambda s: s.get("fumbles.LOST"),
    "sacks": lambda s: s.get("defensive.SACKS"),
    "tackles": lambda s: s.get("defensive.SOLO"),
    "tackles_and_assists": lambda s: s.get("defensive.TOT"),
    "assists": lambda s: (None if s.get("defensive.TOT") is None
                          else s["defensive.TOT"] - (s.get("defensive.SOLO") or 0)),
    "defensive_ints": lambda s: s.get("interceptions.INT"),
    "field_goals_made": lambda s: s.get("kicking.FGM"),
    "extra_points_made": lambda s: s.get("kicking.XPM"),
    "kicking_points": lambda s: s.get("kicking.PTS"),
}
# Stats where 0 is the natural pre-touch value (player appeared, did nothing yet).
ZERO_OK = set(STAT_FUNCS)


@st.cache_data(ttl=3600, show_spinner=False)
def _week_ends(sched_mtime: float) -> pd.DataFrame:
    s = load_csv_if_exists("schedules.csv")
    if s is None:
        return pd.DataFrame(columns=["season", "week", "last_day"])
    reg = s[s["game_type"] == "REG"].assign(gd=lambda d: pd.to_datetime(d["gameday"], errors="coerce"))
    return (reg.groupby(["season", "week"])["gd"].max().rename("last_day").reset_index()
            .sort_values(["season", "week"]))


def bet_week(bet_date) -> tuple[int, int] | None:
    """(season, week) a bet placed on `bet_date` belongs to: the first regular-season week
    whose last game is on or after that date (a Tuesday bet is for the coming week)."""
    d = pd.to_datetime(bet_date, errors="coerce")
    if pd.isna(d):
        return None
    import os
    from utils import RAW_DIR
    p = os.path.join(RAW_DIR, "schedules.csv")
    ends = _week_ends(os.path.getmtime(p) if os.path.exists(p) else 0.0)
    hit = ends[ends["last_day"] >= d.normalize()]
    return (int(hit.iloc[0]["season"]), int(hit.iloc[0]["week"])) if not hit.empty else None


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def fetch_games(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """A week's games (default: ESPN's current week): id, home/away abbr, state
    (pre/in/post), detail, period, clock."""
    params = {"dates": season, "seasontype": 2, "week": week} if season and week else None
    resp = requests.get(SCOREBOARD, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    rows = []
    for e in resp.json().get("events", []):
        c = e["competitions"][0]
        st_ = c["status"]
        teams = {t["homeAway"]: t["team"]["abbreviation"] for t in c["competitors"]}
        scores = {t["homeAway"]: t.get("score") for t in c["competitors"]}
        rows.append({
            "event_id": e["id"], "home": teams.get("home"), "away": teams.get("away"),
            "home_score": scores.get("home"), "away_score": scores.get("away"),
            "state": st_["type"]["state"], "detail": st_["type"]["shortDetail"],
            "period": st_.get("period") or 0, "clock": st_.get("displayClock") or "",
            "kickoff": e.get("date"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def fetch_box(event_id: str) -> dict[str, dict]:
    """normalized player name -> {'team': abbr, 'category.LABEL': value, ...}."""
    resp = requests.get(SUMMARY, params={"event": event_id}, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    out: dict[str, dict] = {}
    for team in resp.json().get("boxscore", {}).get("players", []):
        abbr = team.get("team", {}).get("abbreviation")
        for cat in team.get("statistics", []):
            name, labels = cat.get("name"), cat.get("labels", [])
            for ath in cat.get("athletes", []):
                key = normalize_name(ath["athlete"].get("displayName", ""))
                rec = out.setdefault(key, {"team": abbr, "display": ath["athlete"].get("displayName")})
                for lab, val in zip(labels, ath.get("stats", [])):
                    if name == "passing" and lab == "C/ATT":
                        rec["passing.CMP"], rec["passing.ATT"] = _made(val), _att(val)
                    elif name == "kicking" and lab == "FG":
                        rec["kicking.FGM"] = _made(val)
                    elif name == "kicking" and lab == "XP":
                        rec["kicking.XPM"] = _made(val)
                    elif name == "passing" and lab == "SACKS":
                        continue  # "5-30" = sacks taken, not a prop
                    else:
                        rec[f"{name}.{lab}"] = _num(val)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _player_teams() -> dict[str, str]:
    """normalized name -> ESPN team abbr, from the team-site rosters (for games not
    started yet, where the player isn't in any box score)."""
    df = load_csv_if_exists("nfl_rosters.csv")
    if df is None:
        return {}
    df = df.assign(_k=df["player"].map(normalize_name)).drop_duplicates("_k")
    return {k: NFLVERSE_TO_ESPN.get(t, t) for k, t in zip(df["_k"], df["team_abbr"])}


def _minutes_elapsed(period: int, clock: str) -> float | None:
    m = re.match(r"(\d+):(\d+)", clock or "")
    if not m or not period:
        return None
    left = int(m.group(1)) + int(m.group(2)) / 60
    return min(60.0, (min(period, 4) - 1) * 15 + (15 - left)) if period <= 4 else 60.0


def track_leg(player: str, stat: str, choice: str, line, bet_date=None) -> dict:
    """{'state': upcoming|live|final|untracked|unknown, 'value', 'status': hit|miss|push|
    void|alive|pending, 'text', 'game', 'pace'}

    `bet_date` (the log's date) picks the NFL week the bet was for, so a pick from an
    earlier week is graded from that week's final box score -- not tracked against this
    week's game. Without it, ESPN's current week is used."""
    line = _num(line)
    stat = str(stat or "")
    over = str(choice).lower() in ("over", "higher")
    out = {"state": "unknown", "value": None, "status": "pending", "text": "", "game": "", "pace": None}
    if stat not in STAT_FUNCS:
        out.update(state="untracked", text="not tracked live — check Underdog")
        return out
    try:
        wk = bet_week(bet_date) if bet_date is not None else None
        games = fetch_games(*wk) if wk else fetch_games()
    except Exception as e:  # noqa: BLE001
        out["text"] = f"live feed unavailable ({type(e).__name__})"
        return out
    key = normalize_name(player)
    team = _player_teams().get(key)
    game = None
    if team is not None and not games.empty:
        g = games[(games["home"] == team) | (games["away"] == team)]
        game = g.iloc[0] if not g.empty else None
    # Search started games' box scores (covers trades / roster-file misses too).
    stats = None
    for _, g in games[games["state"].isin(["in", "post"])].iterrows():
        if game is not None and g["event_id"] != game["event_id"] and team is not None:
            continue
        try:
            box = fetch_box(g["event_id"])
        except Exception:  # noqa: BLE001
            continue
        if key in box:
            stats, game = box[key], g
            break
    if game is None:
        out["text"] = "no game found this week"
        return out
    if game["state"] == "pre":
        out["game"] = f"{game['away']} @ {game['home']} · {game['detail']}"
    else:
        out["game"] = f"{game['away']} {game['away_score']} @ {game['home']} {game['home_score']} · {game['detail']}"
    if game["state"] == "pre":
        out.update(state="upcoming", text=f"kicks off {game['detail'].split(' - ')[-1]}")
        return out
    value = STAT_FUNCS[stat](stats) if stats else None
    if value is None and stats is not None:
        value = 0.0  # in the box score, no entry in this category yet
    final = game["state"] == "post"
    if value is None:
        if final:
            out.update(state="final", status="void", text="DNP / not in box score — Underdog voids it")
        else:
            out.update(state="live", value=0.0, text="no stats yet")
            value = 0.0
        if final:
            return out
    out["value"] = value
    out["state"] = "final" if final else "live"
    if line is None:
        out["text"] = f"{value:g}"
        return out
    if final:
        out["status"] = "push" if value == line else ("hit" if (value > line) == over else "miss")
    elif over and value > line:
        out["status"] = "hit"  # can't go back under
    elif not over and value > line:
        out["status"] = "miss"
    else:
        out["status"] = "alive"
    if out["status"] == "alive":
        gap = line - value  # >= 0 here
        # Smallest whole number of additional units that clears the line (6.5 - 4 -> 3),
        # and for unders the most it can still add without going over (6.5 - 4 -> 2).
        out["text"] = (f"needs {math.floor(gap) + 1:g} more" if over
                       else f"can add {math.ceil(gap) - 1 if gap == int(gap) else math.floor(gap):g} more")
    mins = _minutes_elapsed(int(game["period"]), game["clock"]) if not final else None
    if mins and mins >= 5 and stat.endswith(("_yds", "_rec", "_att", "_tgts", "_comps")):
        out["pace"] = round(value / mins * 60, 1)
    return out


ICON = {"hit": "✅", "miss": "❌", "push": "➖", "void": "⚪", "alive": "🟡", "pending": "⏳"}


def entry_status(leg_results: list[str]) -> str:
    """Entry-level status from per-leg statuses (manual results or tracker)."""
    s = [r for r in leg_results if r not in ("push", "void")]
    if any(r in ("miss", "lost") for r in s):
        return "lost"
    if s and all(r in ("hit", "won") for r in s):
        return "won"
    if not s:
        return "push"
    return "live" if any(r in ("alive", "hit", "won") for r in s) else "pending"
