"""
Yahoo Fantasy Sports connection for the Fantasy page.

Read-only. OAuth 2.0 authorization-code flow against the app's own URL as the
redirect: the user clicks an authorize link, Yahoo bounces back with ?code=, we
exchange it for an access + refresh token. The refresh token is what keeps the
connection alive; on Streamlit Community Cloud (no persistent disk) the user
pastes it into App Secrets once as `yahoo_refresh_token` and we bootstrap from
there on every cold start.

Secrets used (App settings -> Secrets, or .streamlit/secrets.toml locally):
    yahoo_client_id      = "..."      # from developer.yahoo.com/apps
    yahoo_client_secret  = "..."
    yahoo_redirect_uri   = "https://<your-app>.streamlit.app"   # must match the Yahoo app exactly
    yahoo_refresh_token  = "..."      # optional; printed by the UI after the first authorize

Yahoo's JSON is awful -- collections come back as {"0": {...}, "1": {...},
"count": n} and objects as lists of single-key dicts, sometimes nested. `_items`
and `_flatten` absorb that; everything above them works with plain dicts/lists.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

from utils import normalize_name, load_csv_if_exists

_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
_API = "https://fantasysports.yahooapis.com/fantasy/v2"

# Yahoo NFL stat ids we care about (stable across seasons).
_STAT_REC = 11          # receptions -> reception point value = PPR / half / std
_STAT_PASS_YD, _STAT_PASS_TD, _STAT_INT = 4, 5, 6
_STAT_RUSH_YD, _STAT_RUSH_TD = 9, 10
_STAT_REC_YD, _STAT_REC_TD = 12, 13

# Yahoo team abbreviations -> the ones the rest of the app uses (nflverse style).
_TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}


# --------------------------------------------------------------- config / secrets

def _secret(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def configured() -> bool:
    """True once the Yahoo app credentials are in secrets. Until then the Fantasy
    page just doesn't show the Yahoo panel."""
    return bool(_secret("yahoo_client_id") and _secret("yahoo_client_secret")
               and _secret("yahoo_redirect_uri"))


# ------------------------------------------------------------------ oauth tokens

def _exchange(grant: dict) -> dict:
    """POST to the token endpoint with HTTP Basic client auth. Returns the token
    dict augmented with an absolute `expires_at`."""
    r = requests.post(
        _TOKEN_URL, data=grant,
        auth=(_secret("yahoo_client_id"), _secret("yahoo_client_secret")),
        headers={"Accept": "application/json"}, timeout=20,
    )
    r.raise_for_status()
    tok = r.json()
    tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
    return tok


def _token_from_code(code: str) -> dict:
    return _exchange({
        "grant_type": "authorization_code",
        "redirect_uri": _secret("yahoo_redirect_uri"),
        "code": code,
    })


def _token_from_refresh(refresh_token: str) -> dict:
    tok = _exchange({
        "grant_type": "refresh_token",
        "redirect_uri": _secret("yahoo_redirect_uri"),
        "refresh_token": refresh_token,
    })
    # Yahoo usually echoes the same refresh_token back; keep the old one if it doesn't.
    tok.setdefault("refresh_token", refresh_token)
    return tok


def authorize_url() -> str:
    """Yahoo removed the 'Fantasy Sports' permission checkbox from app creation, so
    access is requested here via the `fspt-r` scope. If Yahoo ever rejects that
    scope for a given app, set the secret `yahoo_scope = ""` to drop it (the app
    then gets whatever its registered permissions allow)."""
    from urllib.parse import urlencode
    params = {
        "client_id": _secret("yahoo_client_id"),
        "redirect_uri": _secret("yahoo_redirect_uri"),
        "response_type": "code",
    }
    scope = _secret("yahoo_scope")
    scope = "fspt-r" if scope is None else scope
    if scope:
        params["scope"] = scope
    return _AUTH_URL + "?" + urlencode(params)


def _live_token() -> dict | None:
    """Current token dict, refreshing if expired. Session state first, then the
    `yahoo_refresh_token` secret as a cold-start bootstrap. None => not connected."""
    tok = st.session_state.get("yahoo_token")
    if not tok:
        seed = _secret("yahoo_refresh_token")
        if not seed:
            return None
        try:
            tok = _token_from_refresh(seed)
        except Exception as e:  # noqa: BLE001
            st.session_state["yahoo_auth_error"] = f"Stored refresh token rejected: {e}"
            return None
        st.session_state["yahoo_token"] = tok

    if tok.get("expires_at", 0) <= time.time():
        try:
            tok = _token_from_refresh(tok["refresh_token"])
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("yahoo_token", None)
            st.session_state["yahoo_auth_error"] = f"Token refresh failed: {e}"
            return None
        st.session_state["yahoo_token"] = tok
    return tok


_REDIRECT_PARAMS = ("code", "state", "error", "error_description")


def _clear_redirect_params() -> None:
    for k in _REDIRECT_PARAMS:
        try:
            st.query_params.pop(k, None)
        except Exception:  # noqa: BLE001
            pass


def redirect_params_seen() -> list[str]:
    """Which OAuth-redirect params are on the URL right now (names only, no
    values) -- shown in the panel so a failed round-trip is diagnosable."""
    return [k for k in _REDIRECT_PARAMS if st.query_params.get(k)]


def handle_oauth_redirect() -> None:
    """Process a bounce-back from Yahoo: an `error` param, or a `code` to exchange.
    Call once near the top of the Yahoo panel."""
    if st.session_state.get("yahoo_token"):
        _clear_redirect_params()
        return

    if st.query_params.get("error"):
        why = st.query_params.get("error_description") or "(no detail from Yahoo)"
        st.session_state["yahoo_auth_error"] = (
            f"Yahoo rejected the request — `{st.query_params.get('error')}`: {why}. "
            "Usual causes: the redirect URI doesn't exactly match the one registered "
            "on the Yahoo app, or the `fspt-r` scope isn't allowed (set secret "
            '`yahoo_scope = ""` and retry).'
        )
        _clear_redirect_params()
        return

    code = st.query_params.get("code")
    if not code:
        return
    try:
        st.session_state["yahoo_token"] = _token_from_code(code)
        st.session_state.pop("yahoo_auth_error", None)
    except requests.HTTPError as e:  # surface Yahoo's body, which explains most failures
        body = e.response.text[:400] if e.response is not None else ""
        st.session_state["yahoo_auth_error"] = f"Token exchange failed ({e}). Yahoo said: {body}"
    except Exception as e:  # noqa: BLE001
        st.session_state["yahoo_auth_error"] = f"Token exchange failed: {e}"
    finally:
        _clear_redirect_params()


def connected() -> bool:
    return _live_token() is not None


def disconnect() -> None:
    for k in ("yahoo_token", "yahoo_auth_error", "yahoo_league_key", "yahoo_team_key"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------- API plumbing

def _get(path: str) -> dict:
    tok = _live_token()
    if not tok:
        raise RuntimeError("not connected to Yahoo")
    r = requests.get(
        f"{_API}/{path}",
        headers={"Authorization": f"Bearer {tok['access_token']}", "Accept": "application/json"},
        params={"format": "json"}, timeout=25,
    )
    if r.status_code == 401:
        st.session_state.pop("yahoo_token", None)      # force a fresh refresh next run
        raise RuntimeError("Yahoo session expired — reconnect.")
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=300)
def _get_cached(path: str, _token_key: str) -> dict:
    """`_token_key` (a hash of the refresh token) only exists to scope the cache to
    the connected account; the leading underscore keeps Streamlit from hashing it."""
    return _get(path)


def _cached(path: str) -> dict:
    tok = _live_token() or {}
    key = str(hash(tok.get("refresh_token", "")))
    return _get_cached(path, key)


# ---- Yahoo-JSON shape helpers

def _items(node: Any):
    """Yield the members of a Yahoo pseudo-array {"0":..,"1":..,"count":n}."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in ("count",):
                yield v
    elif isinstance(node, list):
        yield from node


def _flatten(node: Any) -> dict:
    """Yahoo represents one object as a list of single-key dicts, sometimes with
    nested lists. Merge it all into one flat dict."""
    out: dict = {}
    if isinstance(node, dict):
        return node
    if isinstance(node, list):
        for part in node:
            if isinstance(part, dict):
                out.update(part)
            elif isinstance(part, list):
                out.update(_flatten(part))
    return out


# ----------------------------------------------------------------------- reads

@st.cache_data(show_spinner=False, ttl=600)
def my_leagues(_token_key: str) -> pd.DataFrame:
    """Every NFL league the connected user is in this season: league_key, name,
    num_teams, scoring_type, season."""
    data = _get("users;use_login=1/games;game_keys=nfl/leagues")
    users = data["fantasy_content"]["users"]
    rows = []
    for u in _items(users):
        user = _flatten(u.get("user", []))
        for g in _items(user.get("games", {})):
            game = _flatten(g.get("game", []))
            for lg in _items(game.get("leagues", {})):
                lg_meta = _flatten(lg.get("league", []))
                if lg_meta.get("league_key"):
                    rows.append({
                        "league_key": lg_meta["league_key"],
                        "name": lg_meta.get("name", lg_meta["league_key"]),
                        "num_teams": int(lg_meta.get("num_teams", 0) or 0),
                        "scoring_type": lg_meta.get("scoring_type", ""),
                        "season": lg_meta.get("season", ""),
                    })
    return pd.DataFrame(rows)


def leagues() -> pd.DataFrame:
    tok = _live_token() or {}
    return my_leagues(str(hash(tok.get("refresh_token", ""))))


def league_settings(league_key: str) -> dict:
    """Roster slots + the reception point value, so the app can auto-pick PPR vs
    Half-PPR and know how many of each position start."""
    data = _cached(f"league/{league_key}/settings")
    league = data["fantasy_content"]["league"]
    meta = _flatten(league[0]) if isinstance(league, list) else {}
    settings = _flatten(league[1].get("settings", [])) if len(league) > 1 else {}

    slots: dict[str, int] = {}
    for rp in _items(settings.get("roster_positions", [])):
        p = _flatten(rp.get("roster_position", {}))
        pos, cnt = p.get("position"), int(p.get("count", 0) or 0)
        if pos:
            slots[pos] = slots.get(pos, 0) + cnt

    rec_val = 0.0
    for sm in _items(settings.get("stat_modifiers", {}).get("stats", [])):
        s = _flatten(sm.get("stat", {}))
        if str(s.get("stat_id")) == str(_STAT_REC):
            try:
                rec_val = float(s.get("value", 0) or 0)
            except ValueError:
                rec_val = 0.0
    scoring = "ppr" if rec_val >= 0.75 else "half" if rec_val >= 0.25 else "std"

    return {
        "name": meta.get("name", league_key),
        "num_teams": int(meta.get("num_teams", 0) or 0),
        "slots": slots,                       # {"QB":1,"RB":2,"WR":2,"TE":1,"W/R/T":1,"BN":6,...}
        "reception_point": rec_val,
        "scoring": scoring,
        "starters": _starters_from_slots(slots),
    }


_FLEX_SLOTS = {"W/R/T", "W/R", "R/W/T", "Q/W/R/T"}


def _starters_from_slots(slots: dict[str, int]) -> dict[str, int]:
    """Collapse Yahoo's slot names into the {QB,RB,WR,TE,FLEX} shape the draft
    board's VBD math expects."""
    out = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0}
    for slot, cnt in slots.items():
        if slot in out:
            out[slot] += cnt
        elif slot in _FLEX_SLOTS:
            out["FLEX"] += cnt
    return out


def my_team_key(league_key: str) -> str | None:
    data = _cached("users;use_login=1/games;game_keys=nfl/teams")
    for u in _items(data["fantasy_content"]["users"]):
        user = _flatten(u.get("user", []))
        for g in _items(user.get("games", {})):
            game = _flatten(g.get("game", []))
            for t in _items(game.get("teams", {})):
                team = _flatten(t.get("team", []))
                tk = team.get("team_key", "")
                if tk.startswith(league_key + ".t."):
                    return tk
    return None


def _parse_players(node: Any) -> pd.DataFrame:
    """Shared shape for roster / free-agent / draft player lists."""
    rows = []
    for p in _items(node):
        pl = p.get("player") if isinstance(p, dict) else None
        if pl is None:
            continue
        meta = _flatten(pl[0]) if isinstance(pl, list) and pl else _flatten(pl)
        extra = _flatten(pl[1]) if isinstance(pl, list) and len(pl) > 1 else {}
        name = _flatten(meta.get("name", {})).get("full", "")
        team = str(meta.get("editorial_team_abbr", "")).upper()
        sel = _flatten(extra.get("selected_position", [])).get("position", "")
        rows.append({
            "yahoo_id": str(meta.get("player_id", "")),
            "player_key": meta.get("player_key", ""),
            "name": name,
            "team": _TEAM_FIX.get(team, team),
            "position": meta.get("display_position", ""),
            "slot": sel,
            "status": meta.get("status", ""),            # "", IR, O, Q, D, SUSP, PUP...
        })
    return pd.DataFrame(rows)


def my_roster(league_key: str, week: int | None = None) -> pd.DataFrame:
    tk = st.session_state.get("yahoo_team_key") or my_team_key(league_key)
    if not tk:
        return pd.DataFrame()
    st.session_state["yahoo_team_key"] = tk
    path = f"team/{tk}/roster" + (f";week={week}" if week else "")
    data = _cached(path)
    team = data["fantasy_content"]["team"]
    roster = _flatten(team[1].get("roster", {})) if isinstance(team, list) and len(team) > 1 else {}
    return _match_players(_parse_players(roster.get("0", {}).get("players", roster.get("players", {}))))


def free_agents(league_key: str, position: str | None = None, count: int = 50) -> pd.DataFrame:
    filt = f";position={position}" if position else ""
    path = f"league/{league_key}/players;status=FA{filt};sort=AR;count={count};start=0"
    data = _cached(path)
    league = data["fantasy_content"]["league"]
    players = _flatten(league[1].get("players", {})) if isinstance(league, list) and len(league) > 1 else {}
    return _match_players(_parse_players(players))


def draft_results(league_key: str) -> pd.DataFrame:
    """pick / round / team_key / yahoo player_key for a completed draft. Empty if
    the league hasn't drafted."""
    try:
        data = _cached(f"league/{league_key}/draftresults")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    league = data["fantasy_content"]["league"]
    drs = _flatten(league[1].get("draft_results", {})) if isinstance(league, list) and len(league) > 1 else {}
    rows = []
    for d in _items(drs):
        dr = _flatten(d.get("draft_result", {}))
        if dr.get("player_key"):
            rows.append({"pick": int(dr.get("pick", 0) or 0),
                         "round": int(dr.get("round", 0) or 0),
                         "team_key": dr.get("team_key", ""),
                         "player_key": dr["player_key"]})
    return pd.DataFrame(rows)


def standings(league_key: str) -> pd.DataFrame:
    data = _cached(f"league/{league_key}/standings")
    league = data["fantasy_content"]["league"]
    st_node = _flatten(league[1].get("standings", [])) if isinstance(league, list) and len(league) > 1 else {}
    rows = []
    for t in _items(st_node.get("0", {}).get("teams", st_node.get("teams", {}))):
        team = _flatten(t.get("team", []))
        outcome = _flatten(team.get("team_standings", {}).get("outcome_totals", {}))
        rows.append({
            "team": team.get("name", ""),
            "rank": int(_flatten(team.get("team_standings", {})).get("rank", 0) or 0),
            "wins": int(outcome.get("wins", 0) or 0),
            "losses": int(outcome.get("losses", 0) or 0),
            "points_for": float(_flatten(team.get("team_standings", {})).get("points_for", 0) or 0),
        })
    return pd.DataFrame(rows).sort_values("rank")


# ------------------------------------------------------------- player matching

def _match_players(df: pd.DataFrame) -> pd.DataFrame:
    """Add `gsis_id` (via the yahoo_id bridge) and `norm_name` so callers can join
    Yahoo players onto our projections / props by id first, name second."""
    if df.empty:
        return df.assign(gsis_id=pd.Series(dtype="string"), norm_name=pd.Series(dtype="string"))
    df = df.copy()
    df["norm_name"] = df["name"].map(normalize_name)
    bridge = load_csv_if_exists("player_ids.csv")
    if bridge is not None and "yahoo_id" in bridge.columns:
        b = bridge.dropna(subset=["yahoo_id"]).copy()
        b["yahoo_id"] = b["yahoo_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        id_map = b.set_index("yahoo_id")["gsis_id"].to_dict()
        df["gsis_id"] = df["yahoo_id"].map(id_map)
    else:
        df["gsis_id"] = pd.NA
    return df
