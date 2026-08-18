"""
Pulls REAL historical player-prop lines from The Odds API, to eventually backtest our
models against actual market lines instead of our own proxy line -- the single
most-repeated "top priority" item in this project's history (see README: 7-8 rounds
of feature testing concluded the remaining lever that could reveal genuine edge, not
just re-measure the same ceiling, is changing the TARGET from a proxy to a real
market line).

SETUP (can't be done automatically -- I don't create accounts, even for a free tier):
  1. Sign up at https://the-odds-api.com (free tier: 500 credits/month).
  2. Copy .env.example to .env and set ODDS_API_KEY to your real key.
  3. `pip install python-dotenv` if not already present (it's in requirements.txt).

CREDIT BUDGET IS THE CENTRAL DESIGN CONSTRAINT, confirmed directly against The Odds
API's own docs 2026-08-18 (not assumed): historical event-odds cost 10 credits PER
MARKET PER REGION PER EVENT; the events-list call is 1 credit (free if it returns
nothing). At 500 free credits/month, one region, and the 6 markets below, that's
10*6=60 credits per game -- only ~8 games/month. This is built around that scarcity:
  1. NEVER re-spends credits on an (event_id, snapshot_date, markets) combo already
     pulled -- tracked permanently in data/raw/historical_odds/_pulled.json, checked
     before every paid call.
  2. Targets ALREADY-PLAYED games only, most recent first -- need real graded
     outcomes to backtest against, not future lines.
  3. Takes an explicit --max-credits budget per run (no silent default that could burn
     a whole month's quota in one call) and stops before exceeding it.
  4. Snapshot requested ~10 minutes before kickoff (closing line, the standard
     "final market consensus" reference point in betting analysis) using
     schedules.csv's own gameday+gametime.

Market keys confirmed against The Odds API's own market-key reference page (not
guessed), mapped to this project's existing prop taxonomy:
  player_rush_reception_tds -> rush_rec_tds (this project's single largest prop family)
  player_anytime_td         -> period_first_touchdown_scored family
  player_reception_yds      -> receiving_yards
  player_rush_yds           -> rushing_yards
  player_pass_yds           -> passing_yards
  player_sacks              -> sacks
Historical player-prop data only exists from 2023-05-03 onward (The Odds API's own
stated start date for "additional markets" -- core game lines go back further, but
those aren't what this project needs).

HONEST CAVEAT: built and reasoned through carefully, but never run against a real
key (none exists yet) -- unlike every other data source added this session, this
one couldn't be verified end-to-end. The team-name mapping (nflverse uses
abbreviations like "BUF", The Odds API is expected to use full names like "Buffalo
Bills" per standard convention for this kind of API) and the exact response field
names beyond what's confirmed in this file's own research are the most likely things
to need a real debugging pass on the first live run.

Usage:
    python data/pull_historical_odds.py --max-credits 400
    python data/pull_historical_odds.py --max-credits 60 --markets player_rush_reception_tds
"""
import argparse
import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
HIST_DIR = os.path.join(RAW_DIR, "historical_odds")
PULLED_LOG_PATH = os.path.join(HIST_DIR, "_pulled.json")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
EARLIEST_PROP_DATE = pd.Timestamp("2023-05-03", tz="UTC")

DEFAULT_MARKETS = ["player_rush_reception_tds", "player_anytime_td", "player_reception_yds",
                    "player_rush_yds", "player_pass_yds", "player_sacks"]

EVENTS_LIST_COST = 1

# Standard nflverse abbreviation -> full team name, for matching against The Odds
# API's events (expected to use full names, standard convention for this kind of
# API -- not verified live, see module docstring's honest caveat).
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY not set. Sign up for a free key at https://the-odds-api.com, "
            "then copy .env.example to .env and set it there -- this can't be done "
            "automatically, even for the free tier.")
    return key


def _load_pulled_log() -> set:
    if not os.path.exists(PULLED_LOG_PATH):
        return set()
    with open(PULLED_LOG_PATH) as f:
        return {tuple(x) for x in json.load(f)}


def _save_pulled_log(pulled: set) -> None:
    with open(PULLED_LOG_PATH, "w") as f:
        json.dump([list(x) for x in sorted(pulled)], f, indent=2)


def get_historical_events(date: pd.Timestamp) -> tuple[list[dict], str | None]:
    resp = requests.get(f"{BASE_URL}/historical/sports/{SPORT}/events", params={
        "apiKey": _api_key(), "date": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", []), resp.headers.get("x-requests-remaining")


def get_historical_event_odds(event_id: str, date: pd.Timestamp, markets: list[str]) -> tuple[dict, str | None]:
    resp = requests.get(f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds", params={
        "apiKey": _api_key(), "date": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regions": "us", "markets": ",".join(markets), "oddsFormat": "american",
    }, timeout=20)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining")


def _flatten_event_odds(response: dict, game_id: str, season: int, week: int) -> list[dict]:
    """One row per (bookmaker, market, outcome) -- player name, stat, line, price."""
    event = response.get("data")
    if not event:
        return []
    rows = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            for outcome in market.get("outcomes", []):
                rows.append({
                    "game_id": game_id, "season": season, "week": week,
                    "event_id": event.get("id"), "commence_time": event.get("commence_time"),
                    "bookmaker": bm.get("key"), "market": market.get("key"),
                    "player_name": outcome.get("description"), "choice": outcome.get("name"),
                    "line": outcome.get("point"), "price": outcome.get("price"),
                    "snapshot_timestamp": response.get("timestamp"),
                })
    return rows


def pull(max_credits: int, markets: list[str]) -> pd.DataFrame:
    os.makedirs(HIST_DIR, exist_ok=True)
    pulled = _load_pulled_log()
    per_event_cost = 10 * len(markets)
    credits_used = 0
    all_rows = []

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"), low_memory=False)
    played = schedules[schedules["home_score"].notna()].copy()
    played["kickoff"] = pd.to_datetime(
        played["gameday"] + " " + played["gametime"].fillna("13:00"), utc=True, errors="coerce")
    played = played.dropna(subset=["kickoff"])
    played = played[played["kickoff"] >= EARLIEST_PROP_DATE]
    played = played.sort_values("kickoff", ascending=False)  # most recent first

    print(f"Budget: {max_credits} credits, {len(markets)} markets/event "
          f"({per_event_cost} credits/event + {EVENTS_LIST_COST}/events-list call).")

    for _, game in played.iterrows():
        if credits_used + EVENTS_LIST_COST + per_event_cost > max_credits:
            print(f"Stopping: {max_credits - credits_used} credits left, "
                  f"next pull needs {EVENTS_LIST_COST + per_event_cost}.")
            break

        snapshot_date = game["kickoff"] - pd.Timedelta(minutes=10)
        home_name = TEAM_NAMES.get(game["home_team"])
        away_name = TEAM_NAMES.get(game["away_team"])
        if home_name is None or away_name is None:
            continue

        try:
            events, remaining = get_historical_events(snapshot_date)
        except requests.RequestException as e:
            print(f"[{game['game_id']}] events lookup FAILED ({e})")
            continue
        credits_used += EVENTS_LIST_COST

        match = next((e for e in events if e.get("home_team") == home_name
                      and e.get("away_team") == away_name), None)
        if match is None:
            print(f"[{game['game_id']}] no matching event found in The Odds API for "
                  f"{away_name} @ {home_name} ({snapshot_date}) -- skipping, 0 credits spent on odds.")
            continue

        key = (match["id"], snapshot_date.isoformat(), ",".join(sorted(markets)))
        if key in pulled:
            continue  # already have this exact snapshot -- don't re-spend credits

        try:
            response, remaining = get_historical_event_odds(match["id"], snapshot_date, markets)
        except requests.RequestException as e:
            print(f"[{game['game_id']}] odds pull FAILED ({e})")
            continue
        credits_used += per_event_cost

        rows = _flatten_event_odds(response, game["game_id"], game["season"], game["week"])
        all_rows.extend(rows)
        pulled.add(key)
        print(f"[{game['game_id']}] {away_name} @ {home_name}: {len(rows)} prop lines. "
              f"({credits_used}/{max_credits} credits used this run, {remaining} remaining on key)")

    _save_pulled_log(pulled)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Pull historical NFL player-prop odds from The Odds API")
    parser.add_argument("--max-credits", type=int, required=True,
                         help="Hard cap on credits to spend this run (required, no silent default)")
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    args = parser.parse_args()

    _api_key()  # fail fast and clean on a missing key, before printing anything else
    df = pull(args.max_credits, args.markets)
    if df.empty:
        print("0 new prop lines pulled this run (budget exhausted immediately, or "
              "everything in range already pulled). Nothing written.")
        return

    out_path = os.path.join(HIST_DIR, "historical_player_props.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, low_memory=False)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["event_id", "bookmaker", "market", "player_name", "choice", "snapshot_timestamp"])
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(df)} total rows)")


if __name__ == "__main__":
    main()
