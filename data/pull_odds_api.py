"""
Pulls CURRENT sportsbook player-prop lines from The Odds API (added 2026-10-02) --
the live sibling of pull_historical_odds.py. Feeds the +EV Finder's "sportsbook
consensus" column (a second market opinion next to Underdog's own prices) and the
line-shopping idea in IDEAS.md.

SETUP: same key as the historical puller -- ODDS_API_KEY in .env (or the
environment / Streamlit secrets). Never created automatically.

CREDITS: live event odds cost 1 credit per MARKET per REGION per EVENT (the events
list itself is free). One region x 6 markets x ~14 games is ~84 credits for a full
week's slate, so --max-credits is required and the puller stops before exceeding it.
The dashboard reads the saved CSV and only re-pulls on an explicit button press,
cached for 5 minutes (st.cache_data(ttl=300)) so reruns never re-spend credits.

Output: data/raw/odds_api_props.csv, one row per (book, player, stat, line, side)
with american/decimal/implied plus a per-book de-vigged fair probability, and
`stat_name` mapped to Underdog's naming so the two boards join directly.

Usage:
    python data/pull_odds_api.py --max-credits 90
    python data/pull_odds_api.py --max-credits 30 --markets player_reception_yds,player_receptions
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
from odds_utils import american_to_decimal, decimal_to_implied, devig_two_way  # noqa: E402

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "odds_api_props.csv")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# The Odds API market key -> Underdog stat_name.
MARKET_TO_STAT = {
    "player_pass_yds": "passing_yds",
    "player_pass_tds": "passing_tds",
    "player_pass_interceptions": "passing_ints",
    "player_rush_yds": "rushing_yds",
    "player_reception_yds": "receiving_yds",
    "player_receptions": "receiving_rec",
    "player_anytime_td": "rush_rec_tds",
}
DEFAULT_MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds",
                   "player_receptions", "player_pass_tds", "player_anytime_td"]


def api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY")
    if key and key != "your_odds_api_key_here":
        return key
    try:  # Streamlit secrets, when imported from the dashboard
        import streamlit as st
        return st.secrets.get("ODDS_API_KEY")
    except Exception:
        return None


def get_events(key: str) -> list[dict]:
    resp = requests.get(f"{BASE_URL}/sports/{SPORT}/events", params={"apiKey": key}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_event_odds(key: str, event_id: str, markets: list[str], regions: str = "us") -> tuple[dict, str | None]:
    resp = requests.get(f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds", params={
        "apiKey": key, "regions": regions, "markets": ",".join(markets), "oddsFormat": "american",
    }, timeout=20)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining")


def flatten_event(event: dict, pulled_at: str) -> list[dict]:
    """One row per (bookmaker, market, outcome). Player props put the player in
    `description` and Over/Under (or Yes/No for anytime TD) in `name`."""
    rows = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            stat = MARKET_TO_STAT.get(market.get("key"))
            if stat is None:
                continue
            for o in market.get("outcomes", []):
                side = str(o.get("name", "")).lower()
                side = {"yes": "over", "no": "under"}.get(side, side)
                line = o.get("point")
                if line is None and market.get("key") == "player_anytime_td":
                    line = 0.5
                rows.append({
                    "book": bm.get("key"), "book_title": bm.get("title"),
                    "event_id": event.get("id"), "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"), "away_team": event.get("away_team"),
                    "player": o.get("description"), "market": market.get("key"), "stat_name": stat,
                    "line": line, "choice": side, "american": o.get("price"),
                    "last_update": market.get("last_update"), "pulled_at": pulled_at,
                })
    return rows


def standardise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["decimal"] = american_to_decimal(pd.to_numeric(df["american"], errors="coerce"))
    df["implied_prob"] = decimal_to_implied(df["decimal"].to_numpy())
    key = ["book", "event_id", "player", "stat_name", "line"]
    over = df[df["choice"] == "over"][key + ["implied_prob"]].drop_duplicates(key)
    under = df[df["choice"] == "under"][key + ["implied_prob"]].drop_duplicates(key)
    pair = over.merge(under, on=key, suffixes=("_o", "_u"))
    pair["fair_over"], _ = devig_two_way(pair["implied_prob_o"].to_numpy(), pair["implied_prob_u"].to_numpy())
    df = df.merge(pair[key + ["fair_over"]], on=key, how="left")
    df["fair_prob"] = df["fair_over"].where(df["choice"] == "over", 1 - df["fair_over"])
    return df.drop(columns=["fair_over"])


def consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Per (player, stat, line): median de-vigged P(over) across books, number of books,
    and the best available over/under price (line shopping)."""
    if df.empty:
        return df
    over = df[df["choice"] == "over"]
    under = df[df["choice"] == "under"]
    g = over.groupby(["player", "stat_name", "line"]).agg(
        consensus_fair_over=("fair_prob", "median"), n_books=("book", "nunique"),
        best_over_decimal=("decimal", "max")).reset_index()
    u = under.groupby(["player", "stat_name", "line"]).agg(best_under_decimal=("decimal", "max")).reset_index()
    return g.merge(u, on=["player", "stat_name", "line"], how="left")


def pull(max_credits: int, markets: list[str], regions: str = "us", key: str | None = None) -> pd.DataFrame:
    key = key or api_key()
    if not key:
        raise RuntimeError("ODDS_API_KEY not set -- add it to .env (see .env.example).")
    pulled_at = datetime.now(timezone.utc).isoformat()
    events = get_events(key)
    now = pd.Timestamp.now(tz="UTC")
    upcoming = [e for e in events if pd.Timestamp(e["commence_time"]) > now]
    cost_per_event = len(markets) * len(regions.split(","))
    rows, spent = [], 0
    for ev in sorted(upcoming, key=lambda e: e["commence_time"]):
        if spent + cost_per_event > max_credits:
            print(f"[odds-api] budget reached ({spent}/{max_credits} credits) -- stopping.")
            break
        data, remaining = get_event_odds(key, ev["id"], markets, regions)
        spent += cost_per_event
        rows.extend(flatten_event(data, pulled_at))
        print(f"[odds-api] {ev['away_team']} @ {ev['home_team']}: ok (~{spent} credits, {remaining} left)")
    df = standardise(rows)
    print(f"[odds-api] {len(df)} outcome rows from {df['book'].nunique() if not df.empty else 0} books.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-credits", type=int, required=True)
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    ap.add_argument("--regions", default="us")
    args = ap.parse_args()
    df = pull(args.max_credits, args.markets.split(","), args.regions)
    if not df.empty:
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
