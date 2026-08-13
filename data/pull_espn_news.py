"""
Pulls current NFL news from ESPN's unofficial public API (no key required):
league-wide news plus per-team news, covering injuries, signings/trades, suspensions,
and general storylines that can affect gameplay but won't show up in box-score data.

NOTE: this uses ESPN's undocumented endpoints. They're widely used and free, but ESPN
can change or remove them without notice — that's why every pull is validated and logged
loudly on failure rather than failing silently.

Usage:
    python data/pull_espn_news.py                  # league news + all 32 teams
    python data/pull_espn_news.py --league-only     # skip per-team pulls (faster)
"""
import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
TIMEOUT = 15


def get_team_ids() -> dict:
    """Returns {team_abbreviation: espn_team_id} for all 32 teams."""
    resp = requests.get(f"{BASE}/teams", headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    teams = {}
    for group in data["sports"][0]["leagues"][0]["teams"]:
        t = group["team"]
        teams[t["abbreviation"]] = t["id"]
    return teams


def _parse_news_response(data: dict, source: str) -> list[dict]:
    rows = []
    for article in data.get("articles", []):
        categories = article.get("categories", [])
        athletes = [c.get("description") for c in categories if c.get("type") == "athlete"]
        teams_tagged = [c.get("description") for c in categories if c.get("type") == "team"]
        rows.append({
            "source": source,
            "headline": article.get("headline"),
            "description": article.get("description"),
            "published": article.get("published"),
            "type": article.get("type"),
            "athletes_tagged": "; ".join(a for a in athletes if a),
            "teams_tagged": "; ".join(t for t in teams_tagged if t),
            "link": (article.get("links", {}).get("web", {}) or {}).get("href"),
        })
    return rows


def pull_league_news(limit: int = 50) -> pd.DataFrame:
    resp = requests.get(f"{BASE}/news", params={"limit": limit}, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    rows = _parse_news_response(resp.json(), source="league")
    df = pd.DataFrame(rows)
    print(f"[league_news] {len(df)} articles.")
    return df


def pull_team_news(teams: dict, pause_seconds: float = 0.3) -> pd.DataFrame:
    all_rows = []
    for abbr, team_id in teams.items():
        try:
            resp = requests.get(f"{BASE}/news", params={"team": team_id}, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            rows = _parse_news_response(resp.json(), source=f"team:{abbr}")
            all_rows.extend(rows)
            print(f"[team_news] {abbr}: {len(rows)} articles.")
        except requests.RequestException as e:
            print(f"[team_news] {abbr}: FAILED ({e})")
        time.sleep(pause_seconds)  # be polite to an unofficial/undocumented API
    return pd.DataFrame(all_rows)


def validate_news(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        print(f"[{name}] WARNING: 0 articles returned. Endpoint may have changed — check manually.")
        return
    n_missing_headline = df["headline"].isna().sum()
    n_missing_date = df["published"].isna().sum()
    print(f"[{name}] {len(df)} total rows.")
    if n_missing_headline or n_missing_date:
        print(f"[{name}] WARNINGS: {n_missing_headline} missing headline, {n_missing_date} missing published date")


def main():
    parser = argparse.ArgumentParser(description="Pull current NFL news from ESPN")
    parser.add_argument("--league-only", action="store_true", help="Skip per-team news pulls")
    parser.add_argument("--limit", type=int, default=50, help="Max league-news articles to pull")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    league_df = pull_league_news(args.limit)
    validate_news(league_df, "league_news")

    if not args.league_only:
        teams = get_team_ids()
        print(f"[teams] found {len(teams)} teams.")
        team_df = pull_team_news(teams)
        validate_news(team_df, "team_news")
        combined = pd.concat([league_df, team_df], ignore_index=True)
    else:
        combined = league_df

    combined = combined.drop_duplicates(subset=["headline", "published"])
    combined["pulled_at"] = pulled_at

    out_path = os.path.join(RAW_DIR, "espn_news.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} unique articles -> {out_path}")


if __name__ == "__main__":
    main()
