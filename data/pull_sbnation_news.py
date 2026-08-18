"""
Pulls daily news/analysis from each NFL team's dedicated SB Nation blog (Vox Media
"Chorus" platform) via their public Atom feeds — injury notes, roster/depth-chart
chatter, and inside coverage that's more team-specific and higher-frequency than
ESPN's league-wide news, which is blocked from this environment (403 Access Denied at
ESPN's Akamai edge, confirmed via curl — unrelated to the sandbox network restriction
that blocks Underdog; not something to route around, so ESPN news is left broken until
tried from a different network).

CONFIRMED WORKING (2026-08-15): all 32 team sites listed below were scraped live from
SB Nation's own NFL nav (https://www.sbnation.com/nfl), not guessed. Every site follows
the same Vox Media Chorus platform convention: an Atom feed at /rss/index.xml
(/rss/current.xml 301-redirects to the same place). Spot-checked 4 different team sites
— all returned HTTP 200 with real, current articles.

Unlike the Underdog puller, this accumulates a running history file (dedup'd by article
link) rather than daily snapshots — individual articles are naturally unique and
timestamped, so there's no "current value at time T" to snapshot, just a growing set of
articles worth keeping across runs since each team's feed only holds its ~20 most
recent posts.

Usage:
    python data/pull_sbnation_news.py
    python data/pull_sbnation_news.py --teams "Philadelphia Eagles,Dallas Cowboys"
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "sbnation_news.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
TIMEOUT = 15
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}

# Verified live from https://www.sbnation.com/nfl on 2026-08-15 — do not guess/replace
# entries without re-checking that page, site names change ownership occasionally.
TEAM_SITES = {
    "Arizona Cardinals": "https://www.revengeofthebirds.com",
    "Atlanta Falcons": "https://www.thefalcoholic.com",
    "Baltimore Ravens": "https://www.baltimorebeatdown.com",
    "Buffalo Bills": "https://www.buffalorumblings.com",
    "Carolina Panthers": "https://www.catscratchreader.com",
    "Chicago Bears": "https://www.windycitygridiron.com",
    "Cincinnati Bengals": "https://www.cincyjungle.com",
    "Cleveland Browns": "https://www.dawgsbynature.com",
    "Dallas Cowboys": "https://www.bloggingtheboys.com",
    "Denver Broncos": "https://www.milehighreport.com",
    "Detroit Lions": "https://www.prideofdetroit.com",
    "Green Bay Packers": "https://www.acmepackingcompany.com",
    "Houston Texans": "https://www.battleredblog.com",
    "Indianapolis Colts": "https://www.stampedeblue.com",
    "Jacksonville Jaguars": "https://www.bigcatcountry.com",
    "Kansas City Chiefs": "https://www.arrowheadpride.com",
    "Las Vegas Raiders": "https://www.silverandblackpride.com",
    "Los Angeles Rams": "https://www.turfshowtimes.com",
    "Miami Dolphins": "https://www.thephinsider.com",
    "Minnesota Vikings": "https://www.dailynorseman.com",
    "New England Patriots": "https://www.patspulpit.com",
    "New Orleans Saints": "https://www.canalstreetchronicles.com",
    "New York Giants": "https://www.bigblueview.com",
    "New York Jets": "https://www.ganggreennation.com",
    "Philadelphia Eagles": "https://www.bleedinggreennation.com",
    "Pittsburgh Steelers": "https://www.behindthesteelcurtain.com",
    "Los Angeles Chargers": "https://www.boltsfromtheblue.com",
    "San Francisco 49ers": "https://www.ninersnation.com",
    "Seattle Seahawks": "https://www.fieldgulls.com",
    "Tampa Bay Buccaneers": "https://www.bucsnation.com",
    "Tennessee Titans": "https://www.musiccitymiracles.com",
    "Washington Commanders": "https://www.hogshaven.com",
}


def _text(entry: ET.Element, tag: str) -> str | None:
    el = entry.find(f"a:{tag}", ATOM_NS)
    return el.text if el is not None else None


def fetch_team_feed(team: str, base_url: str) -> list[dict]:
    resp = requests.get(f"{base_url}/rss/index.xml", headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    rows = []
    for entry in root.findall("a:entry", ATOM_NS):
        link_el = entry.find("a:link[@rel='alternate']", ATOM_NS)
        author_el = entry.find("a:author/a:name", ATOM_NS)
        rows.append({
            "team": team,
            "source": base_url,
            "headline": _text(entry, "title"),
            "summary": _text(entry, "summary"),
            "author": author_el.text if author_el is not None else None,
            "published": _text(entry, "published") or _text(entry, "updated"),
            "link": link_el.get("href") if link_el is not None else None,
        })
    return rows


def pull_all_teams(teams: dict, pause_seconds: float = 0.3) -> pd.DataFrame:
    all_rows = []
    for team, base_url in teams.items():
        try:
            rows = fetch_team_feed(team, base_url)
            all_rows.extend(rows)
            print(f"[{team}] {len(rows)} articles.")
        except (requests.RequestException, ET.ParseError) as e:
            print(f"[{team}] FAILED ({e})")
        time.sleep(pause_seconds)  # be polite to sites that aren't expecting a bot
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Pull daily NFL news from SB Nation team sites")
    parser.add_argument("--teams", default=None, help="Comma-separated team names to limit to (default: all 32)")
    args = parser.parse_args()

    teams = TEAM_SITES
    if args.teams:
        wanted = {t.strip() for t in args.teams.split(",")}
        teams = {k: v for k, v in TEAM_SITES.items() if k in wanted}
        missing = wanted - teams.keys()
        if missing:
            print(f"WARNING: unknown team names ignored: {missing}")

    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    new_df = pull_all_teams(teams)
    if new_df.empty:
        # Raise rather than a silent return -- see pull_espn_news.py's main() for why
        # a quiet exit-0 here is a real trap (run_pull_script reads it as success).
        raise RuntimeError("0 articles pulled across all teams -- feeds may have changed. Nothing written.")
    new_df["pulled_at"] = pulled_at

    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=["link"], keep="first")
    print(f"Deduped {before - len(combined)} already-seen articles ({len(new_df)} fetched this run).")

    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(combined)} total unique articles -> {OUT_PATH}")


if __name__ == "__main__":
    main()
