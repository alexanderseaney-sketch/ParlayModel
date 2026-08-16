"""
Pulls current NFL news from ESPN's public RSS feed.

Previously used ESPN's undocumented JSON API (site.api.espn.com), which let us break
news out per-team. That endpoint now hard-blocks every request behind an Akamai edge
"Access Denied" -- confirmed from both a local dev machine and the deployed Streamlit
Cloud app, with headers ranging from a bare User-Agent to a full browser-mimicking set
(Accept/Accept-Language/Referer/Origin), all identically blocked. That's ESPN's bot
mitigation working as intended on an unofficial endpoint, not something to route
around -- so this switched to their public RSS feed instead, a real, documented,
publicly-served route rather than a workaround of the block.

Trade-off: RSS is league-wide only, no per-team breakdown -- ESPN's old team-specific
and injury-specific RSS routes (e.g. .../rss/nfl/team/_/name/dal) turned out to be dead
too, silently serving the same generic fallback feed instead of 404ing, so they're
skipped rather than pulled as if real. Per-team granularity is still covered by
pull_sbnation_news.py's 32-team pull; this is a second, differently-sourced signal on
top of that, same relationship pull_nbcsports_news.py already has to SB Nation.

Usage:
    python data/pull_espn_news.py
"""
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "espn_news.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
TIMEOUT = 15
DC_NS = {"dc": "http://purl.org/dc/elements/1.1/"}

FEEDS = {
    "league": "https://www.espn.com/espn/rss/nfl/news",
}


def fetch_feed(name: str, url: str) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    rows = []
    for item in root.find("channel").findall("item"):
        creator_el = item.find("dc:creator", DC_NS)
        rows.append({
            "source": name,
            "headline": item.findtext("title"),
            "description": item.findtext("description"),
            "author": creator_el.text if creator_el is not None else None,
            "published": item.findtext("pubDate"),
            "link": item.findtext("link"),
        })
    return rows


def pull_all_feeds() -> pd.DataFrame:
    all_rows = []
    for name, url in FEEDS.items():
        try:
            rows = fetch_feed(name, url)
            all_rows.extend(rows)
            print(f"[{name}] {len(rows)} articles.")
        except (requests.RequestException, ET.ParseError) as e:
            print(f"[{name}] FAILED ({e})")
    return pd.DataFrame(all_rows)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    new_df = pull_all_feeds()
    if new_df.empty:
        print("WARNING: 0 articles pulled -- feed may have changed. Nothing written.")
        return
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
