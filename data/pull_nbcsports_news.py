"""
Pulls NFL rumor/insider news from NBC Sports (ProFootballTalk's new home after a site
consolidation -- the old profootballtalk.nbcsports.com now redirects to
nbcsports.com/nfl/profootballtalk). Public RSS, no auth.

Why this exists alongside pull_sbnation_news.py rather than instead of it: SB Nation's
32 team blogs are longer-form fan analysis/recaps; this is the opposite in kind --
short, atomic, insider-sourced roster/injury blurbs (Mike Florio, Charean Williams,
etc.) that are closer to "ahead of the official injury report" than a game recap is.
Different signal, not a replacement.

CONFIRMED WORKING (2026-08-15): checked several candidate URLs before picking these --
nbcsports.com/nfl.rss is genuinely live (items dated same-day), nbcsports.com/nfl/
profootballtalk.rss is real but ~1 day behind, and a third candidate
(nfl/profootballtalk/rumor-mill.rss) turned out to be stale/abandoned (items 3+ weeks
old despite returning HTTP 200) -- deliberately excluded rather than pulling dead data.

Real limitation worth knowing: each feed only holds ~4 items, refreshed as new ones
publish -- this is NOT a deep archive like SB Nation's ~10-per-team pull. Running this
on the same daily schedule will miss items on busy news days; it accumulates whatever
it catches (deduped by link) rather than guaranteeing full coverage. Worth tightening
the pull interval later if that gap matters more than the current cadence assumes.

Usage:
    python data/pull_nbcsports_news.py
"""
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "nbcsports_news.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
TIMEOUT = 15
DC_NS = {"dc": "http://purl.org/dc/elements/1.1/"}

FEEDS = {
    "nfl_main": "https://www.nbcsports.com/nfl.rss",
    "profootballtalk": "https://www.nbcsports.com/nfl/profootballtalk.rss",
}


def fetch_feed(name: str, url: str) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    rows = []
    for item in root.find("channel").findall("item"):
        creator_el = item.find("dc:creator", DC_NS)
        rows.append({
            "feed": name,
            "headline": item.findtext("title"),
            "summary": item.findtext("description"),
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
            print(f"[{name}] {len(rows)} items.")
        except (requests.RequestException, ET.ParseError) as e:
            print(f"[{name}] FAILED ({e})")
    return pd.DataFrame(all_rows)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    new_df = pull_all_feeds()
    if new_df.empty:
        # Raise rather than a silent return -- a quiet exit-0 here is a real trap,
        # since run_pull_script's subprocess.returncode check reads it as a
        # successful pull even though nothing was written.
        raise RuntimeError("0 items pulled across all feeds -- feeds may have changed. Nothing written.")
    new_df["pulled_at"] = pulled_at

    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=["link"], keep="first")
    print(f"Deduped {before - len(combined)} already-seen items ({len(new_df)} fetched this run).")

    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(combined)} total unique items -> {OUT_PATH}")


if __name__ == "__main__":
    main()
