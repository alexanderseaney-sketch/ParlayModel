"""
Pulls NFL depth charts from Footballguys.com -- all 32 teams, offense + defense +
special teams, with structured per-player status tags (Q/PUP/IR/SUS/NFI/O/CEL/EX).
Free, no login, single page load, explicitly dated on the page itself.

This is the structured injury/role signal this project has been trying to get from
news sources all session -- unlike SB Nation or NBC/PFT (prose that would need NLP to
extract a signal from), Footballguys already publishes it as one tag per player.

CONFIRMED (2026-08-15): fully server-rendered -- all 32 teams present in a single
plain HTTP GET, no JS/browser needed (verified every team name appears exactly once in
the raw response, and every category class -off/-def/-st is present). Checked
alternatives first rather than assuming this was the best option: RotoWire has similar
per-player status tags but paywalls most teams ("reserved for subscribers", only ~6
free alphabetically); Pro Football Network is free and fresh but offense-only with no
status tags; FantasyPros appears subscription-gated; Lineups.com's depth-chart URL is
dead (redirects to its homepage).

Real cross-source disagreement worth knowing, not glossed over: spot-checked PFN vs
Footballguys and found different depth-chart ORDER for at least one team (Falcons QB1:
Penix on PFN, Tagovailoa on Footballguys). Treat exact ordering with some skepticism
regardless of source -- the per-player status tags are the more reliable signal this
puller is actually for, not "is this guy officially QB1 or QB2 today."

Each player is stored with Footballguys' own stable player-page slug as an ID (e.g.
"BrisJa00" from /player/Jacoby+Brissett/BrisJa00) -- likely a Pro-Football-Reference-
style ID, which could eventually crosswalk to nflverse's player_id, but no such
crosswalk has been built yet -- matching by name is the only link for now.

Usage:
    python data/pull_footballguys_depth.py
"""
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "footballguys_depth.csv")
# Footballguys restructured their URLs at some point between 2026-08-15 (this
# puller's last confirmed-working check) and 2026-08-23 -- the old /depthcharts
# 404s outright now. Found the replacement (2026-08-23) via the real "Depth
# Charts" nav link on their homepage: /depth-charts, but ONLY with a ?team=
# query param -- the bare URL 400s/404s with a raw "No input file specified"
# server error even though it's the exact same page and returns all 32 teams
# regardless of which team the param names. Also confirmed their backend is
# now flaky independent of this fix: the identical request succeeded only 1
# of 5 tries in testing (mix of 400/404/200), hence the retry loop below --
# this wasn't a factor before the restructuring.
URL = "https://www.footballguys.com/depth-charts?team=ARI"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
TIMEOUT = 20
MAX_ATTEMPTS = 6
RETRY_DELAY_SECONDS = 3

STATUS_RE = re.compile(r"\(([A-Z]{1,4})\)\s*$")
CATEGORY_MAP = {
    "depth-chart-cat-off": "offense",
    "depth-chart-cat-def": "defense",
    "depth-chart-cat-st": "special_teams",
}


def fetch_page() -> str:
    """Retries because the site itself is currently flaky for this exact URL --
    same request returned 400, 400, 404, 404, 200 across 5 tries in testing, not
    a transient one-off. A bad response never has the real markup in it, so
    checking for that (rather than just status == 200) is what actually
    distinguishes a good fetch from Footballguys' own error pages."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(URL, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and "depth_chart_" in resp.text:
                return resp.text
            last_error = f"status {resp.status_code}, {len(resp.text)} bytes, no depth-chart markup found"
        except requests.exceptions.RequestException as e:
            last_error = str(e)
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f"Failed after {MAX_ATTEMPTS} attempts against {URL}. Last error: {last_error}")


def parse_depth_charts(html: str) -> tuple[pd.DataFrame, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    updated_el = soup.find("p", class_="fs-6")
    updated_text = updated_el.get_text(strip=True) if updated_el else None

    rows = []
    for team_div in soup.select("div.depth-chart"):
        team_id = team_div.get("id", "") or ""
        team_abbr = team_id.replace("depth_chart_", "") if team_id else None
        header = team_div.select_one("span.team-header")
        team_name = header.get_text(strip=True) if header else None

        for pos_li in team_div.select("li.depth-chart-pos"):
            pos_label_el = pos_li.select_one("span.pos-label")
            position = pos_label_el.get_text(strip=True).rstrip(":").strip() if pos_label_el else None

            li_classes = pos_li.get("class", [])
            category = next((v for k, v in CATEGORY_MAP.items() if k in li_classes), "other")

            # Footballguys only links "fantasy relevant" positions (QB/RB/WR/TE/PK/KR/PR,
            # marked with a depth-chart-fantasy class on the <li>) to a player page as
            # <a class="player">. Everyone else -- the offensive line, P/H/LS -- gets the
            # same "player" class on a plain <span> instead, since nobody drafts a punter.
            # Selecting only a.player silently dropped every one of those positions across
            # all 32 teams (confirmed 2026-08-16 by fetching the live page directly and
            # diffing against what this parser actually returned).
            for depth_rank, player_a in enumerate(pos_li.select("a.player, span.player"), start=1):
                name_raw = player_a.get_text(strip=True)
                status_match = STATUS_RE.search(name_raw)
                status = status_match.group(1) if status_match else None
                name = STATUS_RE.sub("", name_raw).strip()

                href = player_a.get("href", "") or ""
                player_slug = href.rstrip("/").rsplit("/", 1)[-1] if href else None

                rows.append({
                    "team_abbr": team_abbr,
                    "team_name": team_name,
                    "category": category,
                    "position": position,
                    "depth_rank": depth_rank,
                    "is_starter": "starter" in player_a.get("class", []),
                    "player_name": name,
                    "status": status,
                    "footballguys_player_id": player_slug,
                })

    return pd.DataFrame(rows), updated_text


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    html = fetch_page()
    df, updated_text = parse_depth_charts(html)

    if df.empty:
        # Raise rather than a silent return -- a quiet exit-0 here is a real trap,
        # since run_pull_script's subprocess.returncode check reads it as a
        # successful pull even though nothing was written.
        raise RuntimeError("0 rows parsed -- page structure may have changed. Nothing written.")

    n_teams = df["team_abbr"].nunique()
    print(f"Parsed {len(df)} player-position rows across {n_teams} teams.")
    print(f"Page's own freshness stamp: {updated_text!r}")
    if n_teams != 32:
        print(f"WARNING: expected 32 teams, got {n_teams} -- page structure may have partially changed.")

    n_status = df["status"].notna().sum()
    print(f"{n_status} players currently carry a status tag (Q/PUP/IR/SUS/NFI/O/etc).")

    df["pulled_at"] = pulled_at
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
