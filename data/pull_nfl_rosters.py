"""
Pulls the official post-cutdown rosters straight from each club's own team site
(e.g. www.giants.com/team/players-roster/, www.chiefs.com/team/players-roster/).

Why the individual team sites and not www.nfl.com/teams/<slug>/roster:
- The club sites are run by each team's own digital staff and reflect a
  transaction within an hour or two of it being filed. Confirmed 2026-08-30
  (final-cutdowns day): ~25 of 32 club sites already showed a 53-man Active
  roster while nfl.com/teams still showed ~90 for all but a handful. Same NFL
  "Realm" CMS underneath, but a fresher cache.
- Fully server-rendered: the roster is plain <table>s in the initial HTML, one
  per roster designation, each with a <caption> naming the designation
  ("Active", "Reserve/Injured", "Reserve/Physically Unable to Perform",
  "Practice Squad", ...). No JS or browser needed.

The gate in models/current_predictions.py keys off roster_status: a player
counts as rostered when their designation starts with "Active" (the 53-man
roster, including Active/PUP and Active/NFI who are on the 53). Everyone on a
Reserve/* list is out for now and is excluded until a re-pull moves them back.

Names only, no ID crosswalk (the table carries no gsis_id) -- matched to the
rest of the pipeline by normalized name, same as the Footballguys puller.
weekly_rosters.csv (data/pull_nflverse.py) is the source to join on for a real
player_id; it's the slower, canonical fallback.

Usage:
    python data/pull_nfl_rosters.py
"""
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "nfl_rosters.csv")

# Pipeline team abbreviation -> the club's own domain. Abbreviations match
# schedules.csv / weekly_stats.csv (Rams = LA not LAR, Washington = WAS not WSH).
TEAMS = {
    "ARI": "azcardinals.com", "ATL": "atlantafalcons.com", "BAL": "baltimoreravens.com",
    "BUF": "buffalobills.com", "CAR": "panthers.com", "CHI": "chicagobears.com",
    "CIN": "bengals.com", "CLE": "clevelandbrowns.com", "DAL": "dallascowboys.com",
    "DEN": "denverbroncos.com", "DET": "detroitlions.com", "GB": "packers.com",
    "HOU": "houstontexans.com", "IND": "colts.com", "JAX": "jaguars.com",
    "KC": "chiefs.com", "LV": "raiders.com", "LAC": "chargers.com", "LA": "therams.com",
    "MIA": "miamidolphins.com", "MIN": "vikings.com", "NE": "patriots.com",
    "NO": "neworleanssaints.com", "NYG": "giants.com", "NYJ": "newyorkjets.com",
    "PHI": "philadelphiaeagles.com", "PIT": "steelers.com", "SF": "49ers.com",
    "SEA": "seahawks.com", "TB": "buccaneers.com", "TEN": "tennesseetitans.com",
    "WAS": "commanders.com",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}
TIMEOUT = 25
MAX_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 3

# Team-site table headers, in order. "#"/"Exp" here vs "No"/"Experience" on
# nfl.com; there's an Age column the nfl.com table didn't have.
HEADER_MAP = {
    "player": "player", "#": "jersey", "pos": "pos", "ht": "height",
    "wt": "weight", "age": "age", "exp": "experience", "college": "college",
}
OUT_COLUMNS = ["team_abbr", "player", "jersey", "pos", "height", "weight",
               "age", "experience", "college", "roster_status"]


def fetch_team(domain: str) -> str:
    url = f"https://www.{domain}/team/players-roster/"
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code == 200 and "<table" in resp.text:
                return resp.text
            last_error = f"status {resp.status_code}, {len(resp.text)} bytes, no table"
        except requests.exceptions.RequestException as e:
            last_error = str(e)
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f"{domain}: failed after {MAX_ATTEMPTS} attempts ({last_error})")


def parse_team(html: str, abbr: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError(f"{abbr}: no <table> in page")

    rows = []
    for table in tables:
        caption = table.find("caption")
        roster_status = caption.get_text(strip=True) if caption else "Unknown"

        header_cells = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        # Map this table's header order to our column names; skip anything unknown.
        col_for_index = {i: HEADER_MAP[h] for i, h in enumerate(header_cells) if h in HEADER_MAP}
        if "player" not in col_for_index.values():
            continue  # not a roster table (e.g. a stray layout table)

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < len(header_cells):
                continue
            rec = {"team_abbr": abbr, "roster_status": roster_status}
            for i, name in col_for_index.items():
                rec[name] = cells[i].get_text(strip=True)
            if rec.get("player"):
                rows.append(rec)

    if not rows:
        raise RuntimeError(f"{abbr}: tables found but 0 player rows parsed (structure changed?)")
    return pd.DataFrame(rows).reindex(columns=OUT_COLUMNS)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    frames, failed = [], []
    for abbr, domain in TEAMS.items():
        try:
            df = parse_team(fetch_team(domain), abbr)
            frames.append(df)
            active = int(df["roster_status"].str.startswith("Active").sum())
            print(f"[{abbr}] {len(df)} rostered, {active} Active "
                  f"({', '.join(sorted(df['roster_status'].unique()))})")
        except Exception as e:
            failed.append(abbr)
            print(f"[{abbr}] FAILED: {e}")
        time.sleep(0.3)

    if not frames:
        raise RuntimeError("Every team failed -- nothing written.")

    out = pd.concat(frames, ignore_index=True)
    out["pulled_at"] = pulled_at

    n_teams = out["team_abbr"].nunique()
    active_mask = out["roster_status"].str.startswith("Active")
    per_team_active = out[active_mask].groupby("team_abbr").size()
    settled = int((per_team_active.between(50, 56)).sum())
    print(f"\nParsed {len(out)} rows across {n_teams}/32 teams. "
          f"{settled}/{n_teams} teams have a settled Active roster (50-56).")
    print(f"Designations seen: {sorted(out['roster_status'].unique())}")
    if failed:
        print(f"WARNING: {len(failed)} team(s) failed: {failed}")
    if settled < n_teams:
        lagging = sorted(per_team_active[~per_team_active.between(50, 56)].to_dict().items())
        print(f"NOTE: not-yet-settled teams (Active count): {lagging} -- re-pull once "
              f"their site catches up.")

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
