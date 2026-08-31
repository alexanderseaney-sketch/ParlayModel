"""
Pulls each team's full coaching staff + football front office from Wikipedia's
per-team `Template:<Team> staff` (the navbox that every team-season article
transcludes, so it's the one canonical, continuously-maintained copy).

Why Wikipedia and not the club sites: www.<team>.com/team/coaches-roster/ is a
JS-hydrated page whose non-JS HTML serves stale/placeholder content (confirmed
2026-08-30 -- giants.com returned Ravens staff, chiefs.com returned a
years-old staff). The Wikipedia template is plain wikitext, grouped by
`;Section` headers, one `*Title <dash> Name` bullet per person.

Output columns: team_abbr, group, title, name, pulled_at
  group in: front_office | head_coach | offense | defense | special_teams |
            strength | other

Names/titles only -- no photos, no bios. The depth-charts page renders this
grouped under each team.

Usage:
    python data/pull_coaching_staff.py
"""
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "coaching_staff.csv")

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ParlayModel/1.0 (personal NFL model; contact via GitHub)"}
TIMEOUT = 25
MAX_ATTEMPTS = 5
RETRY_BASE_SECONDS = 5   # exponential: 5, 10, 20, 40 -- Wikipedia 429s a fast 32-request loop
BETWEEN_REQUESTS_SECONDS = 1.2

# team abbr -> full name for `Template:<full name> staff`
TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Wikipedia `;Section` header (lowercased) -> our group key
GROUP_MAP = {
    "front office": "front_office",
    "ownership": "front_office",
    "owners": "front_office",
    "head coaches": "head_coach",
    "head coach": "head_coach",  # ARI/CIN/DAL/DEN/IND/WAS templates use the singular
    "offensive coaches": "offense",
    "defensive coaches": "defense",
    "special teams coaches": "special_teams",
    "special teams": "special_teams",
    "strength and conditioning": "strength",
    "strength and conditioning coaches": "strength",
}

_WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
_PARENS_NOTE = re.compile(r"\s*\((?:fired|interim|resigned|hired|promoted|reassigned|until|from|through)[^)]*\)", re.I)
_DASH = re.compile(r"\s*[–—‒-]\s*")  # en / em / figure dash / hyphen


def _fetch_template(full_name: str) -> str:
    page = f"Template:{full_name} staff"
    params = {"action": "parse", "page": page, "prop": "wikitext", "format": "json"}
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if "parse" in data:
                    return data["parse"]["wikitext"]["*"]
                last = data.get("error", {}).get("info", "no parse key")
            elif r.status_code == 429:
                last = "HTTP 429 (rate limited)"
                wait = int(r.headers.get("Retry-After", 0)) or RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
            else:
                last = f"HTTP {r.status_code}"
        except (requests.exceptions.RequestException, ValueError) as e:
            last = str(e)
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError(f"{page}: failed after {MAX_ATTEMPTS} attempts ({last})")


def _clean_name(raw: str) -> str:
    raw = _REF.sub("", raw)
    raw = _WIKILINK.sub(r"\1", raw)
    raw = _PARENS_NOTE.sub("", raw)
    raw = re.sub(r"'''?|\{\{[^}]*\}\}", "", raw)
    return raw.strip(" .* ")


def parse_staff(wikitext: str, abbr: str) -> list[dict]:
    rows = []
    group = "other"
    for line in wikitext.splitlines():
        line = line.strip()
        # Section headers come two ways across the 32 templates: definition-list
        # (`;Front office`) and bold (`'''Front office'''` on its own line).
        header = None
        if line.startswith(";"):
            header = line.lstrip(";").strip().lower()
        else:
            m = re.fullmatch(r"'''(.+?)'''", line)
            if m:
                header = m.group(1).strip().lower()
        if header is not None:
            if header in GROUP_MAP:
                group = GROUP_MAP[header]
            continue
        if not line.startswith("*"):
            continue
        body = line.lstrip("*").strip()
        parts = _DASH.split(body, maxsplit=1)
        if len(parts) != 2:
            continue
        title = _clean_name(parts[0])
        name = _clean_name(parts[1])
        if name and title:
            rows.append({"team_abbr": abbr, "group": group, "title": title, "name": name})
    return rows


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    all_rows, failed = [], []
    for abbr, full_name in TEAMS.items():
        try:
            rows = parse_staff(_fetch_template(full_name), abbr)
            if not rows:
                raise RuntimeError("0 staff rows parsed (template format changed?)")
            all_rows.extend(rows)
            coaches = sum(1 for r in rows if r["group"] not in ("front_office", "other"))
            print(f"[{abbr}] {len(rows)} people ({coaches} coaches).")
        except Exception as e:
            failed.append(abbr)
            print(f"[{abbr}] FAILED: {e}")
        time.sleep(0.5)  # be polite to the Wikipedia API

    if not all_rows:
        raise RuntimeError("Every team failed -- nothing written.")

    df = pd.DataFrame(all_rows)
    df["pulled_at"] = pulled_at

    n_teams = df["team_abbr"].nunique()
    print(f"\n{len(df)} staff rows across {n_teams}/32 teams. "
          f"Groups: {df['group'].value_counts().to_dict()}")
    if failed:
        print(f"WARNING: {len(failed)} team(s) failed: {failed}")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
