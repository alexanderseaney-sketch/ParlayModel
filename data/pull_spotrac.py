"""
Pulls contract + roster-status data from each team's Spotrac overview page
(www.spotrac.com/nfl/<slug>/overview) -> data/raw/spotrac_contracts.csv.

What it gives that nothing else in the project has: per-player cap hit, cap % of
the league cap, dead-cap exposure, total cash, contract free-agent year, age, and
-- for reserve players -- the specific injury/reason ("KNEE (ACL)", "UNDISCLOSED").

Each overview page is server-rendered with one <table> per roster designation
("2026 Active Roster", "2026 Injured Reserve", "2026 Reserve/PUP", ...). The
player name is a clean <a>; the leading duplicated last name is a sort-key span.

NOT the roster source: Spotrac's "Active Roster" table still shows the ~90-man
offseason roster for whichever teams it hasn't reconciled to the 53 yet (checked
2026-08-30: Giants/Rams done, Chiefs/49ers/Raiders still 90ish). The roster gate
stays on data/pull_nfl_rosters.py. This is a contracts + reserve-status feed.

Output columns: team_abbr, player, spotrac_id, position, age, roster_status,
reason, cap_hit, cap_pct, dead_cap, cash_total, fa_year, pulled_at

Usage:
    python data/pull_spotrac.py
"""
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "spotrac_contracts.csv")

# pipeline abbr -> spotrac URL slug (kebab-case full name)
TEAMS = {
    "ARI": "arizona-cardinals", "ATL": "atlanta-falcons", "BAL": "baltimore-ravens",
    "BUF": "buffalo-bills", "CAR": "carolina-panthers", "CHI": "chicago-bears",
    "CIN": "cincinnati-bengals", "CLE": "cleveland-browns", "DAL": "dallas-cowboys",
    "DEN": "denver-broncos", "DET": "detroit-lions", "GB": "green-bay-packers",
    "HOU": "houston-texans", "IND": "indianapolis-colts", "JAX": "jacksonville-jaguars",
    "KC": "kansas-city-chiefs", "LV": "las-vegas-raiders", "LAC": "los-angeles-chargers",
    "LA": "los-angeles-rams", "MIA": "miami-dolphins", "MIN": "minnesota-vikings",
    "NE": "new-england-patriots", "NO": "new-orleans-saints", "NYG": "new-york-giants",
    "NYJ": "new-york-jets", "PHI": "philadelphia-eagles", "PIT": "pittsburgh-steelers",
    "SF": "san-francisco-49ers", "SEA": "seattle-seahawks", "TB": "tampa-bay-buccaneers",
    "TEN": "tennessee-titans", "WAS": "washington-commanders",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}
TIMEOUT = 30
MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 4
BETWEEN_REQUESTS_SECONDS = 1.3

_RESERVE_TAIL = {"pup": "PUP", "nfi": "NFI", "suspended": "SUS", "retired": "RET",
                 "did not report camp": "NR", "left squad": "NR"}
_ID_RE = re.compile(r"/id/(\d+)/")
_MONEY_RE = re.compile(r"-?\d[\d,]*")


def _money(text: str):
    if not text or text.strip() in ("-", "--", ""):
        return None
    m = _MONEY_RE.search(text.replace("(", "-").replace(")", ""))
    return int(m.group(0).replace(",", "")) if m else None


def _pct(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group(0)) if m else None


def _fetch(slug: str) -> str:
    url = f"https://www.spotrac.com/nfl/{slug}/overview"
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200 and "Active Roster" in r.text:
                return r.text
            last = f"HTTP {r.status_code}"
        except requests.exceptions.RequestException as e:
            last = str(e)
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError(f"{slug}: failed after {MAX_ATTEMPTS} attempts ({last})")


def _table_status(table) -> str | None:
    """Only the table's OWN immediately-preceding heading counts. Walking further
    up would relabel non-roster tables ("Dead Money", "Cap Totals") as whatever
    roster section happened to render above them."""
    h = table.find_previous(["h2", "h3", "h4", "h5", "h6"])
    if h is None:
        return None
    t = re.sub(r"^\d{4}\s+", "", h.get_text(" ", strip=True)).strip().lower()
    if t.startswith("active roster"):
        return "Active"
    if t.startswith("injured reserve"):
        return "IR"
    if t.startswith("practice squad"):
        return "PS"
    if t.startswith("exempt") or "commissioner" in t:
        return "EXE"
    if t.startswith("reserve/"):
        return _RESERVE_TAIL.get(t.split("/", 1)[1].strip(), "RES")
    return None  # dead money, cap totals, deadlines, free agents, draft picks, ...


def parse_team(html: str, abbr: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table"):
        status = _table_status(table)
        if status is None:
            continue
        header_cells = [th.get_text(strip=True) for th in table.find_all("th")]
        if not any("Player" in h for h in header_cells):
            continue
        idx = {h: i for i, h in enumerate(header_cells)}

        def col(cells, *names):
            for n in names:
                for h, i in idx.items():
                    if h.startswith(n) and i < len(cells):
                        return cells[i].get_text(" ", strip=True)
            return ""

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < len(header_cells):
                continue
            a = cells[1].find("a")
            if a is None:
                continue
            name = a.get_text(strip=True)
            full = cells[1].get_text(" ", strip=True)
            reason = full.split(name, 1)[1].strip(" :-") if name in full else ""
            reason = re.sub(r"^[A-Z/ -]*RESERVE:?\s*", "", reason).strip() or None
            sid = _ID_RE.search(a.get("href", ""))
            rows.append({
                "team_abbr": abbr,
                "player": name,
                "spotrac_id": sid.group(1) if sid else None,
                "position": col(cells, "Pos"),
                "age": _money(col(cells, "Age")),
                "roster_status": status,
                "reason": reason if status != "Active" else None,
                "cap_hit": _money(col(cells, "Cap Hit")),
                "cap_pct": _pct(col(cells, "Cap Hit Pct", "Cap Hit %")),
                "dead_cap": _money(col(cells, "Dead Cap")),
                "cash_total": _money(col(cells, "Cash Total")),
                "fa_year": _money(col(cells, "Free Agent", "Free AgentYear")),
            })
    return rows


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat()

    all_rows, failed = [], []
    for abbr, slug in TEAMS.items():
        try:
            rows = parse_team(_fetch(slug), abbr)
            if not rows:
                raise RuntimeError("0 rows parsed")
            all_rows.extend(rows)
            active = sum(1 for r in rows if r["roster_status"] == "Active")
            print(f"[{abbr}] {len(rows)} players ({active} active"
                  + (", NOT trimmed to 53 yet" if active > 60 else "") + ").")
        except Exception as e:
            failed.append(abbr)
            print(f"[{abbr}] FAILED: {e}")
        time.sleep(BETWEEN_REQUESTS_SECONDS)

    if not all_rows:
        raise RuntimeError("Every team failed -- nothing written.")

    df = pd.DataFrame(all_rows).drop_duplicates(["team_abbr", "spotrac_id", "roster_status"])
    df["pulled_at"] = pulled_at
    n_teams = df["team_abbr"].nunique()
    trimmed = df[df.roster_status == "Active"].groupby("team_abbr").size().le(60).sum()
    print(f"\n{len(df)} rows across {n_teams}/32 teams. "
          f"{trimmed}/{n_teams} teams' active roster is trimmed to ~53. "
          f"Status mix: {df['roster_status'].value_counts().to_dict()}")
    if failed:
        print(f"WARNING: failed teams: {failed}")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
