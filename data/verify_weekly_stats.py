"""
Cross-checks data/raw/weekly_stats.csv against ESPN's official box scores (added
2026-10-07) -- an independent source for the per-game numbers every prop model trains
on and every bet is graded against.

For each completed REG-season game of the season (default: latest season in
weekly_stats), pulls ESPN's summary box score and compares, per player:
  passing: completions, attempts, yards, TDs, INTs
  rushing: carries, yards, TDs
  receiving: receptions, targets, yards, TDs
Players are matched on normalized name + team. Reports coverage (games present /
missing in weekly_stats), the share of player-stat cells that match exactly, and every
mismatch, worst first. Exit code 1 if match rate < --min-match.

Usage:
    python data/verify_weekly_stats.py                 # latest season, all weeks
    python data/verify_weekly_stats.py --weeks 4 5
"""
import argparse
import os
import sys

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
RAW = os.path.join(ROOT, "data", "raw")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}
ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}

# (ESPN category, ESPN label) -> weekly_stats column
FIELDS = {
    ("passing", "CMP"): "completions", ("passing", "ATT"): "attempts",
    ("passing", "YDS"): "passing_yards", ("passing", "TD"): "passing_tds",
    ("passing", "INT"): "interceptions",
    ("rushing", "CAR"): "carries", ("rushing", "YDS"): "rushing_yards", ("rushing", "TD"): "rushing_tds",
    ("receiving", "REC"): "receptions", ("receiving", "TGTS"): "targets",
    ("receiving", "YDS"): "receiving_yards", ("receiving", "TD"): "receiving_tds",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv", " v"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.replace(".", "").replace("'", "").replace("-", " ").strip()


def espn_events(season: int, week: int) -> list[dict]:
    r = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
                     params={"seasontype": 2, "week": week, "dates": season}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        if c["status"]["type"]["state"] != "post":
            continue
        teams = {t["homeAway"]: t["team"]["abbreviation"] for t in c["competitors"]}
        out.append({"event_id": e["id"], "home": teams["home"], "away": teams["away"]})
    return out


def espn_box(event_id: str) -> list[dict]:
    r = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
                     params={"event": event_id}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    players: dict[tuple, dict] = {}
    for team in r.json().get("boxscore", {}).get("players", []):
        abbr = ESPN_TO_NFLVERSE.get(team["team"]["abbreviation"], team["team"]["abbreviation"])
        for cat in team.get("statistics", []):
            for ath in cat.get("athletes", []):
                key = (normalize_name(ath["athlete"]["displayName"]), abbr)
                rec = players.setdefault(key, {"name": ath["athlete"]["displayName"], "team": abbr})
                for lab, val in zip(cat.get("labels", []), ath.get("stats", [])):
                    if cat["name"] == "passing" and lab == "C/ATT":
                        cmp_, att = str(val).split("/")
                        rec["completions"], rec["attempts"] = float(cmp_), float(att)
                    elif (cat["name"], lab) in FIELDS:
                        try:
                            rec[FIELDS[(cat["name"], lab)]] = float(str(val).replace(",", ""))
                        except ValueError:
                            pass
    return list(players.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--weeks", type=int, nargs="*")
    ap.add_argument("--min-match", type=float, default=0.98)
    args = ap.parse_args()

    ws = pd.read_csv(os.path.join(RAW, "weekly_stats.csv"), low_memory=False)
    season = args.season or int(ws["season"].max())
    ws = ws[(ws["season"] == season) & (ws["season_type"].fillna("REG") == "REG")].copy()
    weeks = args.weeks or sorted(ws["week"].unique())
    ws["_k"] = ws["player_display_name"].map(normalize_name)

    cells = matches = 0
    missing_games, missing_players, mismatches = [], [], []
    for week in weeks:
        events = espn_events(season, week)
        ours_wk = ws[ws["week"] == week]
        for ev in events:
            box = espn_box(ev["event_id"])
            home = ESPN_TO_NFLVERSE.get(ev["home"], ev["home"])
            away = ESPN_TO_NFLVERSE.get(ev["away"], ev["away"])
            ours = ours_wk[ours_wk["recent_team"].isin([home, away])]
            if ours.empty:
                missing_games.append(f"{season} wk{week} {away}@{home}")
                continue
            idx = {(k, t): r for k, t, r in zip(ours["_k"], ours["recent_team"], ours.to_dict("records"))}
            for p in box:
                stat_cols = [c for c in p if c in FIELDS.values()]
                if not stat_cols or all(p[c] == 0 for c in stat_cols):
                    continue
                row = idx.get((normalize_name(p["name"]), p["team"]))
                if row is None:
                    missing_players.append(f"wk{week} {p['team']} {p['name']} {dict((c, p[c]) for c in stat_cols)}")
                    continue
                for c in stat_cols:
                    ours_v = row.get(c)
                    ours_v = 0.0 if pd.isna(ours_v) else float(ours_v)
                    cells += 1
                    if abs(ours_v - p[c]) < 1e-9:
                        matches += 1
                    else:
                        mismatches.append((abs(ours_v - p[c]), f"wk{week} {p['team']} {p['name']} {c}: ours {ours_v:g} vs ESPN {p[c]:g}"))
        print(f"wk{week}: {len(events)} completed games checked")

    rate = matches / cells if cells else 0.0
    print(f"\n{season}: {matches}/{cells} player-stat cells match ESPN exactly ({rate:.2%})")
    print(f"games missing from weekly_stats: {len(missing_games)}", *missing_games, sep="\n  ")
    print(f"players with stats on ESPN but no weekly_stats row: {len(missing_players)}")
    for m in missing_players[:25]:
        print("  ", m)
    print(f"mismatched cells: {len(mismatches)} (largest first)")
    for _, m in sorted(mismatches, reverse=True)[:40]:
        print("  ", m)
    sys.exit(0 if rate >= args.min_match and not missing_games else 1)


if __name__ == "__main__":
    main()
