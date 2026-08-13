"""
Pulls current NFL pick'em prop lines from Underdog Fantasy's internal API (the same
endpoint their own web/app frontend uses — not an officially published or documented API).

IMPORTANT: this is unofficial/reverse-engineered access, discovered via an open-source
reference scraper (github.com/aidanhall21/underdog-fantasy-pickem-scraper) rather than
any Underdog documentation. Underdog's Terms of Service likely restrict automated/scraped
access even for read-only data. This carries real account risk, same category as the
sportsbook browser-automation caveat noted elsewhere in this project — just lower stakes
since it's read-only. Worth deciding deliberately whether to run this regularly.

UNTESTED in this environment: api.underdogfantasy.com isn't reachable from this sandbox's
network (confirmed via curl, host_not_allowed). Needs a live test at home before trusting
the schema assumptions below.

Usage:
    python data/pull_underdog.py
    python data/pull_underdog.py --sport NFL
"""
import argparse
import os

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PICKEM_URL = "https://api.underdogfantasy.com/beta/v5/over_under_lines"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json",
}
TIMEOUT = 15


def fetch_pickem_data() -> dict:
    resp = requests.get(PICKEM_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def process(data: dict) -> pd.DataFrame:
    """Flattens players + appearances + over_under_lines into one row per prop option."""
    players = pd.DataFrame(data["players"]).rename(columns={"id": "player_id"})
    appearances = pd.DataFrame(data["appearances"]).rename(columns={"id": "appearance_id"})
    lines = pd.DataFrame(data["over_under_lines"]).reset_index(drop=True)

    player_appearances = players.merge(
        appearances, on=["player_id", "position_id", "team_id"], how="left"
    )

    lines_expanded = lines.explode("options")
    options_df = pd.json_normalize(lines_expanded["options"])
    lines_expanded = pd.concat(
        [lines_expanded.drop("options", axis=1).reset_index(drop=True), options_df.reset_index(drop=True)],
        axis=1,
    )
    lines_expanded["appearance_id"] = lines_expanded["over_under"].apply(
        lambda x: x["appearance_stat"]["appearance_id"]
    )
    lines_expanded["stat_name"] = lines_expanded["over_under"].apply(
        lambda x: x["appearance_stat"]["stat"]
    )
    lines_expanded["choice"] = lines_expanded["choice"].map(
        {"lower": "under", "higher": "over"}
    ).fillna(lines_expanded["choice"])

    props = player_appearances.merge(
        lines_expanded, on="appearance_id", how="left", suffixes=("", "_ou")
    )
    props["full_name"] = props["first_name"] + " " + props["last_name"]
    return props


def validate(df: pd.DataFrame, sport_filter: str | None) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Pulled 0 rows — Underdog's schema may have changed, or the endpoint moved.")

    if sport_filter and "sport_id" in df.columns:
        before = len(df)
        df = df[df["sport_id"] == sport_filter]
        print(f"Filtered to {sport_filter}: {len(df)} of {before} rows.")

    missing_line = df["stat_name"].isna().sum() if "stat_name" in df.columns else len(df)
    print(f"{len(df)} total prop-option rows.")
    if missing_line > 0:
        print(f"WARNING: {missing_line} rows missing stat_name — schema may have partially changed.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Pull current Underdog pick'em prop lines")
    parser.add_argument("--sport", default="NFL", help="Filter to one sport_id (default NFL); pass '' for all")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    data = fetch_pickem_data()
    df = process(data)
    df = validate(df, args.sport or None)

    out_path = os.path.join(RAW_DIR, "underdog_props.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
