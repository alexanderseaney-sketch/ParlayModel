"""
Per-player, per-game box scores for every Underdog NFL prop type (added 2026-10-20).

weekly_stats.csv only carries passing / rushing / receiving totals, but Underdog also
posts lines on tackles, kicking, longest plays, 1st quarter / 1st half receptions,
"25+ yards each half" and so on. This pulls two free nflverse files per season --
stats_player_week_<year>.csv (defense + kicking box scores) and
play_by_play_<year>.parquet (longest plays, quarter / half splits, first and last
TD scorer) -- and writes one slim row per player-game to
data/raw/prop_game_logs.csv, named by Underdog's stat_name, so
models/extra_prop_models.py can predict and grade every type the same way.

Committed (small, ~5 MB) so the hosted app and the refresh workflow don't need the
full pbp.csv.

Usage:  python data/pull_prop_game_logs.py [--years 2024 2025 2026]
"""
import argparse
import io
import os
import urllib.request

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(RAW_DIR, "prop_game_logs.csv")
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "qtr", "game_seconds_remaining",
            "play_id", "play_type", "yards_gained", "complete_pass", "two_point_attempt", "touchdown",
            "td_player_id", "passer_player_id", "receiver_player_id", "rusher_player_id"]

# Thresholds for Underdog's "X+ yards each half / quarter" props.
EACH_HALF = {"rec": [25, 50], "rush": [25, 50], "pass": [100, 150]}
EACH_QUARTER = {"pass": [25, 50]}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _box_scores(year: int) -> pd.DataFrame:
    s = pd.read_csv(io.BytesIO(_get(f"{BASE}/stats_player/stats_player_week_{year}.csv")), low_memory=False)
    f = lambda c: s[c].fillna(0) if c in s.columns else 0  # noqa: E731
    out = pd.DataFrame({
        "player_id": s["player_id"], "player_display_name": s["player_display_name"],
        "position": s["position"], "team": s["team"], "opponent_team": s["opponent_team"],
        "season": s["season"], "week": s["week"], "season_type": s["season_type"], "game_id": s["game_id"],
        "passing_yds": f("passing_yards"), "passing_tds": f("passing_tds"),
        "passing_ints": f("passing_interceptions"), "passing_comps": f("completions"),
        "passing_att": f("attempts"), "rushing_yds": f("rushing_yards"), "rushing_att": f("carries"),
        "receiving_yds": f("receiving_yards"), "receiving_rec": f("receptions"),
        "receiving_tgts": f("targets"),
        "rush_rec_tds": f("rushing_tds") + f("receiving_tds"),
        "fumbles_lost": f("rushing_fumbles_lost") + f("receiving_fumbles_lost") + f("sack_fumbles_lost"),
        "tackles": f("def_tackles_solo"), "assists": f("def_tackle_assists"),
        "sacks": f("def_sacks"), "defensive_ints": f("def_interceptions"),
        "field_goals_made": f("fg_made"), "extra_points_made": f("pat_made"),
    })
    out["rush_rec_yds"] = out["rushing_yds"] + out["receiving_yds"]
    out["passing_and_rushing_yds"] = out["passing_yds"] + out["rushing_yds"]
    out["tackles_and_assists"] = out["tackles"] + out["assists"]
    out["kicking_points"] = 3 * out["field_goals_made"] + out["extra_points_made"]
    # Underdog pick'em fantasy scoring (half PPR, 4-pt passing TDs).
    two_pt = f("passing_2pt_conversions") + f("rushing_2pt_conversions") + f("receiving_2pt_conversions")
    out["fantasy_points"] = (0.04 * out["passing_yds"] + 4 * out["passing_tds"] - out["passing_ints"]
                             + 0.1 * (out["rushing_yds"] + out["receiving_yds"]) + 6 * out["rush_rec_tds"]
                             + 0.5 * out["receiving_rec"] - 2 * out["fumbles_lost"] + 2 * two_pt
                             + 6 * f("special_teams_tds"))
    return out


def _pbp_splits(year: int) -> pd.DataFrame:
    p = pd.read_parquet(io.BytesIO(_get(f"{BASE}/pbp/play_by_play_{year}.parquet")), columns=PBP_COLS)
    p = p[p["two_point_attempt"].fillna(0) == 0]
    p["half"] = np.where(p["qtr"] <= 2, 1, 2)
    keys = ["game_id", "player_id"]
    frames = []

    rec = p[(p["play_type"] == "pass") & (p["complete_pass"] == 1) & p["receiver_player_id"].notna()] \
        .rename(columns={"receiver_player_id": "player_id"})
    rush = p[(p["play_type"] == "run") & p["rusher_player_id"].notna()].rename(columns={"rusher_player_id": "player_id"})
    pas = p[(p["play_type"] == "pass") & (p["complete_pass"] == 1) & p["passer_player_id"].notna()] \
        .rename(columns={"passer_player_id": "player_id"})
    for d, tag, long_name in ((rec, "rec", "receiving_long"), (rush, "rush", "rushing_long"), (pas, "pass", "passing_long")):
        g = d.groupby(keys)
        frames.append(g["yards_gained"].max().rename(long_name))
        by_half = d.pivot_table(index=keys, columns="half", values="yards_gained", aggfunc="sum").reindex(columns=[1, 2]).fillna(0)
        for x in EACH_HALF[tag]:
            frames.append(((by_half[1] >= x) & (by_half[2] >= x)).astype(int).rename(f"{x}_{tag}_yds_each_half"))
        if tag == "pass":
            by_q = d[d["qtr"] <= 4].pivot_table(index=keys, columns="qtr", values="yards_gained", aggfunc="sum") \
                .reindex(columns=[1, 2, 3, 4]).fillna(0)
            for x in EACH_QUARTER[tag]:
                frames.append((by_q >= x).all(axis=1).astype(int).rename(f"{x}_pass_yds_each_quarter"))
        q1 = d[d["qtr"] == 1].groupby(keys)["yards_gained"]
        h1 = d[d["qtr"] <= 2].groupby(keys)["yards_gained"]
        name = {"rec": "receiving", "rush": "rushing", "pass": "passing"}[tag]
        frames += [q1.sum().rename(f"period_1_{name}_yds"), h1.sum().rename(f"period_1_2_{name}_yds")]
        if tag == "rec":
            frames += [q1.size().rename("period_1_receiving_rec"), h1.size().rename("period_1_2_receiving_rec")]

    # First / last touchdown of the game (two-point tries already dropped).
    td = p[(p["touchdown"] == 1) & p["td_player_id"].notna()].sort_values(["game_id", "game_seconds_remaining"],
                                                                           ascending=[True, False])
    first = td.groupby("game_id").head(1).assign(period_first_touchdown_scored=1)
    last = td.groupby("game_id").tail(1).assign(last_touchdown_scored=1)
    for d, c in ((first, "period_first_touchdown_scored"), (last, "last_touchdown_scored")):
        frames.append(d.rename(columns={"td_player_id": "player_id"}).groupby(keys)[c].max())
    return pd.concat(frames, axis=1).reset_index()


def build(years: list[int]) -> pd.DataFrame:
    out = []
    for y in years:
        box = _box_scores(y)
        splits = _pbp_splits(y)
        d = box.merge(splits, on=["game_id", "player_id"], how="left")
        fill = [c for c in splits.columns if c not in ("game_id", "player_id")]
        d[fill] = d[fill].fillna(0)
        out.append(d)
        print(f"[prop game logs] {y}: {len(d)} player-games")
    df = pd.concat(out, ignore_index=True)
    num = df.select_dtypes("number").columns.difference(["season", "week"])
    df[num] = df[num].round(2)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int)
    a = ap.parse_args()
    if a.years:
        years = a.years
    else:
        today = pd.Timestamp.now()
        cur = today.year if today.month >= 9 else today.year - 1
        years = [cur - 2, cur - 1, cur]
    df = build(years)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
