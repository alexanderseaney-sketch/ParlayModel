"""
Derives weekly_stats.csv-shaped box scores directly from pbp.csv, for seasons
nflverse's own aggregated "player_stats" release hasn't published yet.

Real gap found 2026-08-17: nflverse's play-by-play release and its aggregated
player_stats release run on different publishing schedules -- pbp.csv for a
season can be complete (confirmed: 2025 has all 18 regular season weeks plus
playoffs, 285 games) well before player_stats_<year> exists as a file. Since
literally every model in this project is ultimately built on weekly_stats.csv,
that lag freezes every "current" prediction at whatever the last officially
aggregated season is, even when the actual games have already been played and
the raw data to build our own aggregate already exists.

This performs the same rush/target/reception/pass aggregation nflverse's own
release does (rusher_player_id -> carries/rushing_yards/rushing_tds,
receiver_player_id -> targets/receptions/receiving_yards/receiving_tds,
passer_player_id -> attempts/completions/passing_yards/passing_tds/
interceptions), then attaches player_display_name/position from players.csv
(pbp only carries IDs and raw names, not always the same display convention
weekly_stats.csv uses).

Also derives passing_epa/rushing_epa/receiving_epa, sack_fumbles_lost/
rushing_fumbles_lost/receiving_fumbles_lost, receiving_air_yards, sacks, and
target_share -- checked (via grep across models/) to be the full set of
weekly_stats.csv columns any feature-engineering module actually reads.
passing_epa/rushing_epa/receiving_epa specifically feed build_team_week_offense
and def_epa_allowed_rolling (feature_engineering.py), which is used by nearly
every model -- leaving them NaN would silently zero out defensive-strength
signal for any 2025 game once real games are played. Columns nothing reads
(racr, wopr, dakota, pacr, fantasy_points, air_yards_share, first_downs,
2pt conversions, non-lost fumbles) are skipped rather than faked.

Note two pbp gotchas found empirically by cross-checking a derived-from-pbp
2024 against the real official weekly_stats.csv for 2024 (see
validate_against_real): (1) pbp's "rush" column excludes QB scrambles --
must use "rush_attempt" instead, or QB rushing volume is undercounted by
~35%; (2) pbp's "pass_attempt" flag is 1 on sacks too, which the official
"attempts" stat excludes -- must additionally filter sack==0 for the passer's
attempts/completions, or attempts is overcounted by 5-15%.

Validated against a season where BOTH sources exist (2024) before trusting
this for 2025 -- see validate_against_real() below. Output is designed to be
appended into weekly_stats.csv via pull_nflverse.py's existing save() merge
logic (new rows win on key collision), so if nflverse ever does publish the
real player_stats_2025, a normal pull automatically supersedes these derived
rows with the official ones -- nothing here needs to be manually undone.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

PBP_STATS_COLS = [
    "game_id", "season", "week", "season_type", "posteam",
    "rush_attempt", "pass_attempt", "sack", "epa", "qb_epa", "air_yards",
    "fumble_lost", "fumbled_1_player_id",
    "rusher_player_id", "rusher_player_name", "rushing_yards", "rush_touchdown",
    "receiver_player_id", "receiver_player_name", "receiving_yards", "complete_pass", "pass_touchdown",
    "passer_player_id", "passer_player_name", "passing_yards", "interception",
]


def build_weekly_stats_from_pbp(years: list[int]) -> pd.DataFrame:
    pbp = pd.read_csv(os.path.join(RAW_DIR, "pbp.csv"), usecols=PBP_STATS_COLS, low_memory=False)
    pbp = pbp[pbp["season"].isin(years)]

    key = ["season", "week", "season_type", "posteam"]

    # Grouped by player_id + key ONLY -- pbp's own name columns (rusher_player_name
    # etc.) are occasionally inconsistent for the same gsis_id within a single week
    # (e.g. "M.Wilson" vs "Mi.Wilson" on different plays), which would silently split
    # one player's stats across two rows if the name were part of the group key.
    # Canonical display_name is pulled from players.csv below instead.
    rush_fumble_lost = (pbp["fumble_lost"] == 1) & (pbp["fumbled_1_player_id"] == pbp["rusher_player_id"])
    rushing = pbp[pbp["rush_attempt"] == 1].dropna(subset=["rusher_player_id"]).assign(
        _fumble_lost=rush_fumble_lost)
    rushing = rushing.groupby(["rusher_player_id"] + key).agg(
        carries=("rush_attempt", "size"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
        rushing_epa=("epa", "sum"),
        rushing_fumbles_lost=("_fumble_lost", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})

    rec_fumble_lost = (pbp["fumble_lost"] == 1) & (pbp["fumbled_1_player_id"] == pbp["receiver_player_id"])
    targeted = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].assign(
        _fumble_lost=rec_fumble_lost)
    receiving = targeted.groupby(["receiver_player_id"] + key).agg(
        targets=("pass_attempt", "size"),
        receptions=("complete_pass", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("pass_touchdown", "sum"),
        receiving_air_yards=("air_yards", "sum"),
        receiving_epa=("epa", "sum"),
        receiving_fumbles_lost=("_fumble_lost", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})

    team_targets = targeted.groupby(key).size().rename("_team_targets").reset_index()

    sack_fumble_lost = (pbp["fumble_lost"] == 1) & (pbp["fumbled_1_player_id"] == pbp["passer_player_id"])
    passer_plays = pbp[pbp["passer_player_id"].notna()].assign(_fumble_lost=sack_fumble_lost)
    attempted = passer_plays[(passer_plays["pass_attempt"] == 1) & (passer_plays["sack"] == 0)]
    dropback = attempted.groupby(["passer_player_id"] + key).agg(
        attempts=("pass_attempt", "size"),
        completions=("complete_pass", "sum"),
        passing_yards=("passing_yards", "sum"),
        passing_tds=("pass_touchdown", "sum"),
        interceptions=("interception", "sum"),
    ).reset_index().rename(columns={"passer_player_id": "player_id"})
    # passing_epa/sacks/sack_fumbles_lost are attributed across ALL passer dropbacks
    # (attempts + sacks), since a sack's negative value is charged to the passer.
    passer_totals = passer_plays.groupby(["passer_player_id"] + key).agg(
        passing_epa=("qb_epa", "sum"),
        sacks=("sack", "sum"),
        sack_fumbles_lost=("_fumble_lost", "sum"),
    ).reset_index().rename(columns={"passer_player_id": "player_id"})
    dropback = dropback.merge(passer_totals, on=["player_id"] + key, how="outer")

    merged = rushing.merge(receiving, on=["player_id"] + key, how="outer")
    merged = merged.merge(dropback, on=["player_id"] + key, how="outer")
    merged = merged.merge(team_targets, on=key, how="left")

    stat_cols = ["carries", "rushing_yards", "rushing_tds", "rushing_epa", "rushing_fumbles_lost",
                 "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
                 "receiving_epa", "receiving_fumbles_lost",
                 "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
                 "passing_epa", "sacks", "sack_fumbles_lost"]
    for col in stat_cols:
        merged[col] = merged[col].fillna(0)

    merged["target_share"] = (merged["targets"] / merged["_team_targets"]).where(merged["_team_targets"] > 0)

    merged = merged.drop(columns=["_team_targets"])
    merged = merged.rename(columns={"posteam": "recent_team"})

    players = pd.read_csv(os.path.join(RAW_DIR, "players.csv"), low_memory=False)
    players = players.rename(columns={"gsis_id": "player_id", "display_name": "player_display_name"})[
        ["player_id", "player_display_name", "position"]].drop_duplicates("player_id")
    merged = merged.merge(players, on="player_id", how="left")

    merged = merged.dropna(subset=["player_id"])
    merged = merged[(merged[stat_cols].abs().sum(axis=1) > 0)]  # drop rows with literally no recorded stat

    return merged[["player_id", "player_display_name", "position", "recent_team", "season", "week", "season_type"]
                   + stat_cols + ["target_share"]]


def save_derived_weekly_stats(years: list[int], real_data_max_frac: float = 0.5) -> None:
    """Appends derived rows into weekly_stats.csv, guarded against clobbering a
    year nflverse already officially publishes: if the existing file already has
    real rows covering more than real_data_max_frac of a requested year's weeks,
    that year is skipped rather than partially overwritten with our approximation
    -- once nflverse ships the real player_stats release for a year, a normal
    `pull_nflverse.py` re-pull is what should update it, not this module."""
    out_path = os.path.join(RAW_DIR, "weekly_stats.csv")
    existing = pd.read_csv(out_path, low_memory=False)

    safe_years = []
    for year in years:
        real_weeks = existing.loc[existing["season"] == year, "week"].nunique()
        if real_weeks > 18 * real_data_max_frac:  # regular season is 18 weeks
            print(f"[pbp_derived_weekly_stats] {year}: already has {real_weeks} weeks of real "
                  f"data on file, skipping derived overwrite.")
            continue
        safe_years.append(year)

    if not safe_years:
        print("[pbp_derived_weekly_stats] nothing to do -- all requested years already have real data.")
        return

    derived = build_weekly_stats_from_pbp(safe_years)
    key_cols = ["player_id", "season", "week"]
    combined = pd.concat([derived, existing], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="first")
    combined.to_csv(out_path, index=False)
    print(f"[pbp_derived_weekly_stats] added {len(derived)} derived rows for {safe_years} "
          f"-> {out_path} ({len(combined)} rows total)")


def validate_against_real(year: int = 2024) -> None:
    """Sanity check: aggregate the SAME year two ways (from pbp here, vs. the
    real nflverse weekly_stats.csv already on disk) and compare season totals
    for the top volume players -- catches a wrong aggregation logic before it
    ever touches a season we can't cross-check."""
    derived = build_weekly_stats_from_pbp([year])
    real = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    real = real[real["season"] == year]

    for stat in ["rushing_yards", "receiving_yards", "passing_yards", "rushing_tds", "receiving_tds",
                 "carries", "targets", "receptions", "attempts", "completions", "interceptions", "passing_tds",
                 "rushing_epa", "receiving_epa", "passing_epa", "receiving_air_yards", "sacks",
                 "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]:
        d_totals = derived.groupby("player_id")[stat].sum().rename("derived")
        r_totals = real.groupby("player_id")[stat].sum().rename("real")
        cmp = pd.concat([d_totals, r_totals], axis=1).fillna(0)
        cmp = cmp[(cmp["real"] > 0) | (cmp["derived"] > 0)]
        cmp["diff"] = cmp["derived"] - cmp["real"]
        cmp["pct_diff"] = (cmp["diff"] / cmp["real"].replace(0, pd.NA)).abs()
        top = cmp.reindex(cmp["real"].abs().sort_values(ascending=False).index).head(10)
        print(f"\n=== {stat} (top 10 by real volume) ===")
        print(top.to_string())
        big_misses = cmp[cmp["pct_diff"] > 0.05]
        print(f"  {len(big_misses)} of {len(cmp)} players off by >5% (excluding real==0 rows)")

    # target_share is a per-game (not season-summed) ratio -- compare row-by-row instead
    merged_ts = derived.merge(
        real[["player_id", "season", "week", "target_share"]], on=["player_id", "season", "week"],
        suffixes=("_derived", "_real"))
    merged_ts = merged_ts[merged_ts["targets"] > 0]
    merged_ts["ts_diff"] = (merged_ts["target_share_derived"] - merged_ts["target_share_real"]).abs()
    print(f"\n=== target_share (per player-week, {len(merged_ts)} rows with targets>0) ===")
    print(f"  mean abs diff: {merged_ts['ts_diff'].mean():.4f}  max abs diff: {merged_ts['ts_diff'].max():.4f}")
    print(f"  rows off by >0.02: {(merged_ts['ts_diff'] > 0.02).sum()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_against_real(2024)
    else:
        save_derived_weekly_stats([2025])
