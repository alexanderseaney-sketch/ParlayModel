"""
Feature engineering for defensive sack props (season_sacks, sacks) -- the first
defensive-player model in this project. Every other model here reuses
weekly_stats.csv (offensive box-score stats only), which has no defensive
player data at all, so this is built straight from play-by-play instead.

Two things pbp.csv provides that make this possible:
  - Sack credit columns (sack_player_id for a full sack, half_sack_1/2_player_id
    for a split sack) -- attributed per play, in the same GSIS player_id format
    weekly_stats.csv uses.
  - defense_players: a semicolon-delimited list of the 11 GSIS IDs on the field
    on defense for that play, present on ~92% of plays. This is what makes a
    proper zero-filled game log possible -- without it, a rolling sack rate
    could only be computed over games where a player recorded a sack, silently
    ignoring every game they played and got zero (which is most of them).

Defensive players don't appear in weekly_stats.csv, so names/positions come
from players.csv (nflverse's full player ID/name/position crosswalk, pulled
separately -- see pull_players() in data/pull_nflverse.py).
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

DEFENSIVE_POSITIONS = ["LB", "OLB", "ILB", "MLB", "CB", "DE", "DT", "NT", "DL",
                        "S", "FS", "SS", "DB", "EDGE"]
MIN_PLAYS_TO_QUALIFY = 10  # filters token/garbage-time defensive appearances


def _load_pbp_defense_cols() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(RAW_DIR, "pbp.csv"),
        usecols=["game_id", "season", "week", "defteam", "sack",
                  "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
                  "defense_players"],
        low_memory=False,
    )


def _build_defender_game_log() -> pd.DataFrame:
    """One row per (player, defteam, season, week): sacks that game (0 if they
    played but didn't record one) and how many plays they were on the field for."""
    pbp = _load_pbp_defense_cols()

    # Roster + play count: explode defense_players across every play (not just sack
    # plays) so a player who played but never touched a sack still gets a real row.
    roster = pbp.dropna(subset=["defense_players"]).copy()
    roster["player_id"] = roster["defense_players"].str.split(";")
    roster = roster.explode("player_id")
    plays = (
        roster.groupby(["player_id", "defteam", "season", "week"])
        .size().rename("plays_on_defense").reset_index()
    )

    # Sack credits: full sack = 1.0, each half-sack = 0.5, summed per player-week.
    sack_plays = pbp[pbp["sack"] == 1]
    credit_frames = [
        sack_plays[["sack_player_id", "defteam", "season", "week"]]
        .dropna(subset=["sack_player_id"])
        .rename(columns={"sack_player_id": "player_id"})
        .assign(credit=1.0),
    ]
    for col in ["half_sack_1_player_id", "half_sack_2_player_id"]:
        credit_frames.append(
            sack_plays[[col, "defteam", "season", "week"]]
            .dropna(subset=[col])
            .rename(columns={col: "player_id"})
            .assign(credit=0.5)
        )
    credits = pd.concat(credit_frames, ignore_index=True)
    sacks_by_week = (
        credits.groupby(["player_id", "defteam", "season", "week"])["credit"]
        .sum().rename("sacks").reset_index()
    )

    log = plays.merge(sacks_by_week, on=["player_id", "defteam", "season", "week"], how="left")
    log["sacks"] = log["sacks"].fillna(0.0)
    return log


def build_sacks_dataset(min_week: int = 4) -> pd.DataFrame:
    """Single-game sacks prop: one row per defender-game, rolling own sack rate
    plus team pass-rush context, same no-leakage discipline as every other prop
    (shift(1) before any rolling average)."""
    log = _build_defender_game_log()

    players = pd.read_csv(os.path.join(RAW_DIR, "players.csv"), low_memory=False)
    players = players.rename(columns={"gsis_id": "player_id", "display_name": "player_display_name"})
    log = log.merge(players[["player_id", "player_display_name", "position"]], on="player_id", how="left")
    log = log[log["position"].isin(DEFENSIVE_POSITIONS)].copy()

    log = log.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    # Deliberately NOT pre-filtered to plays_on_defense >= MIN_PLAYS_TO_QUALIFY here --
    # same bug as player_prop_features.py's receiving-yards dataset (fixed 2026-08-23):
    # filtering token/garbage-time appearances out before this rolling average meant a
    # defender's own baseline silently excluded those games, not just this week's label.
    # Applied below instead, after the rolling features (and team_sacks, which should
    # count every recorded sack regardless of any individual defender's snap count) are
    # computed.
    for col in ["sacks", "plays_on_defense"]:
        log[f"{col}_rolling"] = (
            log.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        log[f"{col}_last3"] = (
            log.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    # Team pass-rush context: a good pass-rushing DEFENSE creates more sack
    # opportunity to go around, same "team context" idea used for offensive props.
    team_sacks = (
        log.groupby(["defteam", "season", "week"])["sacks"].sum().rename("team_sacks").reset_index()
    )
    team_sacks = team_sacks.sort_values(["defteam", "season", "week"]).reset_index(drop=True)
    team_sacks["team_sacks_rolling"] = (
        team_sacks.groupby(["defteam", "season"])["team_sacks"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    log = log.merge(team_sacks[["defteam", "season", "week", "team_sacks_rolling"]],
                     on=["defteam", "season", "week"], how="left")

    log = log[log["week"] >= min_week].reset_index(drop=True)

    # Applied here, after every rolling/merge step above, not on the raw defender log --
    # see the matching comment where the rolling loop is first built.
    log = log[log["plays_on_defense"] >= MIN_PLAYS_TO_QUALIFY].reset_index(drop=True)

    log["proxy_line"] = log["sacks_rolling"]
    log["over_proxy_line"] = (log["sacks"] > log["proxy_line"]).astype(int)

    # current_predictions.py's next-game lookup joins on "recent_team" for every prop
    log = log.rename(columns={"defteam": "recent_team"})

    return log


# season_sacks was tried and dropped: built the same prior-season-total ->
# target-season pairing used for the other season-long props (prior_sacks,
# prior_games_played as features), but AUC came back 0.52-0.55 in EVERY holdout
# season, 2020-2024 -- no real signal, not just noise. Individual sack totals
# are far more volatile year to year than touches/targets (scheme changes,
# O-line matchups, double-team attention, injury luck), and two features
# aren't enough to explain that. Would need real pass-rush-quality signal
# (pressure rate, snap share trend) to be worth revisiting.


if __name__ == "__main__":
    df = build_sacks_dataset()
    print(df.shape)
    print(df["over_proxy_line"].value_counts(normalize=True))
    print(df[["player_display_name", "position", "season", "week", "sacks", "sacks_rolling",
              "plays_on_defense", "over_proxy_line"]].dropna().sort_values("sacks", ascending=False).head(10))
