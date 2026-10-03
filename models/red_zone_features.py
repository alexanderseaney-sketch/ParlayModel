"""
Red-zone usage features -- the single biggest known gap in the TD-related
models (rush_rec_tds, period_first_touchdown_scored, season_rec_tds,
season_rush_tds): none of them currently know a player's share of their
team's red-zone touches, which is the most direct signal there is for "is
this player the team's actual goal-line/scoring option" as opposed to a
general volume/role signal. Built from pbp.csv (yardline_100 <= 20 defines
the red zone -- 20 yards or closer to the opponent's end zone).

Deliberately built as an ADD-ON feature to test, not assumed to help --
same discipline as everything else in this project: only keep it where a
real before/after AUC comparison shows it actually does.
"""
import os

import pandas as pd

from feature_engineering import cross_season_rolling

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RED_ZONE_YARDLINE = 20
# See feature_engineering.cross_season_rolling's docstring for why this is bounded
# and crosses a season boundary rather than resetting there.
_ROLLING_WINDOW = 16


def build_red_zone_usage(min_week: int = 1) -> pd.DataFrame:
    """One row per player-game: red-zone touches (carries + targets inside the
    20) and that player's share of their team's total red-zone touches that
    game. Own rolling history only -- shift(1) before any averaging, same
    no-leakage discipline as every other feature in this project."""
    pbp = pd.read_csv(
        os.path.join(RAW_DIR, "pbp.csv"),
        usecols=["game_id", "season", "week", "posteam", "yardline_100",
                  "rush", "pass", "rusher_player_id", "receiver_player_id"],
        low_memory=False,
    )
    rz = pbp[pbp["yardline_100"] <= RED_ZONE_YARDLINE]

    rz_rush = rz[rz["rush"] == 1].dropna(subset=["rusher_player_id"])[
        ["rusher_player_id", "posteam", "game_id", "season", "week"]
    ].rename(columns={"rusher_player_id": "player_id"})
    rz_rec = rz[rz["pass"] == 1].dropna(subset=["receiver_player_id"])[
        ["receiver_player_id", "posteam", "game_id", "season", "week"]
    ].rename(columns={"receiver_player_id": "player_id"})

    touches = pd.concat([rz_rush, rz_rec], ignore_index=True)
    player_touches = (
        touches.groupby(["player_id", "posteam", "game_id", "season", "week"])
        .size().rename("red_zone_touches").reset_index()
    )
    team_touches = (
        touches.groupby(["posteam", "game_id", "season", "week"])
        .size().rename("team_red_zone_touches").reset_index()
    )

    # Every game the player actually played (weekly_stats), restricted to team-games
    # pbp.csv covers -- NOT just games with >=1 red-zone touch. Found 2026-10-02: the
    # touch table only had rows for games with a red-zone touch, so (a) a zero-touch
    # game's row got its rolling features fillna(0)'d downstream instead of carrying
    # the player's trailing average -- live, Chase Brown's anytime-TD probability fell
    # from ~45% to 5% because his latest game had no red-zone touch -- and (b) the
    # rolling averages silently skipped every zero-touch game, inflating them.
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False,
                         usecols=["player_id", "recent_team", "season", "week"])
    covered = pbp[["season", "week", "posteam"]].drop_duplicates()
    grid = weekly.drop_duplicates().merge(
        covered.rename(columns={"posteam": "recent_team"}), on=["recent_team", "season", "week"], how="inner")
    df = grid.merge(
        player_touches.rename(columns={"posteam": "recent_team"}).drop(columns=["game_id"]),
        on=["player_id", "recent_team", "season", "week"], how="left")
    df = df.merge(
        team_touches.rename(columns={"posteam": "recent_team"}).drop(columns=["game_id"]),
        on=["recent_team", "season", "week"], how="left")
    df["red_zone_touches"] = df["red_zone_touches"].fillna(0)
    df["red_zone_share"] = (df["red_zone_touches"] / df["team_red_zone_touches"]).fillna(0)
    df = df.rename(columns={"posteam": "recent_team"})
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["red_zone_touches", "red_zone_share"]:
        df[f"{col}_rolling"] = cross_season_rolling(df, "player_id", col, window=_ROLLING_WINDOW)

    df = df[df["week"] >= min_week].reset_index(drop=True)
    return df[["player_id", "recent_team", "season", "week",
                "red_zone_touches_rolling", "red_zone_share_rolling"]]


if __name__ == "__main__":
    df = build_red_zone_usage()
    print(df.shape)
    print(df.dropna().sort_values("red_zone_share_rolling", ascending=False).head(10))
