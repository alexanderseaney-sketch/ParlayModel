"""
Computes each active player's CURRENT rolling features (as of the most recent
available data) and scores them with all trained prop ensembles — receiving
yards, rushing yards, passing yards, receptions, rush+rec TDs — this is what the
dashboard's Parlay Builder loads to show real model confidence next to each prop,
instead of the old manual slider. Previously this only covered receiving yards.

Each model's exact feature list lives inside its own .pkl (saved at training time), so
this file trusts that as the source of truth rather than re-declaring feature lists by
hand — it only needs to know which shared context merges (injury status, div/primetime
flags, weather/Vegas total, snap share) each prop's dataset needs before selecting
those features, which was determined by inspecting each saved model directly.

IMPORTANT, fixed 2026-08-15 ahead of the real season: a player's rolling stats
(target share, efficiency, etc.) correctly come from their own most recent games —
that part was always right. But game-level context (injury status, divisional/
primetime flag, weather, Vegas implied total) used to be pulled from that SAME row,
i.e. whatever game the rolling stats happened to end on. That's fine once a season is
underway, but wrong in exactly the situation this project actually needs to handle:
early season (weeks 1-3 of a new season have no qualifying current-season rolling row
at all — min 3 prior games required — so this used to silently fall back to a
player's LAST game of the PREVIOUS season, and then, worse, describe that stale old
game's context as if it were the upcoming one). Now context is looked up separately
against each player's team's actual NEXT unplayed game from schedules.csv, and the
output makes both dates explicit (`stats_as_of_season/week` vs `next_season/week/
opponent`) instead of hiding the gap between them. Known remaining limitation: this
assumes a player is still on the team from their last recorded game — a trade/signing
elsewhere between then and now wouldn't be caught (no roster/depth-chart pull exists
in this project yet).

Output: one row per player per prop type, their proxy line (their own rolling average
-- see the "no historical Underdog line archive" caveat in the README/feature files),
predicted probability of beating it, and a confidence score (0-1, how far the
prediction sits from a coinflip) -- >=0.4 confidence historically hit ~78-84% accuracy
depending on the prop, per backtesting.
"""
import os
import pickle

import numpy as np
import pandas as pd

from individual_context_features import build_player_injury_status, build_game_flags
from game_context_features import build_game_context, add_snap_share

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(os.path.dirname(__file__), "current_player_predictions.csv")


def _next_game_lookup(schedules: pd.DataFrame) -> pd.DataFrame:
    """Each team's next unplayed game (home_score still null in schedules.csv), one row
    per team. This is what "current" predictions should actually be conditioned on --
    not whatever game a player's rolling-stats row happens to be dated."""
    home = schedules[["season", "week", "home_team", "away_team", "home_score"]].rename(
        columns={"home_team": "team", "away_team": "opponent"})
    away = schedules[["season", "week", "home_team", "away_team", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
    all_games = pd.concat([home, away], ignore_index=True)

    unplayed = all_games[all_games["home_score"].isna()].copy()
    unplayed = unplayed.sort_values(["team", "season", "week"])
    return unplayed.groupby("team").head(1)[["team", "season", "week", "opponent"]]


def _merge_injury(df: pd.DataFrame) -> pd.DataFrame:
    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    inj_status = build_player_injury_status(injuries)
    return df.merge(inj_status, on=["player_id", "season", "week"], how="left")


def _merge_game_flags(df: pd.DataFrame) -> pd.DataFrame:
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})
    return df.merge(flags, on=["recent_team", "season", "week"], how="left")


def _merge_game_context(df: pd.DataFrame) -> pd.DataFrame:
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})
    return df.merge(context, on=["recent_team", "season", "week"], how="left")


def _merge_snap_share(df: pd.DataFrame) -> pd.DataFrame:
    return add_snap_share(df)


CONTEXT_MERGERS = {
    "injury": _merge_injury,
    "game_flags": _merge_game_flags,
    "game_context": _merge_game_context,
    "snap_share": _merge_snap_share,
}

# Verified against each saved model's own `features` list (models/*.pkl) rather than
# assumed — "context" lists only the merges that model's features actually draw from.
PROP_CONFIGS = {
    "receiving_yards": {
        "stat_name": "receiving_yds",
        "model_path": "player_prop_receiving_yards_model.pkl",
        "positions": ["WR", "TE"],
        "proxy_col": "receiving_yards_rolling",
        "context": ["injury", "game_flags"],
    },
    "rushing_yards": {
        "stat_name": "rushing_yds",
        "model_path": "player_prop_rushing_yards_model.pkl",
        "positions": ["RB"],
        "proxy_col": "rushing_yards_rolling",
        "context": ["game_flags", "game_context", "snap_share"],
    },
    # Same real Underdog stat ("receiving_yds") as the "receiving_yards" config above,
    # but a separate model trained on RB-only, non-NGS features (nflverse's NGS
    # receiving pull has zero RB rows -- see train_receiving_yards_rb_props.py).
    # Safe to split like this: a player is only ever one position, so this and
    # "receiving_yards" never produce a duplicate row for the same real player.
    "receiving_yards_rb": {
        "stat_name": "receiving_yds",
        "model_path": "player_prop_receiving_yards_rb_model.pkl",
        "positions": ["RB"],
        "proxy_col": "receiving_yards_rolling",
        "context": ["game_flags"],
    },
    "passing_yards": {
        "stat_name": "passing_yds",
        "model_path": "player_prop_passing_yards_model.pkl",
        "positions": ["QB"],
        "proxy_col": "passing_yards_rolling",
        "context": ["game_context"],
    },
    "receptions": {
        "stat_name": "receptions",
        "model_path": "player_prop_receptions_model.pkl",
        "positions": ["WR", "TE"],
        "proxy_col": "receptions_rolling",
        "context": ["game_flags"],
    },
    "rush_rec_tds": {
        "stat_name": "rush_rec_tds",
        "model_path": "player_prop_rush_rec_tds_model.pkl",
        "positions": ["RB", "WR", "TE"],
        "proxy_col": "proxy_line",
        "context": [],
    },
    # Season-long props: one row per player per season instead of per game (see
    # season_prop_features.py). "stats_as_of" for these means the most recent
    # COMPLETE regular season used as the projection basis, not a rolling window.
    # season_pass_yards/season_pass_tds were tried and dropped -- validated with
    # AUC ~0.5 (no real signal from prior-season stats alone), see
    # train_season_props.py's comment for why.
    "season_receiving_yards": {
        "stat_name": "season_receiving_yards",
        "model_path": "player_prop_season_receiving_yards_model.pkl",
        "positions": ["WR", "TE", "RB"],
        "proxy_col": "proxy_line",
        "context": [],
    },
    "season_rec_tds": {
        "stat_name": "season_rec_tds",
        "model_path": "player_prop_season_rec_tds_model.pkl",
        "positions": ["WR", "TE", "RB"],
        "proxy_col": "proxy_line",
        "context": [],
    },
    "season_rush_yards": {
        "stat_name": "season_rush_yards",
        "model_path": "player_prop_season_rush_yards_model.pkl",
        "positions": ["RB", "QB", "WR", "TE"],
        "proxy_col": "proxy_line",
        "context": [],
    },
    "season_rush_tds": {
        "stat_name": "season_rush_tds",
        "model_path": "player_prop_season_rush_tds_model.pkl",
        "positions": ["RB", "QB", "WR", "TE"],
        "proxy_col": "proxy_line",
        "context": [],
    },
}

# prop_type key -> underlying stat column in season_prop_features.STAT_COLS
SEASON_PROP_STAT_COLS = {
    "season_receiving_yards": "receiving_yards",
    "season_rec_tds": "receiving_tds",
    "season_rush_yards": "rushing_yards",
    "season_rush_tds": "rushing_tds",
}


def _build_base_dataset(prop_type: str, min_week: int) -> pd.DataFrame:
    if prop_type in ("receiving_yards", "receiving_yards_rb"):
        from player_prop_features import build_receiving_yards_dataset
        return build_receiving_yards_dataset(min_week=min_week)
    if prop_type == "rushing_yards":
        from player_prop_rushing_features import build_rushing_yards_dataset
        return build_rushing_yards_dataset(min_week=min_week)
    if prop_type == "passing_yards":
        from player_prop_passing_features import build_passing_yards_dataset
        return build_passing_yards_dataset(min_week=min_week)
    if prop_type == "receptions":
        from player_prop_receptions_features import build_receptions_dataset
        return build_receptions_dataset(min_week=min_week)
    if prop_type == "rush_rec_tds":
        from player_prop_rush_rec_tds_features import build_rush_rec_tds_dataset
        return build_rush_rec_tds_dataset(min_week=min_week)
    if prop_type in SEASON_PROP_STAT_COLS:
        from season_prop_features import build_season_prop_current_dataset
        return build_season_prop_current_dataset(SEASON_PROP_STAT_COLS[prop_type])
    raise ValueError(f"Unknown prop_type: {prop_type}")


def score_prop(prop_type: str, config: dict, min_week: int = 4) -> pd.DataFrame:
    df = _build_base_dataset(prop_type, min_week)
    df = df[df["position"].isin(config["positions"])].copy()
    df = df.dropna(subset=[config["proxy_col"]])
    if df.empty:
        print(f"[{prop_type}] no qualifying rows, skipping.")
        return pd.DataFrame()

    # snap_share is the player's OWN trailing role -- correctly tied to their rolling
    # stats regardless of which game those stats are dated to, so it's merged before
    # taking the latest row like before. injury/game_flags/game_context describe the
    # GAME, not the player, and must describe the upcoming game specifically -- merged
    # further down, after `latest` knows what that upcoming game actually is.
    if "snap_share" in config["context"]:
        df = CONTEXT_MERGERS["snap_share"](df)

    # Most recent row per player = their current rolling form. Rename season/week here
    # so it's never confused with the next-game columns merged in below -- this is the
    # game the STATS are dated to, which during the first 3 weeks of a new season will
    # still be last season's finale (min 3 current-season games required) rather than
    # silently passing as "current" the way it used to.
    latest = df.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(1).copy()
    latest = latest.rename(columns={"season": "stats_as_of_season", "week": "stats_as_of_week"})

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    # Rename BEFORE merging, not after -- the base dataset already has its own
    # "opponent"/"season"/"week" columns (that game's matchup), so merging a
    # same-named second copy would silently get pandas-suffixed (_x/_y) instead of
    # overwriting, and a post-merge rename would then find nothing to rename.
    next_game = _next_game_lookup(schedules).rename(columns={
        "team": "recent_team", "season": "next_season", "week": "next_week", "opponent": "next_opponent",
    })
    latest = latest.merge(next_game, on="recent_team", how="left")

    # No upcoming scheduled game found (stale team assignment, wrong era of data,
    # etc.) -- can't responsibly attach injury/matchup context to a game that doesn't
    # exist, so these players are dropped rather than scored with a fabricated context.
    dropped = latest["next_season"].isna().sum()
    latest = latest.dropna(subset=["next_season"])
    if dropped:
        print(f"[{prop_type}] dropped {dropped} player(s) with no upcoming scheduled game.")
    if latest.empty:
        return pd.DataFrame()

    if "injury" in config["context"]:
        injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
        inj_status = build_player_injury_status(injuries).rename(
            columns={"season": "next_season", "week": "next_week"})
        latest = latest.merge(inj_status, on=["player_id", "next_season", "next_week"], how="left")
    if "game_flags" in config["context"]:
        flags = build_game_flags(schedules).rename(
            columns={"team": "recent_team", "season": "next_season", "week": "next_week"})
        latest = latest.merge(flags, on=["recent_team", "next_season", "next_week"], how="left")
    if "game_context" in config["context"]:
        context = build_game_context(schedules).rename(
            columns={"team": "recent_team", "season": "next_season", "week": "next_week"})
        latest = latest.merge(context, on=["recent_team", "next_season", "next_week"], how="left")

    model_path = os.path.join(os.path.dirname(__file__), config["model_path"])
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    models, features = saved["models"], saved["features"]

    missing = [c for c in features if c not in latest.columns]
    if missing:
        raise ValueError(f"[{prop_type}] missing expected feature columns: {missing}")

    X = latest[features].fillna(0)
    all_probs = np.array([m.predict_proba(X)[:, 1] for m in models])  # shape (100, n_players)
    mean_prob = all_probs.mean(axis=0)
    confidence = np.abs(mean_prob - 0.5) * 2

    latest["prop_type"] = prop_type
    latest["stat_name"] = config["stat_name"]
    latest["proxy_line"] = latest[config["proxy_col"]].round(1)
    latest["predicted_prob_over"] = mean_prob
    latest["confidence"] = confidence

    return latest[[
        "player_id", "player_display_name", "position", "recent_team",
        "stats_as_of_season", "stats_as_of_week", "next_season", "next_week", "next_opponent",
        "prop_type", "stat_name", "proxy_line", "predicted_prob_over", "confidence",
    ]]


def build_current_predictions() -> pd.DataFrame:
    frames = []
    for prop_type, config in PROP_CONFIGS.items():
        try:
            result = score_prop(prop_type, config)
            if not result.empty:
                frames.append(result)
                print(f"[{prop_type}] scored {len(result)} players.")
        except Exception as e:
            print(f"[{prop_type}] FAILED: {e}")

    if not frames:
        return pd.DataFrame(columns=[
            "player_id", "player_display_name", "position", "recent_team",
            "stats_as_of_season", "stats_as_of_week", "next_season", "next_week", "next_opponent",
            "prop_type", "stat_name", "proxy_line", "predicted_prob_over", "confidence",
        ])

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("confidence", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    predictions = build_current_predictions()
    predictions.to_csv(OUT_PATH, index=False)
    print(f"\n{len(predictions)} current player predictions saved -> {OUT_PATH}\n")

    for prop_type in PROP_CONFIGS:
        subset = predictions[predictions["prop_type"] == prop_type]
        n_above_40 = (subset["confidence"] >= 0.4).sum()
        print(f"{prop_type}: {len(subset)} players scored, {n_above_40} clear 0.4 confidence")

    print("\nHighest-confidence predictions right now:")
    print(predictions.head(10)[[
        "player_display_name", "recent_team", "prop_type", "proxy_line",
        "predicted_prob_over", "confidence", "next_season", "next_week", "next_opponent",
    ]])
