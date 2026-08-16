"""
Computes each active player's CURRENT rolling features (as of the most recent
available data) and scores them with all four trained prop ensembles — receiving
yards, rushing yards, passing yards, receptions — this is what the dashboard's Parlay
Builder loads to show real model confidence next to each prop, instead of the old
manual slider. Previously this only covered receiving yards.

Each model's exact feature list lives inside its own .pkl (saved at training time), so
this file trusts that as the source of truth rather than re-declaring feature lists by
hand — it only needs to know which shared context merges (injury status, div/primetime
flags, weather/Vegas total, snap share) each prop's dataset needs before selecting
those features, which was determined by inspecting each saved model directly.

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
}


def _build_base_dataset(prop_type: str, min_week: int) -> pd.DataFrame:
    if prop_type == "receiving_yards":
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
    raise ValueError(f"Unknown prop_type: {prop_type}")


def score_prop(prop_type: str, config: dict, min_week: int = 4) -> pd.DataFrame:
    df = _build_base_dataset(prop_type, min_week)
    df = df[df["position"].isin(config["positions"])].copy()
    df = df.dropna(subset=[config["proxy_col"]])
    if df.empty:
        print(f"[{prop_type}] no qualifying rows, skipping.")
        return pd.DataFrame()

    for merge_name in config["context"]:
        df = CONTEXT_MERGERS[merge_name](df)

    # Most recent row per player = their current rolling form going into their next
    # game. Same caveat as the original receiving-only version: this reflects the
    # LATEST COMPLETED game, not a genuinely future upcoming one (own_injury_severity/
    # div_game/is_primetime/weather all describe that past game, not next week's).
    latest = df.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(1).copy()

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
        "player_id", "player_display_name", "position", "recent_team", "season", "week",
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
            "player_id", "player_display_name", "position", "recent_team", "season", "week",
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
        "predicted_prob_over", "confidence",
    ]])
