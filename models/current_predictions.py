"""
Computes each active WR/TE's CURRENT rolling features (as of the most recent available
data) and scores them with the trained receiving-yards ensemble — this is what the
dashboard's Parlay Builder loads to show real model confidence next to each prop,
instead of the old manual slider.

Output: one row per player, their latest rolling stats, predicted probability of
beating their own recent-form line, and a confidence score (0-1, how far the
prediction sits from a coinflip) — this is exactly the metric validated in the last
round: filtering to confidence >= 0.4 historically hit ~78% accuracy.
"""
import os
import pickle

import pandas as pd

from player_prop_features import build_receiving_yards_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "player_prop_receiving_yards_model.pkl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "current_player_predictions.csv")


def build_current_predictions() -> pd.DataFrame:
    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receiving_yards_rolling"])

    # Take each player's MOST RECENT row — their current rolling form going into their
    # next game. This is a projection base, not a backtest, so it uses whatever the
    # latest real data actually shows.
    latest = df.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(1).copy()

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    models, features = saved["models"], saved["features"]

    X = latest[features].fillna(0)
    import numpy as np
    all_probs = np.array([m.predict_proba(X)[:, 1] for m in models])  # shape (100, n_players)
    mean_prob = all_probs.mean(axis=0)
    confidence = abs(mean_prob - 0.5) * 2

    latest["predicted_prob_over"] = mean_prob
    latest["confidence"] = confidence
    latest["proxy_line_yards"] = latest["receiving_yards_rolling"].round(1)

    out = latest[[
        "player_id", "player_display_name", "position", "recent_team", "season", "week",
        "proxy_line_yards", "predicted_prob_over", "confidence",
        "target_share_rolling", "avg_separation_rolling",
    ]].sort_values("confidence", ascending=False).reset_index(drop=True)

    return out


if __name__ == "__main__":
    predictions = build_current_predictions()
    predictions.to_csv(OUT_PATH, index=False)
    print(f"{len(predictions)} current player predictions saved -> {OUT_PATH}\n")

    print("Highest-confidence predictions right now:")
    print(predictions.head(10)[["player_display_name", "recent_team", "proxy_line_yards", "predicted_prob_over", "confidence"]])

    n_above_40 = (predictions["confidence"] >= 0.4).sum()
    print(f"\n{n_above_40} of {len(predictions)} players currently clear the 0.4 confidence bar "
          f"(historically ~78% accuracy at this threshold).")
