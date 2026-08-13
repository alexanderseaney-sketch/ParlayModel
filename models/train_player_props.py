"""
Trains and validates a receiving-yards player-prop model for WR/TE (RBs excluded — NGS
receiving data has zero coverage for RBs in this pull, an honest gap, not a bug).

Target: does the player beat their own trailing rolling average ("proxy line")? See the
honest caveat in player_prop_features.py about why this proxy is used instead of real
historical Underdog lines (which don't exist to backtest against).

Same discipline as the team-level model: leave-one-season-out CV across the last 5
seasons, not a single lucky split.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from player_prop_features import build_receiving_yards_dataset

FEATURES = [
    "receiving_yards_rolling", "receiving_yards_last3", "targets_rolling", "targets_last3",
    "target_share_rolling", "receiving_air_yards_rolling",
    "avg_separation_rolling", "avg_cushion_rolling", "avg_yac_above_expectation_rolling",
    "def_epa_allowed_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receiving_yards_rolling"])  # need at least some prior-game history

    print(f"Dataset: {len(df)} WR/TE player-games (2019-2024, week 4+)\n")

    print(f"{'Holdout':<10} {'Accuracy':>10} {'n_test':>8} {'Baseline always-over':>22}")
    print("-" * 58)

    accs = []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]

        X_train, y_train = train[FEATURES].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[FEATURES].fillna(0), test["over_proxy_line"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        accs.append(acc)

        always_over_baseline = y_test.mean()  # accuracy if you just always guessed "over"
        print(f"{holdout:<10} {acc*100:>9.1f}% {len(test):>8} {always_over_baseline*100:>25.1f}%")

    print("-" * 58)
    print(f"{'Mean':<10} {np.mean(accs)*100:>9.1f}%")

    # Fit on everything for feature importance read
    X_all, y_all = df[FEATURES].fillna(0), df["over_proxy_line"]
    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X_all, y_all)
    print("\nFeature coefficients (positive = pushes toward OVER):")
    for feat, coef in sorted(zip(FEATURES, final_model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {feat}: {coef:+.4f}")


if __name__ == "__main__":
    main()


def train_and_save_production_model():
    """Trains the final 100-model bootstrap ensemble on ALL data and saves it, same
    approach as the team-win model (final_model.pkl)."""
    import pickle
    import os

    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receiving_yards_rolling"])

    X_full = df[FEATURES].fillna(0).reset_index(drop=True)
    y_full = df["over_proxy_line"].reset_index(drop=True)

    rng = np.random.RandomState(42)
    n = len(X_full)
    models = []
    for i in range(100):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=1000)
        m.fit(X_full.iloc[idx], y_full.iloc[idx])
        models.append(m)

    out_path = os.path.join(os.path.dirname(__file__), "player_prop_receiving_yards_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["WR", "TE"]}, f)
    print(f"\nSaved 100-model receiving-yards prop ensemble -> {out_path}")
