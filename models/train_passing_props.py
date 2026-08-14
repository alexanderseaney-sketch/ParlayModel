"""
Trains and validates the passing-yards QB prop model, same rigor as receiving/rushing:
5-season pooled CV plus confidence-filtered accuracy.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from player_prop_passing_features import build_passing_yards_dataset

FEATURES = [
    "passing_yards_rolling", "passing_yards_last3", "attempts_rolling", "attempts_last3",
    "passing_tds_rolling", "interceptions_rolling",
    "completion_percentage_above_expectation_rolling", "avg_intended_air_yards_rolling",
    "aggressiveness_rolling", "avg_time_to_throw_rolling", "def_epa_allowed_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_passing_yards_dataset(min_week=4)
    df = df.dropna(subset=["passing_yards_rolling"])
    print(f"Dataset: {len(df)} QB player-games (2019-2024, week 4+)\n")

    all_probs, all_correct = [], []
    print(f"{'Holdout':<10} {'Accuracy':>10} {'n_test':>8}")
    print("-" * 32)
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[FEATURES].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[FEATURES].fillna(0), test["over_proxy_line"].values

        model = LogisticRegression(max_iter=3000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"{holdout:<10} {acc*100:>9.1f}% {len(test):>8}")

        all_probs.extend(probs)
        all_correct.extend(preds == y_test)

    all_probs, all_correct = np.array(all_probs), np.array(all_correct)
    print("-" * 32)
    print(f"{'Mean':<10} {all_correct.mean()*100:>9.1f}%\n")

    print("Confidence-filtered accuracy (pooled across all 5 seasons):")
    conf = np.abs(all_probs - 0.5) * 2
    for thresh in [0.0, 0.2, 0.3, 0.4, 0.5]:
        mask = conf >= thresh
        if mask.sum() > 0:
            print(f"  >={thresh}: {all_correct[mask].mean()*100:.1f}% accuracy ({mask.sum()} games, {mask.mean()*100:.1f}% of total)")

    X_all, y_all = df[FEATURES].fillna(0), df["over_proxy_line"]
    final_model = LogisticRegression(max_iter=3000)
    final_model.fit(X_all, y_all)
    print("\nFeature coefficients (positive = pushes toward OVER):")
    for feat, coef in sorted(zip(FEATURES, final_model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {feat}: {coef:+.4f}")

    rng = np.random.RandomState(42)
    n = len(X_all)
    models = []
    for i in range(100):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=3000)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    out_path = os.path.join(os.path.dirname(__file__), "player_prop_passing_yards_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["QB"]}, f)
    print(f"\nSaved production model -> {out_path}")


if __name__ == "__main__":
    main()
