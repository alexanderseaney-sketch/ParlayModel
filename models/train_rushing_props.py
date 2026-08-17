"""
Trains and validates the rushing-yards RB prop model, same rigor as receiving yards:
5-season pooled CV, plus the confidence-filtered accuracy that's the metric that
actually matters for real use.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from player_prop_rushing_features import build_rushing_yards_dataset

FEATURES = [
    "rushing_yards_rolling", "rushing_yards_last3", "carries_rolling", "carries_last3",
    "targets_rolling", "receiving_yards_rolling",
    "rush_yards_over_expected_per_att_rolling", "percent_attempts_gte_eight_defenders_rolling",
    "efficiency_rolling", "def_epa_allowed_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_rushing_yards_dataset(min_week=4)
    # Explicit RB filter -- build_rushing_yards_dataset() now includes QB too (for the
    # separate QB model, see train_rushing_yards_qb_props.py), so this can't rely on
    # the base dataset being RB-only anymore the way it used to.
    df = df[df["position"] == "RB"].copy()
    df = df.dropna(subset=["rushing_yards_rolling"])
    print(f"Dataset: {len(df)} RB player-games (2019-2024, week 4+)\n")

    all_probs, all_correct = [], []
    print(f"{'Holdout':<10} {'Accuracy':>10} {'n_test':>8}")
    print("-" * 32)
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[FEATURES].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[FEATURES].fillna(0), test["over_proxy_line"].values

        model = LogisticRegression(max_iter=2000)
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

    # Fit on all data, print coefficients
    X_all, y_all = df[FEATURES].fillna(0), df["over_proxy_line"]
    final_model = LogisticRegression(max_iter=2000)
    final_model.fit(X_all, y_all)
    print("\nFeature coefficients (positive = pushes toward OVER):")
    for feat, coef in sorted(zip(FEATURES, final_model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {feat}: {coef:+.4f}")

    # Save production ensemble
    rng = np.random.RandomState(42)
    n = len(X_all)
    models = []
    for i in range(100):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=2000)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_rushing_yards_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["RB"]}, f)
    print(f"\nSaved production model -> {out_path}")


if __name__ == "__main__":
    main()
