"""
Trains logistic regression models on increasingly rich feature sets, testing each
against the SAME held-out 2024 season, to see which signals actually add predictive
value beyond the Elo baseline.

Train: 2019-2023 (5 seasons). Test: 2024 season only, completely held out.
This is a stricter, more standard ML evaluation than walk-forward within a single set —
appropriate here since we're comparing feature sets, not re-deriving Elo's per-game logic.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from feature_engineering import build_game_features

ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")


FEATURE_SETS = {
    "v1_epa_only": ["diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling"],
    "v2_epa_plus_injuries": ["diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling", "diff_injury_count_rolling"],
    "v3_epa_injuries_ngs": ["diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling", "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling"],
}


def run_iteration(name: str, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_train = train[features].fillna(0)
    y_train = train["home_win"]
    X_test = test[features].fillna(0)
    y_test = test["home_win"]

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    pred_probs = model.predict_proba(X_test)[:, 1]
    pred_labels = (pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, pred_labels)
    brier = brier_score_loss(y_test, pred_probs)

    coefs = dict(zip(features, model.coef_[0]))

    return {"name": name, "accuracy": acc, "brier": brier, "n_test": len(test), "coefs": coefs}


def main():
    df = build_game_features(min_week=3)
    train = df[df["season"] <= 2023].copy()
    test = df[df["season"] == 2024].copy()
    print(f"Train: {len(train)} games (2019-2023). Test: {len(test)} games (2024 only, held out).\n")

    # Elo's accuracy on this EXACT same test subset (season 2024, week >= 3), for a fair comparison
    elo_results = pd.read_csv(ELO_RESULTS_PATH)
    elo_2024 = elo_results[(elo_results["season"] == 2024) & (elo_results["week"] >= 3)]
    elo_acc = elo_2024["correct"].mean()
    print(f"Elo baseline on this same 2024 (week>=3) subset: {elo_acc*100:.1f}% accuracy, n={len(elo_2024)}\n")

    print(f"{'Model':<25} {'Accuracy':>10} {'Brier':>8} {'n_test':>8}")
    print("-" * 55)
    results = []
    for name, features in FEATURE_SETS.items():
        r = run_iteration(name, features, train, test)
        results.append(r)
        print(f"{r['name']:<25} {r['accuracy']*100:>9.1f}% {r['brier']:>8.3f} {r['n_test']:>8}")

    print("\nFeature coefficients (positive = higher value favors home team):")
    for r in results:
        print(f"\n{r['name']}:")
        for feat, coef in r["coefs"].items():
            print(f"  {feat}: {coef:+.4f}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nBest this round: {best['name']} at {best['accuracy']*100:.1f}% "
          f"(vs. Elo's {elo_acc*100:.1f}% on the same games)")


if __name__ == "__main__":
    main()
