"""
Tests whether model AGREEMENT is itself a useful signal: train the same model
architecture on many different bootstrap resamples of the training data, then check
whether games where those runs agree with each other are actually the ones the model
gets right more often (vs. games where runs disagree/split).

This is standard bagging-style uncertainty quantification — if agreement tracks
accuracy, it means "how confident the model is" is itself informative, which matters a
lot for a betting application: you'd want to bet more when the model is more internally
consistent, not just when its point estimate crosses 50%.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_engineering import build_game_features

ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")

FEATURES = [
    "diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling",
    "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling",
    "diff_turnover_margin_rolling", "rest_diff", "temp", "wind", "is_dome",
    "elo_home_win_prob",
]

N_BOOTSTRAP = 100
RANDOM_SEED = 42


def main():
    df = build_game_features(min_week=3)
    elo = pd.read_csv(ELO_RESULTS_PATH)[
        ["season", "week", "home_team", "away_team", "model_home_win_prob"]
    ].rename(columns={"model_home_win_prob": "elo_home_win_prob"})
    df = df.merge(elo, on=["season", "week", "home_team", "away_team"], how="left")

    train = df[df["season"] <= 2023].copy()
    test = df[df["season"] == 2024].copy()

    X_train_full = train[FEATURES].fillna(0).reset_index(drop=True)
    y_train_full = train["home_win"].reset_index(drop=True)
    X_test = test[FEATURES].fillna(0)
    y_test = test["home_win"].values

    rng = np.random.RandomState(RANDOM_SEED)
    n_train = len(X_train_full)

    all_preds = np.zeros((N_BOOTSTRAP, len(test)))  # predicted probability of home win, per bootstrap model

    print(f"Training {N_BOOTSTRAP} bootstrap models on resampled 2019-2023 data, testing each on the full 2024 season...")
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot, y_boot = X_train_full.iloc[idx], y_train_full.iloc[idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_boot, y_boot)
        all_preds[i] = model.predict_proba(X_test)[:, 1]

    # For each test game: mean predicted prob across all 100 models, and how much they agree
    mean_prob = all_preds.mean(axis=0)
    std_prob = all_preds.std(axis=0)
    majority_pred = (mean_prob > 0.5).astype(int)
    # agreement = fraction of the 100 models that voted for the eventual majority pick
    per_game_votes_for_home = (all_preds > 0.5).mean(axis=0)
    agreement = np.maximum(per_game_votes_for_home, 1 - per_game_votes_for_home)

    correct = (majority_pred == y_test)

    results = pd.DataFrame({
        "season": test["season"].values, "week": test["week"].values,
        "home_team": test["home_team"].values, "away_team": test["away_team"].values,
        "mean_prob_home_win": mean_prob, "std_across_models": std_prob,
        "agreement_pct": agreement * 100,
        "predicted_home_win": majority_pred, "actual_home_win": y_test, "correct": correct,
    })

    out_path = os.path.join(os.path.dirname(__file__), "ensemble_agreement_results.csv")
    results.to_csv(out_path, index=False)

    overall_acc = correct.mean()
    print(f"\nOverall accuracy (majority vote across {N_BOOTSTRAP} bootstrap models): {overall_acc*100:.1f}%")

    print(f"\n{'Agreement bucket':<20} {'n games':>8} {'accuracy':>10}")
    print("-" * 42)
    buckets = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100.01)]
    for lo, hi in buckets:
        mask = (results["agreement_pct"] >= lo) & (results["agreement_pct"] < hi)
        subset = results[mask]
        if len(subset) > 0:
            print(f"{lo}-{hi:.0f}% agreement{'':<3} {len(subset):>8} {subset['correct'].mean()*100:>9.1f}%")

    # Direct correlation check
    corr = results["agreement_pct"].corr(results["correct"].astype(int))
    print(f"\nCorrelation between agreement level and being correct: {corr:.3f}")
    print("(positive = higher agreement genuinely tracks higher accuracy)")

    print(f"\nFull per-game results saved -> {out_path}")


if __name__ == "__main__":
    main()
