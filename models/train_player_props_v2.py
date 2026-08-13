"""
v2 of the receiving-yards prop model: adds team pass_oe (scheme signal), then applies
the bootstrap-agreement approach (validated on the game-winner model) to see how high
accuracy climbs on the subset of predictions with high model consensus.
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
    "def_epa_allowed_rolling", "team_pass_oe_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]
N_BOOTSTRAP = 100


def cv_accuracy(df, features):
    accs = []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[features].fillna(0), test["over_proxy_line"]
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        accs.append(accuracy_score(y_test, preds))
    return accs


def main():
    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receiving_yards_rolling"])

    accs = cv_accuracy(df, FEATURES)
    print("With team pass_oe added:")
    for s, a in zip(HOLDOUT_SEASONS, accs):
        print(f"  {s}: {a*100:.1f}%")
    print(f"  Mean: {np.mean(accs)*100:.1f}% (prior version without pass_oe: 66.7%)\n")

    # --- Bootstrap agreement on the 2024 holdout ---
    train = df[df["season"] <= 2023]
    test = df[df["season"] == 2024]
    X_train_full = train[FEATURES].fillna(0).reset_index(drop=True)
    y_train_full = train["over_proxy_line"].reset_index(drop=True)
    X_test = test[FEATURES].fillna(0)
    y_test = test["over_proxy_line"].values

    rng = np.random.RandomState(42)
    n = len(X_train_full)
    all_preds = np.zeros((N_BOOTSTRAP, len(test)))
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=1000)
        m.fit(X_train_full.iloc[idx], y_train_full.iloc[idx])
        all_preds[i] = m.predict_proba(X_test)[:, 1]

    mean_prob = all_preds.mean(axis=0)
    majority_pred = (mean_prob > 0.5).astype(int)
    votes_for_over = (all_preds > 0.5).mean(axis=0)
    agreement = np.maximum(votes_for_over, 1 - votes_for_over)
    correct = (majority_pred == y_test)

    print(f"2024 holdout, {N_BOOTSTRAP}-model bootstrap ensemble:")
    print(f"Overall accuracy: {correct.mean()*100:.1f}% (n={len(test)})\n")

    print(f"{'Agreement threshold':<22} {'n games':>8} {'accuracy':>10} {'% of total':>11}")
    print("-" * 55)
    for thresh in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        mask = agreement >= thresh
        if mask.sum() > 0:
            acc = correct[mask].mean()
            pct_of_total = mask.mean() * 100
            print(f">={thresh*100:.0f}%{'':<17} {mask.sum():>8} {acc*100:>9.1f}% {pct_of_total:>10.1f}%")


if __name__ == "__main__":
    main()
