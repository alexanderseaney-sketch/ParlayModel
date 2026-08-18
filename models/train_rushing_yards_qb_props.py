"""
Trains a rushing-yards model for QBs specifically -- a real gap found while
auditing live prop coverage (2026-08-17): mobile-QB rushing yards is one of
Underdog's most common real markets (Josh Allen, Lamar Jackson, Jalen Hurts,
Joe Burrow, etc. all had live rushing_yds props), but every one of them was
being silently excluded from rushing_yards, not because of missing data but
because the model was RB-only.

Same root cause as the earlier receiving_yards/RB fix: ngs_rushing.csv has
zero QB rows, so this uses only the non-NGS signal that's already computed
for every position in player_prop_rushing_features.py -- own rolling rushing
yards/carries, opponent defense strength.

Same leave-one-season-out discipline as everything else here.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from player_prop_rushing_features import build_rushing_yards_dataset
from calibration_report import print_calibration_report

FEATURES = ["rushing_yards_rolling", "rushing_yards_last3", "carries_rolling",
            "carries_last3", "def_epa_allowed_rolling"]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_rushing_yards_dataset(min_week=4)
    df = df[df["position"] == "QB"].copy()
    df = df.dropna(subset=FEATURES)
    print(f"Dataset: {len(df)} QB player-games\n")

    all_probs, all_y, all_correct = [], [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[FEATURES], train["over_proxy_line"]
        X_test, y_test = test[FEATURES], test["over_proxy_line"].values
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"{holdout}: {acc*100:.1f}% acc, AUC={auc:.3f}, n={len(test)}")
        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    auc = roc_auc_score(all_y, all_probs)
    print(f"\nPooled: {all_correct.mean()*100:.1f}% acc, AUC={auc:.3f} (0.5=no signal)")
    print_calibration_report(all_probs, all_y, "rushing_yards_qb")
    conf = np.abs(all_probs - 0.5) * 2
    for t in [0.2, 0.3, 0.4]:
        mask = conf >= t
        if mask.sum() > 0:
            print(f"  >={t}: {all_correct[mask].mean()*100:.1f}% ({mask.sum()} rows, {mask.mean()*100:.1f}%)")

    X_all, y_all = df[FEATURES], df["over_proxy_line"]
    rng = np.random.RandomState(42)
    n = len(X_all)
    models = []
    for i in range(100):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=2000)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_rushing_yards_qb_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["QB"]}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
