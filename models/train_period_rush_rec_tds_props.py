"""
Trains period_1_rush_rec_tds and period_1_2_rush_rec_tds. Same leave-one-
season-out discipline, same AUC/lift check as everything else here given
the low base rates.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from period_rush_rec_tds_features import build_period_rush_rec_tds_dataset

FEATURES = ["rush_rec_tds_rolling", "rush_rec_tds_last3", "carries_rolling", "carries_last3",
            "targets_rolling", "targets_last3", "rushing_yards_rolling", "receiving_yards_rolling",
            "def_epa_allowed_rolling", "red_zone_touches_rolling", "red_zone_share_rolling"]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]

CONFIGS = {
    "period_1_rush_rec_tds": "q1",
    "period_1_2_rush_rec_tds": "h1",
}


def train_one(prop_type: str, period: str) -> None:
    df = build_period_rush_rec_tds_dataset(period, min_week=4)
    df = df[df["position"].isin(["RB", "WR", "TE"])].copy()
    df = df.dropna(subset=FEATURES)
    print(f"\n=== {prop_type} ===")
    print(f"Dataset: {len(df)} RB/WR/TE player-games")
    base_rate = df["over_proxy_line"].mean()
    print(f"Base rate: {base_rate*100:.2f}%")

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
        print(f"  {holdout}: {acc*100:.1f}% acc, AUC={auc:.3f}, n={len(test)}")
        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    print(f"  Pooled: {all_correct.mean()*100:.1f}% acc, AUC={roc_auc_score(all_y, all_probs):.3f}")
    for pct in [5, 10, 20]:
        thresh = np.percentile(all_probs, 100 - pct)
        mask = all_probs >= thresh
        print(f"    top {pct}%: hit rate {all_y[mask].mean()*100:.1f}% (vs base {base_rate*100:.1f}%, "
              f"lift {all_y[mask].mean()/base_rate:.2f}x)")

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
    out_path = os.path.join(os.path.dirname(__file__), f"player_prop_{prop_type}_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["RB", "WR", "TE"]}, f)
    print(f"  Saved -> {out_path}")


def main():
    for prop_type, period in CONFIGS.items():
        train_one(prop_type, period)


if __name__ == "__main__":
    main()
