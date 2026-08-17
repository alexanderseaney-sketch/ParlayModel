"""
Trains period_first_touchdown_scored. Same leave-one-season-out discipline,
with the AUC/lift-over-baseline check being especially important here since
the base rate (~6.6%) means raw accuracy is dominated by the trivial
always-under baseline, same lesson as rush_rec_tds and sacks.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from first_td_scorer_features import build_first_td_scorer_dataset

FEATURES = ["rush_rec_tds_rolling", "rush_rec_tds_last3", "carries_rolling", "carries_last3",
            "targets_rolling", "targets_last3", "rushing_yards_rolling", "receiving_yards_rolling",
            "def_epa_allowed_rolling"]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_first_td_scorer_dataset(min_week=4)
    df = df[df["position"].isin(["RB", "WR", "TE"])].copy()
    df = df.dropna(subset=FEATURES)
    print(f"Dataset: {len(df)} RB/WR/TE player-games")
    print(f"Base rate (scored first TD): {df['over_proxy_line'].mean()*100:.2f}%")

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
    print(f"\nPooled: {all_correct.mean()*100:.1f}% acc, AUC={roc_auc_score(all_y, all_probs):.3f}")

    preds = (all_probs > 0.5).astype(int)
    over_mask = preds == 1
    base_rate = all_y.mean()
    print(f"Base rate: {base_rate*100:.2f}%, model called OVER on {over_mask.sum()} games ({over_mask.mean()*100:.2f}%)")
    if over_mask.sum() > 0:
        print(f"  actual hit rate when called OVER: {all_y[over_mask].mean()*100:.2f}% (vs {base_rate*100:.2f}% base)")

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
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_period_first_touchdown_scored_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["RB", "WR", "TE"]}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
