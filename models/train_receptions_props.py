import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from player_prop_receptions_features import build_receptions_dataset

FEATURES = [
    "receptions_rolling", "receptions_last3", "targets_rolling", "targets_last3",
    "target_share_rolling", "receiving_air_yards_rolling", "catch_rate_rolling",
    "avg_separation_rolling", "avg_cushion_rolling", "catch_percentage_rolling",
    "def_epa_allowed_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_receptions_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receptions_rolling"])
    print(f"Dataset: {len(df)} WR/TE player-games\n")

    all_probs, all_correct = [], []
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
        print(f"{holdout}: {acc*100:.1f}%")
        all_probs.extend(probs)
        all_correct.extend(preds == y_test)

    all_probs, all_correct = np.array(all_probs), np.array(all_correct)
    print(f"\nMean: {all_correct.mean()*100:.1f}%")
    conf = np.abs(all_probs - 0.5) * 2
    for t in [0.2, 0.3, 0.4, 0.5]:
        mask = conf >= t
        print(f"  >={t}: {all_correct[mask].mean()*100:.1f}% ({mask.sum()} games, {mask.mean()*100:.1f}%)")

    X_all, y_all = df[FEATURES].fillna(0), df["over_proxy_line"]
    rng = np.random.RandomState(42)
    n = len(X_all)
    models = []
    for i in range(100):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=2000)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_receptions_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["WR", "TE"]}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
