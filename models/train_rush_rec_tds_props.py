import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from player_prop_rush_rec_tds_features import build_rush_rec_tds_dataset
from calibration_report import print_calibration_report

FEATURES = [
    "rush_rec_tds_rolling", "rush_rec_tds_last3", "carries_rolling", "carries_last3",
    "targets_rolling", "targets_last3", "rushing_yards_rolling", "receiving_yards_rolling",
    "def_epa_allowed_rolling", "red_zone_touches_rolling", "red_zone_share_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_rush_rec_tds_dataset(min_week=4)
    df = df[df["position"].isin(["RB", "WR", "TE"])].copy()
    df = df.dropna(subset=["rush_rec_tds_rolling"])
    print(f"Dataset: {len(df)} RB/WR/TE player-games\n")

    all_probs, all_y, all_correct = [], [], []
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
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    print(f"\nMean: {all_correct.mean()*100:.1f}%")
    conf = np.abs(all_probs - 0.5) * 2
    for t in [0.2, 0.3, 0.4, 0.5]:
        mask = conf >= t
        print(f"  >={t}: {all_correct[mask].mean()*100:.1f}% ({mask.sum()} games, {mask.mean()*100:.1f}%)")
    print_calibration_report(all_probs, all_y, "rush_rec_tds")

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
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_rush_rec_tds_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["RB", "WR", "TE"]}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
