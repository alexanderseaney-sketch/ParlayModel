"""
Trains single-game passing_tds and passing_ints models -- both reuse
build_passing_yards_dataset() as-is rather than a new feature file, since it
already computes rolling passing_tds/interceptions columns as a side effect
of building the passing_yards dataset (it rolls all 4 QB counting stats
together). Only the proxy_line/target and which columns count as FEATURES
change per stat; the underlying data pull is identical.

Same leave-one-season-out discipline as everything else here.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from player_prop_passing_features import build_passing_yards_dataset
from calibration_report import print_calibration_report

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]

CONFIGS = {
    "passing_tds": {
        "stat_col": "passing_tds",
        "features": ["passing_tds_rolling", "passing_tds_last3", "attempts_rolling", "attempts_last3",
                      "completion_percentage_above_expectation_rolling", "avg_intended_air_yards_rolling",
                      "def_epa_allowed_rolling"],
    },
    "passing_ints": {
        "stat_col": "interceptions",
        "features": ["interceptions_rolling", "interceptions_last3", "attempts_rolling", "attempts_last3",
                      "aggressiveness_rolling", "completion_percentage_above_expectation_rolling",
                      "def_epa_allowed_rolling"],
    },
}


def train_one(prop_type: str, config: dict) -> None:
    stat_col, features = config["stat_col"], config["features"]
    df = build_passing_yards_dataset(min_week=4)
    df["proxy_line"] = df[f"{stat_col}_rolling"]
    df["over_proxy_line"] = (df[stat_col] > df["proxy_line"]).astype(int)
    df = df.dropna(subset=features)
    print(f"\n=== {prop_type} ===")
    print(f"Dataset: {len(df)} QB player-games")
    print(f"Base rate (over own rolling avg): {df['over_proxy_line'].mean()*100:.1f}%")

    all_probs, all_y, all_correct = [], [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features], train["over_proxy_line"]
        X_test, y_test = test[features], test["over_proxy_line"].values
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"  {holdout}: {acc*100:.1f}% acc, n={len(test)}")
        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    auc = roc_auc_score(all_y, all_probs)
    print(f"  Pooled: {all_correct.mean()*100:.1f}% acc, AUC={auc:.3f} (0.5=no signal)")
    print_calibration_report(all_probs, all_y, prop_type)
    conf = np.abs(all_probs - 0.5) * 2
    for t in [0.2, 0.3, 0.4]:
        mask = conf >= t
        if mask.sum() > 0:
            print(f"    >={t}: {all_correct[mask].mean()*100:.1f}% ({mask.sum()} rows, {mask.mean()*100:.1f}%)")

    X_all, y_all = df[features], df["over_proxy_line"]
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
        pickle.dump({"models": models, "features": features, "position_scope": ["QB"]}, f)
    print(f"  Saved -> {out_path}")


def main():
    for prop_type, config in CONFIGS.items():
        train_one(prop_type, config)


if __name__ == "__main__":
    main()
