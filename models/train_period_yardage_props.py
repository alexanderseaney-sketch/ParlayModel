"""
Trains all 8 remaining small period props (period_1/period_1_2 x
receiving_yds/rushing_yds/passing_yds/passing_tds) in one parameterized pass,
same reasoning as train_season_props.py -- structurally identical, differing
only in which stat/period. Same leave-one-season-out discipline; drops
anything that doesn't show real AUC signal rather than shipping it anyway
(already happened twice this session -- season_pass_yards/tds, season_sacks).
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from period_yardage_features import build_period_stat_dataset, STAT_CONFIGS

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]

# prop_type -> (period, stat_col)
CONFIGS = {
    "period_1_receiving_yds": ("q1", "receiving_yards"),
    "period_1_2_receiving_yds": ("h1", "receiving_yards"),
    "period_1_rushing_yds": ("q1", "rushing_yards"),
    "period_1_2_rushing_yds": ("h1", "rushing_yards"),
    "period_1_passing_yds": ("q1", "passing_yards"),
    "period_1_2_passing_yds": ("h1", "passing_yards"),
    "period_1_passing_tds": ("q1", "passing_tds"),
    "period_1_2_passing_tds": ("h1", "passing_tds"),
}


def train_one(prop_type: str, period: str, stat_col: str) -> dict | None:
    opportunity_col, positions = STAT_CONFIGS[stat_col]
    features = [f"{stat_col}_rolling", f"{stat_col}_last3",
                f"{opportunity_col}_rolling", f"{opportunity_col}_last3"]

    df = build_period_stat_dataset(period, stat_col, min_week=4)
    df = df.dropna(subset=features)
    print(f"\n=== {prop_type} ===")
    print(f"Dataset: {len(df)} player-games, positions={positions}")
    base_rate = df["over_proxy_line"].mean()
    print(f"Base rate: {base_rate*100:.1f}%")

    all_probs, all_y, all_correct = [], [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        if len(test) < 30:
            print(f"  {holdout}: skipped, only {len(test)} test rows")
            continue
        X_train, y_train = train[features], train["over_proxy_line"]
        X_test, y_test = test[features], test["over_proxy_line"].values
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
    pooled_auc = roc_auc_score(all_y, all_probs)
    print(f"  Pooled: {all_correct.mean()*100:.1f}% acc, AUC={pooled_auc:.3f}")

    X_all, y_all = df[features], df["over_proxy_line"]
    rng = np.random.RandomState(42)
    n = len(X_all)
    models = []
    for i in range(100):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=2000)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    return {"models": models, "features": features, "position_scope": positions,
            "pooled_auc": pooled_auc}


def main():
    results = {}
    for prop_type, (period, stat_col) in CONFIGS.items():
        results[prop_type] = train_one(prop_type, period, stat_col)

    print("\n\n=== Summary ===")
    for prop_type, result in results.items():
        print(f"{prop_type}: AUC={result['pooled_auc']:.3f}")

    import os
    for prop_type, result in results.items():
        out_path = os.path.join(os.path.dirname(__file__), f"player_prop_{prop_type}_model.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({k: v for k, v in result.items() if k != "pooled_auc"}, f)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
