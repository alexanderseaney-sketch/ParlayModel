"""
Trains the remaining 3 QB-scoped anytime-TD-style props in one parameterized
pass -- period_first_touchdown_scored_qb, period_1_rush_rec_tds_qb,
period_1_2_rush_rec_tds_qb. Same reasoning as train_rush_rec_tds_qb_props.py
(see that file's docstring): QBs were excluded from all of these purely by
position filter, not data availability, since nothing in the underlying
dataset is NGS-derived. Trained as their own population rather than mixed
into the RB/WR/TE versions.

Pre-validated before building (all consistent across every 2020-2024 holdout
season, never near 0.5):
  period_first_touchdown_scored_qb: AUC 0.663 (0.580-0.799)
  period_1_rush_rec_tds_qb:         AUC 0.684 (0.623-0.800)
  period_1_2_rush_rec_tds_qb:       AUC 0.707 (0.626-0.780)
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from first_td_scorer_features import build_first_td_scorer_dataset
from period_rush_rec_tds_features import build_period_rush_rec_tds_dataset
from calibration_report import print_calibration_report

FEATURES = ["rush_rec_tds_rolling", "rush_rec_tds_last3", "carries_rolling", "carries_last3",
            "targets_rolling", "targets_last3", "rushing_yards_rolling", "receiving_yards_rolling",
            "def_epa_allowed_rolling", "red_zone_touches_rolling", "red_zone_share_rolling"]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def train_one(prop_type: str, df) -> None:
    df = df[df["position"] == "QB"].copy()
    df = df.dropna(subset=FEATURES)
    print(f"\n=== {prop_type} ===")
    print(f"Dataset: {len(df)} QB player-games")
    print(f"Base rate: {df['over_proxy_line'].mean()*100:.1f}%")

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
    print_calibration_report(all_probs, all_y, prop_type)

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
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["QB"]}, f)
    print(f"  Saved -> {out_path}")


def main():
    train_one("period_first_touchdown_scored_qb", build_first_td_scorer_dataset(min_week=4))
    train_one("period_1_rush_rec_tds_qb", build_period_rush_rec_tds_dataset("q1", min_week=4))
    train_one("period_1_2_rush_rec_tds_qb", build_period_rush_rec_tds_dataset("h1", min_week=4))


if __name__ == "__main__":
    main()
