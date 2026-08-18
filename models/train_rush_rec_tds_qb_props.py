"""
Trains a rush_rec_tds model for QBs specifically -- a real gap found while
auditing live prop coverage (2026-08-17): mobile-QB touchdown props (Josh
Allen, Lamar Jackson, Jalen Hurts, etc.) are a real, common Underdog market,
but rush_rec_tds was RB/WR/TE only. Unlike receiving_yards/rushing_yards,
nothing in this dataset is NGS-derived, so QBs use the exact same feature
set as the RB/WR/TE model -- just trained as a separate population, since
QB touchdown-scoring mechanics (goal-line sneaks, scramble TDs) are
different enough that mixing them in risked diluting the already-validated
RB/WR/TE model rather than improving it.

Same leave-one-season-out discipline as everything else here. Validated
before building: AUC 0.731 pooled, consistent 0.70-0.80 across all 5
holdout seasons -- actually stronger signal than the RB/WR/TE version,
likely because starting-QB rushing role is more stable/binary than a
committee backfield.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from player_prop_rush_rec_tds_features import build_rush_rec_tds_dataset
from calibration_report import print_calibration_report

FEATURES = ["rush_rec_tds_rolling", "rush_rec_tds_last3", "carries_rolling", "carries_last3",
            "targets_rolling", "targets_last3", "rushing_yards_rolling", "receiving_yards_rolling",
            "def_epa_allowed_rolling", "red_zone_touches_rolling", "red_zone_share_rolling"]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_rush_rec_tds_dataset(min_week=4)
    df = df[df["position"] == "QB"].copy()
    df = df.dropna(subset=FEATURES)
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
    auc = roc_auc_score(all_y, all_probs)
    print(f"  Pooled: {all_correct.mean()*100:.1f}% acc, AUC={auc:.3f} (0.5=no signal)")
    print_calibration_report(all_probs, all_y, "rush_rec_tds_qb")
    conf = np.abs(all_probs - 0.5) * 2
    for t in [0.2, 0.3, 0.4]:
        mask = conf >= t
        if mask.sum() > 0:
            print(f"    >={t}: {all_correct[mask].mean()*100:.1f}% ({mask.sum()} rows, {mask.mean()*100:.1f}%)")

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
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_rush_rec_tds_qb_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["QB"]}, f)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
