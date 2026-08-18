"""
Trains the single-game sacks model. Same leave-one-season-out discipline as
everything else here, plus AUC (sacks are a rare discrete event -- ~11% base
rate for "beat your own rolling average" -- so raw accuracy alone would be
misleading, same lesson learned from rush_rec_tds).

Uses XGBoost, not LogisticRegression -- head-to-head tested (see the
"making models more accurate" pass) and XGBoost won here by +0.028 AUC
(0.736 vs 0.708), the one clear win out of six models tested. The other
five (rush_rec_tds, receiving_yards_rb, season_rec_tds,
period_first_touchdown_scored, passing_yards) all did the same or worse
with XGBoost, so they stayed on LogisticRegression -- this isn't a
blanket "XGBoost is better" upgrade, it needed testing per model. Same
XGB_PARAMS already proven in production for receiving_yards/rushing_yards/
receptions (see train_momentum_and_receptions_gaps.py).
"""
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from defensive_sacks_features import build_sacks_dataset, DEFENSIVE_POSITIONS
from calibration_report import print_calibration_report

FEATURES = ["sacks_rolling", "sacks_last3", "plays_on_defense_rolling",
            "plays_on_defense_last3", "team_sacks_rolling"]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def main():
    df = build_sacks_dataset(min_week=4)
    df = df.dropna(subset=FEATURES)
    print(f"Dataset: {len(df)} defender-games\n")
    print(f"Base rate (over own rolling avg): {df['over_proxy_line'].mean()*100:.1f}%")

    all_probs, all_y, all_correct = [], [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[FEATURES], train["over_proxy_line"]
        X_test, y_test = test[FEATURES], test["over_proxy_line"].values
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"{holdout}: {acc*100:.1f}% acc, n={len(test)}")
        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    auc = roc_auc_score(all_y, all_probs)
    print(f"\nPooled: {all_correct.mean()*100:.1f}% acc, AUC={auc:.3f} (0.5=no signal)")
    print_calibration_report(all_probs, all_y, "sacks")
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
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_sacks_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": DEFENSIVE_POSITIONS}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
