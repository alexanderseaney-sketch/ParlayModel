"""
Trains and validates the production receptions model for WR/TE.

Reconstructed 2026-08-17 to match what's actually deployed in
player_prop_receptions_model.pkl (XGBoost + divisional-game context) --
see train_player_props.py's docstring for why (a since-lost promotion
script had drifted out of sync with the original version of this file).
Confirmed against the deployed .pkl's own "features" list.

Also the vehicle for the "5 more years of history" pass -- see
train_player_props.py's docstring for the validated finding. This model
doesn't depend on NGS receiving (no avg_separation/cushion here, unlike
receiving_yards), so it can use the full weekly_stats.csv history without
an NGS-imposed floor.

Target: does the player beat their own trailing rolling average ("proxy
line")? See the honest caveat in player_prop_receptions_features.py.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from player_prop_receptions_features import build_receptions_dataset
from individual_context_features import build_game_flags
from calibration_report import print_calibration_report

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

FEATURES = [
    "receptions_rolling", "receptions_last3", "targets_rolling", "targets_last3",
    "target_share_rolling", "receiving_air_yards_rolling", "catch_rate_rolling",
    "avg_separation_rolling", "avg_cushion_rolling", "catch_percentage_rolling",
    "def_epa_allowed_rolling", "div_game",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def _build_dataset() -> pd.DataFrame:
    df = build_receptions_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receptions_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(flags, on=["recent_team", "season", "week"], how="left")
    return df


def main():
    df = _build_dataset()
    df = df.dropna(subset=FEATURES)
    print(f"Dataset: {len(df)} WR/TE player-games ({df['season'].min()}-{df['season'].max()}, week 4+)\n")

    all_probs, all_y, all_correct = [], [], []
    print(f"{'Holdout':<10} {'Accuracy':>10} {'AUC':>8} {'n_test':>8}")
    print("-" * 40)
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
        auc = roc_auc_score(y_test, probs)
        print(f"{holdout:<10} {acc*100:>9.1f}% {auc:>8.3f} {len(test):>8}")

        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    print("-" * 40)
    print(f"{'Pooled':<10} {all_correct.mean()*100:>9.1f}% {roc_auc_score(all_y, all_probs):>8.3f}")
    print_calibration_report(all_probs, all_y, "receptions")
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

    out_path = os.path.join(os.path.dirname(__file__), "player_prop_receptions_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["WR", "TE"]}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
