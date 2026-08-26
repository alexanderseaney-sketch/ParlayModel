"""
Genuine walk-forward holdout backtest of the CURRENT production feature set: train each
prop model using ONLY seasons before 2024 (2019-2023), then score every 2024 game
blind, and compare to what actually happened. This directly answers "if we'd built this
model using only data available before the 2024 season, how would it actually have
performed on games we now know the results of" -- it retrains from scratch on real
held-out data rather than re-reading the previously reported numbers.

Uses the exact same model type/hyperparameters as the saved production models
(introspected from each .pkl: XGBoost n_estimators=200/max_depth=3/lr=0.05/
subsample=0.8/colsample_bytree=0.8 for receiving/rushing/receptions; LogisticRegression
max_iter=3000 for passing) and the exact same feature lists (each .pkl's own `features`
key) -- an apples-to-apples re-creation of the original methodology, not a new one.

Caveat carried over from everywhere else in this project: "accuracy" here still means
beating the player's own trailing rolling average (the proxy line), not a real
historical Underdog line -- that archive is still being built (see README).
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from current_predictions import PROP_CONFIGS, CONTEXT_MERGERS, _build_base_dataset

TEST_SEASON = 2024
N_BOOTSTRAP = 100
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def _load_features_and_model_type(prop_type: str) -> tuple[list[str], str]:
    model_path = os.path.join(os.path.dirname(__file__), PROP_CONFIGS[prop_type]["model_path"])
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    return saved["features"], type(saved["models"][0]).__name__


def backtest_prop(prop_type: str, config: dict) -> dict:
    features, model_type = _load_features_and_model_type(prop_type)

    df = _build_base_dataset(prop_type, min_week=4)
    df = df[df["position"].isin(config["positions"])].copy()
    df = df.dropna(subset=[config["proxy_col"]])
    for merge_name in config["context"]:
        df = CONTEXT_MERGERS[merge_name](df)

    train = df[df["season"] < TEST_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()

    X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
    X_test, y_test = test[features].fillna(0), test["over_proxy_line"].values

    print(f"[{prop_type}] train: {len(train)} rows, seasons {sorted(train['season'].unique())}")
    print(f"[{prop_type}] test:  {len(test)} rows, season {TEST_SEASON} only (never in training)")

    rng = np.random.RandomState(42)
    n = len(X_train)
    models = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        m = XGBClassifier(**XGB_PARAMS) if model_type == "XGBClassifier" else LogisticRegression(max_iter=3000)
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        models.append(m)

    all_probs = np.array([m.predict_proba(X_test)[:, 1] for m in models])
    mean_prob = all_probs.mean(axis=0)
    preds = (mean_prob > 0.5).astype(int)
    correct = preds == y_test
    confidence = np.abs(mean_prob - 0.5) * 2

    print(f"[{prop_type}] overall accuracy on held-out {TEST_SEASON}: {correct.mean()*100:.1f}% ({len(test)} games)")
    thresholds_report = {}
    for t in [0.0, 0.2, 0.3, 0.4, 0.5]:
        mask = confidence >= t
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        thresholds_report[t] = (acc, int(mask.sum()))
        print(f"    >= {t} confidence: {acc*100:.1f}% accuracy on {mask.sum()} games ({mask.mean()*100:.1f}% of games)")
    print()

    return {
        "prop_type": prop_type,
        "n_test": len(test),
        "overall_acc": correct.mean(),
        "thresholds": thresholds_report,
    }


def main():
    results = [backtest_prop(prop_type, config) for prop_type, config in PROP_CONFIGS.items()]

    print("=" * 78)
    print(f"SUMMARY: genuine {TEST_SEASON} holdout backtest (trained ONLY on seasons before {TEST_SEASON})")
    print("=" * 78)
    for r in results:
        acc04, n04 = r["thresholds"].get(0.4, (None, 0))
        acc04_str = f"{acc04*100:.1f}%" if acc04 is not None else "n/a"
        pct04 = f"{n04 / r['n_test'] * 100:.0f}%" if r["n_test"] else "0%"
        print(f"{r['prop_type']:16s} overall: {r['overall_acc']*100:5.1f}%   "
              f"@0.4 confidence: {acc04_str:>6s} on {n04} of {r['n_test']} games ({pct04})")


if __name__ == "__main__":
    main()
