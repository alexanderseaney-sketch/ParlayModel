"""
Tests a specific hypothesis from Alex: should the model actually get smarter as a
season goes on, by incorporating that season's own emerging weeks into training
(reweighted so they're not numerically drowned out by 5+ years of historical rows),
rather than staying frozen from a one-time historical fit?

Simulates real weekly retraining through the 2024 season: for each week W, trains on
2019-2023 + 2024 weeks before W, with RECENCY-WEIGHTED samples (weight = 0.75 **
(2024 - row_season), so 2024's own rows always carry the most weight and get MORE
influential as more of them accumulate week over week), then scores week W blind.
This exactly mirrors how it would work in production if wired into the weekly
pipeline -- retrain after each new week's data lands, score the next week with the
updated model.

Compared against the STATIC baseline already validated in backtest_holdout_2024.py:
trained once on 2019-2023 with every season weighted equally, never updated as 2024
progressed. Same test rows both ways -- a genuine apples-to-apples comparison, not
just a hunch about whether recency weighting sounds like it should help.

Uses single (non-bootstrap) models per week rather than the full 100-model production
ensemble -- this is a validation experiment, not what gets deployed, and 100x fewer
fits keeps ~14 weeks x 4 props tractable. If this shows real promise, the production
retraining script would still use the full bootstrap ensemble.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from current_predictions import PROP_CONFIGS, CONTEXT_MERGERS, _build_base_dataset

TEST_SEASON = 2024
RECENCY_DECAY = 0.75  # each season back from TEST_SEASON counts 25% less
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def _load_model_type(prop_type: str) -> str:
    path = os.path.join(os.path.dirname(__file__), PROP_CONFIGS[prop_type]["model_path"])
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return type(saved["models"][0]).__name__, saved["features"]


def _fit(model_type: str, X, y, sample_weight=None):
    m = XGBClassifier(**XGB_PARAMS) if model_type == "XGBClassifier" else LogisticRegression(max_iter=3000)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def backtest_prop(prop_type: str, config: dict):
    model_type, features = _load_model_type(prop_type)

    df = _build_base_dataset(prop_type, min_week=4)
    df = df[df["position"].isin(config["positions"])].copy()
    df = df.dropna(subset=[config["proxy_col"]])
    for merge_name in config["context"]:
        df = CONTEXT_MERGERS[merge_name](df)

    historical = df[df["season"] < TEST_SEASON]
    current = df[df["season"] == TEST_SEASON].sort_values("week")
    weeks = sorted(current["week"].unique())
    # First simulate-able week needs at least one prior current-season week to train
    # on beyond pure history -- otherwise it's identical to the static model by
    # construction and adds no information to the comparison.
    weeks = [w for w in weeks if w > weeks[0]]

    static_correct, recency_correct = [], []

    for w in weeks:
        train_hist = historical
        train_cur = current[current["week"] < w]
        test = current[current["week"] == w]
        if test.empty or train_cur.empty:
            continue

        train = pd.concat([train_hist, train_cur], ignore_index=True)
        X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[features].fillna(0), test["over_proxy_line"].values

        # --- static baseline: every season weighted equally ---
        static_model = _fit(model_type, X_train, y_train, sample_weight=None)
        static_pred = (static_model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        static_correct.extend(static_pred == y_test)

        # --- recency-weighted: this season's own weeks count most, and count more
        # as more of them accumulate ---
        weights = RECENCY_DECAY ** (TEST_SEASON - train["season"])
        recency_model = _fit(model_type, X_train, y_train, sample_weight=weights.values)
        recency_pred = (recency_model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        recency_correct.extend(recency_pred == y_test)

    static_correct = np.array(static_correct)
    recency_correct = np.array(recency_correct)
    n = len(static_correct)
    static_acc = static_correct.mean() if n else float("nan")
    recency_acc = recency_correct.mean() if n else float("nan")

    print(f"[{prop_type}] {n} test games across weeks {weeks[0] if weeks else '-'}-{weeks[-1] if weeks else '-'}")
    print(f"[{prop_type}] static (equal-weight, frozen):     {static_acc*100:.1f}%")
    print(f"[{prop_type}] recency-weighted (updates weekly): {recency_acc*100:.1f}%")
    delta = recency_acc - static_acc
    verdict = "WIN" if delta > 0.005 else ("WORSE" if delta < -0.005 else "NULL")
    print(f"[{prop_type}] -> {verdict} ({delta*100:+.2f}pt)\n")

    return {"prop_type": prop_type, "n": n, "static_acc": static_acc, "recency_acc": recency_acc, "delta": delta}


def main():
    results = [backtest_prop(p, c) for p, c in PROP_CONFIGS.items()]

    print("=" * 78)
    print("SUMMARY: does recency-weighted weekly retraining beat the static model?")
    print("=" * 78)
    for r in results:
        print(f"{r['prop_type']:16s} static: {r['static_acc']*100:5.1f}%   "
              f"recency-weighted: {r['recency_acc']*100:5.1f}%   ({r['delta']*100:+.2f}pt, n={r['n']})")


if __name__ == "__main__":
    main()
