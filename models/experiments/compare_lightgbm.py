"""
Head-to-head: production learner vs LightGBM vs their average, on the exact
leave-one-season-out folds and datasets the production train_*.py scripts use
(imports each script's own _build_dataset / FEATURES so nothing drifts).

Scored on pooled holdout log loss (primary -- it's what EV math consumes), Brier
and AUC. A learner is only worth promoting if it wins on log loss across most
holdout seasons, not just pooled.

Usage: python models/experiments/compare_lightgbm.py [prop ...]
"""
import importlib
import os
import sys
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import log_loss, brier  # noqa: E402

warnings.filterwarnings("ignore")

# prop -> (train module, production learner)
TARGETS = {
    "receiving_yards": ("train_player_props", "xgb"),
    "receptions": ("train_receptions_props", "xgb"),
    "rushing_yards": ("train_rushing_props", "xgb"),
    "passing_yards": ("train_passing_props", "logreg"),
}
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def _prod(kind, mod):
    if kind == "xgb":
        return XGBClassifier(**mod.XGB_PARAMS)
    return LogisticRegression(max_iter=5000)


def run(prop: str):
    mod_name, kind = TARGETS[prop]
    mod = importlib.import_module(mod_name)
    df = mod._build_dataset().dropna(subset=mod.FEATURES)
    F, H = mod.FEATURES, mod.HOLDOUT_SEASONS
    pooled = {"prod": [], "lgbm": [], "avg": []}
    ys, wins = [], 0
    print(f"\n=== {prop} ({kind} in production) -- {len(df)} rows")
    print(f"{'season':<8}{'prod LL':>9}{'lgbm LL':>9}{'avg LL':>9}")
    for h in H:
        tr, te = df[df["season"] != h], df[df["season"] == h]
        if te.empty:
            continue
        y = te["over_proxy_line"].to_numpy()
        p_prod = _prod(kind, mod).fit(tr[F], tr["over_proxy_line"]).predict_proba(te[F])[:, 1]
        p_lgbm = LGBMClassifier(**LGBM_PARAMS).fit(tr[F], tr["over_proxy_line"]).predict_proba(te[F])[:, 1]
        p_avg = (p_prod + p_lgbm) / 2
        lls = [log_loss(p, y) for p in (p_prod, p_lgbm, p_avg)]
        wins += lls[1] < lls[0]
        print(f"{h:<8}{lls[0]:>9.4f}{lls[1]:>9.4f}{lls[2]:>9.4f}")
        for k, p in zip(pooled, (p_prod, p_lgbm, p_avg)):
            pooled[k].append(p)
        ys.append(y)
    y = np.concatenate(ys)
    print("pooled  " + "".join(f"{log_loss(np.concatenate(v), y):>9.4f}" for v in pooled.values()))
    for k, v in pooled.items():
        p = np.concatenate(v)
        print(f"  {k:5} AUC {roc_auc_score(y, p):.4f}  Brier {brier(p, y):.4f}")
    print(f"  LightGBM beat production in {wins}/{len(ys)} seasons")


if __name__ == "__main__":
    for prop in (sys.argv[1:] or TARGETS):
        run(prop)
