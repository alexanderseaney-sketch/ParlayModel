"""
Experiment (2026-10-02): outcome-conditioned training population via NGS presence.

NGS weekly files only list a player for games where he cleared a volume minimum in
THAT game. The production builders join NGS exactly on (player, season, week) and the
train scripts dropna(FEATURES), so a row survives only if the player got real volume
in the very game being predicted -- the model never sees the 2-carry backup game, and
learns that low-average players almost always beat their average (live: Kendre
Miller 98% OVER 25.5 rush yds vs a 50% market).

Datasets built with feature_engineering.NGS_ASOF (latest NGS rolling as of each game)
and no usage filter, then on identical leave-one-season-out folds:
  population : PRE-game rolling usage >= the production minimum, all outcomes
  A (prod)   : train on label usage >= min AND the game has an NGS row
  B          : train on pre-game usage >= min (as-of NGS)
  C          : train on label usage >= min (as-of NGS, no NGS-row requirement)
All scored on the population (holdout season). Week-0 NGS rows are already excluded.

Usage: python models/experiments/ngs_selection.py [rushing_yards receiving_yards receptions]
"""
import importlib
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_engineering  # noqa: E402
from calibration import log_loss, brier  # noqa: E402

warnings.filterwarnings("ignore")

TARGETS = {
    "rushing_yards": ("train_rushing_props", "player_prop_rushing_features", "MIN_CARRIES_TO_QUALIFY",
                      "carries", "carries_rolling"),
    "receiving_yards": ("train_player_props", "player_prop_features", "MIN_TARGETS_TO_QUALIFY",
                        "targets", "targets_rolling"),
    "receptions": ("train_receptions_props", "player_prop_receptions_features", "MIN_TARGETS_TO_QUALIFY",
                   "targets", "targets_rolling"),
}


def run(prop: str):
    train_mod, feat_mod, const, usage, pre = TARGETS[prop]
    fm, tm = importlib.import_module(feat_mod), importlib.import_module(train_mod)
    minimum = getattr(fm, const)
    feature_engineering.NGS_ASOF = True
    feature_engineering.PREGAME_QUALIFIER = False  # min=-1 below disables filtering anyway
    setattr(fm, const, -1)
    try:
        df = tm._build_dataset()
    finally:
        setattr(fm, const, minimum)
    F = tm.FEATURES
    df = df.dropna(subset=[pre, "proxy_line"]).copy()
    sel = {"A": (df[usage] >= minimum) & df["_ngs_row"], "B": df[pre] >= minimum, "C": df[usage] >= minimum}
    pop = df[pre] >= minimum
    print(f"\n=== {prop}: train rows A={sel['A'].sum()} B={sel['B'].sum()} C={sel['C'].sum()}; "
          f"population={pop.sum()} (base rate {df.loc[pop, 'over_proxy_line'].mean():.1%})")
    preds = {k: [] for k in sel}
    ys = []
    print(f"{'season':<8}" + "".join(f"{k + ' LL':>9}" for k in sel) + f"{'n':>7}")
    for h in tm.HOLDOUT_SEASONS:
        te = df[(df["season"] == h) & pop]
        if te.empty:
            continue
        y = te["over_proxy_line"].to_numpy()
        row = f"{h:<8}"
        for k, mask in sel.items():
            tr = df[(df["season"] != h) & mask]
            p = XGBClassifier(**tm.XGB_PARAMS).fit(tr[F], tr["over_proxy_line"]).predict_proba(te[F])[:, 1]
            preds[k].append(p)
            row += f"{log_loss(p, y):>9.4f}"
        ys.append(y)
        print(row + f"{len(te):>7}")
    y = np.concatenate(ys)
    print("pooled  " + "".join(f"{log_loss(np.concatenate(v), y):>9.4f}" for v in preds.values()))
    for k, v in preds.items():
        p = np.concatenate(v)
        hi = p >= 0.9
        print(f"  {k}: AUC {roc_auc_score(y, p):.4f}  Brier {brier(p, y):.4f}  "
              f"P>=0.9 n={hi.sum()} actual {y[hi].mean() if hi.any() else float('nan'):.1%}")


if __name__ == "__main__":
    for prop in (sys.argv[1:] or TARGETS):
        run(prop)
