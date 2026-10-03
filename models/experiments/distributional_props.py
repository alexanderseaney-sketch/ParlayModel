"""
Experiment (2026-10-02): is a DISTRIBUTIONAL model better than the production
"classifier + normal shift" path at pricing lines that differ from the proxy?

Production: classifier gives P(stat > proxy line); utils.recompute_probability_for_
real_line() converts that to P(stat > real line) by assuming stat ~ Normal(implied_mean,
player_std). That conversion has never been validated -- there were no real lines.

But it can be validated on history WITHOUT real lines: outcomes are known, so grade
both methods at synthetic lines = proxy x {0.7, 0.85, 1.0, 1.15, 1.3} (rounded to .5,
the shape of real lines) on the same leave-one-season-out folds.

  A. production: XGB classifier P(over proxy) -> normal shift with the player's own
     trailing std (as the dashboard does).
  B. distributional: LightGBM regressor for the stat's mean (same features) + a
     second regressor for |residual| (heteroscedastic spread), P(over L) from a
     normal with those two -- no proxy involved.
  C. distributional with the mean regressor only and the player's trailing std.

Usage: python models/experiments/distributional_props.py [receiving_yards|receptions|rushing_yards]
"""
import importlib
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import log_loss, brier  # noqa: E402

warnings.filterwarnings("ignore")

TARGETS = {  # prop -> (train module, stat column)
    "receiving_yards": ("train_player_props", "receiving_yards"),
    "receptions": ("train_receptions_props", "receptions"),
    "rushing_yards": ("train_rushing_props", "rushing_yards"),
}
LINE_MULTS = [0.7, 0.85, 1.0, 1.15, 1.3]
REG = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _player_std(df: pd.DataFrame, stat: str) -> pd.Series:
    """Trailing (strictly prior) 16-game std of the stat per player -- what the
    dashboard's estimate_player_stat_std approximates."""
    d = df.sort_values(["player_id", "season", "week"])
    s = d.groupby("player_id")[stat].transform(lambda x: x.shift(1).rolling(16, min_periods=4).std())
    return s.reindex(df.index)


def run(prop: str):
    mod_name, stat = TARGETS[prop]
    mod = importlib.import_module(mod_name)
    df = mod._build_dataset().dropna(subset=mod.FEATURES).copy()
    F = mod.FEATURES
    df["pstd"] = _player_std(df, stat)
    df["pstd"] = df["pstd"].fillna(df.groupby("position")["pstd"].transform("median")).clip(lower=0.5)
    rows = []
    for h in mod.HOLDOUT_SEASONS:
        tr, te = df[df["season"] != h], df[df["season"] == h]
        if te.empty:
            continue
        clf = XGBClassifier(**mod.XGB_PARAMS).fit(tr[F], tr["over_proxy_line"])
        p_proxy = clf.predict_proba(te[F])[:, 1]
        mu_m = LGBMRegressor(**REG).fit(tr[F], tr[stat])
        mu_tr = mu_m.predict(tr[F])
        sd_m = LGBMRegressor(**REG).fit(tr[F], np.abs(tr[stat] - mu_tr))
        mu = mu_m.predict(te[F])
        sd = np.clip(sd_m.predict(te[F]) * np.sqrt(np.pi / 2), 0.5, None)  # mean|e| -> sigma
        proxy = te["proxy_line"].to_numpy()
        y_stat = te[stat].to_numpy()
        z = norm.ppf(np.clip(p_proxy, 1e-4, 1 - 1e-4))
        for m in LINE_MULTS:
            line = np.floor(proxy * m) + 0.5
            y = (y_stat > line).astype(int)
            pA = norm.sf((line - (proxy + z * te["pstd"].to_numpy())) / te["pstd"].to_numpy())
            pB = norm.sf((line - mu) / sd)
            pC = norm.sf((line - mu) / te["pstd"].to_numpy())
            rows.append(pd.DataFrame({"season": h, "mult": m, "y": y, "A": pA, "B": pB, "C": pC}))
    res = pd.concat(rows)
    print(f"\n=== {prop}: log loss by synthetic line (x proxy) -- A=production, B=distributional, C=mean+player std")
    print(f"{'mult':<6}{'A':>9}{'B':>9}{'C':>9}{'n':>8}")
    for m, g in res.groupby("mult"):
        print(f"{m:<6}{log_loss(g['A'], g['y']):>9.4f}{log_loss(g['B'], g['y']):>9.4f}"
              f"{log_loss(g['C'], g['y']):>9.4f}{len(g):>8}")
    print(f"{'all':<6}{log_loss(res['A'], res['y']):>9.4f}{log_loss(res['B'], res['y']):>9.4f}"
          f"{log_loss(res['C'], res['y']):>9.4f}")
    print(f"Brier all: A {brier(res['A'], res['y']):.4f}  B {brier(res['B'], res['y']):.4f}  C {brier(res['C'], res['y']):.4f}")
    wins = sum(log_loss(g["B"], g["y"]) < log_loss(g["A"], g["y"]) for _, g in res.groupby("season"))
    print(f"B beats A in {wins}/{res['season'].nunique()} seasons (all lines pooled)")


if __name__ == "__main__":
    for prop in (sys.argv[1:] or TARGETS):
        run(prop)
