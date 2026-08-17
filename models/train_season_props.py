"""
Trains all 6 season-long cumulative prop models in one parameterized pass --
they're structurally identical (prior season's own totals -> next season's
total), differing only in which stat and which positions, so one shared
routine instead of 6 near-duplicate scripts. Same leave-one-season-out
backtest discipline as every other prop model, plus AUC (not just accuracy)
since the base rate here is meaningfully off 50/50 -- see
season_prop_features.py's docstring for why the proxy line is cruder at
season grain than at game grain.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from season_prop_features import build_season_prop_training_dataset

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]

REC_FEATURES = ["prior_receiving_yards", "prior_receiving_tds", "prior_receptions",
                 "prior_targets", "prior_games_played", "prior_yards_per_target",
                 "prior_yards_per_game_rec"]
RUSH_FEATURES = ["prior_rushing_yards", "prior_rushing_tds", "prior_carries",
                  "prior_games_played", "prior_yards_per_carry", "prior_yards_per_game_rush"]

CONFIGS = {
    "season_receiving_yards": {"stat_col": "receiving_yards", "positions": ["WR", "TE", "RB"], "features": REC_FEATURES},
    "season_rec_tds": {"stat_col": "receiving_tds", "positions": ["WR", "TE", "RB"], "features": REC_FEATURES},
    "season_rush_yards": {"stat_col": "rushing_yards", "positions": ["RB", "QB", "WR", "TE"], "features": RUSH_FEATURES},
    "season_rush_tds": {"stat_col": "rushing_tds", "positions": ["RB", "QB", "WR", "TE"], "features": RUSH_FEATURES},
}

# season_pass_yards / season_pass_tds were tried and dropped: AUC came back ~0.5
# (0.53 and 0.52 pooled) and, worse, individual holdout seasons landed BELOW 0.5
# (e.g. passing_tds 2020 AUC 0.480, 2023 AUC 0.468) -- not just weaker signal than
# the other four, no reliable signal at all. Makes sense: a QB's season passing
# total is dominated by whether they're even still the starter, which "prior own
# stats" can't see (backup->starter, new team/scheme, benching). Would need a
# depth-chart/starter-status feature to be worth shipping; not built yet.


def train_one(prop_type: str, config: dict) -> None:
    stat_col, positions, features = config["stat_col"], config["positions"], config["features"]
    df = build_season_prop_training_dataset(stat_col)
    df = df[df["position"].isin(positions)].copy()
    df = df.dropna(subset=features + ["over_proxy_line"])
    print(f"\n=== {prop_type} ===")
    print(f"Dataset: {len(df)} player-seasons, positions={positions}")

    base_rate = df["over_proxy_line"].mean()
    print(f"Base rate (target > prior): {base_rate*100:.1f}%")

    all_probs, all_y, all_correct = [], [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["target_season"] != holdout]
        test = df[df["target_season"] == holdout]
        if len(test) < 10:
            print(f"  {holdout}: skipped, only {len(test)} test rows")
            continue
        X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[features].fillna(0), test["over_proxy_line"].values
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"  {holdout}: {acc*100:.1f}% acc, n={len(test)}")
        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    auc = roc_auc_score(all_y, all_probs)
    print(f"  Pooled: {all_correct.mean()*100:.1f}% acc, AUC={auc:.3f} (0.5=no signal)")
    conf = np.abs(all_probs - 0.5) * 2
    for t in [0.2, 0.3, 0.4]:
        mask = conf >= t
        if mask.sum() > 0:
            print(f"    >={t}: {all_correct[mask].mean()*100:.1f}% ({mask.sum()} rows, {mask.mean()*100:.1f}%)")

    X_all, y_all = df[features].fillna(0), df["over_proxy_line"]
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
        pickle.dump({"models": models, "features": features, "position_scope": positions}, f)
    print(f"  Saved -> {out_path}")


def main():
    for prop_type, config in CONFIGS.items():
        train_one(prop_type, config)


if __name__ == "__main__":
    main()
