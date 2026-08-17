"""
Trains a receiving-yards model for RBs specifically -- a real gap found while
auditing prop coverage: pass-catching RBs (Bijan Robinson, Saquon Barkley,
Christian McCaffrey, etc.) were being silently excluded from receiving_yards
entirely, not because of missing data, but because train_player_props.py
filters to WR/TE only.

That WR/TE filter exists for a real reason, though: nflverse's NGS receiving
pull has zero RB rows (checked directly against the pulled data), so 3 of
that model's 10 features (avg_separation/cushion/yac_above_expectation) don't
exist for RBs. Rather than fillna(0)-ing those into a WR/TE-fitted model
(which would just teach it "0 separation" as a proxy for "this is a RB",
distorting the coefficients that are already validated for WR/TE), this
trains a SEPARATE model using only the non-NGS features that
build_receiving_yards_dataset() already computes for every position --
target share, air yards, team pass-rate/efficiency context. Reuses that same
dataset builder rather than a new feature file, since nothing about how the
features are built needs to change, only which columns and which position.

Same leave-one-season-out discipline as everything else here.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from player_prop_features import build_receiving_yards_dataset

FEATURES = [
    "receiving_yards_rolling", "receiving_yards_last3", "targets_rolling", "targets_last3",
    "target_share_rolling", "receiving_air_yards_rolling", "def_epa_allowed_rolling",
    "team_pass_oe_rolling", "team_off_epa_per_play_rolling",
]
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"] == "RB"].copy()
    df = df.dropna(subset=FEATURES)
    print(f"Dataset: {len(df)} RB player-games\n")

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
        print(f"{holdout}: {acc*100:.1f}% acc, n={len(test)}")
        all_probs.extend(probs)
        all_y.extend(y_test)
        all_correct.extend(preds == y_test)

    all_probs, all_y, all_correct = np.array(all_probs), np.array(all_y), np.array(all_correct)
    auc = roc_auc_score(all_y, all_probs)
    print(f"\nPooled: {all_correct.mean()*100:.1f}% acc, AUC={auc:.3f} (0.5=no signal)")
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
        m = LogisticRegression(max_iter=2000)
        m.fit(X_all.iloc[idx], y_all.iloc[idx])
        models.append(m)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "player_prop_receiving_yards_rb_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"models": models, "features": FEATURES, "position_scope": ["RB"]}, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
