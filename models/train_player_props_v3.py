"""
Mix-and-match test: adds team-level PBP efficiency + scheme features to the receiving-
yards prop model, individually and combined, same rigorous 5-season pooled CV plus the
confidence-filtered accuracy check (the metric that actually matters for real use).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from player_prop_features import build_receiving_yards_dataset

BASE_FEATURES = [
    "receiving_yards_rolling", "receiving_yards_last3", "targets_rolling", "targets_last3",
    "target_share_rolling", "receiving_air_yards_rolling",
    "avg_separation_rolling", "avg_cushion_rolling", "avg_yac_above_expectation_rolling",
    "def_epa_allowed_rolling",
]

NEW_FEATURE_GROUPS = {
    "team_pass_oe": ["team_pass_oe_rolling"],
    "team_tempo": ["team_shotgun_rate_rolling", "team_no_huddle_rate_rolling"],
    "team_early_down_pass": ["team_early_down_pass_rate_rolling"],
    "team_off_efficiency": ["team_off_epa_per_play_rolling", "team_off_success_rate_rolling"],
    "team_pass_efficiency": ["team_off_pass_epa_per_play_rolling"],
    "team_explosive_rate": ["team_off_explosive_rate_rolling"],
}

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def cv_pooled(df, features):
    """Returns pooled (probs, correct) across all 5 holdout seasons."""
    all_probs, all_correct = [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[features].fillna(0), test["over_proxy_line"].values
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_correct.extend(preds == y_test)
    return np.array(all_probs), np.array(all_correct)


def summarize(name, probs, correct):
    overall = correct.mean()
    conf = np.abs(probs - 0.5) * 2
    mask = conf >= 0.4
    conf_acc = correct[mask].mean() if mask.sum() > 0 else float("nan")
    conf_pct = mask.mean() * 100
    print(f"{name:<30} overall: {overall*100:>5.1f}%   @0.4 confidence: {conf_acc*100:>5.1f}% ({conf_pct:>4.1f}% of games)")


def main():
    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receiving_yards_rolling"])

    base_probs, base_correct = cv_pooled(df, BASE_FEATURES)
    summarize("BASE (no team features)", base_probs, base_correct)
    print()

    winners = []
    for name, feats in NEW_FEATURE_GROUPS.items():
        probs, correct = cv_pooled(df, BASE_FEATURES + feats)
        summarize(name, probs, correct)
        if correct.mean() > base_correct.mean():
            winners.append(name)

    print(f"\nIndividually beat base: {winners if winners else 'none'}")

    if winners:
        combined_feats = BASE_FEATURES + [f for name in winners for f in NEW_FEATURE_GROUPS[name]]
        probs, correct = cv_pooled(df, combined_feats)
        print()
        summarize("COMBINED (winners only)", probs, correct)

    all_feats = BASE_FEATURES + [f for feats in NEW_FEATURE_GROUPS.values() for f in feats]
    probs, correct = cv_pooled(df, all_feats)
    summarize("ALL new features combined", probs, correct)


if __name__ == "__main__":
    main()
