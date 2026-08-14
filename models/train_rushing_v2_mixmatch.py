"""
Game-context mix-and-match for rushing yards: implied team total, home/away, weather,
rest, snap share. Game script is classically thought to matter more for rushing
specifically (leading teams run out the clock) than receiving.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from player_prop_rushing_features import build_rushing_yards_dataset
from game_context_features import build_game_context, add_snap_share

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

BASE_FEATURES = [
    "rushing_yards_rolling", "rushing_yards_last3", "carries_rolling", "carries_last3",
    "targets_rolling", "receiving_yards_rolling",
    "rush_yards_over_expected_per_att_rolling", "percent_attempts_gte_eight_defenders_rolling",
    "efficiency_rolling", "def_epa_allowed_rolling",
]
NEW_GROUPS = {
    "implied_total": ["team_implied_total"],
    "home_away": ["is_home"],
    "weather": ["temp", "wind", "is_dome"],
    "rest": ["rest_days"],
    "snap_share": ["offense_pct_rolling"],
}
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def cv_pooled(df, features):
    all_probs, all_correct = [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[features].fillna(0), test["over_proxy_line"].values
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_correct.extend(preds == y_test)
    return np.array(all_probs), np.array(all_correct)


def summarize(name, probs, correct):
    conf = np.abs(probs - 0.5) * 2
    mask = conf >= 0.4
    conf_acc = correct[mask].mean() if mask.sum() > 0 else float("nan")
    print(f"{name:<25} overall: {correct.mean()*100:>5.1f}%   @0.4 conf: {conf_acc*100:>5.1f}% ({mask.mean()*100:>4.1f}% of games)")


def main():
    df = build_rushing_yards_dataset(min_week=4)
    df = df.dropna(subset=["rushing_yards_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(context, on=["recent_team", "season", "week"], how="left")
    df = add_snap_share(df, id_col="player_id")

    base_probs, base_correct = cv_pooled(df, BASE_FEATURES)
    summarize("BASE", base_probs, base_correct)
    print()

    winners = []
    for name, feats in NEW_GROUPS.items():
        probs, correct = cv_pooled(df, BASE_FEATURES + feats)
        summarize(name, probs, correct)
        if correct.mean() > base_correct.mean():
            winners.append(name)

    print(f"\nIndividually beat base: {winners if winners else 'none'}")
    if winners:
        combined = BASE_FEATURES + [f for n in winners for f in NEW_GROUPS[n]]
        probs, correct = cv_pooled(df, combined)
        print()
        summarize("COMBINED (winners)", probs, correct)

    all_feats = BASE_FEATURES + [f for feats in NEW_GROUPS.values() for f in feats]
    probs, correct = cv_pooled(df, all_feats)
    summarize("ALL combined", probs, correct)


if __name__ == "__main__":
    main()
