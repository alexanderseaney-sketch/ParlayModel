"""
Game-context mix-and-match for passing yards. Game script hypothesis is strongest here:
trailing teams throw more, leading teams run the clock out.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from player_prop_passing_features import build_passing_yards_dataset
from game_context_features import build_game_context, add_snap_share

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

BASE_FEATURES = [
    "passing_yards_rolling", "passing_yards_last3", "attempts_rolling", "attempts_last3",
    "passing_tds_rolling", "interceptions_rolling",
    "completion_percentage_above_expectation_rolling", "avg_intended_air_yards_rolling",
    "aggressiveness_rolling", "avg_time_to_throw_rolling", "def_epa_allowed_rolling",
]
NEW_GROUPS = {
    "implied_total": ["team_implied_total"],
    "spread": ["spread_line"],
    "home_away": ["is_home"],
    "weather": ["temp", "wind", "is_dome"],
    "rest": ["rest_days"],
}
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def cv_pooled(df, features):
    all_probs, all_correct = [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train["over_proxy_line"]
        X_test, y_test = test[features].fillna(0), test["over_proxy_line"].values
        model = LogisticRegression(max_iter=3000)
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
    df = build_passing_yards_dataset(min_week=4)
    df = df.dropna(subset=["passing_yards_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(context, on=["recent_team", "season", "week"], how="left")
    df = df.merge(schedules[["season", "week", "home_team", "spread_line"]].rename(
        columns={"home_team": "recent_team"}), on=["recent_team", "season", "week"], how="left")
    df["spread_line"] = df["spread_line"].fillna(0)

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
