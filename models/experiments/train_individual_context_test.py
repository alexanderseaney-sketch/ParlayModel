"""
Tests individual-level signals (own injury status, divisional game, primetime, usage
trend) against all four prop models — using the current best model type per prop
(XGBoost for receiving/rushing/receptions, LogReg for passing).
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from individual_context_features import build_player_injury_status, build_game_flags, add_usage_trend

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def cv_pooled(df, features, model_fn, target_col="over_proxy_line"):
    all_probs, all_correct = [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train[target_col]
        X_test, y_test = test[features].fillna(0), test[target_col].values
        model = model_fn()
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
    print(f"{name:<30} overall: {correct.mean()*100:>5.1f}%   @0.4 conf: {conf_acc*100:>5.1f}% ({mask.mean()*100:>4.1f}% of games)")


def xgb():
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss")


def logreg():
    return LogisticRegression(max_iter=3000)


def test_prop(build_fn, base_features, model_fn, position_filter, dataset_name):
    print(f"=== {dataset_name} ===")
    df = build_fn(min_week=4)
    if position_filter:
        df = df[df["position"].isin(position_filter)].copy()
    level_col = [c for c in base_features if c.endswith("_rolling") and "yards" in c.lower()]
    df = df.dropna(subset=[base_features[0]])

    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    inj_status = build_player_injury_status(injuries)
    flags = build_game_flags(schedules).rename(columns={"team": "recent_team"})

    df = df.merge(inj_status, on=["player_id", "season", "week"], how="left")
    df = df.merge(flags, on=["recent_team", "season", "week"], how="left")

    base_probs, base_correct = cv_pooled(df, base_features, model_fn)
    summarize("base", base_probs, base_correct)

    new_groups = {
        "own_injury": ["own_injury_severity"],
        "div_game": ["div_game"],
        "primetime": ["is_primetime"],
    }
    winners = []
    for name, feats in new_groups.items():
        probs, correct = cv_pooled(df, base_features + feats, model_fn)
        summarize(f"+ {name}", probs, correct)
        if correct.mean() > base_correct.mean():
            winners.append(name)

    if winners:
        combo = base_features + [f for n in winners for f in new_groups[n]]
        probs, correct = cv_pooled(df, combo, model_fn)
        summarize("+ combined winners", probs, correct)
    print()


if __name__ == "__main__":
    from player_prop_features import build_receiving_yards_dataset
    from player_prop_rushing_features import build_rushing_yards_dataset
    from player_prop_passing_features import build_passing_yards_dataset
    from player_prop_receptions_features import build_receptions_dataset

    test_prop(build_receiving_yards_dataset,
              ["receiving_yards_rolling", "receiving_yards_last3", "targets_rolling", "targets_last3",
               "target_share_rolling", "receiving_air_yards_rolling", "avg_separation_rolling",
               "avg_cushion_rolling", "avg_yac_above_expectation_rolling", "def_epa_allowed_rolling"],
              xgb, ["WR", "TE"], "RECEIVING YARDS")

    test_prop(build_rushing_yards_dataset,
              ["rushing_yards_rolling", "rushing_yards_last3", "carries_rolling", "carries_last3",
               "targets_rolling", "receiving_yards_rolling", "rush_yards_over_expected_per_att_rolling",
               "percent_attempts_gte_eight_defenders_rolling", "efficiency_rolling", "def_epa_allowed_rolling"],
              xgb, None, "RUSHING YARDS")

    test_prop(build_passing_yards_dataset,
              ["passing_yards_rolling", "passing_yards_last3", "attempts_rolling", "attempts_last3",
               "passing_tds_rolling", "interceptions_rolling",
               "completion_percentage_above_expectation_rolling", "avg_intended_air_yards_rolling",
               "aggressiveness_rolling", "avg_time_to_throw_rolling", "def_epa_allowed_rolling"],
              logreg, None, "PASSING YARDS")

    test_prop(build_receptions_dataset,
              ["receptions_rolling", "receptions_last3", "targets_rolling", "targets_last3",
               "target_share_rolling", "receiving_air_yards_rolling", "catch_rate_rolling",
               "avg_separation_rolling", "avg_cushion_rolling", "catch_percentage_rolling",
               "def_epa_allowed_rolling"],
              xgb, ["WR", "TE"], "RECEPTIONS")
