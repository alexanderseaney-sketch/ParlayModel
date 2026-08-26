"""
Tests real matchup-specific defensive splits (pass vs run defense, defense-vs-position,
sack rate, run-stuff rate) against all three prop models, using the current best/updated
feature sets from the prior round as the base. Same pooled 5-season CV.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matchup_features import build_all_matchup_features
from game_context_features import build_game_context, add_snap_share

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def cv_pooled(df, features, target_col="over_proxy_line"):
    all_probs, all_correct = [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]
        X_train, y_train = train[features].fillna(0), train[target_col]
        X_test, y_test = test[features].fillna(0), test[target_col].values
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
    print(f"{name:<30} overall: {correct.mean()*100:>5.1f}%   @0.4 conf: {conf_acc*100:>5.1f}% ({mask.mean()*100:>4.1f}% of games)")


def test_receiving(pbp, weekly_stats):
    from player_prop_features import build_receiving_yards_dataset

    print("=== RECEIVING ===")
    df = build_receiving_yards_dataset(min_week=4)
    df = df[df["position"].isin(["WR", "TE"])].copy()
    df = df.dropna(subset=["receiving_yards_rolling"])

    matchup = build_all_matchup_features(pbp, weekly_stats)
    rolling_cols = [c for c in matchup.columns if c.endswith("_rolling")]
    df = df.merge(matchup[["team", "season", "week"] + rolling_cols].rename(columns={"team": "opponent"}),
                   on=["opponent", "season", "week"], how="left")

    base = ["receiving_yards_rolling", "receiving_yards_last3", "targets_rolling", "targets_last3",
            "target_share_rolling", "receiving_air_yards_rolling", "avg_separation_rolling",
            "avg_cushion_rolling", "avg_yac_above_expectation_rolling", "def_epa_allowed_rolling"]

    base_probs, base_correct = cv_pooled(df, base)
    summarize("base", base_probs, base_correct)

    # Most relevant: defense's pass defense EPA, and defense-vs-WR/TE specifically
    relevant = ["def_pass_epa_allowed_rolling", "def_pass_success_rate_allowed_rolling",
                "def_epa_allowed_vs_WR_rolling", "def_epa_allowed_vs_TE_rolling"]
    for feat in relevant:
        probs, correct = cv_pooled(df, base + [feat])
        summarize(f"+ {feat}", probs, correct)

    probs, correct = cv_pooled(df, base + relevant)
    summarize("+ ALL matchup features", probs, correct)


def test_rushing(pbp, weekly_stats):
    from player_prop_rushing_features import build_rushing_yards_dataset

    print("\n=== RUSHING ===")
    df = build_rushing_yards_dataset(min_week=4)
    df = df.dropna(subset=["rushing_yards_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(context, on=["recent_team", "season", "week"], how="left")
    df = add_snap_share(df, id_col="player_id")

    matchup = build_all_matchup_features(pbp, weekly_stats)
    rolling_cols = [c for c in matchup.columns if c.endswith("_rolling")]
    df = df.merge(matchup[["team", "season", "week"] + rolling_cols].rename(columns={"team": "opponent"}),
                   on=["opponent", "season", "week"], how="left")

    base = ["rushing_yards_rolling", "rushing_yards_last3", "carries_rolling", "carries_last3",
            "targets_rolling", "receiving_yards_rolling", "rush_yards_over_expected_per_att_rolling",
            "percent_attempts_gte_eight_defenders_rolling", "efficiency_rolling", "def_epa_allowed_rolling",
            "team_implied_total", "temp", "wind", "is_dome", "offense_pct_rolling"]

    base_probs, base_correct = cv_pooled(df, base)
    summarize("base (updated)", base_probs, base_correct)

    relevant = ["def_rush_epa_allowed_rolling", "def_rush_success_rate_allowed_rolling",
                "def_rush_stuff_rate_rolling", "def_epa_allowed_vs_RB_rolling"]
    for feat in relevant:
        probs, correct = cv_pooled(df, base + [feat])
        summarize(f"+ {feat}", probs, correct)

    probs, correct = cv_pooled(df, base + relevant)
    summarize("+ ALL matchup features", probs, correct)


def test_passing(pbp, weekly_stats):
    from player_prop_passing_features import build_passing_yards_dataset

    print("\n=== PASSING ===")
    df = build_passing_yards_dataset(min_week=4)
    df = df.dropna(subset=["passing_yards_rolling"])

    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    context = build_game_context(schedules).rename(columns={"team": "recent_team"})
    df = df.merge(context, on=["recent_team", "season", "week"], how="left")

    matchup = build_all_matchup_features(pbp, weekly_stats)
    rolling_cols = [c for c in matchup.columns if c.endswith("_rolling")]
    df = df.merge(matchup[["team", "season", "week"] + rolling_cols].rename(columns={"team": "opponent"}),
                   on=["opponent", "season", "week"], how="left")

    base = ["passing_yards_rolling", "passing_yards_last3", "attempts_rolling", "attempts_last3",
            "passing_tds_rolling", "interceptions_rolling",
            "completion_percentage_above_expectation_rolling", "avg_intended_air_yards_rolling",
            "aggressiveness_rolling", "avg_time_to_throw_rolling", "def_epa_allowed_rolling",
            "temp", "wind", "is_dome"]

    base_probs, base_correct = cv_pooled(df, base)
    summarize("base (updated)", base_probs, base_correct)

    relevant = ["def_pass_epa_allowed_rolling", "def_pass_success_rate_allowed_rolling", "def_sack_rate_rolling"]
    for feat in relevant:
        probs, correct = cv_pooled(df, base + [feat])
        summarize(f"+ {feat}", probs, correct)

    probs, correct = cv_pooled(df, base + relevant)
    summarize("+ ALL matchup features", probs, correct)


def main():
    from pbp_features import load_pbp
    pbp = load_pbp()
    weekly_stats = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)

    test_receiving(pbp, weekly_stats)
    test_rushing(pbp, weekly_stats)
    test_passing(pbp, weekly_stats)


if __name__ == "__main__":
    main()
