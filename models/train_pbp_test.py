"""
Tests whether real play-by-play efficiency features (EPA/play, success rate, garbage-
time filtered) beat the volume-based weekly_stats EPA totals used in the current best
model. Same rigorous 5-season leave-one-out CV as every other test in this project.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_engineering import build_game_features
from pbp_features import build_pbp_team_week_features

ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")

CURRENT_BEST_FEATURES = [
    "elo_home_win_prob", "diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling",
    "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling",
    "diff_turnover_margin_rolling", "rest_diff", "temp", "wind", "is_dome",
]

PBP_FEATURES_NEW = [
    "diff_off_epa_per_play_rolling", "diff_off_success_rate_rolling",
    "diff_off_pass_epa_per_play_rolling", "diff_off_rush_epa_per_play_rolling",
    "diff_off_explosive_rate_rolling",
    "diff_def_epa_per_play_allowed_rolling", "diff_def_success_rate_allowed_rolling",
]
PBP_REPLACEMENT_FEATURES = [
    f for f in CURRENT_BEST_FEATURES if f not in ("diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling")
] + PBP_FEATURES_NEW

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_game_features(min_week=3)
    elo = pd.read_csv(ELO_RESULTS_PATH)[
        ["season", "week", "home_team", "away_team", "model_home_win_prob"]
    ].rename(columns={"model_home_win_prob": "elo_home_win_prob"})
    df = df.merge(elo, on=["season", "week", "home_team", "away_team"], how="left")

    pbp_tw = build_pbp_team_week_features()
    rolling_cols = [c for c in pbp_tw.columns if c.endswith("_rolling")]
    pbp_pregame = pbp_tw[["team", "season", "week"] + rolling_cols]

    df = df.merge(
        pbp_pregame.rename(columns={c: f"home_{c}" for c in rolling_cols} | {"team": "home_team"}),
        on=["home_team", "season", "week"], how="left",
    )
    df = df.merge(
        pbp_pregame.rename(columns={c: f"away_{c}" for c in rolling_cols} | {"team": "away_team"}),
        on=["away_team", "season", "week"], how="left",
    )
    for c in rolling_cols:
        df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]

    def cv_accuracy(features):
        accs = []
        for holdout in HOLDOUT_SEASONS:
            train = df[df["season"] != holdout]
            test = df[df["season"] == holdout]
            X_train, y_train = train[features].fillna(0), train["home_win"]
            X_test, y_test = test[features].fillna(0), test["home_win"]
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
            accs.append(accuracy_score(y_test, preds))
        return accs

    current_accs = cv_accuracy(CURRENT_BEST_FEATURES)
    pbp_accs = cv_accuracy(PBP_REPLACEMENT_FEATURES)
    combined_accs = cv_accuracy(CURRENT_BEST_FEATURES + PBP_FEATURES_NEW)

    print(f"{'Season':<10} {'Current best':>14} {'PBP replaces EPA':>18} {'Current + PBP':>16}")
    print("-" * 60)
    for s, a, b, c in zip(HOLDOUT_SEASONS, current_accs, pbp_accs, combined_accs):
        print(f"{s:<10} {a*100:>13.1f}% {b*100:>17.1f}% {c*100:>15.1f}%")
    print("-" * 60)
    print(f"{'Mean':<10} {np.mean(current_accs)*100:>13.1f}% {np.mean(pbp_accs)*100:>17.1f}% {np.mean(combined_accs)*100:>15.1f}%")


if __name__ == "__main__":
    main()
