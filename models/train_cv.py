"""
Leave-one-season-out cross-validation across the last 5 seasons (2020-2024), to check
whether the 70.4% result from a single 2024 holdout was real signal or a lucky split.

For each of the 5 seasons: train on all other available seasons (2019-2024 minus that
one), test on the held-out season. This directly answers "does this model generalize,
or did it just get lucky on one particular year."
"""
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from feature_engineering import build_game_features

ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")

STATS_FEATURES = [
    "diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling",
    "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling",
    "diff_turnover_margin_rolling", "rest_diff", "temp", "wind", "is_dome",
]
FULL_FEATURES = STATS_FEATURES + ["elo_home_win_prob"]

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_game_features(min_week=3)
    elo = pd.read_csv(ELO_RESULTS_PATH)[
        ["season", "week", "home_team", "away_team", "model_home_win_prob", "correct"]
    ].rename(columns={"model_home_win_prob": "elo_home_win_prob", "correct": "elo_correct"})
    df = df.merge(elo, on=["season", "week", "home_team", "away_team"], how="left")

    print(f"{'Holdout':<10} {'Elo alone':>10} {'Stats only':>11} {'Elo+Stats':>10}")
    print("-" * 45)

    elo_accs, stats_accs, combo_accs = [], [], []
    for holdout in HOLDOUT_SEASONS:
        train = df[df["season"] != holdout]
        test = df[df["season"] == holdout]

        elo_acc = test["elo_correct"].mean()
        elo_accs.append(elo_acc)

        for features, acc_list in [(STATS_FEATURES, stats_accs), (FULL_FEATURES, combo_accs)]:
            X_train, y_train = train[features].fillna(0), train["home_win"]
            X_test, y_test = test[features].fillna(0), test["home_win"]
            model = LogisticRegression(max_iter=3000)
            model.fit(X_train, y_train)
            preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
            acc_list.append(accuracy_score(y_test, preds))

        print(f"{holdout:<10} {elo_acc*100:>9.1f}% {stats_accs[-1]*100:>10.1f}% {combo_accs[-1]*100:>9.1f}%")

    print("-" * 45)
    print(f"{'Mean':<10} {sum(elo_accs)/5*100:>9.1f}% {sum(stats_accs)/5*100:>10.1f}% {sum(combo_accs)/5*100:>9.1f}%")
    print(f"{'Std dev':<10} {pd.Series(elo_accs).std()*100:>9.1f}  {pd.Series(stats_accs).std()*100:>10.1f}  {pd.Series(combo_accs).std()*100:>9.1f}")


if __name__ == "__main__":
    main()
