"""
Round 2 of iteration: does combining Elo's rating with EPA/injury/NGS features beat
Elo alone? And does a nonlinear model (XGBoost) find interactions logistic regression
can't? Same train/test split as round 1 (2019-2023 train, 2024 held-out test) for a
fair comparison to those results.
"""
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from feature_engineering import build_game_features

ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")


def main():
    df = build_game_features(min_week=3)
    elo = pd.read_csv(ELO_RESULTS_PATH)[
        ["season", "week", "home_team", "away_team", "model_home_win_prob"]
    ].rename(columns={"model_home_win_prob": "elo_home_win_prob"})

    df = df.merge(elo, on=["season", "week", "home_team", "away_team"], how="left")

    base_features = [
        "diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling",
        "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling",
    ]
    with_elo = base_features + ["elo_home_win_prob"]

    train = df[df["season"] <= 2023].copy()
    test = df[df["season"] == 2024].copy()

    elo_only_acc = accuracy_score(test["home_win"], (test["elo_home_win_prob"] > 0.5).astype(int))
    print(f"Elo alone on this exact test set: {elo_only_acc*100:.1f}%\n")

    configs = [
        ("v4_logreg_epa+elo", LogisticRegression(max_iter=3000), with_elo),
        ("v5_xgboost_epa_only", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric="logloss"), base_features),
        ("v6_xgboost_epa+elo", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric="logloss"), with_elo),
    ]

    print(f"{'Model':<25} {'Accuracy':>10} {'Brier':>8}")
    print("-" * 45)
    for name, model, features in configs:
        X_train, y_train = train[features].fillna(0), train["home_win"]
        X_test, y_test = test[features].fillna(0), test["home_win"]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        brier = brier_score_loss(y_test, probs)
        print(f"{name:<25} {acc*100:>9.1f}% {brier:>8.3f}")

        if hasattr(model, "feature_importances_"):
            print("  feature importances:", dict(zip(features, [round(x, 3) for x in model.feature_importances_])))


if __name__ == "__main__":
    main()
