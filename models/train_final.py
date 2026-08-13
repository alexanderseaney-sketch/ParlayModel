"""
Assembles the final model from everything validated so far:
- Elo rating (biggest single contributor, confirmed across all prior tests)
- EPA offense/defense, injuries, turnovers, weather, rest (small but real combined lift)
- Tests QB-specific CPOE as a replacement for team-average CPOE (new this round)
- Also checks the raw Vegas spread as a market-comparison ceiling (diagnostic only,
  NOT part of the actual production feature set — see note below)

Final production model = 100-model bootstrap ensemble (validated last round: high-
agreement predictions hit 70.7% vs 57.7% on low-agreement ones), trained on ALL 6
seasons of data, saved to disk for the dashboard to load.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_engineering import build_game_features

ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")
MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")

TEAM_AVG_FEATURES = [
    "elo_home_win_prob", "diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling",
    "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling",
    "diff_turnover_margin_rolling", "rest_diff", "temp", "wind", "is_dome",
]
QB_SPECIFIC_FEATURES = [f for f in TEAM_AVG_FEATURES if f != "diff_cpoe_rolling"] + ["diff_qb_cpoe_rolling"]

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]
N_BOOTSTRAP = 100
RANDOM_SEED = 42


def cv_accuracy(df: pd.DataFrame, features: list[str]) -> list[float]:
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


def main():
    df = build_game_features(min_week=3)
    elo = pd.read_csv(ELO_RESULTS_PATH)[
        ["season", "week", "home_team", "away_team", "model_home_win_prob"]
    ].rename(columns={"model_home_win_prob": "elo_home_win_prob"})
    df = df.merge(elo, on=["season", "week", "home_team", "away_team"], how="left")

    print("=== Round: team-average CPOE vs. QB-specific CPOE ===\n")
    team_avg_accs = cv_accuracy(df, TEAM_AVG_FEATURES)
    qb_specific_accs = cv_accuracy(df, QB_SPECIFIC_FEATURES)

    print(f"{'Season':<10} {'Team-avg CPOE':>15} {'QB-specific CPOE':>18}")
    for s, a, b in zip(HOLDOUT_SEASONS, team_avg_accs, qb_specific_accs):
        print(f"{s:<10} {a*100:>14.1f}% {b*100:>17.1f}%")
    print(f"{'Mean':<10} {np.mean(team_avg_accs)*100:>14.1f}% {np.mean(qb_specific_accs)*100:>17.1f}%")

    best_features = QB_SPECIFIC_FEATURES if np.mean(qb_specific_accs) > np.mean(team_avg_accs) else TEAM_AVG_FEATURES
    print(f"\nUsing: {'QB-specific' if best_features is QB_SPECIFIC_FEATURES else 'team-average'} CPOE for final model.\n")

    # --- Market comparison (diagnostic only) ---
    print("=== Diagnostic: how much does the market (Vegas spread) already know? ===\n")
    market_features = best_features + ["spread_line"]
    market_accs = cv_accuracy(df, market_features)
    print(f"Our best model:        {np.mean(best_features and (qb_specific_accs if best_features is QB_SPECIFIC_FEATURES else team_avg_accs))*100:.1f}% mean")
    print(f"Our model + Vegas line: {np.mean(market_accs)*100:.1f}% mean")
    print("(If this jumps a lot, it means the market has real information our stats")
    print("don't — which is expected and fine. This feature is diagnostic only, NOT")
    print("used in the production model, since the point is finding OUR edge, not")
    print("just re-deriving what the market already prices in.)\n")

    # --- Final production model: bootstrap ensemble on ALL available data ---
    print(f"=== Training final production model: {N_BOOTSTRAP}-model bootstrap ensemble on all 2019-2024 data ===\n")
    X_full = df[best_features].fillna(0).reset_index(drop=True)
    y_full = df["home_win"].reset_index(drop=True)

    rng = np.random.RandomState(RANDOM_SEED)
    n = len(X_full)
    models = []
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        m = LogisticRegression(max_iter=1000)
        m.fit(X_full.iloc[idx], y_full.iloc[idx])
        models.append(m)

    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump({"models": models, "features": best_features}, f)

    print(f"Saved {N_BOOTSTRAP}-model ensemble -> {MODEL_OUT_PATH}")
    print(f"Features used: {best_features}")
    print(f"\nFinal validated performance (from cross-validation above): "
          f"~{np.mean(qb_specific_accs if best_features is QB_SPECIFIC_FEATURES else team_avg_accs)*100:.1f}% "
          f"average accuracy across 5 held-out seasons, with high-agreement predictions "
          f"(≥90% of bootstrap models agreeing) historically hitting ~70.7% vs ~57.7% on "
          f"low-agreement games (see ensemble_agreement_results.csv from the prior round).")


if __name__ == "__main__":
    main()
