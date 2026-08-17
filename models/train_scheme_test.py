"""
Mix-and-match test: adds scheme/tendency features (pass rate over expected, tempo,
box counts) to the current best model, one at a time and combined, to see which
actually help. Same rigorous 5-season CV as every other test.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_engineering import build_game_features
from scheme_features import build_team_week_scheme

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ELO_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "elo_backtest_results.csv")

BASE_FEATURES = [
    "elo_home_win_prob", "diff_off_epa_total_rolling", "diff_def_epa_allowed_rolling",
    "diff_injury_count_rolling", "diff_cpoe_rolling", "diff_avg_separation_rolling",
    "diff_turnover_margin_rolling", "rest_diff", "temp", "wind", "is_dome",
]

SCHEME_FEATURE_GROUPS = {
    "pass_oe": ["diff_pass_oe_rolling"],
    "tempo (shotgun+no_huddle)": ["diff_shotgun_rate_rolling", "diff_no_huddle_rate_rolling"],
    "early_down_pass_rate": ["diff_early_down_pass_rate_rolling"],
    "defenders_in_box": ["diff_avg_defenders_in_box_rolling"],
}

HOLDOUT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def main():
    df = build_game_features(min_week=3)
    elo = pd.read_csv(ELO_RESULTS_PATH)[
        ["season", "week", "home_team", "away_team", "model_home_win_prob"]
    ].rename(columns={"model_home_win_prob": "elo_home_win_prob"})
    df = df.merge(elo, on=["season", "week", "home_team", "away_team"], how="left")

    pbp = pd.read_csv(os.path.join(RAW_DIR, "pbp.csv"), low_memory=False)
    scheme = build_team_week_scheme(pbp)
    rolling_cols = [c for c in scheme.columns if c.endswith("_rolling")]
    scheme_pregame = scheme[["team", "season", "week"] + rolling_cols]

    df = df.merge(
        scheme_pregame.rename(columns={c: f"home_{c}" for c in rolling_cols} | {"team": "home_team"}),
        on=["home_team", "season", "week"], how="left",
    )
    df = df.merge(
        scheme_pregame.rename(columns={c: f"away_{c}" for c in rolling_cols} | {"team": "away_team"}),
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
            model = LogisticRegression(max_iter=3000)
            model.fit(X_train, y_train)
            preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
            accs.append(accuracy_score(y_test, preds))
        return accs

    base_accs = cv_accuracy(BASE_FEATURES)
    print(f"Base model (current best): {np.mean(base_accs)*100:.1f}% mean\n")

    print(f"{'Addition':<30} {'Mean acc':>10} {'vs base':>10}")
    print("-" * 52)
    individual_results = {}
    for name, feats in SCHEME_FEATURE_GROUPS.items():
        accs = cv_accuracy(BASE_FEATURES + feats)
        mean_acc = np.mean(accs)
        individual_results[name] = mean_acc
        delta = (mean_acc - np.mean(base_accs)) * 100
        print(f"{name:<30} {mean_acc*100:>9.1f}% {delta:>+9.1f}pt")

    # Combine only the additions that individually helped (beat base)
    winners = [name for name, acc in individual_results.items() if acc > np.mean(base_accs)]
    print(f"\nFeatures that individually beat base: {winners if winners else 'none'}")

    if winners:
        combined_feats = BASE_FEATURES + [f for name in winners for f in SCHEME_FEATURE_GROUPS[name]]
        combined_accs = cv_accuracy(combined_feats)
        print(f"\nCombined (winners only): {np.mean(combined_accs)*100:.1f}% mean")
        for s, a in zip(HOLDOUT_SEASONS, combined_accs):
            print(f"  {s}: {a*100:.1f}%")

    # Also try ALL scheme features combined, regardless of individual performance
    all_scheme_feats = [f for feats in SCHEME_FEATURE_GROUPS.values() for f in feats]
    all_accs = cv_accuracy(BASE_FEATURES + all_scheme_feats)
    print(f"\nAll scheme features combined: {np.mean(all_accs)*100:.1f}% mean")


if __name__ == "__main__":
    main()
