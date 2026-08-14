"""
Tests whether extra Elo regression for teams with a new head coach actually improves
real backtested accuracy — not just assumed to help. Runs the same walk-forward
backtest as backtest_elo.py, with and without the adjustment, across several regression
amounts, and checks specifically how well each version predicts games involving
new-HC teams (where the effect should show up most, if it's real).
"""
import os

import pandas as pd

from elo_baseline import EloRatings
from coaching_changes import get_new_hc_teams

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def run_backtest(schedules: pd.DataFrame, extra_regression: float = 0.0) -> pd.DataFrame:
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    schedules = schedules.dropna(subset=["home_score", "away_score"])
    schedules = schedules.sort_values(["season", "week"]).reset_index(drop=True)

    elo = EloRatings()
    results = []
    current_season = None

    for _, game in schedules.iterrows():
        if current_season is not None and game["season"] != current_season:
            new_hc_teams = get_new_hc_teams(game["season"])
            elo.regress_to_mean(new_hc_teams=new_hc_teams, extra_regression=extra_regression)
        current_season = game["season"]

        pred = elo.predict(game["home_team"], game["away_team"])
        actual_home_won = game["home_score"] > game["away_score"]
        predicted_home_win = pred["home_win_prob"] > 0.5

        new_hc_teams_this_season = get_new_hc_teams(game["season"])
        involves_new_hc = game["home_team"] in new_hc_teams_this_season or game["away_team"] in new_hc_teams_this_season

        results.append({
            "season": game["season"], "week": game["week"],
            "correct": predicted_home_win == actual_home_won,
            "involves_new_hc": involves_new_hc,
        })

        elo.update(game["home_team"], game["away_team"], game["home_score"], game["away_score"])

    return pd.DataFrame(results)


def main():
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    print(f"{'Extra regression':<18} {'Overall acc':>12} {'New-HC games acc':>18} {'n new-HC games':>15}")
    print("-" * 65)
    for extra in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        results = run_backtest(schedules, extra_regression=extra)
        overall = results["correct"].mean()
        new_hc_subset = results[results["involves_new_hc"]]
        new_hc_acc = new_hc_subset["correct"].mean()
        print(f"{extra:<18} {overall*100:>11.1f}% {new_hc_acc*100:>17.1f}% {len(new_hc_subset):>15}")


if __name__ == "__main__":
    main()
