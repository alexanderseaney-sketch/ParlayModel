"""
Walk-forward backtest of the baseline Elo model against real historical games.

Critical detail: this predicts each game using ONLY ratings built from games that
happened BEFORE it, then updates ratings using that game's result before moving to the
next one. This avoids the classic backtesting mistake of leaking future information
(e.g. using a team's end-of-season rating to "predict" an early-season game).

Usage:
    python backtesting/backtest_elo.py
"""
import os

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.elo_baseline import EloRatings

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def run_backtest(schedules: pd.DataFrame) -> pd.DataFrame:
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    schedules = schedules.dropna(subset=["home_score", "away_score"])  # only played games
    schedules = schedules.sort_values(["season", "week"]).reset_index(drop=True)

    elo = EloRatings()
    results = []
    current_season = None

    for _, game in schedules.iterrows():
        if current_season is not None and game["season"] != current_season:
            elo.regress_to_mean()
        current_season = game["season"]

        # Predict BEFORE updating ratings with this game's result
        pred = elo.predict(game["home_team"], game["away_team"])

        actual_home_won = game["home_score"] > game["away_score"]
        predicted_home_win = pred["home_win_prob"] > 0.5
        correct = predicted_home_win == actual_home_won

        vegas_spread = game.get("spread_line")
        spread_error = None
        if pd.notna(vegas_spread):
            spread_error = abs(pred["implied_home_spread"] - (-vegas_spread))
            # Verified against real data: nflverse spread_line is positive when home is
            # favored; model's implied_home_spread is negative when home is favored.
            # Negating vegas_spread puts both on the same scale before comparing.

        results.append({
            "season": game["season"], "week": game["week"],
            "home_team": game["home_team"], "away_team": game["away_team"],
            "home_score": game["home_score"], "away_score": game["away_score"],
            "model_home_win_prob": pred["home_win_prob"],
            "model_implied_spread": pred["implied_home_spread"],
            "vegas_spread": vegas_spread,
            "spread_error": spread_error,
            "predicted_home_win": predicted_home_win,
            "actual_home_win": actual_home_won,
            "correct": correct,
        })

        elo.update(game["home_team"], game["away_team"], game["home_score"], game["away_score"])

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> None:
    total = len(results)
    accuracy = results["correct"].mean()
    print(f"Games backtested: {total}")
    print(f"Straight-up accuracy (picking the win-prob favorite): {accuracy * 100:.1f}%")

    home_win_rate = results["actual_home_win"].mean()
    print(f"(Baseline: home teams won {home_win_rate * 100:.1f}% of these games — "
          f"model should beat this by a meaningful margin to be worth anything)")

    valid_spread = results.dropna(subset=["spread_error"])
    if len(valid_spread):
        mae = valid_spread["spread_error"].mean()
        print(f"\nMean absolute error vs. Vegas closing spread: {mae:.2f} points "
              f"(on {len(valid_spread)} games with a spread available)")
        print("This measures whether the model's implied line tracks the market at all —")
        print("it is NOT an edge-finding metric. A model that matches Vegas closely here")
        print("is 'sane', not 'profitable' — real edge only shows up as small, real")
        print("disagreements with the market, tracked via CLV once betting starts.")

    # Per-season breakdown, since early seasons include cold-start noise
    print("\nBy season:")
    for season, group in results.groupby("season"):
        print(f"  {season}: {group['correct'].mean() * 100:.1f}% accuracy over {len(group)} games")


if __name__ == "__main__":
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    results = run_backtest(schedules)

    out_path = os.path.join(os.path.dirname(__file__), "elo_backtest_results.csv")
    results.to_csv(out_path, index=False)
    print(f"Full results saved -> {out_path}\n")

    summarize(results)
