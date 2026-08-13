"""
Baseline Elo-style power rating model for NFL teams.

This is intentionally simple — the Betting Strategy section of the README calls this
the starting point before any ML model, specifically so we have something to backtest
against and compare later, more complex models to.

Core idea: every team has a rating. After each game, ratings update based on the gap
between expected and actual outcome, scaled by margin of victory. Home field advantage
is a fixed bump applied to the home team's effective rating for the win-probability calc.
"""
import math

DEFAULT_RATING = 1500.0
HOME_FIELD_ADV = 55.0   # rating points, roughly standard for NFL Elo implementations
K_FACTOR = 20.0         # how much one game moves a team's rating
MOV_MULTIPLIER_BASE = 2.2
SEASON_REGRESSION = 0.33  # fraction each team's rating regresses toward the mean between seasons


def win_probability(rating_a: float, rating_b: float) -> float:
    """Probability team A beats team B, given their ratings (logistic curve)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_of_victory_multiplier(point_diff: float, rating_diff: float) -> float:
    """Blowouts move ratings more than close games, but a big favorite winning big
    moves ratings less than a big underdog winning big (this is the standard 538/FiveThirtyEight
    NFL Elo margin-of-victory adjustment)."""
    return math.log(max(abs(point_diff), 1) + 1) * (MOV_MULTIPLIER_BASE / ((rating_diff * 0.001) + MOV_MULTIPLIER_BASE))


class EloRatings:
    def __init__(self):
        self.ratings = {}  # team -> current rating

    def get(self, team: str) -> float:
        return self.ratings.get(team, DEFAULT_RATING)

    def regress_to_mean(self):
        """Called between seasons — teams regress partway back toward average."""
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - SEASON_REGRESSION) + DEFAULT_RATING * SEASON_REGRESSION
            )

    def update(self, home_team: str, away_team: str, home_score: float, away_score: float):
        """Updates both teams' ratings after one game's result."""
        home_rating = self.get(home_team)
        away_rating = self.get(away_team)

        home_effective = home_rating + HOME_FIELD_ADV
        expected_home_win_prob = win_probability(home_effective, away_rating)

        actual_home_result = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)

        point_diff = home_score - away_score
        rating_diff = home_effective - away_rating
        mov_mult = margin_of_victory_multiplier(point_diff, rating_diff)

        shift = K_FACTOR * mov_mult * (actual_home_result - expected_home_win_prob)

        self.ratings[home_team] = home_rating + shift
        self.ratings[away_team] = away_rating - shift

    def predict(self, home_team: str, away_team: str) -> dict:
        """Returns model's pre-game prediction — win prob and an implied point spread."""
        home_rating = self.get(home_team)
        away_rating = self.get(away_team)
        home_effective = home_rating + HOME_FIELD_ADV

        home_win_prob = win_probability(home_effective, away_rating)
        # Rough, commonly-used conversion from Elo rating gap to point spread (25 Elo points ≈ 1 point of spread)
        implied_spread = (away_rating - home_effective) / 25.0

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": 1 - home_win_prob,
            "implied_home_spread": implied_spread,  # negative = home favored, matches Vegas convention
        }
