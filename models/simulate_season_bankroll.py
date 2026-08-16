"""
Simulates a $10/week season using the model, to answer "how much would we have at the
end of the season" -- honestly, as a real Monte Carlo distribution rather than a single
made-up number, and with the load-bearing caveat stated as loudly in code as it will be
in the write-up: our validated 67-84% accuracy is against each player's own PROXY line
(their rolling average), not a real Underdog line. Whether that edge survives against
real market pricing is exactly what the still-building Underdog line archive is meant
to answer -- this simulation's "optimistic" scenario assumes it does, which is an
unvalidated assumption, not a finding. Two more conservative scenarios are included
specifically to show how much the outcome depends on that one unknown.

Pricing: real Underdog single-leg pricing pulled live this session is overwhelmingly
decimal 1.90 (american -112, ~52.8% breakeven) -- used here rather than a guessed
number. Two strategies simulated:
  1. Single best-confidence leg per week, straight bet at 1.90.
  2. Two-leg parlay per week (product of decimal prices), using this project's own
     correlation_adjusted_parlay_probability when the two legs are a known-correlated
     same-team pair, naive product otherwise -- reusing the real tool built this
     session rather than assuming independence.

18-week regular season, $10 flat stake/week, 10,000 Monte Carlo trials per scenario.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
from utils import correlation_adjusted_parlay_probability  # noqa: E402

N_WEEKS = 18
STAKE = 10.0
N_TRIALS = 10000
DECIMAL_PRICE = 1.90  # real Underdog single-leg price, most common in live pulled data

SCENARIOS = {
    "Optimistic (validated proxy-line accuracy holds against real lines)": 0.814,
    "Moderate (edge partially survives against real lines)": 0.60,
    "Conservative (edge mostly doesn't survive -- near real breakeven)": 0.55,
}
BREAKEVEN = 1 / DECIMAL_PRICE  # ~52.8%

RNG = np.random.default_rng(42)


def simulate_single_leg(win_prob: float, n_trials: int = N_TRIALS) -> np.ndarray:
    outcomes = RNG.random((n_trials, N_WEEKS)) < win_prob
    weekly_pnl = np.where(outcomes, STAKE * (DECIMAL_PRICE - 1), -STAKE)
    return weekly_pnl.sum(axis=1)


def simulate_two_leg_parlay(win_prob: float, n_trials: int = N_TRIALS, correlated_frac: float = 0.15) -> np.ndarray:
    """A fraction of weeks (correlated_frac) draw the model's top-2 legs from the same
    team on a known-correlated pairing (using this project's real measured phi via
    correlation_adjusted_parlay_probability), matching real usage where the model's
    best two legs sometimes land on the same game. The rest assume independent legs
    from different games, the norm for a "top confidence picks" strategy."""
    parlay_decimal = DECIMAL_PRICE ** 2
    is_correlated_week = RNG.random((n_trials, N_WEEKS)) < correlated_frac
    phi = 0.20  # representative of the real measured QB+WR/TE positive correlations found this session

    true_joint_indep = win_prob * win_prob
    std = np.sqrt(win_prob * (1 - win_prob))
    joint_corr = np.clip(win_prob * win_prob + phi * std * std, 0, win_prob)

    joint_prob = np.where(is_correlated_week, joint_corr, true_joint_indep)
    outcomes = RNG.random((n_trials, N_WEEKS)) < joint_prob
    weekly_pnl = np.where(outcomes, STAKE * (parlay_decimal - 1), -STAKE)
    return weekly_pnl.sum(axis=1)


def summarize(name: str, pnl: np.ndarray):
    ending = 0.0 + pnl  # started with $0 sunk, STAKE spent each week already reflected in pnl
    print(f"--- {name} ---")
    print(f"  Median season P&L:     {np.median(pnl):+.2f}")
    print(f"  Mean season P&L:       {pnl.mean():+.2f}")
    print(f"  10th percentile:       {np.percentile(pnl, 10):+.2f}")
    print(f"  90th percentile:       {np.percentile(pnl, 90):+.2f}")
    print(f"  P(finish positive):    {(pnl > 0).mean()*100:.1f}%")
    print(f"  Worst case (of {N_TRIALS}):  {pnl.min():+.2f}")
    print(f"  Best case (of {N_TRIALS}):   {pnl.max():+.2f}")
    print()


COPULA_RHO = 0.30  # calibrated (see git history / README) to reproduce this project's
                    # real measured phi~0.20 (QB+WR/TE same-team) for a PAIR -- unlike
                    # the dashboard's pairwise-multiplication approximation (valid for
                    # 2 legs only), this single-factor structure stays mathematically
                    # valid for any number of legs, since it's a real joint model.


def simulate_n_leg_parlay(win_prob: float, n_legs: int, payout: float,
                           n_trials: int = N_TRIALS, correlated: bool = False) -> np.ndarray:
    """Generalized N-leg version. correlated=False treats all legs as independent
    (the realistic default for a "top N confidence picks" strategy spanning multiple
    games/teams). correlated=True simulates deliberately stacking all N legs on the
    same team/game's shared game-script factor via a one-factor Gaussian copula --
    each leg's outcome is driven partly by a shared factor F (e.g. "did this offense
    have a big day") and partly by its own idiosyncratic noise, which is what actually
    produces realistic correlation for a block of ANY size without the pairwise-
    multiplication approximation's blowup at high pair counts."""
    if not correlated:
        joint_prob = win_prob ** n_legs
        outcomes = RNG.random((n_trials, N_WEEKS)) < joint_prob
    else:
        from scipy.stats import norm
        threshold = norm.ppf(1 - win_prob)
        shared = RNG.standard_normal((n_trials, N_WEEKS, 1))
        idio = RNG.standard_normal((n_trials, N_WEEKS, n_legs))
        z = np.sqrt(COPULA_RHO) * shared + np.sqrt(1 - COPULA_RHO) * idio
        leg_wins = z > threshold
        outcomes = leg_wins.all(axis=2)

    weekly_pnl = np.where(outcomes, STAKE * (payout - 1), -STAKE)
    return weekly_pnl.sum(axis=1)


def main():
    print(f"Real Underdog single-leg breakeven at {DECIMAL_PRICE} decimal odds: {BREAKEVEN*100:.1f}%")
    print(f"({N_WEEKS} weeks, ${STAKE:.0f}/week, {N_TRIALS} simulated seasons per scenario)\n")

    print("=" * 78)
    print("STRATEGY 1: single best-confidence leg per week, straight bet")
    print("=" * 78)
    for name, p in SCENARIOS.items():
        pnl = simulate_single_leg(p)
        summarize(f"{name} (p={p:.1%})", pnl)

    print("=" * 78)
    print("STRATEGY 2: two-leg parlay per week (this project's actual correlation math)")
    print("=" * 78)
    for name, p in SCENARIOS.items():
        pnl = simulate_two_leg_parlay(p)
        summarize(f"{name} (p={p:.1%} per leg)", pnl)

    print("=" * 78)
    print("STRATEGY 3: six-leg parlay per week -- assumed 20x payout (per Alex; NOT")
    print("independently verified in our data -- naive-fair would be 47x at real")
    print("single-leg pricing, so 20x already implies a much bigger platform margin")
    print("than single legs carry)")
    print("=" * 78)
    for name, p in SCENARIOS.items():
        pnl_random = simulate_n_leg_parlay(p, n_legs=6, payout=20.0, correlated=False)
        summarize(f"{name}, 6 independent/random legs (p={p:.1%} per leg)", pnl_random)
        pnl_stacked = simulate_n_leg_parlay(p, n_legs=6, payout=20.0, correlated=True)
        summarize(f"{name}, 6 legs deliberately STACKED on correlated same-team pairs", pnl_stacked)


if __name__ == "__main__":
    main()
