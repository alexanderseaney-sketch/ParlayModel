"""
Generates a concrete, ranked weekly bet suggestion list against a fixed budget --
Alex places every bet themselves; this produces WHAT to consider and HOW MUCH to
stake, using real Kelly-fraction edge ranking against REAL live Underdog prices (not
assumed odds).

CRITICAL CAVEAT, stated here and repeated in every output: model confidence is
validated against each player's own PROXY line (rolling average), not a real
historical Underdog line -- see README, this is still the single biggest open
question in the whole project. Real accuracy against actual market prices is
UNVALIDATED.

Kelly criterion: for a bet with true win probability p and decimal odds d, the
bankroll-growth-optimal fraction is f* = p - (1-p)/(d-1), positive only when p*d > 1
(genuinely +EV at that REAL price -- a high model confidence alone doesn't guarantee
this, since the real price may already reflect similar information). Using this
project's own correlation_adjusted_parlay_probability for 2-leg same-team
combinations rather than assuming independence.

The weekly budget here is a small, fixed dollar amount ($10 by default), not a
bankroll to take a Kelly percentage OF -- so f* is used for what it's actually good
for in this context: RANKING opportunities by real edge strength and weighting how
the fixed budget splits across them, not as a literal fraction-of-bankroll dollar
formula (that scale mismatch would produce cents-sized "correct" Kelly stakes against
a $10 budget, which isn't what a fixed weekly amount is for). Prudence against
uncertain edge estimates is applied differently here: a minimum Kelly-edge bar
(MIN_KELLY_EDGE) an opportunity must clear before it's even considered, and spreading
the budget across up to 3 opportunities rather than concentrating it on one.

Usage:
    python models/generate_weekly_bet_slip.py --budget 10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console default codepage mangles em-dashes otherwise

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
from utils import normalize_name, load_leg_correlations, line_matches_proxy, pretty_stat_name  # noqa: E402

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), "current_player_predictions.csv")

MIN_CONFIDENCE = 0.4    # matches this project's validated confidence threshold
MIN_KELLY_EDGE = 0.02   # the real prudence lever: raw Kelly fraction must clear 2%
                         # before an opportunity is even considered -- quarter-Kelly-
                         # style caution isn't applied as a literal dollar multiplier
                         # here (the budget is a fixed $10/week, not a bankroll to take
                         # a percentage of), it's applied as "don't dilute the week's
                         # budget across marginal edges," which is the same underlying
                         # goal of not over-betting an uncertain edge estimate.
                         # line_matches_proxy() (the other correctness gate, for
                         # the proxy-line-vs-real-line bug) lives in dashboard/utils.py
                         # -- shared with the Parlay Builder rather than duplicated here.
TOP_N_OPPORTUNITIES = 3
MIN_STAKE = 0.50


def _leg_detail(row: dict) -> dict:
    """Standardized per-leg record, structurally matching what the dashboard's
    st.session_state.slip entries need -- lets the Weekly Bet Slip page push a
    suggestion directly into the Parlay Builder's slip with one click, leg by leg,
    rather than the two tools only being able to show the same data side by side."""
    return {
        "player": row["full_name"],
        "stat": row["stat_name"],
        "choice": row["my_side"],
        "line": row["stat_value"],
        "decimal_price": row["decimal_price"],
        "my_prob": row["my_prob"],
        "team": row["recent_team"],
        "position_prop": f"{row['position']} {row['prop_type']}",
    }


def kelly_fraction(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f = p - (1 - p) / b
    return max(f, 0.0)


def load_matched_props() -> pd.DataFrame:
    """One row per (player, stat_name), matched to whichever side (over/under) the
    model actually favors -- NOT hardcoded to "over". predicted_prob_over < 0.5 means
    the model favors UNDER, and must be matched against the "under" row's own price,
    not the "over" row's. Confidence alone doesn't tell you which side; it's symmetric
    around 50/50 by construction (confidence = |prob - 0.5| * 2)."""
    predictions = pd.read_csv(PREDICTIONS_PATH)
    props = pd.read_csv(os.path.join(RAW_DIR, "underdog_props.csv"), low_memory=False)

    predictions["_match_key"] = predictions["player_display_name"].apply(normalize_name)
    props["_match_key"] = props["full_name"].apply(normalize_name)

    predictions = predictions.copy()
    predictions["my_side"] = np.where(predictions["predicted_prob_over"] >= 0.5, "over", "under")
    predictions["my_prob"] = np.where(
        predictions["predicted_prob_over"] >= 0.5,
        predictions["predicted_prob_over"],
        1 - predictions["predicted_prob_over"],
    )

    merged = props.merge(
        predictions[["_match_key", "stat_name", "my_side", "my_prob", "confidence",
                     "recent_team", "position", "prop_type", "proxy_line", "next_week", "next_gameday"]],
        on=["_match_key", "stat_name"], how="inner",
    )
    merged = merged[merged["choice"].str.lower() == merged["my_side"]]
    merged = merged.dropna(subset=["decimal_price", "my_prob"])
    merged = merged[merged["decimal_price"] > 1]
    merged = merged.drop_duplicates(subset=["full_name", "stat_name"])

    # The correctness gate: my_prob only answers "beats OUR proxy_line", which is
    # only a valid stand-in for "beats Underdog's real stat_value" when the two
    # numbers are actually close. See line_matches_proxy's comment (in
    # dashboard/utils.py) for the real bug this fixes, why the threshold varies by
    # prop grain (season/period-scoped proxies are structurally noisier), and why
    # TD/INT/sack counts are gated on absolute difference instead of percentage.
    # line_divergence itself is kept only for the human-readable "X% off" bet-slip
    # description below -- it's not the actual pass/fail check for count stats.
    before = len(merged)
    merged["line_divergence"] = (merged["stat_value"] - merged["proxy_line"]).abs() / merged["proxy_line"].replace(0, np.nan)
    merged["_line_ok"] = merged.apply(
        lambda r: line_matches_proxy(r["stat_value"], r["proxy_line"], r["stat_name"]), axis=1)
    merged = merged[merged["_line_ok"]].drop(columns=["_line_ok"])
    excluded = before - len(merged)
    if excluded:
        print(f"Excluded {excluded} of {before} matched props: Underdog's real line diverges "
              f"too far from our proxy line for its prop type, so our model's probability "
              f"isn't a trustworthy stand-in for beating THAT specific number.")
    return merged


def build_single_leg_candidates(matched: pd.DataFrame) -> list[dict]:
    qualifying = matched[matched["confidence"] >= MIN_CONFIDENCE].copy()

    candidates = []
    for _, row in qualifying.iterrows():
        p = row["my_prob"]
        d = row["decimal_price"]
        f = kelly_fraction(p, d)
        if f < MIN_KELLY_EDGE:
            continue  # not enough edge at the REAL price to justify a slice of the budget
        candidates.append({
            "type": "single",
            "description": f"{row['full_name']} — {pretty_stat_name(row['stat_name'])} {row['my_side']} {row['stat_value']} "
                            f"(our proxy: {row['proxy_line']:.1f}, {row['line_divergence']*100:.0f}% off) "
                            f"[Wk {int(row['next_week'])} · {row['next_gameday']}]",
            "legs": [row["full_name"]],
            "leg_details": [_leg_detail(row)],
            "model_prob": p,
            "decimal_odds": d,
            "kelly_fraction": f,
            "team": row["recent_team"],
        })
    return candidates


def _joint_prob_with_phi(p_a: float, p_b: float, phi: float) -> float:
    """Same phi-coefficient identity as dashboard/utils.py's
    correlation_adjusted_parlay_probability, exposed here so a sign-corrected phi can
    be passed directly (see note below on over/under sign flipping)."""
    std_term = np.sqrt(max(p_a * (1 - p_a) * p_b * (1 - p_b), 0))
    joint = p_a * p_b + phi * std_term
    return min(max(joint, max(0.0, p_a + p_b - 1)), min(p_a, p_b))


def build_parlay_candidates(matched: pd.DataFrame) -> list[dict]:
    qualifying = matched[matched["confidence"] >= MIN_CONFIDENCE].copy()
    correlations = load_leg_correlations()

    candidates = []
    teams = qualifying["recent_team"].dropna().unique()
    for team in teams:
        team_legs = qualifying[qualifying["recent_team"] == team]
        if len(team_legs) < 2:
            continue
        rows = team_legs.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                key = frozenset([f"{a['position']} {a['prop_type']}", f"{b['position']} {b['prop_type']}"])
                phi = correlations.get(key)
                if phi is None:
                    continue  # only surface parlays where we have a REAL measured correlation, not a guess

                # Measured phi is for two "over" (over_proxy_line) outcomes. If a leg's
                # model-favored side is actually "under", that's the complementary
                # event -- corr(A, 1-B) = -corr(A, B), a standard identity for binary
                # variables. Flip once per leg that's on the "under" side.
                n_under = (a["my_side"] == "under") + (b["my_side"] == "under")
                effective_phi = -phi if n_under % 2 == 1 else phi

                p_joint = _joint_prob_with_phi(a["my_prob"], b["my_prob"], effective_phi)
                d_parlay = a["decimal_price"] * b["decimal_price"]
                f = kelly_fraction(p_joint, d_parlay)
                if f < MIN_KELLY_EDGE:
                    continue
                candidates.append({
                    "type": "parlay",
                    # a and b are always a same-team same-game pair (see the team loop
                    # above), so their next_week/next_gameday are identical -- only
                    # need to show it once.
                    "description": f"{a['full_name']} ({pretty_stat_name(a['stat_name'])} {a['my_side']}) + "
                                    f"{b['full_name']} ({pretty_stat_name(b['stat_name'])} {b['my_side']}) "
                                    f"[{team}, Wk {int(a['next_week'])} · {a['next_gameday']}, "
                                    f"correlation {effective_phi:+.2f}]",
                    "legs": [a["full_name"], b["full_name"]],
                    "leg_details": [_leg_detail(a), _leg_detail(b)],
                    "model_prob": p_joint,
                    "decimal_odds": d_parlay,
                    "kelly_fraction": f,
                    "team": team,
                })
    return candidates


def allocate_budget(candidates: list[dict], budget: float, top_n: int = TOP_N_OPPORTUNITIES) -> list[dict]:
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda c: c["kelly_fraction"], reverse=True)[:top_n]
    total_fraction = sum(c["kelly_fraction"] for c in ranked)
    if total_fraction <= 0:
        return []

    for c in ranked:
        c["suggested_stake"] = budget * (c["kelly_fraction"] / total_fraction)

    ranked = [c for c in ranked if c["suggested_stake"] >= MIN_STAKE]
    if not ranked:
        return []
    scale = budget / sum(c["suggested_stake"] for c in ranked)
    for c in ranked:
        c["suggested_stake"] = round(c["suggested_stake"] * scale, 2)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Generate this week's suggested bet slip")
    parser.add_argument("--budget", type=float, default=10.0, help="Total weekly budget in dollars")
    args = parser.parse_args()

    print("=" * 78)
    print("WEEKLY BET SLIP -- SUGGESTIONS ONLY. You place every bet yourself.")
    print("=" * 78)
    print("CAVEAT: model confidence is validated against each player's own rolling")
    print("average (proxy line), NOT a real historical Underdog line. Real accuracy")
    print("against actual market prices is unvalidated -- everything below only")
    print("surfaces bets that clear a real edge bar at REAL live prices, which is a")
    print("much stricter test than confidence alone. Only wager what you can afford to lose.")
    print()

    matched = load_matched_props()
    print(f"{len(matched)} live prop rows matched to a model prediction this week.\n")

    candidates = build_single_leg_candidates(matched) + build_parlay_candidates(matched)
    print(f"{len(candidates)} genuinely +EV opportunities found (model prob x real price > 1).\n")

    allocated = allocate_budget(candidates, args.budget)

    if not allocated:
        print("No +EV opportunities clear the bar this week. Suggestion: skip this week.")
        return

    print(f"Suggested split of ${args.budget:.2f}:\n")
    for c in allocated:
        edge = c["model_prob"] * c["decimal_odds"] - 1
        print(f"[{c['type'].upper():6s}] ${c['suggested_stake']:.2f}  {c['description']}")
        print(f"          model prob: {c['model_prob']*100:.1f}%   real price: {c['decimal_odds']:.2f}x   "
              f"implied edge: {edge*100:+.1f}%   raw Kelly fraction: {c['kelly_fraction']*100:.2f}% (of a full bankroll -- "
              f"used here for relative ranking, not as a literal fraction of this fixed weekly budget)")
        print()

    total = sum(c["suggested_stake"] for c in allocated)
    print(f"Total suggested: ${total:.2f} of ${args.budget:.2f} budget.")
    if total < args.budget:
        print(f"(${args.budget - total:.2f} unallocated -- not enough qualifying +EV opportunities to use the full budget this week. That's fine; forcing it isn't.)")


if __name__ == "__main__":
    main()
