"""
Generates a concrete, ranked weekly bet suggestion list against a fixed budget --
Alex places every bet themselves; this produces WHAT to consider and HOW MUCH to
stake, using real Kelly-fraction edge ranking against REAL live Underdog prices (not
assumed odds).

CRITICAL CAVEAT, stated here and repeated in every output: as of 2026-08-26, my_prob
is recomputed against Underdog's REAL posted line (not the raw proxy-based number) --
see recompute_probability_for_real_line() in dashboard/utils.py. But that
recomputation is itself a statistical approximation (assumes roughly-normal week-to-
week variance around each model's implied performance level), not something trained
or validated against real historical Underdog lines with graded outcomes -- no such
archive exists yet (see README, still the single biggest open question in the whole
project). Real accuracy against actual market prices remains UNVALIDATED; this is the
best real-time estimate available given that gap, not a proven number.

Kelly criterion: for a bet with true win probability p and decimal odds d, the
bankroll-growth-optimal fraction is f* = p - (1-p)/(d-1), positive only when p*d > 1
(genuinely +EV at that REAL price -- a high model confidence alone doesn't guarantee
this, since the real price may already reflect similar information).

Every suggestion is a placeable Underdog pick'em entry: 2-3 picks, one pick per
player, players from at least 2 teams. Entries are searched and scored by
models/parlay_calculator.py (find_best_entries / evaluate_entry / entry_validity) --
the same rules and correlation-adjusted joint probability the Parlay Builder and +EV
Finder use -- so a correlated same-team pair can still ride in an entry, as long as
a pick from another team comes with it. There are no one-pick "singles": Underdog
doesn't take them.

The weekly budget here is a small, fixed dollar amount ($10 by default), not a
bankroll to take a Kelly percentage OF -- so f* is used for what it's actually good
for in this context: RANKING opportunities by real edge strength and weighting how
the fixed budget splits across them, not as a literal fraction-of-bankroll dollar
formula (that scale mismatch would produce cents-sized "correct" Kelly stakes against
a $10 budget, which isn't what a fixed weekly amount is for). Prudence against
uncertain edge estimates is applied differently here: a minimum Kelly-edge bar
(MIN_KELLY_EDGE) an entry must clear before it's even considered, and spreading
the budget across up to 3 entries that share no player rather than concentrating it
on one.

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
from utils import (
    normalize_name, load_leg_correlations, pretty_stat_name,
    estimate_player_stat_std, recompute_probability_for_real_line, _STAT_WEEKLY_COL,
    prop_scope, is_low_noise_line,
)  # noqa: E402
from odds_utils import add_underdog_market_probs, blend_with_market  # noqa: E402
from parlay_calculator import entry_validity, find_best_entries  # noqa: E402

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
TOP_N_OPPORTUNITIES = 3
MIN_STAKE = 0.50
ENTRY_LEGS = (2, 3)      # Underdog's minimum is 2; past 3 picks every leg's error compounds
LEG_POOL_SIZE = 18       # best legs by edge, searched exhaustively (~970 combos at 2-3 picks)


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
        "team": row["team"],
        "position_prop": f"{row['position']} {row['prop_type']}",
    }


def _current_team(merged: pd.DataFrame) -> pd.Series:
    """Team per row for Underdog's 2-team rule. Underdog's own team_id is what that
    rule checks, so each id is labelled with its players' most common current
    abbreviation (32 club team sites, else weekly_stats' recent_team, which goes
    stale for anyone traded since their last game) -- equal labels <=> equal ids, so
    a stale roster row can't split one real team into two."""
    team = merged["recent_team"]
    roster_path = os.path.join(RAW_DIR, "nfl_rosters.csv")
    if os.path.exists(roster_path):
        ros = pd.read_csv(roster_path)
        ros = ros.assign(_k=ros["player"].apply(normalize_name)).sort_values(
            "roster_status", key=lambda s: ~s.fillna("").str.startswith("Active"))
        team = merged["_match_key"].map(ros.drop_duplicates("_k").set_index("_k")["team_abbr"]).fillna(team)
    if "team_id" in merged.columns and merged["team_id"].notna().any():
        label = team.groupby(merged["team_id"]).agg(
            lambda s: s.mode().iat[0] if s.notna().any() else None)
        team = merged["team_id"].map(label).where(merged["team_id"].notna(), team)
    return team


def kelly_fraction(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f = p - (1 - p) / b
    return max(f, 0.0)


def load_matched_props(props: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (player, stat_name), matched to whichever side (over/under) the
    model actually favors -- NOT hardcoded to "over". predicted_prob_over < 0.5 means
    the model favors UNDER, and must be matched against the "under" row's own price,
    not the "over" row's. Confidence alone doesn't tell you which side; it's symmetric
    around 50/50 by construction (confidence = |prob - 0.5| * 2)."""
    predictions = pd.read_csv(PREDICTIONS_PATH)
    # The dashboard passes its live (<=5 min old) Underdog board; the CLI reads the file.
    if props is None:
        props = pd.read_csv(os.path.join(RAW_DIR, "underdog_props.csv"), low_memory=False)
    props = props.copy()

    predictions["_match_key"] = predictions["player_display_name"].apply(normalize_name)
    props["_match_key"] = props["full_name"].apply(normalize_name)

    # A "weekly bet slip" is single-game props only: season-long and quarter/half
    # props are a different bet shape and shouldn't sit next to a Week-1 game leg.
    # And a sub-floor game line (rushing 3.5, receptions 1.5) is deep-backup noise
    # -- the model is trivially 99% there and the payout is a rounding error. Both
    # filters mirror the Parlay Builder's defaults.
    props = props[props["stat_name"].apply(prop_scope) == "game"]
    props = props[~props.apply(
        lambda r: is_low_noise_line(r["stat_name"], r.get("stat_value")), axis=1)]

    # Underdog labels receptions "receiving_rec"; the model's stat_name for it is
    # "receptions". Every other stat_name already lines up. Without this remap the
    # inner merge below silently drops every receptions prop (a high-volume,
    # low-variance market the model is good at) -- same fix as score_underdog_board.
    props["stat_name"] = props["stat_name"].replace({"receiving_rec": "receptions"})
    # De-vigged market probability per option (Underdog prices both sides), for the
    # model/market blend applied after the real-line recompute below.
    props = add_underdog_market_probs(props)

    predictions = predictions.copy()
    predictions["my_side"] = np.where(predictions["predicted_prob_over"] >= 0.5, "over", "under")
    predictions["my_prob"] = np.where(
        predictions["predicted_prob_over"] >= 0.5,
        predictions["predicted_prob_over"],
        1 - predictions["predicted_prob_over"],
    )

    # Both dataframes have their own "player_id" -- Underdog's internal ID (props)
    # and nflverse's gsis ID (predictions). A plain merge would silently produce
    # player_id_x/player_id_y instead of a clean column (a real KeyError caught by
    # actually running this, not just reading it) -- renamed explicitly so the
    # nflverse ID (the one weekly_stats.csv/estimate_player_stat_std actually need)
    # survives the merge under an unambiguous name.
    predictions = predictions.rename(columns={"player_id": "_nflverse_player_id"})
    merged = props.merge(
        predictions[["_match_key", "stat_name", "my_side", "my_prob", "confidence",
                     "recent_team", "position", "prop_type", "proxy_line", "next_week", "next_gameday",
                     "_nflverse_player_id"]],
        on=["_match_key", "stat_name"], how="inner",
    )
    merged = merged[merged["choice"].str.lower() == merged["my_side"]]
    merged = merged.dropna(subset=["decimal_price", "my_prob"])
    merged = merged[merged["decimal_price"] > 1]
    merged = merged.drop_duplicates(subset=["full_name", "stat_name"])

    # Real fix (2026-08-26): there should never be a "line mismatch" concept here
    # anymore -- my_prob only ever answered "beats OUR proxy_line", so it's
    # recomputed against Underdog's REAL stat_value for every matched row instead
    # of being excluded when the two numbers diverge. Same fix as the Parlay
    # Builder and player card (see recompute_probability_for_real_line's docstring
    # in dashboard/utils.py for the full reasoning). line_divergence is kept only
    # as informational context in the bet-slip description below, not a pass/fail
    # gate -- nothing gets dropped for this reason anymore.
    merged["line_divergence"] = (merged["stat_value"] - merged["proxy_line"]).abs() / merged["proxy_line"].replace(0, np.nan)

    weekly_stats = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)

    def _recompute_row(row):
        # my_prob/proxy_line are stated in terms of whichever side the model
        # favors (my_side) -- recompute_probability_for_real_line expects a
        # probability of exceeding a line, so this recomputes P(exceed real
        # stat_value) directly when my_side is "over", or works in "under" terms
        # (both flipped) when my_side is "under", then returns whichever matches
        # my_side so downstream code (Kelly calc etc.) keeps meaning the same thing.
        # estimate_player_stat_std wants a weekly_stats.csv column, not the model's
        # stat_name -- without this map it always returned None and the real-line
        # recompute below silently no-op'd (same trap fixed in score_underdog_board).
        wcol = _STAT_WEEKLY_COL.get(row["stat_name"])
        std = (estimate_player_stat_std(row["_nflverse_player_id"], wcol, weekly_stats,
                                        position=row["position"]) if wcol else None)
        if row["my_side"] == "over":
            recomputed = recompute_probability_for_real_line(row["my_prob"], row["proxy_line"], row["stat_value"], std)
        else:
            # my_prob is P(under proxy_line) here; recompute_probability_for_real_line
            # computes P(exceed X), so flip in and flip back out.
            recomputed_over = recompute_probability_for_real_line(1 - row["my_prob"], row["proxy_line"], row["stat_value"], std)
            recomputed = None if recomputed_over is None else 1 - recomputed_over
        return recomputed if recomputed is not None else row["my_prob"]  # fall back to proxy-based if std unavailable

    merged["my_prob"] = merged.apply(_recompute_row, axis=1)
    # Real-line under-lean correction (calibration.adjust_over_prob), applied in
    # P(over) space then mapped back to the side the model favors.
    from calibration import adjust_over_prob
    p_over = np.where(merged["my_side"] == "over", merged["my_prob"], 1 - merged["my_prob"])
    p_over = adjust_over_prob(p_over)
    merged["my_prob"] = np.where(merged["my_side"] == "over", p_over, 1 - p_over)
    # Blend with the de-vigged market (2026-10-02): on real Underdog lines the blend
    # beat both the model and the market alone, and it tames the model's worst
    # failure -- 99%+ calls where the line sits far from the player's baseline
    # because of a role change the model can't see. model_prob keeps the raw value.
    # The MIN_CONFIDENCE gate keeps its meaning ("the model has a strong opinion") by
    # reading the raw model; Kelly/EV sizing uses the blended, market-anchored my_prob.
    merged["model_prob"] = merged["my_prob"]
    merged["my_prob"] = blend_with_market(merged["my_prob"], merged["market_fair_prob"])
    merged["confidence"] = (merged["model_prob"] - 0.5).abs() * 2
    merged["team"] = _current_team(merged)

    return merged


def build_leg_pool(matched: pd.DataFrame) -> pd.DataFrame:
    """Picks entries are built from -- not bets on their own (Underdog pick'em needs
    2+ picks). Columns follow parlay_calculator.find_best_entries; "choice" is the side
    the model favors, which joint_probability needs to sign each correlation. Legs
    without a known team are dropped: the 2-team rule can't be checked for them."""
    q = matched[(matched["confidence"] >= MIN_CONFIDENCE) & matched["team"].notna()]
    if q.empty:
        return pd.DataFrame()
    # One-sided options have no de-vigged price; the raw implied probability (margin
    # included) stands in, which understates their edge rather than inflating it.
    market = q["market_fair_prob"].fillna(q["implied_prob"])
    return q.assign(
        player=q["full_name"], line=q["stat_value"], choice=q["my_side"], prob=q["my_prob"],
        decimal=q["decimal_price"], market_prob=market, edge=q["my_prob"] - market,
        position_prop=q["position"].astype(str) + " " + q["prop_type"].astype(str),
    )


def _entry_description(legs: list[dict], adjustments: list) -> str:
    picks = " + ".join(f"{l['player']} ({l['team']} · {pretty_stat_name(l['stat_name'])} "
                       f"{l['choice']} {l['line']})" for l in legs)
    weeks = sorted({int(l["next_week"]) for l in legs if pd.notna(l.get("next_week"))})
    days = sorted({str(l["next_gameday"]) for l in legs if pd.notna(l.get("next_gameday"))})
    when = " · ".join(([f"Wk {'/'.join(map(str, weeks))}"] if weeks else []) + days)
    corr = "".join(f", {a} / {b} correlation {phi:+.2f}" for a, b, phi in adjustments)
    return f"{picks} [{when}{corr}]"


def build_entry_candidates(matched: pd.DataFrame, correlations: dict | None = None) -> list[dict]:
    """Highest-EV valid Underdog entries (2-3 picks, 2+ teams, one pick per player)
    from the leg pool, kept only if the entry's Kelly fraction at its real payout
    clears MIN_KELLY_EDGE."""
    pool = build_leg_pool(matched)
    if pool.empty:
        return []
    best = find_best_entries(pool, n_legs_range=ENTRY_LEGS, top_k=50, pool_size=LEG_POOL_SIZE,
                             min_leg_edge=0.0,
                             correlations=load_leg_correlations() if correlations is None else correlations)
    if best.empty:
        return []
    candidates = []
    for r in best.itertuples():
        f = kelly_fraction(r.joint_prob, r.payout)
        if not r.valid or f < MIN_KELLY_EDGE:
            continue
        candidates.append({
            "type": f"{r.n_legs}-pick entry",
            "description": _entry_description(r.legs, r.adjustments),
            "legs": [l["player"] for l in r.legs],
            "leg_details": [_leg_detail(l) for l in r.legs],
            "model_prob": r.joint_prob,
            "decimal_odds": r.payout,
            "kelly_fraction": f,
            "teams": sorted({l["team"] for l in r.legs}),
        })
    return candidates


def allocate_budget(candidates: list[dict], budget: float, top_n: int = TOP_N_OPPORTUNITIES) -> list[dict]:
    # Last line of defence: nothing reaches the slip unless Underdog would accept it.
    candidates = [c for c in candidates if entry_validity(c["leg_details"])[0]]
    if not candidates:
        return []
    # The best entries tend to share the same top legs; three entries riding on one
    # player is one bet tripled, not a spread budget. Each player appears once.
    ranked, used = [], set()
    for c in sorted(candidates, key=lambda c: c["kelly_fraction"], reverse=True):
        if used.isdisjoint(c["legs"]):
            ranked.append(c)
            used.update(c["legs"])
        if len(ranked) == top_n:
            break
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
    print("CAVEAT: probabilities below are recomputed against Underdog's REAL posted")
    print("line (not a raw proxy-based number) -- but that recomputation is a")
    print("statistical approximation, not something trained on real historical")
    print("Underdog lines with graded outcomes (no such archive exists yet). Real")
    print("accuracy against actual market prices remains unvalidated -- everything")
    print("below only surfaces bets that clear a real edge bar at REAL live prices,")
    print("which is a much stricter test than confidence alone. Only wager what you can afford to lose.")
    print()

    matched = load_matched_props()
    print(f"{len(matched)} live prop rows matched to a model prediction this week.\n")

    candidates = build_entry_candidates(matched)
    print(f"{len(candidates)} valid +EV Underdog entries found (2-3 picks, 2+ teams, "
          f"joint prob x real payout > 1).\n")

    allocated = allocate_budget(candidates, args.budget)

    if not allocated:
        print("No valid +EV entries clear the bar this week. Suggestion: skip this week.")
        return

    print(f"Suggested split of ${args.budget:.2f}:\n")
    for c in allocated:
        edge = c["model_prob"] * c["decimal_odds"] - 1
        print(f"[{c['type'].upper()}] ${c['suggested_stake']:.2f}  {c['description']}")
        print(f"          joint prob: {c['model_prob']*100:.1f}%   real payout: {c['decimal_odds']:.2f}x   "
              f"implied edge: {edge*100:+.1f}%   raw Kelly fraction: {c['kelly_fraction']*100:.2f}% (of a full bankroll -- "
              f"used here for relative ranking, not as a literal fraction of this fixed weekly budget)")
        print()

    total = sum(c["suggested_stake"] for c in allocated)
    print(f"Total suggested: ${total:.2f} of ${args.budget:.2f} budget.")
    if total < args.budget:
        print(f"(${args.budget - total:.2f} unallocated -- not enough qualifying +EV entries to use the full budget this week. That's fine; forcing it isn't.)")


if __name__ == "__main__":
    main()
