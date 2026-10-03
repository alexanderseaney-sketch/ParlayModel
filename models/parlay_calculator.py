"""
Parlay / Underdog pick'em entry math + automatic +EV combination search
(added 2026-10-02). Pure functions, no Streamlit -- used by the dashboard's +EV
Finder page and the parlay backtest.

Joint probability
-----------------
Legs are not independent when they share a team-game (QB passing yards and his WR1's
receiving yards move together; a QB's passing yards and his RB's rushing yards move
apart). joint_probability() applies the measured same-team phi coefficients from
models/parlay_leg_correlations.csv (analyze_parlay_correlations.py) pairwise:
    P(A and B) = pA*pB + phi*sqrt(pA(1-pA)pB(1-pB)),
clamped to the Frechet bounds, and multiplies each pair's correction into the naive
product. For 3+ mutually correlated legs that's an approximation, not a full joint
distribution. dashboard/utils.correlation_adjusted_parlay_probability() delegates here.

Payout
------
Underdog's odds-based pick'em (2026) prices every option individually (decimal_price,
over and under separately). An entry's payout is modelled as the product of its picks'
decimal prices -- the same convention the Parlay Builder and Weekly Bet Slip already
use. Check the payout Underdog shows in-app before entering; if your account still
gets the legacy fixed table, pass payout_mode="standard" (STANDARD_PAYOUTS, x each
pick's payout_multiplier).

EV per $1:  EV = P(all hit) * payout - 1  (== P*profit - (1-P)).
"""
import itertools
import os

import numpy as np
import pandas as pd

LEG_CORRELATIONS_PATH = os.path.join(os.path.dirname(__file__), "parlay_leg_correlations.csv")

# Legacy Underdog "standard" (all-must-hit) pick'em multipliers by entry size. Verify
# against the app -- Underdog changes these; only used when payout_mode="standard".
STANDARD_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5, 7: 65.0, 8: 120.0}

MIN_LEGS, MAX_LEGS = 2, 8


def load_leg_correlations(path: str = LEG_CORRELATIONS_PATH) -> dict[frozenset, float]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {frozenset([r["position_prop_a"], r["position_prop_b"]]): r["phi"] for _, r in df.iterrows()}


def joint_probability(legs: list[dict], correlations: dict[frozenset, float] | None = None) -> dict:
    """legs: [{"prob": float, "team": str|None, "position_prop": str|None}, ...].
    Returns {"naive_prob", "adjusted_prob", "adjustments": [(a, b, phi), ...]}."""
    if correlations is None:
        correlations = load_leg_correlations()
    naive = float(np.prod([leg["prob"] for leg in legs])) if legs else 0.0
    adjusted = naive
    adjustments = []
    for a, b in itertools.combinations(legs, 2):
        if not a.get("team") or a.get("team") != b.get("team"):
            continue
        if a.get("position_prop") is None or b.get("position_prop") is None:
            continue
        phi = correlations.get(frozenset([a["position_prop"], b["position_prop"]]))
        if phi is None:
            continue
        pa, pb = a["prob"], b["prob"]
        joint = pa * pb + phi * np.sqrt(max(pa * (1 - pa) * pb * (1 - pb), 0))
        joint = min(max(joint, max(0.0, pa + pb - 1)), min(pa, pb))
        adjusted *= joint / (pa * pb) if pa * pb > 0 else 1.0
        adjustments.append((a["position_prop"], b["position_prop"], phi))
    return {"naive_prob": naive, "adjusted_prob": float(adjusted), "adjustments": adjustments}


def entry_payout(legs: list[dict], payout_mode: str = "product") -> float | None:
    """Total return multiple (stake included) if every leg hits."""
    n = len(legs)
    if n < MIN_LEGS or n > MAX_LEGS:
        return None
    if payout_mode == "standard":
        base = STANDARD_PAYOUTS.get(n)
        if base is None:
            return None
        mult = np.prod([float(leg.get("payout_multiplier") or 1.0) for leg in legs])
        return float(base * mult)
    prices = [leg.get("decimal") for leg in legs]
    if any(p is None or not np.isfinite(p) or p <= 1 for p in prices):
        return None
    return float(np.prod(prices))


def entry_validity(legs: list[dict]) -> tuple[bool, str]:
    """Underdog entry rules already enforced by the Parlay Builder: 2-8 picks, one pick
    per player, and players from at least 2 different teams."""
    n = len(legs)
    if n < MIN_LEGS:
        return False, f"needs at least {MIN_LEGS} picks"
    if n > MAX_LEGS:
        return False, f"max {MAX_LEGS} picks"
    players = [leg.get("player") for leg in legs]
    if len(set(players)) < n:
        return False, "same player twice"
    teams = {leg.get("team") for leg in legs if leg.get("team")}
    if len(teams) < 2 and all(leg.get("team") for leg in legs):
        return False, "all picks from one team"
    return True, "ok"


def evaluate_entry(legs: list[dict], payout_mode: str = "product",
                   correlations: dict[frozenset, float] | None = None) -> dict:
    jp = joint_probability(legs, correlations)
    payout = entry_payout(legs, payout_mode)
    market = float(np.prod([leg["market_prob"] for leg in legs])) \
        if all(leg.get("market_prob") is not None and np.isfinite(leg.get("market_prob")) for leg in legs) else None
    valid, why = entry_validity(legs)
    ev = jp["adjusted_prob"] * payout - 1 if payout else None
    return {
        "n_legs": len(legs), "valid": valid, "validity_note": why,
        "naive_prob": jp["naive_prob"], "joint_prob": jp["adjusted_prob"],
        "market_joint_prob": market, "payout": payout,
        "fair_payout": 1 / jp["adjusted_prob"] if jp["adjusted_prob"] > 0 else None,
        "ev": ev, "adjustments": jp["adjustments"],
    }


def riskiest_leg(legs: list[dict]) -> dict | None:
    """The leg most likely to sink the entry: lowest model probability, ties broken by
    the smallest edge over the market."""
    if not legs:
        return None
    return min(legs, key=lambda l: (l["prob"], (l["prob"] - (l.get("market_prob") or l["prob"]))))


def find_best_entries(candidates: pd.DataFrame, n_legs_range=(2, 4), top_k: int = 15,
                      pool_size: int = 18, min_leg_edge: float = 0.0, payout_mode: str = "product",
                      correlations: dict[frozenset, float] | None = None) -> pd.DataFrame:
    """Exhaustive search over the `pool_size` best single legs (by edge) for the highest-
    EV valid entries of each size in n_legs_range. candidates needs columns: player,
    team, stat_name, line, choice, prob, decimal, market_prob, edge, position_prop,
    payout_multiplier (optional). pool_size 18 keeps a 4-leg search at ~3k combos."""
    if candidates.empty:
        return pd.DataFrame()
    if correlations is None:
        correlations = load_leg_correlations()
    pool = candidates[candidates["edge"] >= min_leg_edge].sort_values("edge", ascending=False)
    pool = pool.drop_duplicates(subset=["player"]).head(pool_size)
    legs_all = pool.to_dict("records")
    rows = []
    for n in range(n_legs_range[0], n_legs_range[1] + 1):
        for combo in itertools.combinations(legs_all, n):
            combo = list(combo)
            if not entry_validity(combo)[0]:
                continue
            res = evaluate_entry(combo, payout_mode, correlations)
            if res["ev"] is None:
                continue
            res["legs"] = combo
            res["label"] = " + ".join(
                f"{l['player']} {str(l['choice']).upper()} {l['line']} {l['stat_name']}" for l in combo)
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("ev", ascending=False)
    # Diversify: keep each size's best entries rather than letting one size dominate.
    return out.groupby("n_legs", group_keys=False).head(top_k).sort_values("ev", ascending=False).reset_index(drop=True)
