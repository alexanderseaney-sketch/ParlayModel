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

Since 2026-10-20 legs that carry "leg_key" (position + Underdog stat, e.g. "QB passing_tds")
and "game" (Underdog match_id) are looked up first in the wider same-game table,
models/leg_correlations.csv (build_leg_correlations.py): every bet stat, teammates AND
opponents. The old position_prop table is the fallback for callers that don't pass those.

Underdog prices links too: it lowers the multiplier on picks that tend to hit together and
raises it on picks that pull apart, and shows the adjusted number in its entry builder.
That adjustment isn't in the board pull, so underdog_adjusted_payout() estimates it the way
a book would: reprice the entry from the market's own fair chances with the same links.

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
SAME_GAME_LINKS_PATH = os.path.join(os.path.dirname(__file__), "leg_correlations.csv")

# Legacy Underdog "standard" (all-must-hit) pick'em multipliers by entry size. Verify
# against the app -- Underdog changes these; only used when payout_mode="standard".
STANDARD_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5, 7: 65.0, 8: 120.0}

MIN_LEGS, MAX_LEGS = 2, 8

# From the real-line record (60.7% per pick on proven stat types, 2026 wks 1-6): 3 picks
# grows a bankroll fastest on Underdog's payouts; 2 pays too little, and from 6 picks a
# long losing run gets likely and any error in the per-pick rate compounds.
RECOMMENDED_LEGS = 3
TOO_MANY_LEGS = 6

# Underdog stat names the model's prop_type/stat columns can arrive as.
_STAT_ALIASES = {"receptions": "receiving_rec"}


def load_leg_correlations(path: str = LEG_CORRELATIONS_PATH) -> dict[frozenset, float]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {frozenset([r["position_prop_a"], r["position_prop_b"]]): r["phi"] for _, r in df.iterrows()}


def load_same_game_links(path: str = SAME_GAME_LINKS_PATH) -> dict[tuple, float]:
    """{(frozenset({leg_a, leg_b}), "same_team"|"opponent"): phi}"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {(frozenset([r.leg_a, r.leg_b]), r.relation): float(r.phi) for r in df.itertuples()}


_SAME_GAME_LINKS: dict | None = None


def _same_game_links() -> dict:
    global _SAME_GAME_LINKS
    if _SAME_GAME_LINKS is None:
        _SAME_GAME_LINKS = load_same_game_links()
    return _SAME_GAME_LINKS


def leg_key(position, stat_name) -> str | None:
    if not position or not stat_name or str(position) == "nan":
        return None
    stat = _STAT_ALIASES.get(str(stat_name), str(stat_name))
    return f"{position} {stat}"


def _same_game(a: dict, b: dict) -> bool:
    ga, gb = a.get("game"), b.get("game")
    if ga is not None and gb is not None and str(ga) != "nan" and str(gb) != "nan":
        return str(ga) == str(gb)
    return bool(a.get("team")) and a.get("team") == b.get("team")


def pair_link(a: dict, b: dict, correlations: dict | None = None,
              links: dict | None = None) -> tuple[float, str] | None:
    """(phi between the two OVER outcomes, relation) for two legs in one game, or None
    when they're unlinked (different games, or no measured link)."""
    if not _same_game(a, b):
        return None
    same_team = bool(a.get("team")) and a.get("team") == b.get("team")
    relation = "same_team" if same_team else "opponent"
    ka, kb = a.get("leg_key"), b.get("leg_key")
    if ka and kb:
        links = _same_game_links() if links is None else links
        phi = links.get((frozenset([ka, kb]), relation))
        if phi is not None:
            return phi, relation
    if same_team and a.get("position_prop") and b.get("position_prop"):
        phi = (correlations or {}).get(frozenset([a["position_prop"], b["position_prop"]]))
        if phi is not None:
            return phi, relation
    return None


def _pair_joint(pa: float, pb: float, phi: float) -> float:
    joint = pa * pb + phi * np.sqrt(max(pa * (1 - pa) * pb * (1 - pb), 0))
    return min(max(joint, max(0.0, pa + pb - 1)), min(pa, pb))


def _is_under(leg: dict) -> bool:
    return str(leg.get("choice") or "").lower() in ("under", "lower")


def joint_probability(legs: list[dict], correlations: dict[frozenset, float] | None = None,
                      prob_field: str = "prob", links: dict | None = None) -> dict:
    """legs: [{"prob": float, "team": str|None, "position_prop": str|None,
    "choice": "over"|"under"|None, "leg_key"?: str, "game"?: str}, ...].
    Returns {"naive_prob", "adjusted_prob", "adjustments": [(a, b, phi), ...],
    "links": [{"a", "b", "phi", "relation"}, ...]}. phi is signed for the sides picked."""
    probs = [leg.get(prob_field) for leg in legs]
    if any(p is None or not np.isfinite(p) for p in probs):
        return {"naive_prob": None, "adjusted_prob": None, "adjustments": [], "links": []}
    if correlations is None:
        correlations = load_leg_correlations()
    naive = float(np.prod(probs)) if legs else 0.0
    adjusted = naive
    adjustments, found = [], []
    for (a, pa), (b, pb) in itertools.combinations(list(zip(legs, probs)), 2):
        link = pair_link(a, b, correlations, links)
        if link is None:
            continue
        phi, relation = link
        # The measured phi is between two OVER outcomes. An under is the complement
        # event, and corr(A, not B) = -corr(A, B): flip once per under leg in the pair
        # (same identity models/generate_weekly_bet_slip.py applies). Legs without a
        # "choice" are treated as overs, as before.
        if (_is_under(a) + _is_under(b)) % 2 == 1:
            phi = -phi
        adjusted *= _pair_joint(pa, pb, phi) / (pa * pb) if pa * pb > 0 else 1.0
        adjustments.append((a.get("leg_key") or a.get("position_prop"), b.get("leg_key") or b.get("position_prop"), phi))
        found.append({"a": a.get("player"), "b": b.get("player"), "phi": phi, "relation": relation})
    return {"naive_prob": naive, "adjusted_prob": float(adjusted), "adjustments": adjustments, "links": found}


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


def underdog_adjusted_payout(legs: list[dict], payout: float | None,
                             correlations: dict | None = None, links: dict | None = None) -> float | None:
    """Estimate of the multiplier Underdog shows after its same-game adjustment: the plain
    product of prices scaled by (market chance if independent) / (market chance with the
    links). Picks that hit together pay less, picks that pull apart pay more; legs from
    different games are untouched. Needs every leg's market_prob, else returns payout."""
    if payout is None:
        return None
    m = joint_probability(legs, correlations, prob_field="market_prob", links=links)
    if not m["naive_prob"] or not m["adjusted_prob"]:
        return payout
    return float(payout * m["naive_prob"] / m["adjusted_prob"])


def games_in(legs: list[dict]) -> int:
    return len({str(l.get("game")) for l in legs if l.get("game") is not None and str(l.get("game")) != "nan"})


def leg_count_note(n: int) -> str | None:
    """Plain-English guidance on entry size (see RECOMMENDED_LEGS)."""
    if n == 2:
        return ("2 picks pays the least per pick, so each one has to hit about 58% of the time "
                "to come out ahead. 3 picks is the sweet spot.")
    if n == 4:
        return "4 picks pays relatively little for the extra risk. 3 or 5 picks usually pays better per pick."
    if n >= TOO_MANY_LEGS:
        return (f"{n} picks is a long shot: even good picks all hit only about "
                f"{0.6 ** n * 100:.0f}% of the time. 3 picks is the sweet spot.")
    return None


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
                   correlations: dict[frozenset, float] | None = None,
                   links: dict | None = None) -> dict:
    """base_payout = Underdog's plain multiplier from the board; payout = after its
    estimated same-game adjustment (== base_payout when no legs share a game). EV uses
    payout."""
    jp = joint_probability(legs, correlations, links=links)
    base_payout = entry_payout(legs, payout_mode)
    payout = underdog_adjusted_payout(legs, base_payout, correlations, links)
    market = float(np.prod([leg["market_prob"] for leg in legs])) \
        if all(leg.get("market_prob") is not None and np.isfinite(leg.get("market_prob")) for leg in legs) else None
    valid, why = entry_validity(legs)
    joint = jp["adjusted_prob"]
    ev = joint * payout - 1 if payout and joint is not None else None
    return {
        "n_legs": len(legs), "valid": valid, "validity_note": why,
        "naive_prob": jp["naive_prob"], "joint_prob": jp["adjusted_prob"],
        "market_joint_prob": market, "payout": payout, "base_payout": base_payout,
        "links": jp["links"],
        "fair_payout": 1 / joint if joint else None,
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
                      correlations: dict[frozenset, float] | None = None,
                      mode: str = "any", game=None) -> pd.DataFrame:
    """Exhaustive search over the `pool_size` best single legs (by edge) for the highest-
    EV valid entries of each size in n_legs_range. candidates needs columns: player,
    team, stat_name, line, choice, prob, decimal, market_prob, edge, position_prop,
    payout_multiplier (optional), and leg_key/game for same-game links.
    pool_size 18 keeps a 4-leg search at ~3k combos.
    mode: "any"; "same_game" (every leg from `game`, both teams allowed); or
    "multi_game" (at most one leg per game, so no leg's game script sinks another)."""
    if candidates.empty:
        return pd.DataFrame()
    if correlations is None:
        correlations = load_leg_correlations()
    pool = candidates[candidates["edge"] >= min_leg_edge].sort_values("edge", ascending=False)
    if mode == "same_game" and game is not None:
        pool = pool[pool["game"].astype(str) == str(game)]
    pool = pool.drop_duplicates(subset=["player"])
    if mode == "multi_game" and "game" in pool.columns:
        # The best pick from each game first, so the pool spans as many games as it can.
        pool = pool.drop_duplicates(subset=["game"])
    pool = pool.head(pool_size)
    legs_all = pool.to_dict("records")
    rows = []
    for n in range(n_legs_range[0], n_legs_range[1] + 1):
        for combo in itertools.combinations(legs_all, n):
            combo = list(combo)
            if not entry_validity(combo)[0]:
                continue
            if mode == "multi_game" and games_in(combo) < len(combo):
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
