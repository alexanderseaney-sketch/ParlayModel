"""Price-aware pick rules for Underdog Champions (added 2026-11-22).

Champions pays each pick its own multiplier (the slip shows it: 0.75x for a heavy
favourite, 1.00x for a plain line, 1.18x for an underdog side), and an entry pays about
2^n x the product of those x k(n) (k = 0.82 for 3 picks). So a pick's break-even is
about 1 / (2 x multiplier x 0.82^(1/3)): 53.4% at 1.00x, 62.6% at 0.85x, 71% at 0.75x.

Underdog's per-pick prices are in every committed data/raw/underdog_props.csv; they were
rebuilt for 2026 weeks 4-10 (98% of graded picks, median 4 hours before kickoff) and are
now archived going forward (data/archive_underdog_prices.py). Judged walk-forward on
real closing lines, 2026 weeks 5-10, by Champions value per pick (hit x 2 x mult x 0.936;
over 1.00 makes money in a 3-pick):

    proven picks (pick_rules, before this)          1288  61.7%  value 1.044  +57 units
      ... priced under 0.80x (heavy favourites)      116  66.4%  value 0.924   -9
      ... priced 1.00x                               377  57.0%  value 1.068
      ... priced over 1.00x                          115  58.3%  value 1.183

    flat-price lean Unders, model on the Under,
    not already proven                               273  60.1%  value 1.125  +34

1. PRICED OUT. A proven pick whose own multiplier is under 0.80x hits a lot (66%) but
   not the 70%+ that price needs. It loses money on Champions in 5 of 6 weeks, so it is
   no longer proven. Moving the bar 0.75-0.85 changes little.

2. FLAT-PRICE LEAN. When a line is close to 50/50 Underdog rounds both multipliers to
   1.00x, but its american prices still lean (Under -118 / Over -105 is a 52.6% Under at
   a 50% price). Those leaning Unders hit 59% on the whole board (n=616, wks 4-10, above
   53.4% in 6 of 7 weeks) and 60-63% when the model also takes the Under (flat across
   model confidence 50-60%+ and lean size 51-53%). Leaning Overs at 1.00x hit only 47%:
   Underdog's Overs are priced a little rich everywhere (devigged 50% Overs hit 47.8%).
   So an Under at 1.00x whose devigged chance is 51%+ is proven when the model is on the
   Under at all (over 50%) and the side isn't a losing one.

Also tested: a walk-forward logistic blend of model and market (log loss .6787 vs
market alone .6797, raw model .6898). Picking by its expected value did not beat the
two rules above (value 1.01-1.10 on fewer picks). The market price alone is a better
coin-flip forecast than the raw model on every prop; the model's edge lives in the
picks the rules keep.
"""
import numpy as np
import pandas as pd

MIN_CHAMPIONS_MULT = 0.80   # proven picks priced under this lose on Champions
FLAT_MULT = 1.0
LEAN_MIN = 0.51             # devigged chance of the Under, both sides at 1.00x
K3 = 0.82 ** (1 / 3)        # 3-pick entry factor per pick (measured on the slip 11/21)


def break_even(mult) -> float:
    """Chance a pick needs to break even in a 3-pick Champions entry at this multiplier."""
    return 1 / (2 * float(mult) * K3)


def priced_out(mult) -> bool:
    try:
        m = float(mult)
    except (TypeError, ValueError):
        return False
    return np.isfinite(m) and m < MIN_CHAMPIONS_MULT


def flat_price_lean(choice, mult, fair_prob) -> bool:
    """An Under paying 1.00x whose own devigged chance (fair_prob, this side) is 51%+."""
    if str(choice).lower() not in ("under", "lower"):
        return False
    try:
        m, p = float(mult), float(fair_prob)
    except (TypeError, ValueError):
        return False
    return np.isfinite(m) and np.isfinite(p) and abs(m - FLAT_MULT) < 1e-6 and p >= LEAN_MIN


def _implied(a):
    a = np.asarray(pd.to_numeric(a, errors="coerce"), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a >= 0, 100 / (a + 100), -a / (-a + 100))


def closing_prices(history: pd.DataFrame, props: pd.DataFrame, kickoff_col: str = "kick") -> pd.DataFrame:
    """For each props row (full_name, stat_name, stat_value, kickoff UTC), the last price
    data/raw/underdog_price_history.csv logged for that exact line before kickoff (within
    the 8 days before; the file only logs changes). Adds over_mult, under_mult and
    fair_over (devigged Over chance)."""
    h = history.rename(columns={"line": "stat_value"}).copy()
    h["px_ts"] = pd.to_datetime(h["seen_at"], utc=True, errors="coerce")
    h = h.dropna(subset=["px_ts"]).sort_values("px_ts")
    p = props.copy()
    p["_row"] = np.arange(len(p))
    p["_t"] = pd.to_datetime(p[kickoff_col], utc=True, errors="coerce") - pd.Timedelta(minutes=5)
    have = p.dropna(subset=["_t"]).sort_values("_t")
    m = pd.merge_asof(have, h[["full_name", "stat_name", "stat_value", "px_ts", "over_price", "under_price",
                               "over_mult", "under_mult"]],
                      left_on="_t", right_on="px_ts", by=["full_name", "stat_name", "stat_value"],
                      direction="backward", tolerance=pd.Timedelta(days=8))
    io, iu = _implied(m["over_price"]), _implied(m["under_price"])
    m["fair_over"] = io / (io + iu)
    out = p[["_row"]].merge(m[["_row", "over_mult", "under_mult", "fair_over"]], on="_row", how="left")
    res = props.copy()
    for c in ("over_mult", "under_mult", "fair_over"):
        res[c] = out[c].to_numpy()
    return res
