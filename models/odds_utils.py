"""
Odds-format conversion + de-vigging, shared by the Underdog board, The Odds API
puller, the +EV Finder and the backtests (added 2026-10-02).

Every price that enters the EV math is normalised to a DECIMAL price (total return
per $1 staked, stake included) and a probability. Underdog's own pulls already carry
american_price/decimal_price per option (over and under each priced separately since
the 2026 odds-based pick'em); sportsbooks via The Odds API arrive as American or
decimal depending on the request.

De-vig: a two-way market's raw implied probabilities sum to >1 (the book's margin).
`devig_two_way` strips it multiplicatively (each side / sum) -- the simplest standard
method; with near-even two-way props the choice of method (multiplicative / power /
Shin) moves fair probabilities by well under a point.
"""
import numpy as np
import pandas as pd


def american_to_decimal(american):
    a = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / np.abs(a))
    out = np.where(np.isfinite(a) & (a != 0), out, np.nan)
    return out if out.ndim else float(out)


def decimal_to_american(decimal):
    d = np.asarray(decimal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d >= 2, (d - 1) * 100.0, -100.0 / (d - 1))
    out = np.where(np.isfinite(d) & (d > 1), out, np.nan)
    return out if out.ndim else float(out)


def decimal_to_implied(decimal):
    d = np.asarray(decimal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d > 1, 1.0 / d, np.nan)
    return out if out.ndim else float(out)


def american_to_implied(american):
    return decimal_to_implied(american_to_decimal(american))


def prob_to_decimal(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((p > 0) & (p < 1), 1.0 / p, np.nan)
    return out if out.ndim else float(out)


def devig_two_way(implied_a, implied_b):
    """Fair (vig-free) probabilities for a two-outcome market, multiplicative method."""
    a = np.asarray(implied_a, dtype=float)
    b = np.asarray(implied_b, dtype=float)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = np.where(total > 0, a / total, np.nan)
    return fa, 1 - fa


# Model weight in the model/market blend, chosen on the 2026 wk1 real-line backtest
# (backtesting/backtest_underdog_lines.py, 304 gated props): blend log loss 0.6589 vs
# model 0.670 / market 0.662; flat between w=0.2 and 0.35. Re-check as weeks accrue.
BLEND_WEIGHT_MODEL = 0.35


def blend_with_market(model_p, market_p, w_model: float = BLEND_WEIGHT_MODEL):
    """Linear mix of the model's probability and the de-vigged market's; where the
    market probability is missing, returns the model's unchanged. The market knows
    things the model can't (role changes, injuries, news) -- exactly where the model's
    proxy->real-line conversion produced its worst 95%+ misses. Linear rather than a
    logit pool: tied on gated props and clearly better on the ungated tail (0.681 vs
    0.683), because a logit pool lets a 99.9% model value drag the result to ~86%."""
    m = np.asarray(model_p, dtype=float)
    k = np.asarray(market_p, dtype=float)
    out = np.where(np.isfinite(k), w_model * m + (1 - w_model) * k, m)
    return out if out.ndim else float(out)


def ev_per_dollar(prob, decimal_price):
    """Expected profit per $1 staked: p * decimal - 1 (decimal includes the stake).
    Equivalent to the (P x Payout) - (1 - P) form with Payout = decimal - 1."""
    return np.asarray(prob, dtype=float) * np.asarray(decimal_price, dtype=float) - 1.0


def kelly_fraction(prob, decimal_price, fraction: float = 0.25):
    """Fractional Kelly stake as a share of bankroll; 0 when the bet is -EV."""
    p = np.asarray(prob, dtype=float)
    b = np.asarray(decimal_price, dtype=float) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (b * p - (1 - p)) / b
    k = np.where(np.isfinite(k), np.clip(k, 0, 1), 0.0) * fraction
    return k if k.ndim else float(k)


def add_underdog_market_probs(board: pd.DataFrame) -> pd.DataFrame:
    """Adds market columns to an Underdog board (one row per option):
      decimal          -- the option's decimal price (decimal_price, else from american_price)
      implied_prob     -- 1/decimal (includes Underdog's margin)
      market_fair_prob -- de-vigged probability of THIS option, paired with the opposite
                          option of the same (player, stat, line); NaN when the other
                          side isn't offered (one-sided boosts/specials)
      market_hold      -- the pair's overround (sum of implied - 1)
    """
    b = board.copy()
    dec = pd.to_numeric(b.get("decimal_price"), errors="coerce")
    if "american_price" in b.columns:
        dec = dec.fillna(pd.Series(american_to_decimal(pd.to_numeric(b["american_price"], errors="coerce")),
                                   index=b.index))
    b["decimal"] = dec
    b["implied_prob"] = decimal_to_implied(b["decimal"].to_numpy())

    key = ["full_name", "stat_name", "stat_value"]
    if not all(k in b.columns for k in key) or "choice" not in b.columns:
        b["market_fair_prob"] = np.nan
        b["market_hold"] = np.nan
        return b
    choice = b["choice"].astype(str).str.lower()
    over = b[choice == "higher"].copy() if (choice == "higher").any() else b[choice == "over"].copy()
    under = b[choice == "lower"].copy() if (choice == "lower").any() else b[choice == "under"].copy()
    pair = over[key + ["implied_prob"]].drop_duplicates(key).merge(
        under[key + ["implied_prob"]].drop_duplicates(key), on=key, suffixes=("_over", "_under"))
    fo, _ = devig_two_way(pair["implied_prob_over"].to_numpy(), pair["implied_prob_under"].to_numpy())
    pair["_fair_over"] = fo
    pair["market_hold"] = pair["implied_prob_over"] + pair["implied_prob_under"] - 1
    b = b.merge(pair[key + ["_fair_over", "market_hold"]], on=key, how="left")
    is_over = b["choice"].astype(str).str.lower().isin(["over", "higher"])
    b["market_fair_prob"] = np.where(is_over, b["_fair_over"], 1 - b["_fair_over"])
    b["market_fair_prob_over"] = b["_fair_over"]
    return b.drop(columns=["_fair_over"])
