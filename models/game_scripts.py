"""
Game-script chances for each game: shootout, blowout and slog (2026-11-09 research).

What the research found (nflverse 2014-2026, each 2021-2026 week predicted only from
earlier weeks):
- The game's own total and spread are the best pregame guide to the script. A shootout
  (both teams score 27+) happens in about 8% of games with a total of 43 or less and
  about 30% with a total of 52+. Team stats (EPA per play, pace, pass rate, points form)
  and QB changes added nothing once the total and spread were known: the market already
  prices them in.
- Wind of 10+ mph outdoors is the one extra signal that held up: those games went Under
  the total 58% of the time (2014-2025, n=741) and teams threw for 8-30 fewer yards than
  their average.
- Nothing predicted Over vs Under the total itself, or comebacks.
- Same-game props hit or miss together (per-game hit rates spread out about twice as
  much as independent legs would), so a same-game stack is a bet on the script.
  On real 2026 Underdog lines, 3-Under stacks hit about twice as often as 3-Over stacks
  in low and mid-total games (21% vs 12%), and about the same (14% vs 14%) in the
  highest-total third.
- Big favorites (by 10+) run for ~8 more yards than their average; big underdogs throw
  for ~20 fewer, so "garbage time" pass Overs on big underdogs don't pay off.

The model is a small logistic regression on total, |spread| and windy, refit from
schedules.csv whenever it changes. Probabilities were calibrated out of sample on 2026
(predicted 19 shootouts, 23 happened).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["total_line", "abs_spread", "windy"]
WINDY_MPH = 10
SHOOTOUT_SPOT = 0.20   # about 1.5x the 14% base rate (total ~49.5+ with a close spread)
LOW_TOTAL_SPOT = 0.10  # total ~43 or less
BIG_SPREAD = 9.5


def _prep(g: pd.DataFrame, wind_col: str = "wind") -> pd.DataFrame:
    g = g.copy()
    g["total_line"] = pd.to_numeric(g["total_line"], errors="coerce")
    g["abs_spread"] = pd.to_numeric(g["spread_line"], errors="coerce").abs()
    # A retractable roof ("open" in the record) is usually shut on a windy day.
    indoors = g["roof"].isin(["dome", "closed", "open"]) if "roof" in g else False
    wind = pd.to_numeric(g[wind_col], errors="coerce") if wind_col in g else pd.Series(np.nan, index=g.index)
    g["windy"] = ((wind >= WINDY_MPH) & ~indoors).astype(int)
    return g


def label_games(schedules: pd.DataFrame) -> pd.DataFrame:
    s = _prep(schedules[schedules["home_score"].notna() & schedules["total_line"].notna()])
    hs, aw = s["home_score"], s["away_score"]
    s["y_shootout"] = ((hs >= 27) & (aw >= 27)).astype(int)
    s["y_blowout"] = ((hs - aw).abs() >= 17).astype(int)
    s["y_slog"] = ((hs + aw) <= s["total_line"] - 10).astype(int)
    return s


def fit(schedules: pd.DataFrame) -> dict:
    """One logistic model per script, fit on every finished game with a total."""
    s = label_games(schedules)
    models = {}
    for y in ["y_shootout", "y_blowout", "y_slog"]:
        if s[y].nunique() < 2:
            continue
        models[y[2:]] = LogisticRegression(max_iter=500).fit(s[FEATURES], s[y])
    return models


def fill_roof(games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """nflverse leaves roof blank for some unplayed games; use the last known roof at that
    stadium (else the home team's last home game) so a dome game isn't flagged windy."""
    g = games.copy()
    if "roof" not in g:
        g["roof"] = np.nan
    past = schedules[schedules["roof"].notna()].sort_values("gameday")
    by_stadium = past.drop_duplicates("stadium_id", keep="last").set_index("stadium_id")["roof"] if "stadium_id" in past else pd.Series(dtype=object)
    by_home = past.drop_duplicates("home_team", keep="last").set_index("home_team")["roof"]
    miss = g["roof"].isna()
    if "stadium_id" in g:
        g.loc[miss, "roof"] = g.loc[miss, "stadium_id"].map(by_stadium)
    miss = g["roof"].isna()
    g.loc[miss, "roof"] = g.loc[miss, "home_team"].map(by_home)
    return g


def script_chances(games: pd.DataFrame, models: dict, wind_col: str = "wind") -> pd.DataFrame:
    """Adds p_shootout / p_blowout / p_slog, a short script label and a stacking tip."""
    g = _prep(games, wind_col)
    ok = g[FEATURES[:2]].notna().all(axis=1)
    for name, m in models.items():
        g[f"p_{name}"] = np.nan
        if ok.any():
            g.loc[ok, f"p_{name}"] = m.predict_proba(g.loc[ok, FEATURES])[:, 1]
    tips = [script_tip(r) for r in g.to_dict("records")]
    g["script"] = [t[0] for t in tips]
    g["script_tip"] = [t[1] for t in tips]
    g["stack_side"] = [t[2] for t in tips]
    return g


def script_tip(r: dict) -> tuple[str, str, str | None]:
    """(label, one-line tip, side a same-game stack should lean: 'over', 'under' or None)."""
    ps = r.get("p_shootout")
    if ps is None or ps != ps:
        return "No line yet", "Wait for the game total before stacking this game.", None
    if r.get("windy"):
        return ("Windy", "Wind 10+ mph: these games go Under 58% of the time and passing drops. "
                "Lean Unders, skip Over stacks.", "under")
    big = (r.get("abs_spread") or 0) >= BIG_SPREAD
    if ps >= SHOOTOUT_SPOT:
        tip = ("Shootout spot: a 2-3 Over stack telling the passing story is fine here. "
               "Under stacks don't beat Over stacks in games like this.")
        return "Shootout spot", tip, "over"
    if ps < LOW_TOTAL_SPOT:
        tip = ("Low total: shootouts are rare here (under 10%) and Under stacks have hit more often "
               "than Over stacks. Skip Over stacks.")
        return "Low-scoring", tip, "under"
    if big:
        return ("Blowout risk", "Big favorite: their run game tends to run hot, the underdog's passing cold "
                "(no garbage-time boost). Keep legs in different games.", None)
    return ("No clear script", "Middle-of-the-road game: Under stacks have still been steadier "
            "(about 2 to 1), or keep legs in different games.", "under")


def stack_warning(stack_side: str | None, n_over: int, n_under: int) -> str | None:
    """A warning when a same-game slip leans the other way from the game's script."""
    if stack_side == "under" and n_over >= 2 and n_over > n_under:
        return "This slip stacks Overs in a game that doesn't look like a shootout. Same-game Overs mostly hit or miss together."
    if stack_side == "over" and n_under >= 3 and n_over == 0:
        return "All Unders in a likely high-scoring game: in games like this, Under stacks haven't beaten Over stacks."
    return None
