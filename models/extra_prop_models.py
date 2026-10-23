"""
Predictions for the Underdog NFL prop types the trained XGBoost models don't cover
(added 2026-10-20): tackles, kicking, attempts / completions / targets, longest plays,
fantasy points, combo yards, 1Q / 1H receptions, "X+ yards each half", last TD scorer,
and more.

One simple, honest method for all of them, so every type is graded the same way:

  1. Recent form: a recency-weighted average of the player's last 12 games
     (data/raw/prop_game_logs.csv, from free nflverse box scores and play-by-play),
     last season's games down-weighted. (Shrinking toward the position average was
     tried and dropped: the players Underdog posts lines for are the starters, so it
     dragged every expected value ~10% low and turned the model into an Under machine.)
  2. A spread around that average, fitted per stat on every game from 2025 up to the
     week being predicted (so a graded week is never in its own fit):
       count stats      negative binomial (dispersion fitted by likelihood)
       yardage / points the stat's own spread of actual / expected, binned by expected
       yes / no props   the form rate, recalibrated with a 2-number logistic fit
  3. P(over the real Underdog line) straight from that distribution -- no proxy line,
     so the real line is always used.

Stats a trained model already prices (receptions, yards, TDs, 1Q / 1H yards) are in
FALLBACK_STATS: the form model only fills in for the players those models skip or price
against a proxy line far from the real one.

backtesting/backtest_line_history.py grades these against real Underdog closing lines
and the Picks screen marks each type proven / unproven from that record.

Files written by `python models/extra_prop_models.py`:
  models/extra_prop_dists.json  fitted spreads per stat
  models/extra_prop_form.csv    every player's current expected value per stat
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(ROOT, "data", "raw", "prop_game_logs.csv")
DISTS_PATH = os.path.join(ROOT, "models", "extra_prop_dists.json")
FORM_PATH = os.path.join(ROOT, "models", "extra_prop_form.csv")

COUNT, CONT, BINARY = "count", "cont", "binary"
OFF = ("QB", "RB", "FB", "WR", "TE")
DEF = ("DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "SAF", "FS", "SS", "DB", "DL", "EDGE")

# Underdog stat_name -> (kind, positions it is offered for). Positions keep each stat's
# baseline (position average, fitted spread) to the players who actually get the line.
STATS = {
    "passing_comps": (COUNT, ("QB",)),
    "passing_att": (COUNT, ("QB",)),
    "passing_long": (CONT, ("QB",)),
    "passing_and_rushing_yds": (CONT, ("QB",)),
    "rushing_att": (COUNT, ("QB", "RB", "FB", "WR")),
    "rushing_long": (CONT, ("QB", "RB", "FB", "WR")),
    "receiving_tgts": (COUNT, ("RB", "FB", "WR", "TE")),
    "receiving_long": (CONT, ("RB", "FB", "WR", "TE")),
    "rush_rec_yds": (CONT, ("RB", "FB", "WR", "TE")),
    "period_1_receiving_rec": (COUNT, ("RB", "FB", "WR", "TE")),
    "period_1_2_receiving_rec": (COUNT, ("RB", "FB", "WR", "TE")),
    "fantasy_points": (CONT, OFF),
    "fumbles_lost": (COUNT, OFF),
    "last_touchdown_scored": (BINARY, ("QB", "RB", "FB", "WR", "TE")),
    "25_rec_yds_each_half": (BINARY, ("RB", "FB", "WR", "TE")),
    "50_rec_yds_each_half": (BINARY, ("RB", "FB", "WR", "TE")),
    "25_rush_yds_each_half": (BINARY, ("QB", "RB", "FB", "WR")),
    "50_rush_yds_each_half": (BINARY, ("QB", "RB", "FB", "WR")),
    "100_pass_yds_each_half": (BINARY, ("QB",)),
    "150_pass_yds_each_half": (BINARY, ("QB",)),
    "25_pass_yds_each_quarter": (BINARY, ("QB",)),
    "50_pass_yds_each_quarter": (BINARY, ("QB",)),
    "tackles": (COUNT, DEF),
    "assists": (COUNT, DEF),
    "tackles_and_assists": (COUNT, DEF),
    "sacks": (COUNT, DEF),  # in half-sack units, see _units
    "defensive_ints": (COUNT, DEF),
    "field_goals_made": (COUNT, ("K",)),
    "extra_points_made": (COUNT, ("K",)),
    "kicking_points": (CONT, ("K",)),
}

# Stats a trained model already prices (2026-10-23). The form model only fills in for
# the players those models skip -- running backs' catches (the receptions model is
# WR / TE only), depth players' TDs, QBs' 1st quarter / 1st half rushing, rookies and
# players with too few games -- so nearly every live Underdog prop gets a number.
FALLBACK_STATS = {
    "receiving_rec": (COUNT, ("RB", "FB", "WR", "TE")),
    "receiving_yds": (CONT, ("RB", "FB", "WR", "TE")),
    "rushing_yds": (CONT, ("QB", "RB", "FB", "WR")),
    "passing_yds": (CONT, ("QB",)),
    "passing_tds": (COUNT, ("QB",)),
    "passing_ints": (COUNT, ("QB",)),
    "rush_rec_tds": (COUNT, ("QB", "RB", "FB", "WR", "TE")),
    "period_1_receiving_yds": (CONT, ("RB", "FB", "WR", "TE")),
    "period_1_2_receiving_yds": (CONT, ("RB", "FB", "WR", "TE")),
    "period_1_rushing_yds": (CONT, ("QB", "RB", "FB", "WR")),
    "period_1_2_rushing_yds": (CONT, ("QB", "RB", "FB", "WR")),
    "period_1_passing_yds": (CONT, ("QB",)),
    "period_1_2_passing_yds": (CONT, ("QB",)),
    "period_first_touchdown_scored": (BINARY, ("QB", "RB", "FB", "WR", "TE")),
    "period_1_rush_rec_tds": (COUNT, ("QB", "RB", "FB", "WR", "TE")),
    "period_1_2_rush_rec_tds": (COUNT, ("QB", "RB", "FB", "WR", "TE")),
}
STATS.update(FALLBACK_STATS)

N_GAMES = 12        # games of form
DECAY = 0.85        # weight per game further back
LAST_SEASON = 0.6   # extra weight factor on last season's games
MIN_GAMES = 2       # fewer real games than this -> no prediction
# Spreads are fitted on every game from FIT_FROM_SEASON up to (not including) the week
# being predicted, so 2026 games count as soon as they are played (2026-10-23; before,
# 2025 only, which missed 2026's higher passing volume -- QB attempts / completions /
# yards ran ~8% above form) and every graded week stays out of its own fit.
FIT_FROM_SEASON = 2025
# A QB game with fewer attempts than this (injury exit, benching, kneel-down cameo)
# isn't his form and is skipped, as the trained models do (PR #11). Without it the form
# model took the Under on 60%+ of QB props and hit 43% on attempts / completions.
QB_MIN_ATT = 10
N_BINS = 8
# Logit multiplier on the final probability. Form alone is overconfident against real
# lines. 0.65 was fitted on 2026 week 1 lines only; re-checked 2026-10-23 on weeks 1
# and 4-6 (9,259 graded props), 0.5 has lower log loss on every week but week 4, where
# it ties (0.6141 -> 0.6085 overall). It only changes how sure a pick is, never its side.
TEMPERATURE = 0.5


def _units(stat: str) -> float:
    """Sacks come in halves (lines 0.25 / 0.5 / 0.75): model them as half-sack counts."""
    return 2.0 if stat == "sacks" else 1.0


def load_logs() -> pd.DataFrame:
    d = pd.read_csv(LOGS_PATH, low_memory=False)
    return d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def form_games(logs: pd.DataFrame) -> pd.DataFrame:
    """The games form is built from: QB cameo games (< QB_MIN_ATT attempts) dropped."""
    return logs[~((logs["position"] == "QB") & (logs["passing_att"].fillna(0) < QB_MIN_ATT))]


def _with_targets(logs: pd.DataFrame, targets: pd.DataFrame | None) -> pd.DataFrame:
    """logs plus one empty 'target' row per (player, season, week) to forecast."""
    d = logs.assign(_target=False)
    if targets is not None and not targets.empty:
        d = pd.concat([d, targets.assign(_target=True)], ignore_index=True)
    return d.sort_values(["player_id", "season", "week", "_target"]).reset_index(drop=True)


def form(logs: pd.DataFrame, stats: list[str], targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recency-weighted mean of each stat over each row's PREVIOUS games (never the row's
    own game). Returns the input rows (logs +
    targets) with `<stat>__mu` and `n_prior` columns."""
    d = _with_targets(logs, targets)
    g = d.groupby("player_id", sort=False)
    num = {s: 0.0 for s in stats}
    den = 0.0
    n_prior = 0
    for lag in range(1, N_GAMES + 1):
        sh = g.shift(lag)
        # Lagged rows must be real games (not other target rows).
        real = sh["_target"].eq(False)
        w = DECAY ** (lag - 1) * np.where(sh["season"] < d["season"], LAST_SEASON, 1.0)
        w = np.where(real, w, 0.0)
        den = den + w
        n_prior = n_prior + real.astype(int)
        for s in stats:
            num[s] = num[s] + w * sh[s].fillna(0).to_numpy(float)
    safe = np.where(den > 0, den, np.nan)
    mus = {f"{s}__mu": num[s] / safe for s in stats}
    return pd.concat([d, pd.DataFrame(mus, index=d.index).assign(n_prior=n_prior)], axis=1)


def _nb_p_over(mu, r, line):
    """P(X > line) for a negative binomial with mean mu and dispersion r."""
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    return sps.nbinom.sf(np.floor(line), r, r / (r + mu))


def _fit_nb(mu, y) -> float:
    rs = np.exp(np.linspace(np.log(0.3), np.log(500), 60))
    mu = np.maximum(mu, 1e-6)
    ll = [sps.nbinom.logpmf(y, r, r / (r + mu)).sum() for r in rs]
    return float(rs[int(np.argmax(ll))])


def fit(logs: pd.DataFrame | None = None, before: tuple[int, int] | None = None) -> dict:
    """Fit every stat's spread on games from FIT_FROM_SEASON up to `before` (season,
    week; exclusive -- default: every game), with form from the games before each.
    QB cameo games are left out of the fit too: Underdog only posts lines for the
    expected starter, and fitting on cameos widened the spread toward the Under (QB
    completions 45% vs 53% on real lines)."""
    logs = load_logs() if logs is None else logs
    stats = list(STATS)
    d = form(form_games(logs), stats)
    d = d[(d["season"] >= FIT_FROM_SEASON) & ~d["_target"] & (d["n_prior"] >= MIN_GAMES)]
    if before is not None:
        d = d[(d["season"] < before[0]) | ((d["season"] == before[0]) & (d["week"] < before[1]))]
    out = {}
    for s, (kind, pos) in STATS.items():
        x = d[d["position"].isin(pos)]
        mu, y = x[f"{s}__mu"].to_numpy(float), x[s].to_numpy(float)
        if len(x) < 200:
            continue
        if kind == COUNT:
            u = _units(s)
            out[s] = {"kind": kind, "r": _fit_nb(mu * u, np.round(y * u)), "n": int(len(x))}
        elif kind == CONT:
            edges = np.unique(np.quantile(mu, np.linspace(0, 1, N_BINS + 1)[1:-1]))
            b = np.digitize(mu, edges)
            pools = []
            for i in range(len(edges) + 1):
                ratio = y[b == i] / np.maximum(mu[b == i], 1e-6)
                pools.append(np.round(np.quantile(ratio, np.linspace(0.005, 0.995, 100)), 4).tolist()
                             if len(ratio) else [1.0])
            out[s] = {"kind": kind, "edges": edges.round(4).tolist(), "ratios": pools, "n": int(len(x))}
        else:
            z = np.log(np.clip(mu, 1e-3, 0.999) / (1 - np.clip(mu, 1e-3, 0.999)))
            yy = (y > 0).astype(float)
            best, best_ll = (0.0, 1.0), np.inf
            for a in np.linspace(-4, 2, 61):
                for bb in np.linspace(0.05, 2.0, 40):
                    p = np.clip(1 / (1 + np.exp(-(a + bb * z))), 1e-4, 1 - 1e-4)
                    ll = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
                    if ll < best_ll:
                        best, best_ll = (float(a), float(bb)), ll
            out[s] = {"kind": kind, "a": best[0], "b": best[1], "n": int(len(x))}
    return out


def prob_over(stat: str, mu, line, dists: dict) -> np.ndarray:
    """P(stat > line) for expected values `mu` (arrays broadcast), as shown in the app."""
    p = np.clip(_raw_prob_over(stat, mu, line, dists), 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-TEMPERATURE * np.log(p / (1 - p))))


def _raw_prob_over(stat: str, mu, line, dists: dict) -> np.ndarray:
    p = dists[stat]
    mu = np.asarray(mu, float)
    line = np.broadcast_to(np.asarray(line, float), mu.shape)
    if p["kind"] == COUNT:
        u = _units(stat)
        return _nb_p_over(mu * u, p["r"], line * u)
    if p["kind"] == CONT:
        edges = np.asarray(p["edges"])
        b = np.digitize(mu, edges)
        out = np.empty(mu.shape)
        for i in np.unique(b):
            m = b == i
            ratios = np.asarray(p["ratios"][int(i)])
            out[m] = (mu[m][:, None] * ratios[None, :] > line[m][:, None]).mean(axis=1)
        return np.clip(out, 0.01, 0.99)
    z = np.log(np.clip(mu, 1e-3, 0.999) / (1 - np.clip(mu, 1e-3, 0.999)))
    return 1 / (1 + np.exp(-(p["a"] + p["b"] * z)))


def form_as_of(season: int, week: int, logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each player's expected value per stat for (season, week), from games before it.
    Long format: player_id, player_display_name, position, team, stat_name, mu, n_prior."""
    logs = load_logs() if logs is None else logs
    prior = logs[(logs["season"] < season) | ((logs["season"] == season) & (logs["week"] < week))]
    last = prior.groupby("player_id").tail(1)
    prior = form_games(prior)
    # Only players active within the last season.
    last = last[last["season"] >= season - 1]
    targets = last[["player_id", "player_display_name", "position", "team"]].assign(season=season, week=week)
    d = form(prior, list(STATS), targets)
    d = d[d["_target"] & (d["n_prior"] >= MIN_GAMES)]
    rows = []
    for s, (_, pos) in STATS.items():
        x = d[d["position"].isin(pos)]
        rows.append(pd.DataFrame({"player_id": x["player_id"], "player_display_name": x["player_display_name"],
                                  "position": x["position"], "team": x["team"], "stat_name": s,
                                  "mu": x[f"{s}__mu"].round(4), "n_prior": x["n_prior"]}))
    return pd.concat(rows, ignore_index=True)


def load_dists() -> dict:
    try:
        with open(DISTS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    logs = load_logs()
    dists = fit(logs)
    with open(DISTS_PATH, "w") as f:
        json.dump(dists, f)
    print(f"Fitted {len(dists)} stat types on {FIT_FROM_SEASON}+ games -> {DISTS_PATH}")
    cur = logs[logs["season"] == logs["season"].max()]
    season, week = int(cur["season"].max()), int(cur["week"].max()) + 1
    f = form_as_of(season, week, logs)
    f.to_csv(FORM_PATH, index=False)
    print(f"Form as of {season} wk{week}: {len(f)} player-stat rows -> {FORM_PATH}")


if __name__ == "__main__":
    main()
