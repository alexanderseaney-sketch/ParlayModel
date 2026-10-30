"""
Measures how two Underdog picks in the SAME GAME move together (added 2026-10-20), for
same game parlays and for pricing any entry that has two legs from one game.

Replaces the narrower analyze_parlay_correlations.py table (8 same-team pairs, 4 yardage
props, needs pbp) with one built straight from nflverse weekly stats, so it:
  - covers every stat the app bets: passing yds/TDs/INTs, rushing yds, receiving yds,
    receptions and rush + rec TDs, keyed by Underdog's own stat names;
  - measures OPPOSING-team pairs too (QB vs QB in a shootout, a QB vs the other side's
    RB in a blowout), not just teammates;
  - skips same-player pairs: Underdog allows one pick per player per entry.

Each player-game gets a line like Underdog's: half a point off the player's trailing
average over his previous (up to 8) games, and "over" = beat that line. Players without
3 prior games, or below the volume Underdog posts lines for, are left out.

The correlation of two over/under outcomes is the phi coefficient. A pair is kept when
it has 300+ instances and |phi| clears twice its standard error (~1/sqrt(n)); weaker
pairs are treated as independent. Fit on 2019-2024, then checked on 2025 (held out):
the script prints how far the adjusted joint hit rate lands from the real one vs the
independent product. The saved table is refit on every season through the latest.

Defensive props (tackles, assists) aren't in weekly_stats.csv, so their links come from the
real Underdog lines the app has graded (backtesting/line_history_graded.csv), keyed by
"DEF <stat>" since Underdog's defensive positions (LB, ILB, SAF, CB, DT ...) are too thin
to split. Every pair from one team-game shares that game, so a link is kept only when a
bootstrap over team-games puts its 95% interval clear of zero. Teammates' assists go
over together (+0.10): how freely a game's stat crew hands out assists moves every
defender's number at once, and the 10/29 TNF test lost four assists Overs together.

Output: models/leg_correlations.csv (leg_a, leg_b, relation, n, phi), read by
models/parlay_calculator.py. Run: python models/build_leg_correlations.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEEKLY_PATH = os.path.join(ROOT, "data", "raw", "weekly_stats.csv")
GRADED_PATH = os.path.join(ROOT, "backtesting", "line_history_graded.csv")
OUT_PATH = os.path.join(ROOT, "models", "leg_correlations.csv")

FIRST_SEASON = 2019
HOLDOUT_SEASON = 2025
MIN_PAIRS = 300
LOOKBACK = 8
MIN_PRIOR_GAMES = 3

# Underdog stat name -> (positions, how to compute it from weekly stats, minimum trailing
# average for Underdog to plausibly post a line).
STATS = {
    "passing_yds": (["QB"], lambda w: w["passing_yards"], 150),
    "passing_tds": (["QB"], lambda w: w["passing_tds"], 0.8),
    "passing_ints": (["QB"], lambda w: w["interceptions"], 0.3),
    "rushing_yds": (["QB", "RB"], lambda w: w["rushing_yards"], 20),
    "receiving_yds": (["WR", "TE", "RB"], lambda w: w["receiving_yards"], 20),
    "receiving_rec": (["WR", "TE", "RB"], lambda w: w["receptions"], 2),
    "rush_rec_tds": (["RB", "WR", "TE"], lambda w: w["rushing_tds"] + w["receiving_tds"], 0.2),
}
COUNT_STATS = {"passing_tds", "passing_ints", "receiving_rec", "rush_rec_tds"}
DEF_STATS = ["tackles", "assists", "tackles_and_assists"]
BOOTSTRAPS = 500


def leg_key(position: str, stat_name: str) -> str:
    return f"{position} {stat_name}"


def build_outcomes(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, game, stat) with the over/under outcome against a
    trailing-average line."""
    w = weekly[(weekly["season_type"] == "REG") & (weekly["season"] >= FIRST_SEASON - 1)].copy()
    w = w.sort_values(["player_id", "season", "week"])
    frames = []
    for stat, (positions, value_fn, min_avg) in STATS.items():
        s = w[w["position"].isin(positions)].copy()
        s["value"] = value_fn(s).fillna(0).astype(float)
        g = s.groupby("player_id")["value"]
        s["prior_avg"] = g.transform(lambda v: v.shift(1).rolling(LOOKBACK, min_periods=MIN_PRIOR_GAMES).mean())
        s = s[s["prior_avg"].notna() & (s["prior_avg"] >= min_avg) & (s["season"] >= FIRST_SEASON)]
        if stat in COUNT_STATS:
            # Count props sit on half-points around the typical game: 0.5, 1.5, 4.5 ...
            s["line"] = np.maximum(np.round(s["prior_avg"]) - 0.5, 0.5)
        else:
            s["line"] = np.floor(s["prior_avg"]) + 0.5
        s["over"] = (s["value"] > s["line"]).astype(int)
        s["stat_name"] = stat
        frames.append(s[["player_id", "position", "recent_team", "opponent_team", "season", "week",
                         "stat_name", "over"]])
    out = pd.concat(frames, ignore_index=True)
    out["leg"] = out["position"] + " " + out["stat_name"]
    return out


def build_pairs(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Every pair of different players in the same game, labelled same_team or opponent,
    with the two legs in a canonical order."""
    o = outcomes[["player_id", "recent_team", "opponent_team", "season", "week", "leg", "over"]]
    same = o.merge(o, on=["season", "week", "recent_team"], suffixes=("_a", "_b"))
    same["relation"] = "same_team"
    opp = o.merge(o.drop(columns="opponent_team").rename(columns={"recent_team": "opponent_team"}),
                  on=["season", "week", "opponent_team"], suffixes=("_a", "_b"))
    opp["relation"] = "opponent"
    cols = ["season", "week", "relation", "player_id_a", "player_id_b", "leg_a", "leg_b", "over_a", "over_b"]
    p = pd.concat([same[cols], opp[cols]], ignore_index=True)
    p = p[p["player_id_a"] != p["player_id_b"]]
    # Two QBs on one team only happens when the starter gets hurt or benched mid-game;
    # nobody bets the backup, so those pairs would just be noise.
    p = p[~((p["relation"] == "same_team") & p["leg_a"].str.startswith("QB ") & p["leg_b"].str.startswith("QB "))]
    # Each unordered pair once: order by (leg, player) so (a, b) and (b, a) collapse.
    swap = (p["leg_a"] > p["leg_b"]) | ((p["leg_a"] == p["leg_b"]) & (p["player_id_a"] > p["player_id_b"]))
    for x in ("player_id", "leg", "over"):
        a, b = p.loc[swap, f"{x}_a"].copy(), p.loc[swap, f"{x}_b"].copy()
        p.loc[swap, f"{x}_a"], p.loc[swap, f"{x}_b"] = b, a
    return p.drop_duplicates(["season", "week", "relation", "player_id_a", "player_id_b", "leg_a", "leg_b"])


def phi_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (leg_a, leg_b, rel), g in pairs.groupby(["leg_a", "leg_b", "relation"]):
        n = len(g)
        if n < MIN_PAIRS:
            continue
        a, b = g["over_a"].to_numpy(), g["over_b"].to_numpy()
        pa, pb, pj = a.mean(), b.mean(), (a & b).mean()
        denom = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
        if denom <= 0:
            continue
        phi = (pj - pa * pb) / denom
        if abs(phi) < 2 / np.sqrt(n):
            continue
        rows.append({"leg_a": leg_a, "leg_b": leg_b, "relation": rel, "n": n, "phi": round(float(phi), 4)})
    return pd.DataFrame(rows).sort_values("phi", key=abs, ascending=False).reset_index(drop=True)


def holdout_check(pairs: pd.DataFrame, table: pd.DataFrame) -> None:
    """For each pair type in the held-out season: predicted joint hit rate (independent
    vs correlation-adjusted, using that season's own single-leg hit rates) against what
    actually happened. Weighted mean absolute error, in percentage points."""
    test = pairs[pairs["season"] == HOLDOUT_SEASON]
    lookup = {(r.leg_a, r.leg_b, r.relation): r.phi for r in table.itertuples()}
    err_ind, err_adj, total = 0.0, 0.0, 0
    for (leg_a, leg_b, rel), g in test.groupby(["leg_a", "leg_b", "relation"]):
        phi = lookup.get((leg_a, leg_b, rel))
        if phi is None or len(g) < 50:
            continue
        a, b = g["over_a"].to_numpy(), g["over_b"].to_numpy()
        pa, pb, actual = a.mean(), b.mean(), (a & b).mean()
        ind = pa * pb
        adj = ind + phi * np.sqrt(pa * (1 - pa) * pb * (1 - pb))
        err_ind += abs(ind - actual) * len(g)
        err_adj += abs(adj - actual) * len(g)
        total += len(g)
    if total:
        print(f"Held-out {HOLDOUT_SEASON}, {total:,} linked pairs: joint hit rate off by "
              f"{err_ind / total * 100:.2f} pts if treated as independent vs "
              f"{err_adj / total * 100:.2f} pts with the measured links.")


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = a.mean(), b.mean()
    denom = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
    return float(((a & b).mean() - pa * pb) / denom) if denom > 0 else 0.0


def real_line_defense_links(path: str = GRADED_PATH) -> pd.DataFrame:
    """Same-team links between defensive props, measured against the real lines."""
    if not os.path.exists(path):
        return pd.DataFrame()
    g = pd.read_csv(path, low_memory=False)
    g = g[g["stat_name"].isin(DEF_STATS) & g["actual"].notna() & (g["actual"] != g["stat_value"])]
    g = g.drop_duplicates(["season", "week", "full_name", "stat_name"])
    rng = np.random.default_rng(0)
    rows = []
    for i, sa in enumerate(DEF_STATS):
        for sb in DEF_STATS[i:]:
            a = g[g["stat_name"] == sa][["season", "week", "_team", "full_name", "went_over"]]
            b = g[g["stat_name"] == sb][["season", "week", "_team", "full_name", "went_over"]]
            p = a.merge(b, on=["season", "week", "_team"])
            p = p[p["full_name_x"] < p["full_name_y"]] if sa == sb else p[p["full_name_x"] != p["full_name_y"]]
            if len(p) < MIN_PAIRS:
                continue
            x, y = p["went_over_x"].to_numpy().astype(int), p["went_over_y"].to_numpy().astype(int)
            phi = _phi(x, y)
            groups = list(p.groupby(["season", "week", "_team"]).indices.values())
            boot = []
            for _ in range(BOOTSTRAPS):
                idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                boot.append(_phi(x[idx], y[idx]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            if lo > 0 or hi < 0:
                rows.append({"leg_a": f"DEF {sa}", "leg_b": f"DEF {sb}", "relation": "same_team",
                             "n": len(p), "phi": round(phi, 4)})
    return pd.DataFrame(rows)


def main():
    weekly = pd.read_csv(WEEKLY_PATH, low_memory=False)
    outcomes = build_outcomes(weekly)
    print(f"{len(outcomes):,} player-game-stat outcomes, {FIRST_SEASON}-{int(outcomes['season'].max())}.")
    pairs = build_pairs(outcomes)
    print(f"{len(pairs):,} same-game pairs of different players.")

    train = phi_table(pairs[pairs["season"] < HOLDOUT_SEASON])
    holdout_check(pairs, train)

    table = phi_table(pairs)
    defense = real_line_defense_links()
    print(f"{len(defense)} defensive link(s) from real lines:\n{defense.to_string(index=False)}")
    table = pd.concat([table, defense], ignore_index=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(table)} linked pair types -> {OUT_PATH}")
    print(table.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
