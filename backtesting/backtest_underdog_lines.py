"""
Backtest against REAL Underdog lines (added 2026-10-02).

Every accuracy number in the README is measured against a proxy line (the player's
own rolling average), because no archive of real lines existed. The archive does
exist now: data/pull_underdog.py has been writing timestamped snapshots to
data/raw/underdog_history/ since 2026-08-16, and since the 2026 odds-based pick'em
each option carries its own price. So this script grades the model where it
actually matters:

  1. For each snapshot, the target week = the first REG-season week whose games are
     on/after the snapshot date. Only the LAST snapshot before each week is used
     (closest thing to a closing line). Preseason-only snapshots are skipped.
  2. Point-in-time model probabilities: current_predictions.build_current_predictions(
     as_of=(season, week)) rebuilds every player's form using only games before that
     week, then the same score_underdog_board() the dashboard uses converts each
     proxy-line probability to the real line -- with the variance estimate also
     restricted to pre-week games.
  3. Grades each single-game prop against weekly_stats.csv (no stat row = DNP = void,
     same as Underdog). Pushes (line == actual) are dropped.
  4. Reports, per stat and overall: log loss + Brier for the MODEL, the de-vigged
     MARKET, and a model/market BLEND; hit rate and flat-$1 ROI at Underdog's own
     decimal prices for picks where the model's edge over the market clears a bar.

Outputs backtesting/underdog_line_backtest.csv (graded rows) and
backtesting/underdog_line_backtest_summary.json (read by the Model Performance page).

Sample-size warning: the archive has a gap between 2026-09-09 and 2026-09-30, so
2026 weeks 2-3 have no lines -- run data/pull_underdog.py at least daily in season
or this backtest can't grow.

Usage:  python backtesting/backtest_underdog_lines.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))

from odds_utils import add_underdog_market_probs, blend_with_market, BLEND_WEIGHT_MODEL  # noqa: E402
from calibration import log_loss, brier  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
OUT_CSV = os.path.join(ROOT, "backtesting", "underdog_line_backtest.csv")
OUT_JSON = os.path.join(ROOT, "backtesting", "underdog_line_backtest_summary.json")
CACHE_DIR = os.path.join(ROOT, "backtesting", "_pit_cache")

# Underdog stat_name -> how to compute the actual from weekly_stats.csv
ACTUALS = {
    "receiving_yds": lambda w: w["receiving_yards"],
    "rushing_yds": lambda w: w["rushing_yards"],
    "passing_yds": lambda w: w["passing_yards"],
    "receiving_rec": lambda w: w["receptions"],
    "rush_rec_tds": lambda w: w["rushing_tds"].fillna(0) + w["receiving_tds"].fillna(0),
    "passing_tds": lambda w: w["passing_tds"],
    "passing_ints": lambda w: w["interceptions"],
}

EDGE_BARS = [0.0, 0.03, 0.05, 0.08]


def blend(model_p, market_p, w_model=BLEND_WEIGHT_MODEL):
    return blend_with_market(model_p, market_p, w_model)


def _snapshot_files() -> list[str]:
    files = sorted(glob.glob(os.path.join(RAW, "underdog_history", "underdog_props_*.csv")))
    cur = os.path.join(RAW, "underdog_props.csv")
    if os.path.exists(cur):
        files.append(cur)
    return files


def _target_weeks(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["game_type"] == "REG"] if "game_type" in schedules.columns else schedules
    wk = reg.groupby(["season", "week"])["gameday"].agg(["min", "max"]).reset_index()
    wk["min"] = pd.to_datetime(wk["min"])
    wk["max"] = pd.to_datetime(wk["max"])
    return wk.sort_values("min")


def collect_closing_lines(schedules: pd.DataFrame) -> pd.DataFrame:
    """Every snapshot pulled in the run-up to (or during) each REG week, tagged with its
    pull time in US/Eastern. grade() then keeps, per player/stat/side, the LAST line
    pulled before that player's own kickoff -- the closing line for his game -- so a
    Sunday game is graded at Saturday's lines, not at a Tuesday snapshot (2026-10-07;
    previously one snapshot per week, skipping any pulled after the week's first game)."""
    weeks = _target_weeks(schedules)
    by_week: dict[tuple, list] = {}
    for f in _snapshot_files():
        head = pd.read_csv(f, nrows=1, usecols=lambda c: c == "pulled_at")
        if head.empty:
            continue
        ts = pd.to_datetime(head["pulled_at"].iloc[0], utc=True).tz_convert("US/Eastern").tz_localize(None)
        upcoming = weeks[weeks["max"] + pd.Timedelta(days=1) > ts]
        if upcoming.empty:
            continue
        tgt = upcoming.iloc[0]
        if ts < tgt["min"] - pd.Timedelta(days=7):
            continue  # preseason / too early to be this week's board
        by_week.setdefault((int(tgt["season"]), int(tgt["week"])), []).append((ts, f))
    frames = []
    for (season, week), snaps in sorted(by_week.items()):
        for ts, f in snaps:
            df = pd.read_csv(f, low_memory=False)
            df = df[df["stat_name"].isin(ACTUALS.keys())].copy()
            if "match_type" in df.columns:
                df = df[df["match_type"].fillna("Game") == "Game"]
            df = add_underdog_market_probs(df)
            df["season"], df["week"], df["snapshot"], df["snapshot_et"] = season, week, os.path.basename(f), ts
            frames.append(df)
        print(f"[lines] {season} wk{week}: {len(snaps)} snapshot(s): " + ", ".join(os.path.basename(f)[15:30] for _, f in snaps))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _kickoffs(schedules: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> kickoff, US/Eastern naive (schedules' gametime is ET)."""
    s = schedules.copy()
    s["kickoff"] = pd.to_datetime(s["gameday"].astype(str) + " " + s["gametime"].fillna("13:00").astype(str),
                                  errors="coerce")
    home = s[["season", "week", "home_team", "kickoff"]].rename(columns={"home_team": "team"})
    away = s[["season", "week", "away_team", "kickoff"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True)


def point_in_time_predictions(season: int, week: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"preds_{season}_{week}.csv")
    if os.path.exists(path) and os.path.getmtime(path) > max(
            os.path.getmtime(p) for p in glob.glob(os.path.join(ROOT, "models", "*.pkl"))):
        return pd.read_csv(path)
    from current_predictions import build_current_predictions
    preds = build_current_predictions(as_of=(season, week))
    preds.to_csv(path, index=False)
    return preds


def grade(lines: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    from utils import score_underdog_board, normalize_name

    graded = []
    for (season, week), wk_lines in lines.groupby(["season", "week"]):
        preds = point_in_time_predictions(season, week)
        prior_stats = weekly[(weekly["season"] < season) | ((weekly["season"] == season) & (weekly["week"] < week))]
        # Raw (unadjusted) probabilities: the real-line adjustment is FITTED from these.
        board = score_underdog_board(wk_lines, preds, prior_stats, real_line_adjust=False)

        actual_rows = weekly[(weekly["season"] == season) & (weekly["week"] == week)].copy()
        actual_rows["_k"] = actual_rows["player_display_name"].apply(normalize_name)
        actual_rows = actual_rows.drop_duplicates("_k")
        for stat, fn in ACTUALS.items():
            actual_rows[f"_act_{stat}"] = fn(actual_rows)
        board = board.merge(actual_rows[["_k", "recent_team"] + [f"_act_{s}" for s in ACTUALS]].rename(
            columns={"recent_team": "_team"}), on="_k", how="inner", suffixes=("", "_act"))
        board["actual"] = board.apply(lambda r: r.get(f"_act_{r['stat_name']}"), axis=1)
        board = board.drop(columns=[f"_act_{s}" for s in ACTUALS])
        # Closing line per player: last snapshot pulled before HIS game's kickoff.
        ko = _kickoffs(SCHEDULES)
        ko = ko[(ko["season"] == season) & (ko["week"] == week)][["team", "kickoff"]]
        board = board.merge(ko.rename(columns={"team": "_team"}), on="_team", how="left")
        board = board[board["snapshot_et"] < board["kickoff"]]
        board = (board.sort_values("snapshot_et")
                 .drop_duplicates(["_k", "stat_name", "choice"], keep="last"))
        graded.append(board)
    if not graded:
        return pd.DataFrame()
    g = pd.concat(graded, ignore_index=True)
    g = g[g["actual"].notna() & (g["actual"] != g["stat_value"])].copy()
    choice = g["choice"].astype(str).str.lower()
    g["is_over"] = choice.isin(["over", "higher"])
    g["went_over"] = (g["actual"] > g["stat_value"]).astype(int)
    g["won"] = np.where(g["is_over"], g["went_over"] == 1, g["went_over"] == 0).astype(int)
    # The same gates the +EV Finder / Parlay Builder apply before showing a model
    # probability: meaningful lines only, and the real line close enough to the
    # model's proxy line for the proxy->real conversion to be trusted.
    from utils import is_low_noise_line, line_matches_proxy
    model_stat = g["stat_name"].replace({"receiving_rec": "receptions"})
    g["gated"] = g["has_model"] & ~g.apply(lambda r: is_low_noise_line(r["stat_name"], r["stat_value"]), axis=1) &         pd.Series([line_matches_proxy(v, pl, s) for v, pl, s in zip(g["stat_value"], g["proxy_line"], model_stat)],
                  index=g.index)
    return g


def summarise(g: pd.DataFrame) -> dict:
    """Over-side rows only for probability scoring (each prop counted once); both
    sides for the betting simulation (the model may like either). Only props that
    pass the dashboard's gates (`gated`) are scored. Also fits the best model weight
    for the model/market blend on these rows -- reported, not auto-applied, given the sample."""
    out = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "weeks": sorted(
        {f"{int(s)}-wk{int(w)}" for s, w in g[["season", "week"]].drop_duplicates().values}),
        "by_stat": {}, "betting": []}
    scored = g[g["gated"] & g["is_over"] & g["market_fair_prob_over"].notna()].copy()
    scored["blend_prob_over"] = blend(scored["model_prob_over"], scored["market_fair_prob_over"])

    def _metrics(d: pd.DataFrame) -> dict:
        y = d["went_over"].to_numpy()
        res = {"n": int(len(d)), "base_rate_over": float(y.mean()) if len(d) else None}
        for name, col in (("model", "model_prob_over"), ("market", "market_fair_prob_over"),
                          ("blend", "blend_prob_over")):
            res[f"{name}_log_loss"] = log_loss(d[col], y) if len(d) else None
            res[f"{name}_brier"] = brier(d[col], y) if len(d) else None
        res["model_accuracy"] = float(((d["model_prob_over"] > 0.5) == (y == 1)).mean()) if len(d) else None
        return res

    out["overall"] = _metrics(scored)
    if len(scored):
        ws = np.round(np.arange(0, 1.01, 0.05), 2)
        lls = [log_loss(blend(scored["model_prob_over"], scored["market_fair_prob_over"], w), scored["went_over"]) for w in ws]
        out["best_blend_weight_model"] = float(ws[int(np.argmin(lls))])
        out["blend_curve"] = [{"w_model": float(w), "log_loss": float(l)} for w, l in zip(ws, lls)]
    for stat, d in scored.groupby("stat_name"):
        out["by_stat"][stat] = _metrics(d)

    bets = g[g["gated"] & g["market_fair_prob"].notna() & g["decimal"].notna()].copy()
    bets["edge"] = bets["side_prob"] - bets["market_fair_prob"]
    for bar in EDGE_BARS:
        b = bets[bets["edge"] > bar]
        if b.empty:
            out["betting"].append({"min_edge": bar, "n": 0})
            continue
        profit = np.where(b["won"] == 1, b["decimal"] - 1, -1.0)
        out["betting"].append({"min_edge": bar, "n": int(len(b)), "hit_rate": float(b["won"].mean()),
                               "avg_price": float(b["decimal"].mean()),
                               "roi": float(profit.mean()), "units": float(profit.sum())})
    return out


SCHEDULES = pd.read_csv(os.path.join(RAW, "schedules.csv"))


def fit_real_line_adjustment(g: pd.DataFrame) -> None:
    """Fit the single logit shift that best calibrates the model's raw P(over) on every
    graded week (over-side rows, dashboard-gated), report how it does when fitted on
    the other weeks only, and save it for calibration.adjust_over_prob()."""
    from calibration import REAL_LINE_ADJ_PATH, REAL_LINE_MAX_SHIFT
    ov = g[g["gated"] & g["is_over"] & g["market_fair_prob_over"].notna()]
    if len(ov) < 100:
        print("[real-line adjustment] fewer than 100 graded props -- not refitted.")
        return
    cs = np.linspace(-REAL_LINE_MAX_SHIFT, REAL_LINE_MAX_SHIFT, 201)

    def best(d):
        m = np.clip(d["model_prob_over"].to_numpy(float), 1e-4, 1 - 1e-4)
        z, y = np.log(m / (1 - m)), d["went_over"].to_numpy()
        return float(cs[int(np.argmin([log_loss(1 / (1 + np.exp(-(z + c))), y) for c in cs]))])

    per_week = {f"{int(s)}-wk{int(w)}": best(d) for (s, w), d in ov.groupby(["season", "week"])}
    c = best(ov)
    out = {"logit_shift": c, "n_props": int(len(ov)), "per_week_fit": per_week,
           "raw_mean_p_over": float(ov["model_prob_over"].mean()), "actual_over_rate": float(ov["went_over"].mean()),
           "fitted_at": pd.Timestamp.now(tz="UTC").isoformat()}
    with open(REAL_LINE_ADJ_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[real-line adjustment] logit shift {c:+.3f} (per week {per_week}) on {len(ov)} props; "
          f"raw mean P(over) {out['raw_mean_p_over']:.3f} vs actual {out['actual_over_rate']:.3f} -> {REAL_LINE_ADJ_PATH}")


def main():
    schedules = SCHEDULES
    weekly = pd.read_csv(os.path.join(RAW, "weekly_stats.csv"), low_memory=False)
    lines = collect_closing_lines(schedules)
    if lines.empty:
        print("No usable snapshots.")
        return
    g = grade(lines, weekly)
    if g.empty:
        print("Nothing gradable yet (weeks not played?).")
        return
    keep = ["gated", "season", "week", "snapshot", "snapshot_et", "kickoff", "_team", "full_name", "stat_name", "stat_value", "choice", "decimal",
            "implied_prob", "market_fair_prob", "market_fair_prob_over", "has_model", "model_prob_over",
            "side_prob", "real_line_used", "proxy_line", "prop_type", "actual", "went_over", "won"]
    g[[c for c in keep if c in g.columns]].to_csv(OUT_CSV, index=False)
    summary = summarise(g)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    fit_real_line_adjustment(g)

    o = summary["overall"]
    print(f"\nGraded weeks: {summary['weeks']}  |  props scored (over side, model+market): {o['n']}")
    if o["n"]:
        print(f"{'':10}{'log loss':>10}{'Brier':>9}")
        for name in ("model", "market", "blend"):
            print(f"{name:10}{o[name + '_log_loss']:>10.4f}{o[name + '_brier']:>9.4f}")
        print(f"model accuracy (side with >50%): {o['model_accuracy']*100:.1f}%")
        print(f"best blend model weight on this sample: {summary['best_blend_weight_model']:.2f}")
    print("\nBy stat:")
    for stat, m in summary["by_stat"].items():
        print(f"  {stat:16} n={m['n']:>4}  model LL {m['model_log_loss']:.4f}  market LL {m['market_log_loss']:.4f}"
              f"  blend LL {m['blend_log_loss']:.4f}")
    print("\nFlat $1 singles at Underdog prices, model edge over de-vigged market:")
    for b in summary["betting"]:
        if b["n"]:
            print(f"  edge > {b['min_edge']:.2f}: n={b['n']:>4}  hit {b['hit_rate']*100:5.1f}%  "
                  f"avg price {b['avg_price']:.2f}  ROI {b['roi']*100:+.1f}%")
    print(f"\nSaved -> {OUT_CSV}\n      -> {OUT_JSON}")


if __name__ == "__main__":
    main()
