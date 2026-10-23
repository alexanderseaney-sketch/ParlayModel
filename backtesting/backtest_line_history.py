"""
Per-stat track record against REAL Underdog closing lines, from the line-move log
(added 2026-10-20).

backtest_underdog_lines.py needs full board snapshots (data/raw/underdog_history/,
which only lives on the machine that pulled them and has gaps -- 2026 wk2-3 are
missing), so it graded just 2 weeks. data/raw/underdog_line_history.csv is committed
and records every line open/move since 2026-08-16, so it covers every week -- but
carries no prices. That's enough to answer the question the app's Picks screen needs:
for each stat, does the model pick the right side of the real closing line more often
than the dumb baseline (always taking whichever side hit more often)?

Same machinery as backtest_underdog_lines.py: closing line = last line seen before
the player's own kickoff, point-in-time predictions as of that week, the dashboard's
own gates (`gated`).

Writes backtesting/line_history_track_record.json (read by the Picks screen) and
backtesting/line_history_graded.csv.

Usage:  python backtesting/backtest_line_history.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_underdog_lines as B  # noqa: E402

OUT_CSV = os.path.join(B.ROOT, "backtesting", "line_history_graded.csv")
OUT_JSON = os.path.join(B.ROOT, "backtesting", "line_history_track_record.json")

# A stat is "trusted" when the model beats the always-take-the-commoner-side baseline
# by at least this much on at least MIN_N graded props; "avoid" when it is below a
# coin flip. Anything else is "unproven".
TRUST_MARGIN = 0.03
MIN_N = 30


def closing_lines_from_history(weeks_played: list[int], stats=None) -> pd.DataFrame:
    h = pd.read_csv(os.path.join(B.RAW, "underdog_line_history.csv"))
    # A "pulled" row is the line coming OFF the board, not a line you could bet.
    h = h[h["stat_name"].isin(list(stats or B.ACTUALS.keys())) & (h["event"] != "pulled")].copy()
    h["snapshot_et"] = (pd.to_datetime(h["seen_at"], utc=True)
                        .dt.tz_convert("US/Eastern").dt.tz_localize(None))
    weeks = B._target_weeks(B.SCHEDULES)

    def target(ts):
        up = weeks[weeks["max"] + pd.Timedelta(days=1) > ts]
        if up.empty:
            return (np.nan, np.nan)
        t = up.iloc[0]
        if ts < t["min"] - pd.Timedelta(days=7):
            return (np.nan, np.nan)
        return (t["season"], t["week"])

    tgt = {ts: target(ts) for ts in h["snapshot_et"].drop_duplicates()}
    h["season"] = h["snapshot_et"].map(lambda t: tgt[t][0])
    h["week"] = h["snapshot_et"].map(lambda t: tgt[t][1])
    h = h.dropna(subset=["season"]).rename(columns={"line": "stat_value"})
    h[["season", "week"]] = h[["season", "week"]].astype(int)
    h = h[h["week"].isin(weeks_played)]
    # Prices aren't logged, so the market columns stay empty; grade() only needs a row
    # per side to score.
    sides = []
    for choice in ("over", "under"):
        s = h.copy()
        s["choice"] = choice
        sides.append(s)
    lines = pd.concat(sides, ignore_index=True)
    for c in ("decimal", "market_fair_prob", "market_fair_prob_over"):
        lines[c] = np.nan
    lines["snapshot"] = "line_history"
    return lines


def grade_extra(weeks_played: list[int]) -> pd.DataFrame:
    """Grade models/extra_prop_models.py (every prop type the trained models don't
    cover, plus its fallback for the ones they do) against the same closing lines.
    Point-in-time: each week's expected values use only earlier games, and the spreads
    are refitted on the games before that week, so nothing here was tuned on these
    lines. Same output columns as B.grade (gated is always True: the real line is used
    directly, no proxy to drift from)."""
    import extra_prop_models as X
    from utils import normalize_name

    stats = list(X.STATS)
    lines = closing_lines_from_history(weeks_played, stats)
    lines = lines[lines["choice"] == "over"]
    if lines.empty:
        return pd.DataFrame()
    logs = X.load_logs()
    logs["_k"] = logs["player_display_name"].apply(normalize_name)
    lines["_k"] = lines["full_name"].apply(normalize_name)
    ko = B._kickoffs(B.SCHEDULES)
    out = []
    for (season, week), wk in lines.groupby(["season", "week"]):
        dists = X.fit(logs, before=(season, week))
        f = X.form_as_of(season, week, logs)
        f["_k"] = f["player_display_name"].apply(normalize_name)
        act = logs[(logs["season"] == season) & (logs["week"] == week)].drop_duplicates("_k")
        act_long = act.melt(id_vars=["_k", "team"], value_vars=stats, var_name="stat_name", value_name="actual")
        b = wk.merge(f[["_k", "stat_name", "mu", "position"]], on=["_k", "stat_name"], how="inner")
        b = b.merge(act_long.rename(columns={"team": "_team"}), on=["_k", "stat_name"], how="inner")
        k = ko[(ko["season"] == season) & (ko["week"] == week)][["team", "kickoff"]]
        b = b.merge(k.rename(columns={"team": "_team"}), on="_team", how="left")
        b = b[b["snapshot_et"] < b["kickoff"]].sort_values("snapshot_et") \
            .drop_duplicates(["_k", "stat_name"], keep="last")
        if b.empty:
            continue
        b["model_prob_over"] = np.nan
        for s, x in b[b["stat_name"].isin(list(dists))].groupby("stat_name"):
            b.loc[x.index, "model_prob_over"] = X.prob_over(s, x["mu"].to_numpy(), x["stat_value"].to_numpy(), dists)
        out.append(b)
    if not out:
        return pd.DataFrame()
    g = pd.concat(out, ignore_index=True)
    g = g[g["actual"].notna() & (g["actual"] != g["stat_value"])].copy()
    g["went_over"] = (g["actual"] > g["stat_value"]).astype(int)
    g["gated"] = True
    g["real_line_used"] = True
    g["proxy_line"] = g["stat_value"]
    g["prop_type"] = "form_" + g["stat_name"]
    return g


def _model_label(prop_type: pd.Series) -> str:
    form = prop_type.astype(str).str.startswith("form_")
    return "form" if form.all() else "trained" if not form.any() else "trained+form"


def track_record(g: pd.DataFrame) -> dict:
    d = g[g["gated"] & (g["choice"] == "over") & g["model_prob_over"].notna()].copy()
    d["hit"] = (d["model_prob_over"] > 0.5) == (d["went_over"] == 1)
    out = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
           "weeks": sorted(f"{int(s)}-wk{int(w)}" for s, w in d[["season", "week"]].drop_duplicates().values),
           "by_stat": {}}
    for stat, x in d.groupby("stat_name"):
        n = len(x)
        acc = float(x["hit"].mean())
        over_rate = float(x["went_over"].mean())
        baseline = max(over_rate, 1 - over_rate)
        if acc < 0.5:
            verdict = "avoid"
        elif n >= MIN_N and acc - baseline >= TRUST_MARGIN:
            verdict = "trusted"
        else:
            verdict = "unproven"
        picks_over = x["model_prob_over"] > 0.5
        out["by_stat"][stat] = {"n": n, "accuracy": acc, "baseline": baseline,
                                "over_rate": over_rate, "verdict": verdict,
                                "model": _model_label(x["prop_type"]),
                                "picked_over": int(picks_over.sum()), "picked_under": int((~picks_over).sum()),
                                "over_pick_accuracy": float(x.loc[picks_over, "hit"].mean()) if picks_over.any() else None,
                                "under_pick_accuracy": float(x.loc[~picks_over, "hit"].mean()) if (~picks_over).any() else None}
    po = d["model_prob_over"] > 0.5
    out["overall"] = {"n": int(len(d)), "accuracy": float(d["hit"].mean()) if len(d) else None,
                      "picked_over": int(po.sum()), "picked_under": int((~po).sum()),
                      "over_pick_accuracy": float(d.loc[po, "hit"].mean()) if po.any() else None,
                      "under_pick_accuracy": float(d.loc[~po, "hit"].mean()) if (~po).any() else None}
    return out


def refit_real_line_shift(g: pd.DataFrame) -> None:
    """Refit calibration.adjust_over_prob's logit shift for the trained models on every
    graded week (2026-10-20). It used to be fitted only by backtest_underdog_lines.py,
    on the two weeks of full snapshots it has, and at a time when receptions were still
    scored against the proxy line (the receiving_rec bug fixed in PR #1). That stale
    +0.19 shift pushed every pick toward the Over and cost the trusted stats 2-7
    points of accuracy on real lines. Fitted on the raw (unshifted) probabilities;
    each week's fit on the OTHER weeks is reported as the held-out check."""
    from calibration import REAL_LINE_ADJ_PATH, REAL_LINE_MAX_SHIFT, log_loss
    if not os.path.exists(os.path.join(B.RAW, "pbp.csv")):
        print("[real-line shift] no pbp.csv, so the TD models didn't grade -- not refitted.")
        return
    d = g[g["gated"] & (g["choice"] == "over") & g["model_prob_over"].notna()]
    if len(d) < 100:
        print("[real-line shift] fewer than 100 graded props -- not refitted.")
        return
    cs = np.linspace(-REAL_LINE_MAX_SHIFT, REAL_LINE_MAX_SHIFT, 101)

    def best(x):
        m = np.clip(x["model_prob_over"].to_numpy(float), 1e-4, 1 - 1e-4)
        z, y = np.log(m / (1 - m)), x["went_over"].to_numpy()
        return float(cs[int(np.argmin([log_loss(1 / (1 + np.exp(-(z + c))), y) for c in cs]))])

    wk = d["season"].astype(str) + "-wk" + d["week"].astype(str)
    held_out = {w: best(d[wk != w]) for w in sorted(wk.unique())}
    out = {"logit_shift": best(d), "n_props": int(len(d)), "fit_without_week": held_out,
           "raw_mean_p_over": float(d["model_prob_over"].mean()),
           "actual_over_rate": float(d["went_over"].mean()),
           "source": "backtesting/backtest_line_history.py",
           "fitted_at": pd.Timestamp.now(tz="UTC").isoformat()}
    with open(REAL_LINE_ADJ_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[real-line shift] {out['logit_shift']:+.3f} on {len(d)} props "
          f"(raw mean P(over) {out['raw_mean_p_over']:.3f} vs actual {out['actual_over_rate']:.3f})")


# 1st quarter / 1st half props the trained models price. Their lines are archived from
# 2026-10-20 on (data/archive_underdog_lines.py), so they grade as soon as weeks land.
PERIOD_ACTUALS = ["period_1_receiving_yds", "period_1_2_receiving_yds", "period_1_rushing_yds",
                  "period_1_2_rushing_yds", "period_1_passing_yds", "period_1_2_passing_yds",
                  "period_1_rush_rec_tds", "period_1_2_rush_rec_tds"]


def _add_period_actuals(weekly: pd.DataFrame) -> pd.DataFrame:
    import extra_prop_models as X
    if not os.path.exists(X.LOGS_PATH):
        return weekly
    logs = pd.read_csv(X.LOGS_PATH, usecols=lambda c: c in ["player_id", "season", "week"] + PERIOD_ACTUALS)
    weekly = weekly.merge(logs.drop_duplicates(["player_id", "season", "week"]),
                          on=["player_id", "season", "week"], how="left")
    for c in [c for c in PERIOD_ACTUALS if c in logs.columns]:
        B.ACTUALS.setdefault(c, lambda w, c=c: w[c])
    return weekly


def main():
    weekly = _add_period_actuals(pd.read_csv(os.path.join(B.RAW, "weekly_stats.csv"), low_memory=False))
    season = int(weekly["season"].max())
    played = sorted(weekly.loc[weekly["season"] == season, "week"].unique().tolist())
    lines = closing_lines_from_history(played)
    if lines.empty:
        print("No line history for played weeks.")
        return
    g = B.grade(lines, weekly)
    if g.empty:
        print("Nothing gradable.")
        return
    refit_real_line_shift(g)
    # Grade what the app shows: the trained models' probabilities with that shift.
    from calibration import adjust_over_prob
    has = g["model_prob_over"].notna()
    g.loc[has, "model_prob_over"] = adjust_over_prob(g.loc[has, "model_prob_over"].astype(float))
    extra = grade_extra(played)
    if not extra.empty:
        extra["choice"] = "over"
        # Fallback stats: the form model's number only counts where the app shows it --
        # props the trained model left empty or didn't pass the Picks gate on
        # (utils.score_extra_props).
        import extra_prop_models as X
        from utils import normalize_name
        key = lambda d: d["season"].astype(str) + "|" + d["week"].astype(str) + "|" \
            + d["full_name"].apply(normalize_name) + "|" + d["stat_name"]  # noqa: E731
        shown = g[g["gated"] & (g["choice"] == "over") & g["model_prob_over"].notna()]
        taken = set(key(shown)) if not shown.empty else set()
        from utils import is_low_noise_line
        fb = extra["stat_name"].isin(list(X.FALLBACK_STATS))
        low = pd.Series([is_low_noise_line(s, v) for s, v in zip(extra["stat_name"], extra["stat_value"])],
                        index=extra.index)
        extra = extra[~(fb & (key(extra).isin(taken) | low))]
        g = pd.concat([g, extra], ignore_index=True)
    keep = ["gated", "season", "week", "_team", "full_name", "stat_name", "stat_value", "choice",
            "model_prob_over", "real_line_used", "proxy_line", "prop_type", "actual", "went_over"]
    g[[c for c in keep if c in g.columns]].to_csv(OUT_CSV, index=False)
    rec = track_record(g)
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Weeks: {rec['weeks']}  |  graded props: {rec['overall']['n']}  "
          f"overall accuracy {rec['overall']['accuracy']*100:.1f}%")
    for stat, r in rec["by_stat"].items():
        print(f"  {stat:26} n={r['n']:>4}  model {r['accuracy']*100:5.1f}%  "
              f"baseline {r['baseline']*100:5.1f}%  overs hit {r['over_rate']*100:5.1f}%  -> {r['verdict']}")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
