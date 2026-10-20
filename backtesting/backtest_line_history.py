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


def closing_lines_from_history(weeks_played: list[int]) -> pd.DataFrame:
    h = pd.read_csv(os.path.join(B.RAW, "underdog_line_history.csv"))
    h = h[h["stat_name"].isin(B.ACTUALS.keys())].copy()
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
        out["by_stat"][stat] = {"n": n, "accuracy": acc, "baseline": baseline,
                                "over_rate": over_rate, "verdict": verdict}
    out["overall"] = {"n": int(len(d)), "accuracy": float(d["hit"].mean()) if len(d) else None}
    return out


def main():
    weekly = pd.read_csv(os.path.join(B.RAW, "weekly_stats.csv"), low_memory=False)
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
    keep = ["gated", "season", "week", "_team", "full_name", "stat_name", "stat_value", "choice",
            "model_prob_over", "real_line_used", "proxy_line", "prop_type", "actual", "went_over"]
    g[[c for c in keep if c in g.columns]].to_csv(OUT_CSV, index=False)
    rec = track_record(g)
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Weeks: {rec['weeks']}  |  graded props: {rec['overall']['n']}  "
          f"overall accuracy {rec['overall']['accuracy']*100:.1f}%")
    for stat, r in rec["by_stat"].items():
        print(f"  {stat:16} n={r['n']:>4}  model {r['accuracy']*100:5.1f}%  "
              f"baseline {r['baseline']*100:5.1f}%  -> {r['verdict']}")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
