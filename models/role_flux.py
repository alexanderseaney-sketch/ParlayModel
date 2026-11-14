"""
"Snap dip" watch flag (added 2026-11-14). NOT a pick rule yet: it is graded every week
in the track record's pick_rule_check so it gets an honest out-of-sample test first.

On real closing lines, 2026 weeks 4-9, the model's catch picks (receptions, targets,
receiving yards; either side) for a WR / TE / RB whose offensive snap share in his last
game was 8+ points below his average over the three games before it hit 65.5% (n=342),
against 54.0% (n=1362) for the rest, and beat the rest in all six weeks. Overs and Unders
both: the lines themselves were not off (45% went Over in both groups), the model just
read these players better than usual. The same split on the quarter / half catch lines,
which the search never looked at, went the same way but much weaker (58.9% vs 55.9%),
and the split was found by looking at about a dozen cuts of the same six weeks, so the
true size is unknown. Rushing props showed nothing (carries 46% vs 56%).

Depth-chart moves (nflverse depth_charts: promoted / demoted since last week) and snap
share going UP did not separate hits from misses, and route participation can't be
tested: nflverse has no 2026 participation data.
"""
import numpy as np
import pandas as pd

STATS = {"receiving_rec", "receiving_tgts", "receiving_yds"}
POSITIONS = {"WR", "TE", "RB"}
DIP = 0.08       # last game's snap share this far below the three games before it
MIN_GAMES = 3    # games this season before the flag can fire


def snap_dips(snaps: pd.DataFrame, normalize, season: int, before_week: int | None = None) -> set:
    """Normalized names of WR / TE / RB whose last game's offensive snap share (this
    season, before `before_week` when given) dipped DIP or more below the mean of up to
    three games before it."""
    if snaps is None or snaps.empty:
        return set()
    s = snaps[(snaps["season"] == season) & snaps["position"].isin(list(POSITIONS))]
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    if before_week is not None:
        s = s[s["week"] < before_week]
    if s.empty:
        return set()
    s = s.assign(_k=s["player"].apply(normalize)).sort_values("week").drop_duplicates(["_k", "week"])
    g = s.groupby("_k")["offense_pct"]
    prior = g.transform(lambda x: x.shift(1).rolling(3, min_periods=MIN_GAMES - 1).mean())
    last = s.assign(_prior=prior).groupby("_k").tail(1)
    return set(last.loc[(last["offense_pct"] - last["_prior"]) <= -DIP, "_k"])


def flag(stat_name, key, dipped: set) -> np.ndarray:
    """True where the prop is a catch stat for a player in `dipped`."""
    stat = pd.Series(stat_name).astype(str)
    return (stat.isin(list(STATS)) & pd.Series(key).isin(list(dipped)).to_numpy()).to_numpy()
