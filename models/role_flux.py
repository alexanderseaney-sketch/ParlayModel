"""
"Workload in flux": catch picks for a player whose last game was disrupted (2026-11-14).

Tested with every other kind of news and info the repo has history for (injury tags and
Friday practice status, body part, teammates newly out and who inherits their targets or
carries, a Questionable teammate sitting, QB changes, offensive line and opposing
defenders out, short weeks and byes, wind / rain / temperature forecasts, lines pulled
and reposted, how much a player is in the news) on real Underdog closing lines, 2026
weeks 4-9. Underdog's lines already carry almost all of it: none of those moved how
often props went Over, and the few that looked like they did (Questionable players going
Under, a WR going Under when another WR is newly out) flipped on the quarter / half
lines, which were held aside as a second test.

What did hold up is narrower, and it is about the model rather than the line. For a
WR / TE / RB whose own workload was just disrupted:

    snap dip   last game's offensive snap share 8+ points under the mean of the three
               games before it
    returning  missed his team's last game after playing earlier this season

the model's catch picks (receptions, targets, receiving yards; Over or Under) hit 65%
on full-game lines against 54% for everyone else, in all six weeks, and 61% against 56%
on the held-aside quarter / half catch lines. The two groups share no players and both
show it. Props went Over equally often in both (45%), so the line is not biased; these
are the players whose role is in question, and the model's read of their recent games
separates them better than usual. Picks this adds to the proven rule (model 55%+, not
already proven): 130 at 66.9% walk-forward on weeks 4-9, against 51.4% (n=405) for the
same picks on players with a steady workload. Rushing and fantasy-point props showed
nothing, so only catch props are flagged.

pick_rules counts a flagged pick as proven from 55% model confidence. The track
record's pick_rule_check grades those picks on their own (`workload_flux`): the split
was found on these same six weeks, so check that line after each new week.
"""
import numpy as np
import pandas as pd

STATS = {"receiving_rec", "receiving_tgts", "receiving_yds"}
POSITIONS = {"WR", "TE", "RB"}
DIP = 0.08       # last game's snap share this far below the three games before it
MIN_GAMES = 3    # games this season before either flag can fire


def workload_flux(snaps: pd.DataFrame, normalize, season: int, before_week: int | None = None) -> dict:
    """{normalized name: "snap dip" | "returning"} for WR / TE / RB, from this season's
    snap counts before `before_week` (default: every game played so far)."""
    if snaps is None or snaps.empty:
        return {}
    s = snaps[snaps["season"] == season]
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    if before_week is not None:
        s = s[s["week"] < before_week]
    team_last = s.groupby("team")["week"].max()
    s = s[s["position"].isin(list(POSITIONS)) & (s["offense_snaps"] > 0)]
    if s.empty:
        return {}
    s = s.assign(_k=s["player"].apply(normalize)).sort_values("week").drop_duplicates(["_k", "week"])
    prior = s.groupby("_k")["offense_pct"].transform(lambda x: x.shift(1).rolling(3, min_periods=MIN_GAMES - 1).mean())
    games = s.groupby("_k")["week"].transform("size")
    last = s.assign(_prior=prior, _games=games).groupby("_k").tail(1)
    last = last.assign(_team_last=last["team"].map(team_last))
    out = {k: "snap dip" for k in last.loc[(last["offense_pct"] - last["_prior"]) <= -DIP, "_k"]}
    out.update({k: "returning" for k in last.loc[(last["week"] < last["_team_last"]) & (last["_games"] >= MIN_GAMES - 1), "_k"]})
    return out


def flag(stat_name, key, flux: dict) -> np.ndarray:
    """True where the prop is a catch stat for a player in `flux`."""
    stat = pd.Series(stat_name).astype(str)
    return (stat.isin(list(STATS)) & pd.Series(key).isin(list(flux)).to_numpy()).to_numpy()
