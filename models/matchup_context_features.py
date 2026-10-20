"""
Matchup and injury context for the player-prop models (added 2026-10-20, after Alex
asked that matchup strength, injuries on both teams, game script and usage trends all
count). Every feature is known before kickoff and is built for every scheduled game,
played or not, so training rows and the upcoming game get the same numbers.

  opp_allowed_rolling   what this opponent's defense allowed in the SAME stat over its
                        last 8 games (pass yards / pass TDs / INTs / catches to WR+TE /
                        RB rushing yards), not the all-purpose EPA the models had
  team_spread           points this team is favoured by (game script: big favourites
                        run more late, big underdogs throw more)
  team_out_tgt_share    target share of teammates listed Out / Doubtful that week
                        (volume that has to go somewhere)
  team_out_carry_share  same for carries
  opp_def_regulars_out  opponent defenders with a 60%+ snap role listed Out / Doubtful
  usage_trend           last-3-game volume minus the longer average (role growing or
                        shrinking)

The injury report for a week is published Wednesday-Friday, so early in the week the
upcoming game's injury features are 0 (nobody listed yet) -- same as any game where
nobody is hurt.
"""
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_STATUSES = {"Out", "Doubtful"}
DEF_WINDOW = 8

# model stat column -> (weekly_stats column summed, positions whose production counts)
OPP_ALLOWED = {
    "passing_yards": ("passing_yards", ("QB",)),
    "passing_tds": ("passing_tds", ("QB",)),
    "interceptions": ("interceptions", ("QB",)),
    "receptions": ("receptions", ("WR", "TE")),
    "rushing_yards": ("rushing_yards", ("RB",)),
}


def _team_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season, week) for every scheduled game, with the opponent."""
    s = schedules[["season", "week", "home_team", "away_team", "spread_line"]]
    # nflverse spread_line > 0 = home team favoured; team_spread > 0 = THIS team favoured.
    home = s.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(
        team_spread=lambda d: d["spread_line"])
    away = s.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(
        team_spread=lambda d: -d["spread_line"])
    out = pd.concat([home, away], ignore_index=True).drop(columns="spread_line")
    return out.drop_duplicates(["team", "season", "week"]).sort_values(["season", "week"])


def build_opp_allowed(weekly: pd.DataFrame, schedules: pd.DataFrame, stat: str) -> pd.DataFrame:
    """(team, season, week) -> opp_allowed_rolling: the OPPONENT defense's average allowed
    of `stat` over its previous DEF_WINDOW games (never including this game)."""
    col, positions = OPP_ALLOWED[stat]
    w = weekly[weekly["position"].isin(positions)]
    allowed = (w.groupby(["opponent_team", "season", "week"])[col].sum()
               .rename("allowed").reset_index().rename(columns={"opponent_team": "defense"}))
    games = _team_games(schedules).rename(columns={"team": "defense", "opponent": "_vs"})
    d = games.merge(allowed, on=["defense", "season", "week"], how="left")
    played = d["season"].astype(int) * 100 + d["week"].astype(int)
    d = d.assign(_k=played).sort_values(["defense", "_k"])
    # Unplayed games have no "allowed" yet; rolling skips them and they sit at the end.
    d["def_rolling"] = (d.groupby("defense")["allowed"]
                        .transform(lambda x: x.shift(1).rolling(DEF_WINDOW, min_periods=3).mean()))
    games = _team_games(schedules)
    return games.merge(d[["defense", "season", "week", "def_rolling"]].rename(
        columns={"defense": "opponent", "def_rolling": "opp_allowed_rolling"}),
        on=["opponent", "season", "week"], how="left")[["team", "season", "week", "opp_allowed_rolling"]]


def _player_share_asof(weekly: pd.DataFrame) -> pd.DataFrame:
    """Each player's target share and carry share over his last 8 games before each week."""
    w = weekly[["player_id", "recent_team", "season", "week", "targets", "carries"]].copy()
    team = w.groupby(["recent_team", "season", "week"])[["targets", "carries"]].transform("sum")
    w["tgt_share"] = w["targets"].fillna(0) / team["targets"].replace(0, np.nan)
    w["car_share"] = w["carries"].fillna(0) / team["carries"].replace(0, np.nan)
    w = w.sort_values(["player_id", "season", "week"])
    for c in ("tgt_share", "car_share"):
        # Includes the latest game: this is "his role going into the NEXT week".
        w[c] = w.groupby("player_id")[c].transform(lambda x: x.rolling(8, min_periods=1).mean())
    w["_k"] = w["season"] * 100 + w["week"]
    return w[["player_id", "_k", "tgt_share", "car_share"]]


def build_injury_absences(weekly: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """(team, season, week) -> team_out_tgt_share, team_out_carry_share, opp_def_regulars_out."""
    inj = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    inj = inj[inj["report_status"].isin(OUT_STATUSES)].dropna(subset=["season", "week"])
    inj = inj.drop_duplicates(["gsis_id", "season", "week"])
    inj[["season", "week"]] = inj[["season", "week"]].astype(int)
    inj["_k"] = inj["season"] * 100 + inj["week"] - 1  # role as of the previous week
    shares = _player_share_asof(weekly).sort_values("_k")
    inj = pd.merge_asof(inj.sort_values("_k"), shares.rename(columns={"player_id": "gsis_id"}),
                        on="_k", by="gsis_id", direction="backward")
    off = (inj.groupby(["team", "season", "week"])[["tgt_share", "car_share"]].sum()
           .rename(columns={"tgt_share": "team_out_tgt_share", "car_share": "team_out_carry_share"})
           .reset_index())

    snaps = pd.read_csv(os.path.join(RAW_DIR, "snap_counts.csv"), low_memory=False)
    snaps = snaps.sort_values(["pfr_player_id", "season", "week"])
    snaps["def_role"] = snaps.groupby("pfr_player_id")["defense_pct"].transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    regulars = snaps[snaps["def_role"] >= 0.6].assign(_k=lambda d: d["season"] * 100 + d["week"])
    regulars["_name"] = regulars["player"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    inj["_name"] = inj["full_name"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    last_role = regulars.sort_values("_k").groupby(["_name", "team"]).tail(1)[["_name", "team", "_k"]]
    d = inj.merge(last_role.rename(columns={"_k": "_role_k"}), on=["_name", "team"], how="inner")
    # A regular as of a recent game (not someone who lost the job a season ago).
    d = d[(d["_k"] - d["_role_k"]) < 100]
    dfn = d.groupby(["team", "season", "week"]).size().rename("def_regulars_out").reset_index()

    games = _team_games(schedules)
    out = games.merge(off, on=["team", "season", "week"], how="left")
    out = out.merge(dfn.rename(columns={"team": "opponent", "def_regulars_out": "opp_def_regulars_out"}),
                    on=["opponent", "season", "week"], how="left")
    out[["team_out_tgt_share", "team_out_carry_share", "opp_def_regulars_out"]] = \
        out[["team_out_tgt_share", "team_out_carry_share", "opp_def_regulars_out"]].fillna(0)
    return out[["team", "season", "week", "team_spread", "team_out_tgt_share",
                "team_out_carry_share", "opp_def_regulars_out"]]


def build_matchup_context(stat: str, weekly: pd.DataFrame | None = None,
                          schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    """(team, season, week) -> every feature above for one model's stat."""
    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False) if weekly is None else weekly
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv")) if schedules is None else schedules
    a = build_opp_allowed(weekly, schedules, stat)
    b = build_injury_absences(weekly, schedules)
    return a.merge(b, on=["team", "season", "week"], how="outer")


MATCHUP_FEATURES = ["opp_allowed_rolling", "team_spread", "team_out_tgt_share",
                    "team_out_carry_share", "opp_def_regulars_out", "usage_trend"]

# Last season the trained models are fitted on. 2026 stays out so the real-Underdog-line
# track record (backtesting/backtest_line_history.py) grades games the models never saw.
FINAL_FIT_MAX_SEASON = 2025


def add_matchup_context(df: pd.DataFrame, stat: str, last3_col: str, rolling_col: str,
                        season_col: str = "season", week_col: str = "week") -> pd.DataFrame:
    """Merge MATCHUP_FEATURES onto player rows by (recent_team, season_col, week_col): the
    row's own game for training, the upcoming game at inference (next_season/next_week)."""
    ctx = build_matchup_context(stat).rename(
        columns={"team": "recent_team", "season": season_col, "week": week_col})
    out = df.drop(columns=[c for c in MATCHUP_FEATURES if c in df.columns]).merge(
        ctx, on=["recent_team", season_col, week_col], how="left")
    out["usage_trend"] = out[last3_col] - out[rolling_col]
    fill = ["team_spread", "team_out_tgt_share", "team_out_carry_share", "opp_def_regulars_out"]
    out[fill] = out[fill].fillna(0)
    return out
