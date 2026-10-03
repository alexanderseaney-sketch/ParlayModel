"""
Builds pre-game team-week features from the raw nflverse pulls, then joins home/away
features onto each scheduled game.

Every feature here is computed using ONLY data from games strictly before the game being
predicted (rolling season-to-date average, shifted by one game) — this is the same
no-leakage discipline as the Elo backtest, just applied to a richer feature set.
"""
import os

import numpy as np
import pandas as pd

from inference_mode import lag

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# Training-population switches. Both ON since 2026-10-02 (experiments/ngs_selection.py):
#   NGS_ASOF: attach each player's latest NGS rolling as of the game. NGS weekly only
#     lists players who cleared a volume minimum IN THAT GAME, so the old exact join +
#     the train scripts' dropna kept only games where the player got volume -- an
#     outcome-conditioned population.
#   PREGAME_QUALIFIER: filter rows on PRE-game rolling usage instead of same-game
#     carries/targets/attempts (post-game information).
# Scored on the realistic population (pre-game role >= minimum, every outcome), the
# old setup was near coin-flip (AUC ~0.51, log loss 0.94-1.37; its "90%+" calls hit
# 39-46%) vs AUC 0.58-0.62 / log loss ~0.65-0.66 with both switches on -- better in
# every holdout season for rushing yards, receiving yards and receptions.
NGS_ASOF = True
PREGAME_QUALIFIER = True


def merge_ngs(df: pd.DataFrame, ngs: pd.DataFrame) -> pd.DataFrame:
    """Left-join NGS rolling columns onto player-game rows by (player_id, season, week),
    or as-of (latest NGS row at/before the game) when NGS_ASOF. Adds `_ngs_row` = the
    game itself has an NGS row."""
    exact = ngs[["player_id", "season", "week"]].drop_duplicates().assign(_ngs_row=True)
    if not NGS_ASOF:
        out = df.merge(ngs, on=["player_id", "season", "week"], how="left")
    else:
        left = df.assign(_k=df["season"] * 100 + df["week"]).sort_values("_k")
        right = (ngs.dropna(subset=["player_id"]).assign(_k=ngs["season"] * 100 + ngs["week"])
                 .drop(columns=["season", "week"]).drop_duplicates(["player_id", "_k"]).sort_values("_k"))
        out = pd.merge_asof(left, right, on="_k", by="player_id", direction="backward").drop(columns=["_k"])
        out = out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    out = out.merge(exact, on=["player_id", "season", "week"], how="left")
    out["_ngs_row"] = out["_ngs_row"].fillna(False).astype(bool)
    return out


def qualify(df: pd.DataFrame, label_col: str, pregame_col: str, minimum: float) -> pd.DataFrame:
    col = pregame_col if PREGAME_QUALIFIER else label_col
    return df[df[col] >= minimum].reset_index(drop=True)


def load_ngs(kind: str) -> pd.DataFrame:
    """ngs_{passing,rushing,receiving}.csv WITHOUT nflverse's week-0 rows. Week 0 is
    the full-SEASON aggregate, not a game; sorted by (season, week) it lands first in
    each season, so every "strictly prior" rolling window for that season's games
    silently included the whole season's totals -- future games included. Found
    2026-10-02; every NGS feature (prop models and the game model) loads through here."""
    df = pd.read_csv(os.path.join(RAW_DIR, f"ngs_{kind}.csv"))
    return df[df["week"] > 0].reset_index(drop=True)


def cross_season_rolling(df: pd.DataFrame, group_col: str, value_col: str,
                         window: int | None = None, min_periods: int = 1) -> pd.Series:
    """Trailing rolling average of value_col within group_col, using only strictly
    PRIOR rows (shift(1)) in (season, week) order -- carries across a season
    boundary instead of resetting there. `df` must already be sorted by
    [group_col, "season", "week"] (or [group_col, "season", "week"] equivalent);
    caller is responsible for that sort.

    Why this exists (found 2026-09-22, two weeks into the 2026 season): every
    rolling feature in this project used to reset at each season's week 1 --
    groupby([id, "season"]) -- so a player's or team's rolling stats were
    undefined until ~3 games into a new season (the min_week=4 filters
    throughout models/player_prop_*_features.py exist for exactly this reason).
    That bit at BOTH training time and live inference: current_predictions.py's
    "most recent qualifying row" fell back to a player's LAST GAME OF THE
    PREVIOUS SEASON, frozen, for the first month of every year -- see its own
    2026-08-15 comment. That's backwards for a league where rosters,
    coordinators and schemes turn over heavily every offseason: the exact weeks
    that most need a fresh in-season signal were structurally unable to get one.

    window=None -> expanding (all available prior history, unbounded);
    window=N -> trailing N-game average. Bounded windows are used for anything
    that should react to a real role/scheme change within a season rather than
    being diluted by years of older data (see call sites)."""
    g = df.groupby(group_col)[value_col]
    if window is None:
        return g.apply(lambda s: lag(s).expanding(min_periods=min_periods).mean()).reset_index(level=0, drop=True)
    return g.apply(lambda s: lag(s).rolling(window, min_periods=min_periods).mean()).reset_index(level=0, drop=True)


def build_team_week_offense(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregates player-week stats up to team-week offensive production."""
    for col in ["passing_epa", "rushing_epa", "receiving_epa"]:
        weekly_stats[col] = weekly_stats[col].fillna(0)

    team_week = weekly_stats.groupby(["recent_team", "season", "week"]).agg(
        off_epa=("passing_epa", lambda s: s.sum()),
    ).reset_index()

    # total EPA = sum of passing + rushing + receiving EPA across all players that team-week
    epa_sum = weekly_stats.groupby(["recent_team", "season", "week"])[
        ["passing_epa", "rushing_epa", "receiving_epa"]
    ].sum().sum(axis=1).reset_index(name="off_epa_total")

    team_week = team_week.drop(columns=["off_epa"]).merge(
        epa_sum, on=["recent_team", "season", "week"]
    )
    return team_week.rename(columns={"recent_team": "team"})


def build_team_week_defense(offense: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """A team's defensive EPA allowed = the opponent's offensive EPA that week."""
    games = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team"]
    ].copy()

    home_side = games.rename(columns={"home_team": "team", "away_team": "opponent"})
    away_side = games.rename(columns={"away_team": "team", "home_team": "opponent"})
    matchups = pd.concat([home_side, away_side], ignore_index=True)

    matchups = matchups.merge(
        offense.rename(columns={"team": "opponent", "off_epa_total": "def_epa_allowed"}),
        on=["opponent", "season", "week"], how="left",
    )
    return matchups[["team", "season", "week", "def_epa_allowed"]]


def build_team_week_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    """Counts players listed as Out/Doubtful/Questionable per team-week — a crude but
    real proxy for team health going into a game."""
    injuries = injuries.drop_duplicates(subset=["gsis_id", "season", "week"])
    concerning = injuries[injuries["report_status"].isin(["Out", "Doubtful", "Questionable"])]
    counts = concerning.groupby(["team", "season", "week"]).size().reset_index(name="injury_count")
    return counts


def build_team_week_ngs(ngs_passing: pd.DataFrame, ngs_rushing: pd.DataFrame, ngs_receiving: pd.DataFrame) -> pd.DataFrame:
    """Team-week averages of the Next Gen Stats that most plausibly carry predictive
    signal beyond box-score EPA: CPOE (passing efficiency vs. expectation) and average
    separation (receiving — open receivers = easier offense)."""
    passing = ngs_passing.groupby(["team_abbr", "season", "week"]).agg(
        cpoe=("completion_percentage_above_expectation", "mean"),
    ).reset_index().rename(columns={"team_abbr": "team"})

    receiving = ngs_receiving.groupby(["team_abbr", "season", "week"]).agg(
        avg_separation=("avg_separation", "mean"),
    ).reset_index().rename(columns={"team_abbr": "team"})

    merged = passing.merge(receiving, on=["team", "season", "week"], how="outer")
    return merged


def build_team_week_turnovers(weekly_stats: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Turnovers committed by a team's offense that week (interceptions thrown + fumbles
    lost, across passing/rushing/receiving/sacks), and turnovers forced (= opponent's
    turnovers committed that week, same matchup-lookup pattern as defensive EPA)."""
    for col in ["interceptions", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]:
        weekly_stats[col] = weekly_stats[col].fillna(0)

    committed = weekly_stats.groupby(["recent_team", "season", "week"]).apply(
        lambda g: (g["interceptions"] + g["sack_fumbles_lost"] + g["rushing_fumbles_lost"] + g["receiving_fumbles_lost"]).sum(),
        include_groups=False,
    ).reset_index(name="turnovers_committed").rename(columns={"recent_team": "team"})

    games = schedules[schedules["game_type"] == "REG"][["season", "week", "home_team", "away_team"]].copy()
    home_side = games.rename(columns={"home_team": "team", "away_team": "opponent"})
    away_side = games.rename(columns={"away_team": "team", "home_team": "opponent"})
    matchups = pd.concat([home_side, away_side], ignore_index=True)

    matchups = matchups.merge(
        committed.rename(columns={"team": "opponent", "turnovers_committed": "turnovers_forced"}),
        on=["opponent", "season", "week"], how="left",
    )
    turnovers = committed.merge(
        matchups[["team", "season", "week", "turnovers_forced"]], on=["team", "season", "week"], how="left"
    )
    turnovers["turnover_margin"] = turnovers["turnovers_forced"].fillna(0) - turnovers["turnovers_committed"]
    return turnovers[["team", "season", "week", "turnover_margin"]]



def build_qb_rolling_cpoe(ngs_passing: pd.DataFrame) -> pd.DataFrame:
    """Each starting QB's own rolling CPOE (completion % above expectation), computed
    the same no-leakage way (strictly prior games only) — sharper than team-average NGS
    since it isolates the actual starter's skill rather than blending in backup appearances."""
    qb = ngs_passing[["player_gsis_id", "season", "week", "completion_percentage_above_expectation"]].copy()
    qb = qb.rename(columns={"completion_percentage_above_expectation": "qb_cpoe"})
    qb = qb.sort_values(["player_gsis_id", "season", "week"]).reset_index(drop=True)
    qb["qb_cpoe_rolling"] = (
        qb.groupby(["player_gsis_id", "season"])["qb_cpoe"]
        .apply(lambda s: lag(s).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    return qb[["player_gsis_id", "season", "week", "qb_cpoe_rolling"]]



def add_rolling_pregame_features(team_week: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """For each feature, computes the team's season-to-date average using only STRICTLY
    PRIOR weeks (shift(1) before the expanding mean) — this is what makes it usable as a
    pre-game prediction feature without leaking the current week's own result into itself."""
    team_week = team_week.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in feature_cols:
        team_week[f"{col}_rolling"] = (
            team_week.groupby(["team", "season"])[col]
            .apply(lambda s: lag(s).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    return team_week


def build_game_features(min_week: int = 3) -> pd.DataFrame:
    """Full pipeline: builds the game-level dataset with pre-game rolling features for
    both home and away teams. min_week=3 drops the first two weeks of each season, where
    rolling averages are built from too little data to mean much."""
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    weekly_stats = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    ngs_passing = load_ngs("passing")
    ngs_rushing = load_ngs("rushing")
    ngs_receiving = load_ngs("receiving")

    offense = build_team_week_offense(weekly_stats)
    defense = build_team_week_defense(offense, schedules)
    inj = build_team_week_injuries(injuries)
    ngs = build_team_week_ngs(ngs_passing, ngs_rushing, ngs_receiving)
    turnovers = build_team_week_turnovers(weekly_stats, schedules)

    team_week = offense.merge(defense, on=["team", "season", "week"], how="outer")
    team_week = team_week.merge(inj, on=["team", "season", "week"], how="left")
    team_week = team_week.merge(ngs, on=["team", "season", "week"], how="left")
    team_week = team_week.merge(turnovers, on=["team", "season", "week"], how="left")
    team_week["injury_count"] = team_week["injury_count"].fillna(0)

    feature_cols = ["off_epa_total", "def_epa_allowed", "injury_count", "cpoe", "avg_separation", "turnover_margin"]
    team_week = add_rolling_pregame_features(team_week, feature_cols)

    rolling_cols = [f"{c}_rolling" for c in feature_cols]
    team_week_pregame = team_week[["team", "season", "week"] + rolling_cols]

    games = schedules[schedules["game_type"] == "REG"].copy()
    games = games.dropna(subset=["home_score", "away_score"])
    games = games[games["week"] >= min_week]

    qb_cpoe = build_qb_rolling_cpoe(ngs_passing)
    games = games.merge(
        qb_cpoe.rename(columns={"player_gsis_id": "home_qb_id", "qb_cpoe_rolling": "home_qb_cpoe_rolling"}),
        on=["home_qb_id", "season", "week"], how="left",
    )
    games = games.merge(
        qb_cpoe.rename(columns={"player_gsis_id": "away_qb_id", "qb_cpoe_rolling": "away_qb_cpoe_rolling"}),
        on=["away_qb_id", "season", "week"], how="left",
    )
    games["diff_qb_cpoe_rolling"] = games["home_qb_cpoe_rolling"] - games["away_qb_cpoe_rolling"]

    games = games.merge(
        team_week_pregame.rename(columns={c: f"home_{c}" for c in rolling_cols} | {"team": "home_team"}),
        on=["home_team", "season", "week"], how="left",
    )
    games = games.merge(
        team_week_pregame.rename(columns={c: f"away_{c}" for c in rolling_cols} | {"team": "away_team"}),
        on=["away_team", "season", "week"], how="left",
    )

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # Weather and rest are known BEFORE kickoff already — no rolling needed, use directly.
    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    games["temp"] = games["temp"].fillna(games["temp"].median())
    games["wind"] = games["wind"].fillna(0)
    games["is_dome"] = games["roof"].isin(["dome", "closed"]).astype(int)

    for c in rolling_cols:
        home_col, away_col = f"home_{c}", f"away_{c}"
        games[f"diff_{c}"] = games[home_col] - games[away_col]

    return games


if __name__ == "__main__":
    df = build_game_features()
    print(df.shape)
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    print("Feature columns:", diff_cols)
    print("Null counts:\n", df[diff_cols].isna().sum())
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "game_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
