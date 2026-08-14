"""
Individual player-level and additional game-context signals not yet tried:
- The PLAYER'S OWN injury/practice status that week (not team-wide count — this is
  specifically "is THIS player himself banged up going into this game")
- Divisional game flag (more familiarity, historically different game dynamics)
- Primetime flag (Thursday/Sunday night/Monday night games)
- Usage TREND (is the player's role growing or shrinking — last3 vs. full-season
  rolling average as a momentum signal, not just the level of either alone)
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

INJURY_SEVERITY = {"Out": 3, "Doubtful": 2, "Questionable": 1}


def build_player_injury_status(injuries: pd.DataFrame) -> pd.DataFrame:
    """THIS week's own injury designation for each player — known before kickoff, not
    rolling (a player's injury status resets/updates weekly, no reason to average it)."""
    inj = injuries.drop_duplicates(subset=["gsis_id", "season", "week"]).copy()
    inj["own_injury_severity"] = inj["report_status"].map(INJURY_SEVERITY).fillna(0)
    return inj[["gsis_id", "season", "week", "own_injury_severity"]].rename(columns={"gsis_id": "player_id"})


def build_game_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    sched = schedules.copy()
    sched["is_primetime"] = sched["weekday"].isin(["Thursday", "Monday"]) | (
        (sched["weekday"] == "Sunday") & (sched["gametime"] >= "18:00")
    )
    sched["is_primetime"] = sched["is_primetime"].astype(int)

    home = sched[["season", "week", "home_team", "div_game", "is_primetime"]].rename(columns={"home_team": "team"})
    away = sched[["season", "week", "away_team", "div_game", "is_primetime"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True).drop_duplicates()


def add_usage_trend(df: pd.DataFrame, level_col: str, last3_col: str) -> pd.DataFrame:
    """Momentum: is recent usage above or below the season-long baseline? Positive =
    role is growing, negative = role is shrinking — a signal separate from either level."""
    df = df.copy()
    df["usage_trend"] = df[last3_col] - df[level_col]
    return df


if __name__ == "__main__":
    injuries = pd.read_csv(os.path.join(RAW_DIR, "injuries.csv"), low_memory=False)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))

    inj_status = build_player_injury_status(injuries)
    print("Injury status:", inj_status.shape)
    print(inj_status["own_injury_severity"].value_counts())

    flags = build_game_flags(schedules)
    print("\nGame flags:", flags.shape)
    print(flags[["div_game", "is_primetime"]].mean())
