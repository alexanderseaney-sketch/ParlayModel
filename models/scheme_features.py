"""
Scheme/tendency features from real play-by-play data:
- pass_oe: pass rate OVER EXPECTED (nflverse's own situation-adjusted metric) — the
  standard proxy for offensive coordinator aggression/identity, cleaner than raw pass
  rate since it already accounts for down/distance/score/time situation
- shotgun_rate, no_huddle_rate: formation/tempo tendencies
- avg_defenders_in_box: defensive scheme signal (stacked box vs. light box)
- early_down_pass_rate: 1st/2nd down pass rate specifically — scheme identity before
  game script (trailing/leading) starts forcing play-calling
"""
import os

import pandas as pd

from pbp_features import filter_garbage_time

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def build_team_week_scheme(pbp: pd.DataFrame) -> pd.DataFrame:
    clean = filter_garbage_time(pbp)

    off = clean.groupby(["posteam", "season", "week"]).agg(
        pass_oe=("pass_oe", "mean"),
        shotgun_rate=("shotgun", "mean"),
        no_huddle_rate=("no_huddle", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    early_down = clean[clean["down"].isin([1, 2])]
    early_pass_rate = early_down.groupby(["posteam", "season", "week"])["pass"].mean().reset_index(
        name="early_down_pass_rate"
    ).rename(columns={"posteam": "team"})

    defense = clean.groupby(["defteam", "season", "week"])["defenders_in_box"].mean().reset_index(
        name="avg_defenders_in_box"
    ).rename(columns={"defteam": "team"})

    team_week = off.merge(early_pass_rate, on=["team", "season", "week"], how="left")
    team_week = team_week.merge(defense, on=["team", "season", "week"], how="left")

    team_week = team_week.sort_values(["team", "season", "week"]).reset_index(drop=True)
    cols = ["pass_oe", "shotgun_rate", "no_huddle_rate", "early_down_pass_rate", "avg_defenders_in_box"]
    for col in cols:
        team_week[f"{col}_rolling"] = (
            team_week.groupby(["team", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    return team_week


if __name__ == "__main__":
    pbp = pd.read_csv(os.path.join(RAW_DIR, "pbp.csv"), low_memory=False)
    df = build_team_week_scheme(pbp)
    print(df.shape)
    rolling_cols = [c for c in df.columns if c.endswith("_rolling")]
    print(df[["team", "season", "week"] + rolling_cols].dropna().head(5))
