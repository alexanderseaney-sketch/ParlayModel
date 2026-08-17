"""
Feature engineering for the remaining small period props: period_1/period_1_2
receiving_yds, rushing_yds, passing_yds, passing_tds. Unlike
period_first_touchdown_scored / period_1_rush_rec_tds (anytime-TD style,
constant 0.5 lines), these are yardage/count thresholds with real variable
Underdog lines, so -- same as every non-anytime-TD prop in this project --
the proxy line is the player's own trailing rolling average, just computed
over the PERIOD-scoped stat (own quarter/half history) rather than full-game.

Kept intentionally simple: one shared builder parameterized by which
period-scoped stat to target, using period_features.build_period_totals()
for the target plus full-game opportunity features from weekly_stats.csv
(targets/carries/attempts) as additional predictors -- a player's overall
role should carry real signal for their period-scoped share too, without
needing separate NGS-dependent feature engineering for what are, individually,
very small markets (3-12 legs each).
"""
import os

import pandas as pd

from period_features import build_period_totals

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# stat_col -> (opportunity column already in weekly_stats.csv, eligible positions)
STAT_CONFIGS = {
    "receiving_yards": ("targets", ["WR", "TE", "RB"]),
    "rushing_yards": ("carries", ["RB"]),
    "passing_yards": ("attempts", ["QB"]),
    "passing_tds": ("attempts", ["QB"]),
}


def build_period_stat_dataset(period: str, stat_col: str, min_week: int = 4) -> pd.DataFrame:
    opportunity_col, positions = STAT_CONFIGS[stat_col]

    weekly = pd.read_csv(os.path.join(RAW_DIR, "weekly_stats.csv"), low_memory=False)
    weekly = weekly[weekly["position"].isin(positions)].copy()
    keep_cols = ["player_id", "player_display_name", "position", "recent_team", "season", "week",
                 opportunity_col]
    weekly = weekly[list(dict.fromkeys(keep_cols))].sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    totals = build_period_totals(period)[["player_id", "season", "week", stat_col]]
    df = weekly.merge(totals, on=["player_id", "season", "week"], how="left")
    df[stat_col] = df[stat_col].fillna(0.0)

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in [stat_col, opportunity_col]:
        df[f"{col}_rolling"] = (
            df.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
        df[f"{col}_last3"] = (
            df.groupby(["player_id", "season"])[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )

    df = df[df["week"] >= min_week].reset_index(drop=True)

    df["proxy_line"] = df[f"{stat_col}_rolling"]
    df["over_proxy_line"] = (df[stat_col] > df["proxy_line"]).astype(int)

    return df


if __name__ == "__main__":
    for period in ["q1", "h1"]:
        for stat_col in STAT_CONFIGS:
            df = build_period_stat_dataset(period, stat_col)
            n_nonzero = (df[stat_col] > 0).mean()
            print(f"{period} {stat_col}: {df.shape}, nonzero rate {n_nonzero*100:.1f}%")
