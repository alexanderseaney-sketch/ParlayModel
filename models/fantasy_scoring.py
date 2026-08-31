"""
One place for fantasy scoring, so the matchups builder and the dashboard agree.

Why compute from raw stat columns instead of nflverse's fantasy_points /
fantasy_points_ppr: those are NaN for any season weekly_stats.csv fills by
deriving from play-by-play (currently 2025 -- nflverse hasn't published official
player_stats for it yet). The raw yard/TD/reception columns are always there.

RECEPTION_PT: 1.0 = PPR, 0.5 = half-PPR, 0.0 = standard.
"""
import pandas as pd

# points per unit, everything except the per-reception value (which varies by format)
PASS_YD = 0.04
PASS_TD = 4.0
INTERCEPTION = -2.0
RUSH_YD = 0.10
RUSH_TD = 6.0
REC_YD = 0.10
REC_TD = 6.0
FUMBLE_LOST = -2.0
TWO_PT = 2.0
RETURN_TD = 6.0

SCORING_LABELS = {"ppr": "PPR", "half": "Half-PPR", "std": "Standard"}
RECEPTION_PT = {"ppr": 1.0, "half": 0.5, "std": 0.0}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)


def fantasy_points_from_weekly(df: pd.DataFrame, scoring: str = "ppr") -> pd.Series:
    """Per-row fantasy points for a weekly_stats.csv-shaped frame."""
    rec_pt = RECEPTION_PT[scoring]
    fumbles_lost = (_col(df, "sack_fumbles_lost") + _col(df, "rushing_fumbles_lost")
                    + _col(df, "receiving_fumbles_lost"))
    two_pt = (_col(df, "passing_2pt_conversions") + _col(df, "rushing_2pt_conversions")
              + _col(df, "receiving_2pt_conversions"))
    return (
        _col(df, "passing_yards") * PASS_YD
        + _col(df, "passing_tds") * PASS_TD
        + _col(df, "interceptions") * INTERCEPTION
        + _col(df, "rushing_yards") * RUSH_YD
        + _col(df, "rushing_tds") * RUSH_TD
        + _col(df, "receiving_yards") * REC_YD
        + _col(df, "receiving_tds") * REC_TD
        + _col(df, "receptions") * rec_pt
        + fumbles_lost * FUMBLE_LOST
        + two_pt * TWO_PT
        + _col(df, "special_teams_tds") * RETURN_TD
    )


def project_points(stats: dict, scoring: str = "ppr") -> float:
    """Projected fantasy points from a dict of projected stat values (a player's
    proxy_line per prop_type, rolled up). Missing / NaN keys count as 0."""
    def g(k):
        v = stats.get(k)
        return float(v) if v is not None and pd.notna(v) else 0.0
    rec_pt = RECEPTION_PT[scoring]

    # The prop model has a receptions market for WR/TE but not RB. When a player
    # has an RB receiving-yards projection and no receptions one, back out an
    # estimate at ~7.5 yards per RB catch so PPR/half-PPR RB projections aren't
    # missing a real chunk of scoring.
    receptions = g("receptions")
    if receptions == 0 and g("receiving_yards_rb") > 0:
        receptions = g("receiving_yards_rb") / 7.5

    return (
        g("passing_yards") * PASS_YD
        + g("passing_tds") * PASS_TD
        + g("passing_ints") * INTERCEPTION
        + (g("rushing_yards") + g("rushing_yards_qb")) * RUSH_YD
        + (g("receiving_yards") + g("receiving_yards_rb")) * REC_YD
        + receptions * rec_pt
        + (g("rush_rec_tds") + g("rush_rec_tds_qb")) * RUSH_TD
    )
