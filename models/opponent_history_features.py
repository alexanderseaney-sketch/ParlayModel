"""
Player-vs-specific-opponent history: has THIS player historically performed above or
below their own baseline specifically against THIS opponent? Different from general
opponent-strength features (def_epa_allowed) — this captures player-specific matchup
patterns (a WR who torches a certain defense's coverage scheme repeatedly, etc.),
computed with strict no-leakage (only games played before the current one, and only
against this specific opponent).
"""
import pandas as pd


def add_vs_opponent_history(df: pd.DataFrame, stat_col: str, min_games: int = 1) -> pd.DataFrame:
    """Adds vs_opponent_avg: this player's average in that stat across all PRIOR
    meetings with this specific opponent (any season). NaN if no prior meetings —
    filled with the player's overall rolling average as a neutral fallback."""
    df = df.sort_values(["player_id", "opponent", "season", "week"]).reset_index(drop=True)

    df["vs_opponent_avg"] = (
        df.groupby(["player_id", "opponent"])[stat_col]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["vs_opponent_n_games"] = (
        df.groupby(["player_id", "opponent"]).cumcount()
    )
    return df
