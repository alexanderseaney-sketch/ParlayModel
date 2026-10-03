"""
Training-vs-inference switch for every "strictly prior games" rolling feature.

Every rolling feature in this project is computed as shift(1) -> rolling/expanding,
so a row for game G only sees games BEFORE G -- correct for training (no leakage of
G's own result into its features). But current_predictions.py scores a player by
taking their MOST RECENT played row and using its features to predict the NEXT,
unplayed game. With shift(1) that row's features exclude the most recent game
itself, so every live prediction was one game stale. Found 2026-10-02: Zay Flowers'
live receiving-yards proxy was 71.4 (16 games ending 2026 wk1) when his true
trailing-16 average through wk3 (84 yds) was 75.9; his `_last3` features were
likewise a game behind.

current_predictions.py flips INCLUDE_LATEST_GAME on before building datasets, which
makes lag() a no-op: the latest row's rolling features then cover every game played
so far -- exactly the pre-game state of the upcoming game. Training never touches it.
"""
INCLUDE_LATEST_GAME = False


def lag(s):
    """s.shift(1) for training (strictly prior games); identity at inference so the
    latest played row's rolling window includes that game."""
    return s if INCLUDE_LATEST_GAME else s.shift(1)
