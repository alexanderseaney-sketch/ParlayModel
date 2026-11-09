"""Which seasons the weekly prop models train on (2026-11-09).

Every train_*.py script fits on finished seasons only, through FIT_MAX_SEASON (the last
completed season), so this season's real Underdog lines grade games no model has seen.
That track record decides which picks count as proven (pick_rules.py), so it has to
stay honest. Before this, only the five context models (receptions, rushing yards,
passing yards / TDs / INTs) stopped at 2025; the other nine had 2026 weeks 1-5 in
training, which flattered their real-line record.

Adding this season's finished weeks was tested walk-forward on 2026 weeks 4-9 (week W
scored by a model fit through 2025 plus 2026 weeks before W, against one fit through
2025): log loss got slightly worse for receptions (.6785 -> .6808), rushing yards
(.6238 -> .6275) and passing TDs, about flat for passing yards, better only for INTs
(n=194). Net worse, so the cutoff stays at the last full season. It moves on by
itself when the next season starts.

Usage in a training script:
    df = training_rows(df)                      # before the leave-one-season-out loop
    path = model_path("player_prop_x_model.pkl")
"""
import os
from datetime import date

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def current_season(today: date | None = None) -> int:
    """NFL season in progress (a season's January / February games belong to it)."""
    d = today or date.today()
    return d.year if d.month >= 3 else d.year - 1


FIT_MAX_SEASON = current_season() - 1


def training_rows(df):
    return df[df["season"] <= FIT_MAX_SEASON]


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)
