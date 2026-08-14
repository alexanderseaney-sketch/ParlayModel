"""
Head coaching changes by season — used to apply EXTRA Elo regression for teams with a
new HC, on top of the standard between-season regression. Rationale: a team's prior-
season performance is a less trustworthy predictor under new leadership than the
standard regression already assumes (e.g. the Giants under new HC John Harbaugh for
2026 shouldn't be projected the same way as if Brian Daboll's staff were still there).

Historical years (2020-2024) are compiled from known coaching-change history, used to
VALIDATE this idea against real backtested games before trusting it — not just assumed.
2026 is current-year research (Feb-March 2026 reporting). This needs manual updating
each offseason; there's no clean nflverse dataset for coaching changes.
"""

NEW_HC_BY_SEASON = {
    2020: {"CAR", "NYG", "CLE", "WAS"},
    2021: {"ATL", "NYJ", "PHI", "LAC", "DET", "JAX", "HOU"},
    2022: {"DEN", "MIA", "NYG", "MIN", "HOU", "LV", "JAX"},
    2023: {"ARI", "CAR", "IND", "DEN", "HOU"},
    2024: {"ATL", "CAR", "LAC", "WAS", "NE", "LV", "SEA", "TEN"},
    2026: {"NYG", "BUF", "CLE", "PIT", "BAL", "TEN", "MIA", "ATL", "ARI", "LV", "WAS"},
}


def get_new_hc_teams(season: int) -> set:
    return NEW_HC_BY_SEASON.get(season, set())


def is_new_hc(team: str, season: int) -> bool:
    return team in get_new_hc_teams(season)
