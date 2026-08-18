"""
Pulls real weather FORECASTS for upcoming (not yet played) outdoor games.

Real gap found 2026-08-18: game_context_features.py's temp/wind features already
exist and are used by several models (train_passing_props.py, train_rushing_props.py),
but schedules.csv only records ACTUAL observed conditions -- filled in by nflverse
AFTER a game is played. For any upcoming game, temp/wind are simply NaN there, and
build_game_context() silently falls back to the historical median temp and 0 wind for
every single one -- meaning every current prediction implicitly assumes a mild, calm
day regardless of whether the real upcoming game is a blizzard in Buffalo or a 100degF
day in Miami. This closes that gap using Open-Meteo (free, no API key, worldwide
coverage including the 2026 Melbourne game) for any upcoming game within its ~16-day
forecast horizon.

Deliberately day-level, not hour-level: matching a forecast's hourly timestamps to a
game's exact kickoff requires knowing schedules.csv's gametime timezone convention with
certainty, which isn't verified here -- getting that wrong would silently mis-time
every value. Averaging across the game day's afternoon/evening hours instead sidesteps
that whole risk while still capturing the actual signal that matters for a prediction
("cold and windy that day" vs "mild and calm"), not sub-hour precision.

Indoor games (dome/closed roof) are skipped entirely -- climate-controlled, weather is
a non-factor, not worth an API call or a row that would just say "70degF, no wind" for
every one of them.

Usage:
    python data/pull_weather_forecast.py
"""
import os
from datetime import datetime, timezone

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_PATH = os.path.join(RAW_DIR, "weather_forecast.csv")
TIMEOUT = 15

# (stadium name, lat, lon) for each team's normal home venue. Stable public
# geographic facts, not time-sensitive data -- doesn't need a live lookup.
# LAR and LAC share SoFi Stadium; NYG and NYJ share MetLife Stadium.
TEAM_STADIUMS = {
    "ARI": ("State Farm Stadium", 33.5276, -112.2626),
    "ATL": ("Mercedes-Benz Stadium", 33.7554, -84.4008),
    "BAL": ("M&T Bank Stadium", 39.2780, -76.6227),
    "BUF": ("Highmark Stadium", 42.7738, -78.7870),
    "CAR": ("Bank of America Stadium", 35.2258, -80.8528),
    "CHI": ("Soldier Field", 41.8623, -87.6167),
    "CIN": ("Paycor Stadium", 39.0954, -84.5160),
    "CLE": ("Huntington Bank Field", 41.5061, -81.6995),
    "DAL": ("AT&T Stadium", 32.7473, -97.0945),
    "DEN": ("Empower Field at Mile High", 39.7439, -105.0201),
    "DET": ("Ford Field", 42.3400, -83.0456),
    "GB": ("Lambeau Field", 44.5013, -88.0622),
    "HOU": ("NRG Stadium", 29.6847, -95.4107),
    "IND": ("Lucas Oil Stadium", 39.7601, -86.1639),
    "JAX": ("EverBank Stadium", 30.3239, -81.6373),
    "KC": ("GEHA Field at Arrowhead Stadium", 39.0489, -94.4839),
    "LA": ("SoFi Stadium", 33.9535, -118.3392),
    "LAC": ("SoFi Stadium", 33.9535, -118.3392),
    "LV": ("Allegiant Stadium", 36.0909, -115.1833),
    "MIA": ("Hard Rock Stadium", 25.9580, -80.2389),
    "MIN": ("U.S. Bank Stadium", 44.9738, -93.2575),
    "NE": ("Gillette Stadium", 42.0909, -71.2643),
    "NO": ("Caesars Superdome", 29.9511, -90.0812),
    "NYG": ("MetLife Stadium", 40.8135, -74.0745),
    "NYJ": ("MetLife Stadium", 40.8135, -74.0745),
    "PHI": ("Lincoln Financial Field", 39.9008, -75.1675),
    "PIT": ("Acrisure Stadium", 40.4468, -80.0158),
    "SEA": ("Lumen Field", 47.5952, -122.3316),
    "SF": ("Levi's Stadium", 37.4032, -121.9698),
    "TB": ("Raymond James Stadium", 27.9759, -82.5033),
    "TEN": ("Nissan Stadium", 36.1665, -86.7713),
    "WAS": ("Northwest Stadium", 38.9077, -76.8645),
}

# Known neutral-site/international games where the actual venue isn't the home
# team's normal stadium -- keyed by a substring of schedules.csv's own "stadium"
# column so this is detected from the data itself rather than hardcoded per game.
NEUTRAL_SITE_STADIUMS = {
    "Melbourne Cricket Ground": ("Melbourne Cricket Ground", -37.8199, 144.9834),
}

DOME_ROOFS = {"dome", "closed"}


def _venue_for_game(row) -> tuple[str, float, float] | None:
    stadium_text = row.get("stadium")
    if isinstance(stadium_text, str):
        for name, coords in NEUTRAL_SITE_STADIUMS.items():
            if name in stadium_text:
                return coords
    return TEAM_STADIUMS.get(row["home_team"])


def fetch_day_forecast(lat: float, lon: float, date: str) -> dict | None:
    """Mean temp/max wind/max precip-probability across a game day's afternoon and
    evening hours (12:00-23:00 local to the venue) -- covers every real NFL kickoff
    slot (early/late Sunday, SNF/MNF/TNF) without needing to match an exact hour."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "auto", "start_date": date, "end_date": date,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    hourly = resp.json().get("hourly")
    if not hourly or not hourly.get("time"):
        return None  # date is outside Open-Meteo's forecast horizon (~16 days out)

    day_hours = [i for i, t in enumerate(hourly["time"]) if 12 <= int(t[11:13]) <= 23]
    if not day_hours:
        return None
    temps = [hourly["temperature_2m"][i] for i in day_hours]
    winds = [hourly["wind_speed_10m"][i] for i in day_hours]
    precip = [hourly["precipitation_probability"][i] for i in day_hours]
    return {
        "temp_forecast": sum(temps) / len(temps),
        "wind_forecast": max(winds),
        "precip_prob_forecast": max(precip),
    }


FORECAST_HORIZON_DAYS = 16  # Open-Meteo's free forecast API's real limit


def pull_forecasts() -> pd.DataFrame:
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"), low_memory=False)
    upcoming = schedules[schedules["home_score"].isna()].copy()
    upcoming = upcoming[~upcoming["roof"].isin(DOME_ROOFS)]

    # Filter to the forecast horizon LOCALLY before making any network calls --
    # most of a season's upcoming games are always weeks/months out, so querying
    # every one of them individually (as a first version of this did) means
    # hundreds of API round-trips just to find the handful actually close enough
    # to have a real forecast yet.
    today = pd.Timestamp.now().normalize()
    game_dates = pd.to_datetime(upcoming["gameday"])
    upcoming = upcoming[(game_dates >= today) & (game_dates <= today + pd.Timedelta(days=FORECAST_HORIZON_DAYS))]

    rows = []
    skipped_no_venue, skipped_out_of_horizon, failed = 0, 0, 0
    for _, game in upcoming.iterrows():
        venue = _venue_for_game(game)
        if venue is None:
            skipped_no_venue += 1
            continue
        stadium_name, lat, lon = venue
        try:
            forecast = fetch_day_forecast(lat, lon, game["gameday"])
        except requests.RequestException as e:
            print(f"[{game['game_id']}] FAILED ({e})")
            failed += 1
            continue
        if forecast is None:
            skipped_out_of_horizon += 1
            continue
        rows.append({
            "game_id": game["game_id"], "season": game["season"], "week": game["week"],
            "gameday": game["gameday"], "home_team": game["home_team"], "away_team": game["away_team"],
            "stadium": stadium_name, **forecast,
        })

    print(f"{len(rows)} forecasts pulled, {skipped_out_of_horizon} outside the ~16-day "
          f"forecast horizon, {skipped_no_venue} unknown venue, {failed} failed.")
    return pd.DataFrame(rows)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    df = pull_forecasts()
    if df.empty:
        print("0 forecasts available right now (no outdoor games within the forecast "
              "horizon) -- not an error, just nothing to fetch yet. Nothing written.")
        return
    df["pulled_at"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
