"""Shared helpers for the ParlayModel dashboard."""
import email.utils
import os
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
BET_LOG_DIR = os.path.join(ROOT_DIR, "bet_logs")
BET_LOG_PATH = os.path.join(BET_LOG_DIR, "bets.csv")

# CRITICAL correctness constant, shared by the dashboard and
# models/generate_weekly_bet_slip.py -- a model's predicted_prob_over answers
# "beats OUR proxy line" (the player's own rolling average), not "beats Underdog's
# actual posted line." Those are only the same question when the two numbers are
# close. A real bug found 2026-08-15: matching by player+stat alone let a
# prediction get displayed/used against a wildly different real line (e.g. a 4.1
# proxy shown as "model: 96%" next to a real 65.5 line) -- and because a big
# mismatch produces an artificially huge-looking edge, ranking by edge without this
# gate actively surfaces the worst mismatches as the "best" opportunities. Verified
# against live data (per-game weekly props): ~74% of matched props are naturally
# within 20% on their own, so this excludes genuinely incomparable cases, not most
# of the board.
MAX_LINE_DIVERGENCE = 0.20

# Season-long and period-scoped (quarter/half) props are NOT held to the same 20% --
# checked against live data 2026-08-17 and they're structurally noisier, not just
# unluckier: a season total has vastly more room to drift from a same-player prior-
# season proxy than one game's yardage does (real roster/scheme/depth-chart changes
# a box-score rolling average can't see), and a quarter/half slice is an even
# smaller, noisier sample than a full game. Measured divergence quantiles against
# real live props: weekly yardage props hit their 74th percentile at ~30% (close to
# the existing 20% cutoff's own calibration target), season props hit 74th
# percentile at ~50%, period-scoped ones at ~55%. Applying the WEEKLY-calibrated 20%
# to those grains was never validated for them -- it just happened to be the only
# number that existed yet. These wider values chase the same calibration philosophy
# (let through the natural majority of props whose proxy is roughly in the right
# neighborhood) rather than a tighter bar that was calibrated for a different grain.
# Tried a smarter proxy first (2-season blended average instead of prior-season-
# only) to see if the gap could just be closed instead of widening the gate --
# reduced the mean divergence noticeably but barely moved how many legs actually
# cleared 20% (16->17 of 57 for season_rec_tds), because the biggest gaps are driven
# by real season-to-season role changes no combination of past-season stats can see.
SEASON_MAX_LINE_DIVERGENCE = 0.50
PERIOD_MAX_LINE_DIVERGENCE = 0.55

SEASON_GRAIN_STATS = {"season_receiving_yards", "season_rec_tds", "season_rush_yards", "season_rush_tds"}
PERIOD_GRAIN_STATS = {
    "period_1_receiving_yds", "period_1_2_receiving_yds",
    "period_1_rushing_yds", "period_1_2_rushing_yds",
    "period_1_passing_yds", "period_1_2_passing_yds",
}

# Weekly rushing yards specifically, not the general 20% weekly bar -- checked
# against live data 2026-08-17: passing_yds's 74th-percentile divergence is ~10%
# and receiving_yds's is ~22% (both fine under the general 20% bar), but
# rushing_yds's is ~42% -- a real, distinct pattern (committee backfields and
# game-script-dependent volume swing week to week far more than an entrenched
# starter's passing volume does), not noise. "Weekly" was never one uniform
# distribution; this splits out the one stat where that assumption broke down.
RUSHING_YDS_MAX_LINE_DIVERGENCE = 0.45

# TD/INT/sack counts (weekly and quarter/half-scoped) -- percentage divergence is
# the wrong metric here, not just a badly-calibrated threshold. A percentage-of-
# proxy check breaks down when the proxy itself is a fraction (e.g. a QB averaging
# 0.3 pass TDs/quarter): a completely normal, small real-world gap of 0.3 TDs
# computes as literally 100% "divergence" purely from the denominator being tiny,
# not because the match is actually bad. Checked against live data 2026-08-17: the
# real ABSOLUTE gap for these stats is small and stable regardless of the count
# scale (74th-percentile abs gap 0.20-0.54 across passing_tds/passing_ints/sacks/
# period_1(_2)_passing_tds, max observed 1.40) -- so these are gated on absolute
# difference instead of percentage, same underlying philosophy (let through the
# natural majority of props whose proxy is roughly in the right neighborhood) with
# a metric that doesn't fall apart at a small denominator. period_1_passing_tds and
# period_1_2_passing_tds move here from PERIOD_GRAIN_STATS for the same reason --
# even the widened 55% period bar didn't fix them, because the problem was never
# the threshold number.
COUNT_MAX_ABS_DIVERGENCE = 1.0
COUNT_GRAIN_STATS = {"passing_tds", "passing_ints", "sacks", "period_1_passing_tds", "period_1_2_passing_tds"}


def max_line_divergence_for(stat_name: str) -> float:
    """Which PERCENTAGE divergence gate applies to this stat. Only meaningful for
    stats not in COUNT_GRAIN_STATS -- see line_matches_proxy, the actual gate used
    everywhere, which dispatches to an absolute-difference check for count stats
    instead of calling this at all. Anytime-TD style props (rush_rec_tds,
    period_first_touchdown_scored, etc.) aren't listed here since their proxy is a
    constant 0.5 matching the real market almost exactly -- they essentially never
    need the wider bands."""
    if stat_name in SEASON_GRAIN_STATS:
        return SEASON_MAX_LINE_DIVERGENCE
    if stat_name in PERIOD_GRAIN_STATS:
        return PERIOD_MAX_LINE_DIVERGENCE
    if stat_name == "rushing_yds":
        return RUSHING_YDS_MAX_LINE_DIVERGENCE
    return MAX_LINE_DIVERGENCE


def line_matches_proxy(stat_value, proxy_line, stat_name: str) -> bool:
    """The actual correctness gate: is Underdog's real posted line close enough to
    our own proxy line that the model's probability (computed against the proxy) is
    still a trustworthy stand-in for beating Underdog's real number? Single shared
    implementation for the dashboard and generate_weekly_bet_slip.py, since this
    used to be reimplemented in both places and only one of them would need to
    change to drift out of sync with the other."""
    if pd.isna(stat_value) or pd.isna(proxy_line):
        return False
    stat_value, proxy_line = float(stat_value), float(proxy_line)
    if stat_name in COUNT_GRAIN_STATS:
        return abs(stat_value - proxy_line) <= COUNT_MAX_ABS_DIVERGENCE
    if proxy_line == 0:
        return False
    divergence = abs(stat_value - proxy_line) / abs(proxy_line)
    return divergence <= max_line_divergence_for(stat_name)


def pretty_stat_name(stat_name) -> str:
    """receiving_yds -> Receiving Yds -- a readable label instead of a raw column
    name, used anywhere a prop's stat type is shown to a human rather than matched
    against other data. Shared between the dashboard and generate_weekly_bet_slip.py
    so both surfaces describe a prop the same way instead of one of them leaking the
    raw stat_name string into a bet suggestion's description."""
    if not isinstance(stat_name, str) or not stat_name:
        return "?"
    return stat_name.replace("_", " ").title()

# Every raw data file the pull scripts are expected to produce, and which script produces it.
EXPECTED_FILES = {
    "schedules.csv": "data/pull_nflverse.py",
    "pbp.csv": "data/pull_nflverse.py (skipped by default, use without --skip-pbp)",
    "weekly_stats.csv": "data/pull_nflverse.py",
    "ngs_passing.csv": "data/pull_nflverse.py",
    "ngs_rushing.csv": "data/pull_nflverse.py",
    "ngs_receiving.csv": "data/pull_nflverse.py",
    "injuries.csv": "data/pull_nflverse.py",
    "snap_counts.csv": "data/pull_nflverse.py",
    "players.csv": "data/pull_nflverse.py",
    "sbnation_news.csv": "data/pull_sbnation_news.py",
    "nbcsports_news.csv": "data/pull_nbcsports_news.py",
    "footballguys_depth.csv": "data/pull_footballguys_depth.py",
    "underdog_props.csv": "data/pull_underdog.py",
    "weather_forecast.csv": "data/pull_weather_forecast.py (skipped by freshness check -- "
                             "legitimately empty most of the time, see data_freshness_check)",
}

def _default_pull_years(lookback: int = 3) -> list[str]:
    """NFL seasons are labeled by their START year (the "2025 season" runs Sep 2025 -
    Feb 2026), so the most recently completed season is still last calendar year's
    label through Feb -- before March, current_season steps back a year so that
    boundary doesn't get missed. Previously this was a hardcoded ["2023", "2024"] in
    PULL_SCRIPTS below, which is exactly why it went stale: nothing recomputed it, so
    it just silently stopped covering the current season once enough time passed
    (confirmed 2026-08-16 -- it had never once included the already-complete 2025
    season). data/pull_nflverse.py's save() merges into existing files rather than
    overwriting, so widening this is safe -- it adds whatever's missing without
    touching or re-fetching years already on disk."""
    today = datetime.now()
    current_season = today.year if today.month >= 3 else today.year - 1
    return [str(y) for y in range(current_season - lookback + 1, current_season + 1)]


# sys.executable, not a hardcoded "python3"/"python" string -- a bare name is resolved
# via PATH in the subprocess's environment, which is NOT guaranteed to be the same
# interpreter running this app (and on Windows there's usually no "python3" at all,
# only "python"/"py"). sys.executable is always the exact interpreter currently
# running, so it's guaranteed to have every package in requirements.txt installed,
# on both local dev and Streamlit Cloud's container.
PULL_SCRIPTS = {
    "nflverse (schedules + stats + NGS + injuries + snaps)": [sys.executable, "data/pull_nflverse.py", "--years", *_default_pull_years(), "--skip-pbp"],
    "SB Nation team news": [sys.executable, "data/pull_sbnation_news.py"],
    "NBC Sports / PFT rumor mill": [sys.executable, "data/pull_nbcsports_news.py"],
    "Footballguys depth charts": [sys.executable, "data/pull_footballguys_depth.py"],
    "Underdog pick'em props": [sys.executable, "data/pull_underdog.py"],
    "Weather forecasts (upcoming outdoor games)": [sys.executable, "data/pull_weather_forecast.py"],
}


def file_status(filename: str) -> dict:
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        return {"exists": False}
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    size_kb = os.path.getsize(path) / 1024
    try:
        rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        rows = None
    return {"exists": True, "modified": mtime, "size_kb": size_kb, "rows": rows}


@st.cache_data(show_spinner=False)
def load_csv(filename: str, _mtime: float) -> pd.DataFrame:
    """_mtime is passed in purely to bust the cache when the underlying file changes."""
    path = os.path.join(RAW_DIR, filename)
    return pd.read_csv(path, low_memory=False)


def load_csv_if_exists(filename: str) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return load_csv(filename, mtime)


def load_bet_log() -> pd.DataFrame:
    if not os.path.exists(BET_LOG_PATH):
        return pd.DataFrame(columns=[
            "date", "sport", "player", "stat", "choice", "line",
            "multiplier_or_odds", "stake", "result", "notes", "logged_at",
        ])
    return pd.read_csv(BET_LOG_PATH)


def append_bet(row: dict) -> None:
    os.makedirs(BET_LOG_DIR, exist_ok=True)
    df = load_bet_log()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(BET_LOG_PATH, index=False)


CURRENT_PREDICTIONS_PATH = os.path.join(ROOT_DIR, "models", "current_player_predictions.csv")


def load_current_predictions() -> pd.DataFrame | None:
    if not os.path.exists(CURRENT_PREDICTIONS_PATH):
        return None
    return pd.read_csv(CURRENT_PREDICTIONS_PATH)


def normalize_name(name: str) -> str:
    """Loose name matching between Underdog's player names and nflverse's — strips
    suffixes/punctuation that commonly differ between sources."""
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.replace(".", "").replace("'", "").strip()



    """Converts American odds to implied probability (0-1)."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Returns the first matching column name from candidates that exists in df, else None.
    Used because we don't know Underdog's exact odds/multiplier field names until the
    puller has actually been run once against live data."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


LEG_CORRELATIONS_PATH = os.path.join(ROOT_DIR, "models", "parlay_leg_correlations.csv")


def load_leg_correlations() -> dict[frozenset, float]:
    """Real, empirically-measured correlations between same-team same-game prop pairs
    (models/analyze_parlay_correlations.py, 2019-2024 data) -- e.g. QB passing yards
    and RB rushing yards on the same team are measurably NEGATIVELY correlated (pass-
    heavy and run-heavy game scripts trade off), while QB passing yards and WR
    receiving yards are measurably POSITIVELY correlated. Only pairs with |phi| >= 0.05
    are in this file -- weaker measured correlations were judged noise, not signal, at
    these sample sizes (see the analysis script for the full breakdown including nulls)."""
    if not os.path.exists(LEG_CORRELATIONS_PATH):
        return {}
    df = pd.read_csv(LEG_CORRELATIONS_PATH)
    return {
        frozenset([row["position_prop_a"], row["position_prop_b"]]): row["phi"]
        for _, row in df.iterrows()
    }


def correlation_adjusted_parlay_probability(legs: list[dict]) -> dict:
    """legs: list of {"team": str, "position_prop": str, "prob": float}. Only legs with
    both team and position_prop set participate in correlation lookups; others just
    contribute their probability to the naive product untouched.

    Naive combined probability assumes every leg is independent (current default
    everywhere else in this file). This applies a real, measured correction for each
    same-team pair that has a known correlation, using the phi-coefficient identity
    P(A and B) = p_A*p_B + phi*sqrt(p_A(1-p_A)*p_B(1-p_B)), clamped to the Frechet-
    Hoeffding bounds a joint probability can never violate regardless of phi
    (P(A and B) can never exceed min(p_A, p_B) or be negative). For 3+ mutually-
    correlated legs this multiplies independent pairwise corrections together rather
    than solving a full joint distribution -- an approximation, not exact, but far
    more honest than assuming zero correlation everywhere.

    Returns {"naive_prob", "adjusted_prob", "adjustments": [(leg_a, leg_b, phi), ...]}."""
    correlations = load_leg_correlations()

    naive_prob = 1.0
    for leg in legs:
        naive_prob *= leg["prob"]

    adjusted_prob = naive_prob
    adjustments = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = legs[i], legs[j]
            if not a.get("team") or not b.get("team") or a["team"] != b["team"]:
                continue
            if a.get("position_prop") is None or b.get("position_prop") is None:
                continue
            key = frozenset([a["position_prop"], b["position_prop"]])
            phi = correlations.get(key)
            if phi is None:
                continue

            p_a, p_b = a["prob"], b["prob"]
            joint = p_a * p_b + phi * np.sqrt(max(p_a * (1 - p_a) * p_b * (1 - p_b), 0))
            joint = min(max(joint, max(0.0, p_a + p_b - 1)), min(p_a, p_b))

            correction = joint / (p_a * p_b) if p_a * p_b > 0 else 1.0
            adjusted_prob *= correction
            adjustments.append((a["position_prop"], b["position_prop"], phi))

    return {"naive_prob": naive_prob, "adjusted_prob": adjusted_prob, "adjustments": adjustments}


def get_player_detail(player_name: str) -> dict:
    """Gathers everything we know about a player across every other data source,
    matched by normalize_name() -- the same loose-matching convention already used
    to join predictions to Underdog props in generate_weekly_bet_slip.py. This is a
    best-effort "here's what we have" lookup, not a required join: any source that
    hasn't been pulled yet, or has no match for this player, is simply absent from
    the returned dict rather than raising.

    Returns a dict that may contain any of:
      "injury_report"  -- most recent nflverse injury-report row (a Series)
      "recent_games"   -- last 5 rows of weekly_stats.csv, oldest first (a DataFrame)
      "predictions"    -- all current_player_predictions.csv rows for this player
      "props"          -- all underdog_props.csv rows for this player
    """
    key = normalize_name(player_name)
    result: dict = {}

    injuries = load_csv_if_exists("injuries.csv")
    if injuries is not None and "full_name" in injuries.columns:
        injuries = injuries.copy()
        injuries["_match_key"] = injuries["full_name"].apply(normalize_name)
        matches = injuries[injuries["_match_key"] == key]
        if not matches.empty:
            result["injury_report"] = matches.sort_values(["season", "week"]).iloc[-1]

    weekly = load_csv_if_exists("weekly_stats.csv")
    if weekly is not None and "player_display_name" in weekly.columns:
        weekly = weekly.copy()
        weekly["_match_key"] = weekly["player_display_name"].apply(normalize_name)
        matches = weekly[weekly["_match_key"] == key].sort_values(["season", "week"])
        if not matches.empty:
            result["recent_games"] = matches.tail(5)

    predictions = load_current_predictions()
    if predictions is not None and "player_display_name" in predictions.columns:
        predictions = predictions.copy()
        predictions["_match_key"] = predictions["player_display_name"].apply(normalize_name)
        matches = predictions[predictions["_match_key"] == key]
        if not matches.empty:
            result["predictions"] = matches

    props = load_csv_if_exists("underdog_props.csv")
    if props is not None and "full_name" in props.columns:
        props = props.copy()
        props["_match_key"] = props["full_name"].apply(normalize_name)
        matches = props[props["_match_key"] == key]
        if not matches.empty:
            result["props"] = matches

    snaps = load_csv_if_exists("snap_counts.csv")
    if snaps is not None and "player" in snaps.columns:
        snaps = snaps.copy()
        snaps["_match_key"] = snaps["player"].apply(normalize_name)
        matches = snaps[snaps["_match_key"] == key].sort_values(["season", "week"])
        if not matches.empty:
            result["snap_trend"] = matches.tail(5)

    return result


def load_player_photos() -> dict[str, str]:
    """normalize_name() -> Underdog headshot image_url, for whichever players
    currently have an active Underdog prop. Coverage is strong for offensive skill
    positions (~87% of starters, measured 2026-08-16) but weak for most defenders
    (~14%) since Underdog mostly only carries props for skill positions and a
    handful of premium defenders -- callers need a placeholder fallback for
    everyone else, not an assumption every player has a photo."""
    props = load_csv_if_exists("underdog_props.csv")
    if props is None or "full_name" not in props.columns or "image_url" not in props.columns:
        return {}
    deduped = props.drop_duplicates("full_name")
    return {
        normalize_name(name): url
        for name, url in zip(deduped["full_name"], deduped["image_url"])
        if isinstance(url, str) and url
    }


def load_player_jersey_numbers() -> dict[str, str]:
    """normalize_name() -> jersey number, same source/coverage caveats as
    load_player_photos() (both come from underdog_props.csv)."""
    props = load_csv_if_exists("underdog_props.csv")
    if props is None or "full_name" not in props.columns or "jersey_number" not in props.columns:
        return {}
    deduped = props.drop_duplicates("full_name")
    return {
        normalize_name(name): num
        for name, num in zip(deduped["full_name"], deduped["jersey_number"])
        if pd.notna(num)
    }


def get_player_news(player_name: str, max_items: int = 8) -> list[dict]:
    """Live, on-demand news search via Google News' public RSS search (no key, no
    login -- explicitly offered by Google for "rendering Google News results within
    a personal feed reader for personal, non-commercial use," which is exactly this).
    Deliberately NOT one of the bulk scheduled pulls (sbnation/nbcsports_news.csv)
    -- those are team/league-wide feeds that only catch a player if they're a big
    enough story to appear in a general feed. This is a real-time, player-specific
    search instead, triggered by the user clicking "Pull recent news" on that one
    player's card, so it works for any player regardless of today's bulk pulls.
    Quoting the name + "NFL" biases results toward football coverage and away from
    unrelated people who share the name."""
    query = urllib.parse.quote(f'"{player_name}" NFL')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ParlayModel/1.0)"}, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError):
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    # Google returns these in RELEVANCE order, not chronological -- pulling from the
    # full result set (not just the first max_items) before sorting so a genuinely
    # more recent article ranked lower on relevance isn't dropped before it gets the
    # chance to sort to the top.
    items = []
    for item in channel.findall("item"):
        source_el = item.find("source")
        pub_date_raw = item.findtext("pubDate")
        try:
            pub_dt = email.utils.parsedate_to_datetime(pub_date_raw) if pub_date_raw else None
        except (TypeError, ValueError):
            pub_dt = None
        items.append({
            "headline": item.findtext("title"),
            "link": item.findtext("link"),
            "published": pub_date_raw,
            "source": source_el.text if source_el is not None else None,
            "_pub_dt": pub_dt,
        })

    epoch = datetime.min.replace(tzinfo=timezone.utc)
    items.sort(key=lambda x: x["_pub_dt"] or epoch, reverse=True)
    for it in items:
        del it["_pub_dt"]
    return items[:max_items]


def run_pull_script(cmd: list[str]) -> tuple[bool, str]:
    """Runs a data-pull script and returns (success, combined_output)."""
    try:
        result = subprocess.run(
            cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=600,
        )
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Timed out after 10 minutes."
    except Exception as e:
        return False, f"Failed to run: {e}"


STALE_THRESHOLD_HOURS = 24


def data_freshness_check() -> dict:
    """Checks every EXPECTED_FILE (except pbp.csv, skipped by default and not part of
    the core freshness story, and weather_forecast.csv, which is often LEGITIMATELY
    empty -- see pull_weather_forecast.py: nothing to fetch whenever no outdoor game
    falls within Open-Meteo's ~16-day forecast horizon, true for most of a season, not
    just the off-season) for missing/stale/ok, based on file mtime. Returns
    {'missing': [filename, ...], 'stale': [(filename, age_hours), ...],
    'ok': [filename, ...]}. Used by the app to show a freshness banner instead of
    silently rendering pages against empty or day(s)-old data -- this matters most
    right after a fresh Streamlit Community Cloud deploy, where data/raw/ starts
    completely empty since it's gitignored (deliberately -- it's regenerated data,
    not source code) and never ships with the deployment itself."""
    missing, stale, ok = [], [], []
    for filename in EXPECTED_FILES:
        if filename in ("pbp.csv", "weather_forecast.csv"):
            continue
        status = file_status(filename)
        if not status["exists"]:
            missing.append(filename)
        else:
            age_hours = (datetime.now() - status["modified"]).total_seconds() / 3600
            if age_hours > STALE_THRESHOLD_HOURS:
                stale.append((filename, age_hours))
            else:
                ok.append(filename)
    return {"missing": missing, "stale": stale, "ok": ok}


def run_all_pulls() -> list[tuple[str, bool, str]]:
    """Runs every pull script in PULL_SCRIPTS in sequence. Returns
    [(label, success, output), ...] for each. Used by the freshness banner's
    one-click refresh, and reusable from Run Data Pulls page too."""
    results = []
    for label, cmd in PULL_SCRIPTS.items():
        success, output = run_pull_script(cmd)
        results.append((label, success, output))
    return results

