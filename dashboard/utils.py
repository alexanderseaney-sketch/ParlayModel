"""Shared helpers for the ParlayModel dashboard."""
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
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
# against live data: ~74% of matched props are naturally within 20% on their own,
# so this excludes genuinely incomparable cases, not most of the board.
MAX_LINE_DIVERGENCE = 0.20

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
    "espn_news.csv": "data/pull_espn_news.py",
    "sbnation_news.csv": "data/pull_sbnation_news.py",
    "nbcsports_news.csv": "data/pull_nbcsports_news.py",
    "footballguys_depth.csv": "data/pull_footballguys_depth.py",
    "underdog_props.csv": "data/pull_underdog.py",
}

# sys.executable, not a hardcoded "python3"/"python" string -- a bare name is resolved
# via PATH in the subprocess's environment, which is NOT guaranteed to be the same
# interpreter running this app (and on Windows there's usually no "python3" at all,
# only "python"/"py"). sys.executable is always the exact interpreter currently
# running, so it's guaranteed to have every package in requirements.txt installed,
# on both local dev and Streamlit Cloud's container.
PULL_SCRIPTS = {
    "nflverse (schedules + stats + NGS + injuries + snaps)": [sys.executable, "data/pull_nflverse.py", "--years", "2023", "2024", "--skip-pbp"],
    "ESPN news": [sys.executable, "data/pull_espn_news.py"],
    "SB Nation team news": [sys.executable, "data/pull_sbnation_news.py"],
    "NBC Sports / PFT rumor mill": [sys.executable, "data/pull_nbcsports_news.py"],
    "Footballguys depth charts": [sys.executable, "data/pull_footballguys_depth.py"],
    "Underdog pick'em props": [sys.executable, "data/pull_underdog.py"],
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
