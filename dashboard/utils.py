"""Shared helpers for the ParlayModel dashboard."""
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
BET_LOG_DIR = os.path.join(ROOT_DIR, "bet_logs")
BET_LOG_PATH = os.path.join(BET_LOG_DIR, "bets.csv")

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
    "underdog_props.csv": "data/pull_underdog.py",
}

PULL_SCRIPTS = {
    "nflverse (schedules + stats + NGS + injuries + snaps)": ["python3", "data/pull_nflverse.py", "--years", "2023", "2024", "--skip-pbp"],
    "ESPN news": ["python3", "data/pull_espn_news.py"],
    "Underdog pick'em props": ["python3", "data/pull_underdog.py"],
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


def american_to_prob(odds: float) -> float:
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


def parlay_combined_multiplier(individual_probs: list[float]) -> float:
    """Fair (no-vig) combined payout multiplier for independent legs, from true probabilities."""
    combined_prob = 1.0
    for p in individual_probs:
        combined_prob *= p
    return 1 / combined_prob if combined_prob > 0 else float("inf")

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
