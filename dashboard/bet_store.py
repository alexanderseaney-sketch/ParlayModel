"""
Persistent bet-log storage (added 2026-10-04).

The hosted app (Streamlit Community Cloud) has no persistent disk: bet_logs/bets.csv
written there vanished on every redeploy -- which happens every 6 hours when the data
refresh commits. So the bet log lives in its own PRIVATE GitHub repo
(alexanderseaney-sketch/parlaymodel-bets, file bets.csv on main), read and written
through the GitHub Contents API. A separate repo, not a branch of ParlayModel, because
ParlayModel is public; and writing there never touches ParlayModel's main, so logging a
bet never triggers a redeploy.

Configure with a GitHub token that can read/write that repo's contents:
  BET_LOG_GITHUB_TOKEN = "github_pat_..."         (Streamlit secrets, or .env locally)
  BET_LOG_REPO = "owner/repo"                     (optional, default below)
Without a token everything falls back to the local bet_logs/bets.csv, exactly as before.

Every write is a commit, so the repo's history doubles as an audit trail of the log.
Concurrent writers are handled with the file's blob sha (optimistic locking): a stale
write gets 409/422 from GitHub and append() re-reads and retries.
"""
import base64
import io
import os

import pandas as pd
import requests

DEFAULT_REPO = "alexanderseaney-sketch/parlaymodel-bets"
FILE_PATH = "bets.csv"
BRANCH = "main"
API = "https://api.github.com"
COLUMNS = ["date", "sport", "player", "stat", "choice", "line", "multiplier_or_odds",
           "stake", "result", "notes", "logged_at", "entry_id", "entry_payout"]


class BetStoreError(RuntimeError):
    pass


def _setting(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _token() -> str | None:
    return _setting("BET_LOG_GITHUB_TOKEN")


def repo() -> str:
    return _setting("BET_LOG_REPO") or DEFAULT_REPO


def enabled() -> bool:
    return bool(_token())


def _headers() -> dict:
    return {"Authorization": f"Bearer {_token()}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _url() -> str:
    return f"{API}/repos/{repo()}/contents/{FILE_PATH}"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def read() -> tuple[pd.DataFrame, str | None]:
    """(bets, blob sha). sha is None when the file doesn't exist yet."""
    resp = requests.get(_url(), headers=_headers(), params={"ref": BRANCH}, timeout=15)
    if resp.status_code == 404:
        return _empty(), None
    if resp.status_code in (401, 403):
        raise BetStoreError(f"GitHub rejected the bet-log token ({resp.status_code}) -- check "
                            f"BET_LOG_GITHUB_TOKEN has Contents read/write on {repo()}.")
    resp.raise_for_status()
    body = resp.json()
    raw = base64.b64decode(body["content"]).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw)) if raw.strip() else _empty()
    return df, body["sha"]


def write(df: pd.DataFrame, sha: str | None, message: str) -> str:
    """Commit df as bets.csv. Raises BetStoreError on a stale sha (someone else wrote
    first) so the caller can re-read; returns the new blob sha."""
    payload = {"message": message, "branch": BRANCH,
               "content": base64.b64encode(df.to_csv(index=False).encode("utf-8")).decode("ascii")}
    if sha:
        payload["sha"] = sha
    resp = requests.put(_url(), headers=_headers(), json=payload, timeout=20)
    if resp.status_code in (409, 422):
        raise BetStoreError("stale")
    if resp.status_code in (401, 403, 404):
        raise BetStoreError(f"GitHub rejected the bet-log write ({resp.status_code}) -- check "
                            f"BET_LOG_GITHUB_TOKEN has Contents read/write on {repo()}.")
    resp.raise_for_status()
    return resp.json()["content"]["sha"]


def append(rows: list[dict], attempts: int = 3) -> None:
    """Append one or more bet rows in a single commit (a whole entry's legs together)."""
    for i in range(attempts):
        df, sha = read()
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        first = rows[0]
        msg = (f"Log bet: {first.get('player')} {first.get('choice')} {first.get('line')} {first.get('stat')}"
               + (f" (+{len(rows) - 1} more legs)" if len(rows) > 1 else ""))
        try:
            write(df, sha, msg)
            return
        except BetStoreError as e:
            if str(e) != "stale" or i == attempts - 1:
                raise


KEY = ["logged_at", "player", "stat", "line"]


def update_results(edited: pd.DataFrame, attempts: int = 3) -> None:
    """Apply the Bet Log page's result edits onto the LATEST stored log (re-read, not the
    copy the page rendered), matched on KEY -- so a bet logged meanwhile (another tab, the
    assistant) is never overwritten."""
    for i in range(attempts):
        latest, sha = read()
        if latest.empty:
            return
        keyed = edited.assign(_k=edited[KEY].astype(str).agg("|".join, axis=1)).set_index("_k")["result"]
        k = latest[KEY].astype(str).agg("|".join, axis=1)
        new_results = k.map(keyed)
        latest["result"] = new_results.where(new_results.notna(), latest["result"])
        try:
            write(latest, sha, "Update bet results")
            return
        except BetStoreError as e:
            if str(e) != "stale" or i == attempts - 1:
                raise
