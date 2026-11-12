"""Which of a player's past games were only PART of a game (2026-11-12).

Found on J.J. McCarthy before TNF 11/12 (WAS @ NYG): his 2026 box scores were one game
off the bench (31 of 78 snaps) and one where he left with a hip injury (51 of 67), but
every rolling baseline counted them as full games. His per-game TD / INT / yards
averages were diluted, so the model's INT Under 0.5 said 63% when ~48% was right. The
same thing happens to any player who leaves hurt, gets benched or comes in late: the
box score is real, but it isn't evidence of what he does in a full game, which is what
an Underdog line is posted for.

A game is "partial" when his snap share that game is well below his own usual share:

- every position: share < PARTIAL_RATIO x his usual share, where "usual" is the median
  of his previous USUAL_GAMES games with at least MIN_SHARE of the snaps (so earlier
  cameos don't drag "usual" down). Offense share for offensive players, defense share
  for defenders.
- QBs as well: share < QB_FULL_SHARE when he usually plays QB_FULL_SHARE+ (starters
  play ~100% of snaps, so 76% means he missed a quarter).

A one-time drop is flagged; a lasting role change is not for long, because "usual"
follows his last few games. Games with no snap row (no ID match) count as full, which
is what every baseline did before.

Rolling baselines can skip partial games (feature_engineering.cross_season_rolling's
count_col); the game stays a training LABEL, since a line settles on it either way.

RESULT (tested 2026-11-12): it does NOT help, so MODE defaults to "off" and nothing
changes. Kept so the test can be re-run (PARTIAL_GAMES=all or =qb).
- Held out, 2018-2026, correlation of the baseline with the next game, expected
  starters/regulars only: plain per-game average beats every adjustment. QB (n=4,210):
  pass yds .331 plain vs .304 dropping partial games vs .307 per-snap rate vs .319
  dropping only relief appearances (pbp: first dropback after the 1st quarter); att
  .277 vs .234; TDs .256 vs .241. WR/TE receptions .519 vs .513, RB carries .619 vs .600.
  Partial games carry signal (fragile or shaky players keep getting hurt or pulled), and
  dropping them stretches the window into older games.
- 2026 real Underdog lines through the form model (weeks 1, 4-9, 18,866 props): log loss
  .5996 -> .6023 dropping partial games for everyone. QBs only: overall unchanged;
  pass att / comps / yds slightly better (still under 50% right), INTs .673 -> .681 and
  TDs .652 -> .660 worse -- worse on exactly the INT Under that prompted this.
"""
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

PARTIAL_RATIO = 0.6
QB_FULL_SHARE = 0.85
USUAL_GAMES = 4
MIN_SHARE = 0.15
ROLE_CHANGE_GAMES = 3

DEFENSIVE = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "DB", "S", "SS", "FS", "SAF", "EDGE", "DL"}


def _crosswalk() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(RAW_DIR, "players.csv"), low_memory=False,
                    usecols=["gsis_id", "pfr_id"])
    return p.dropna().drop_duplicates("pfr_id").rename(columns={"gsis_id": "player_id",
                                                                 "pfr_id": "pfr_player_id"})


def _flag(s: pd.DataFrame):
    """Walk each player's games in order. "Usual" = median share of his last USUAL_GAMES
    accepted games; a partial game isn't accepted (one injury exit mustn't lower the bar
    for the next), unless ROLE_CHANGE_GAMES real games in a row come in low -- then it's
    his new role and those games become his usual."""
    pid = s["pfr_player_id"].to_numpy()
    share = s["share"].to_numpy(float)
    is_qb = (s["position"] == "QB").to_numpy()
    usual = np.full(len(s), np.nan)
    partial = np.zeros(len(s), bool)
    acc, pend, prev = [], [], None
    for i in range(len(s)):
        if pid[i] != prev:
            acc, pend, prev = [], [], pid[i]
        x = share[i]
        if np.isnan(x):
            continue
        u = float(np.median(acc[-USUAL_GAMES:])) if acc else np.nan
        usual[i] = u
        if is_qb[i]:
            # A starter plays ~every snap: under QB_FULL_SHARE he missed real time.
            partial[i] = x < QB_FULL_SHARE
            continue
        low = bool(u == u and x < PARTIAL_RATIO * u)
        partial[i] = low
        if x < MIN_SHARE:
            continue
        if low:
            pend.append(x)
            if len(pend) >= ROLE_CHANGE_GAMES:
                acc, pend = acc + pend, []
        else:
            acc.append(x)
            pend = []
    return usual, partial


def snap_flags(snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """(player_id [gsis], season, week) -> share, usual_share, partial."""
    if snaps is None:
        snaps = pd.read_csv(os.path.join(RAW_DIR, "snap_counts.csv"), low_memory=False)
    s = snaps.drop_duplicates(["pfr_player_id", "season", "week"]).copy()
    defense = s["position"].isin(DEFENSIVE) | (s["defense_snaps"].fillna(0) > s["offense_snaps"].fillna(0))
    s["share"] = np.where(defense, s["defense_pct"], s["offense_pct"]).astype(float)
    s = s.sort_values(["pfr_player_id", "season", "week"]).reset_index(drop=True)
    s["usual_share"], s["partial"] = _flag(s)
    s = s.merge(_crosswalk(), on="pfr_player_id", how="inner")
    return (s[["player_id", "season", "week", "share", "usual_share", "partial"]]
            .drop_duplicates(["player_id", "season", "week"]))


# "off" (default, see RESULT above), "all" or "qb" (QBs only).
MODE = os.environ.get("PARTIAL_GAMES", "off")

_CACHE: dict = {}


def full_game(df: pd.DataFrame, id_col: str = "player_id") -> pd.Series:
    """Boolean Series aligned to df: False for that player's partial games."""
    if MODE == "off":
        return pd.Series(True, index=df.index)
    if "flags" not in _CACHE:
        _CACHE["flags"] = snap_flags()
    f = _CACHE["flags"]
    m = df[[id_col, "season", "week"]].astype({id_col: str}).merge(
        f.astype({"player_id": str}).rename(columns={"player_id": id_col})[[id_col, "season", "week", "partial"]],
        on=[id_col, "season", "week"], how="left")
    part = m["partial"].fillna(False).astype(bool).to_numpy()
    if MODE == "qb" and "position" in df.columns:
        part = part & (df["position"].to_numpy() == "QB")
    return pd.Series(~part, index=df.index)


if __name__ == "__main__":
    f = snap_flags()
    print(f["partial"].mean(), len(f))
