"""
Fantasy football page: weekly projections & rankings, start/sit, defense-vs-position
matchups, and boom/bust profiles. Own module (like compare.py) -- app.py imports
page_fantasy() and registers it.

Projections roll up the prop model's own proxy_line per stat (its trailing rolling
average for that player) into fantasy points -- see models/fantasy_scoring.py. The
"lean" column is the model's average P(over) across that player's props: a real
signal for which way the model expects them to break from that baseline, kept
separate from the projection itself rather than baked in.

Known gap: the prop model has no RB-receptions market, so RB projections carry
receiving yards + TDs but no per-reception points. Flagged in the UI.
"""
import os

import pandas as pd
import streamlit as st

from utils import (
    CURRENT_PREDICTIONS_PATH, RAW_DIR, load_csv_if_exists, load_current_predictions,
    normalize_name,
)
from fantasy_scoring import SCORING_LABELS, project_points, fantasy_points_from_weekly

GAME_PROPS = ["passing_yards", "passing_tds", "passing_ints", "rushing_yards",
              "rushing_yards_qb", "receiving_yards", "receiving_yards_rb",
              "receptions", "rush_rec_tds", "rush_rec_tds_qb"]
# props where "over" cleanly means "more fantasy production" -- the lean signal uses
# only these (TD props sit near a 0.5 line and passing_ints inverts, so averaging
# P(over) across everything just pins every player to "under").
VOLUME_PROPS = ["passing_yards", "rushing_yards", "rushing_yards_qb",
                "receiving_yards", "receiving_yards_rb", "receptions"]
POSITIONS = ["QB", "RB", "WR", "TE"]
FLEX = ["RB", "WR", "TE"]

# fixed per-game fantasy thresholds for boom / bust rate (position-relative)
BOOM = {"QB": 25, "RB": 20, "WR": 20, "TE": 15}
BUST = {"QB": 12, "RB": 6, "WR": 6, "TE": 4}


def _pred_mtime() -> float:
    return os.path.getmtime(CURRENT_PREDICTIONS_PATH) if os.path.exists(CURRENT_PREDICTIONS_PATH) else 0.0


def _weekly_mtime() -> float:
    p = os.path.join(RAW_DIR, "weekly_stats.csv")
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def _lean_score(p) -> float:
    """The prop model's average over-probability on a player's yardage/reception
    props, as a 0-100 number. 50 is neutral; higher = the model expects them above
    their own recent form. Shown as a secondary signal, not folded into the
    projection."""
    return round(p * 100) if pd.notna(p) else None


# --------------------------------------------------------------- projections

@st.cache_data(show_spinner=False)
def _projections(pred_mtime: float, scoring: str) -> pd.DataFrame:
    preds = load_current_predictions()
    if preds is None:
        return pd.DataFrame()
    g = preds[preds["prop_type"].isin(GAME_PROPS)].copy()
    if g.empty:
        return pd.DataFrame()

    wide = g.pivot_table(index="player_id", columns="prop_type", values="proxy_line", aggfunc="first")
    for col in GAME_PROPS:
        if col not in wide.columns:
            wide[col] = pd.NA
    meta = (g.groupby("player_id")
            .agg(player=("player_display_name", "first"), position=("position", "first"),
                 team=("recent_team", "first"), opp=("next_opponent", "first"),
                 week=("next_week", "first")))
    lean = (g[g["prop_type"].isin(VOLUME_PROPS)]
            .groupby("player_id")["predicted_prob_over"].mean().rename("lean"))

    out = meta.join(lean).join(wide)
    out = out[out["position"].isin(POSITIONS)].copy()
    out["proj"] = out.apply(lambda r: round(project_points(r.to_dict(), scoring), 1), axis=1)

    out["pass_yds"] = out["passing_yards"]
    out["rush_yds"] = out[["rushing_yards", "rushing_yards_qb"]].sum(axis=1, min_count=1)
    out["rec_yds"] = out[["receiving_yards", "receiving_yards_rb"]].sum(axis=1, min_count=1)
    # receptions: real market for WR/TE, estimated from RB receiving yards otherwise
    # (matches project_points()'s own fallback)
    out["rec"] = out["receptions"].where(
        out["receptions"].notna(), out["receiving_yards_rb"] / 7.5).round(1)
    out["td"] = out[["rush_rec_tds", "rush_rec_tds_qb"]].sum(axis=1, min_count=1)

    out = out.reset_index(drop=True).sort_values("proj", ascending=False)
    out["pos_rank"] = out.groupby("position")["proj"].rank(method="first", ascending=False).astype(int)
    return out


def _tiers(proj: pd.Series) -> list:
    """New tier whenever the drop from the previous player is >= 1.5 fantasy points --
    a rough but readable way to show where the cliffs are within a position."""
    tiers, t, prev = [], 1, None
    for v in proj:
        if prev is not None and prev - v >= 1.5:
            t += 1
        tiers.append(t)
        prev = v
    return tiers


def _render_projections(scoring: str):
    df = _projections(_pred_mtime(), scoring)
    if df.empty:
        st.warning("No current predictions — run **Run Data Pulls** and regenerate predictions.")
        return

    pos = st.radio("Position", ["All", "QB", "RB", "WR", "TE", "FLEX"], horizontal=True,
                   key="ff_proj_pos")
    if pos == "All":
        view = df
    elif pos == "FLEX":
        view = df[df["position"].isin(FLEX)]
    else:
        view = df[df["position"] == pos]
    view = view.sort_values("proj", ascending=False).copy()
    if pos in POSITIONS:
        view["Tier"] = _tiers(view["proj"])
    view["Model"] = view["lean"].map(_lean_score)

    wk = int(df["week"].mode().iloc[0]) if not df["week"].mode().empty else "?"
    st.caption(f"{SCORING_LABELS[scoring]} · projected points for week {wk} · rolled up from "
               f"the prop model's per-stat trailing averages. **Model** = its avg "
               f"over-probability on this player's yardage/reception props (50 = neutral, "
               f"higher = expects them above recent form). RB receptions estimated from RB "
               f"receiving yards (no RB-receptions market in the model).")

    cols = (["pos_rank"] + (["Tier"] if "Tier" in view.columns else [])
            + ["player", "position", "team", "opp", "proj", "Model",
               "pass_yds", "rush_yds", "rec_yds", "rec", "td"])
    labels = {"pos_rank": "Pos#", "player": "Player", "position": "Pos", "team": "Team",
              "opp": "Opp", "proj": "Proj", "pass_yds": "PaYd", "rush_yds": "RuYd",
              "rec_yds": "ReYd", "rec": "Rec", "td": "TD"}
    st.dataframe(view[cols].rename(columns=labels).round(1),
                 hide_index=True, width="stretch", height=560)


# ------------------------------------------------------------------ start/sit

def _next_game_totals() -> dict:
    sched = load_csv_if_exists("schedules.csv")
    if sched is None or "total_line" not in sched.columns:
        return {}
    unplayed = sched[sched["home_score"].isna()].sort_values(["season", "week"])
    totals = {}
    for side in ("home_team", "away_team"):
        for _, r in unplayed.groupby(side).head(1).iterrows():
            totals[r[side]] = r.get("total_line")
    return totals


def _matchup_rank(matchups, def_team: str, position: str, scoring: str):
    if matchups is None:
        return None, None
    row = matchups[(matchups["def_team"] == def_team) & (matchups["position"] == position)]
    if row.empty:
        return None, None
    rank_col = f"rank_{scoring}" if f"rank_{scoring}" in row.columns else "rank_ppr"
    pg_col = f"fp_{scoring}_pg" if f"fp_{scoring}_pg" in row.columns else "fp_ppr_pg"
    return int(row.iloc[0][rank_col]), float(row.iloc[0][pg_col])


def _render_start_sit(scoring: str):
    df = _projections(_pred_mtime(), scoring)
    if df.empty:
        st.warning("No current predictions available.")
        return
    matchups = load_csv_if_exists("fantasy_matchups.csv")
    rosters = load_csv_if_exists("nfl_rosters.csv")
    totals = _next_game_totals()

    df = df.sort_values("proj", ascending=False)
    options = df["player"].tolist()
    picks = st.multiselect("Players to compare (2–6)", options, default=options[:2],
                           max_selections=6, key="ff_ss_players")
    if len(picks) < 2:
        st.info("Pick at least two players.")
        return

    active = set()
    if rosters is not None and "roster_status" in rosters.columns:
        act = rosters[rosters["roster_status"].fillna("").str.startswith("Active")]
        active = set(act["player"].apply(normalize_name))

    rows = []
    for name in picks:
        r = df[df["player"] == name].iloc[0]
        rank, pg = _matchup_rank(matchups, r["opp"], r["position"], scoring)
        on_roster = (normalize_name(name) in active) if active else None
        rows.append({
            "Player": name, "Pos": r["position"], "Team": r["team"], "Opp": r["opp"],
            "Proj": r["proj"], "Model": _lean_score(r["lean"]),
            "Matchup": f"#{rank}" if rank else "—",
            "Pts allowed/gm": round(pg, 1) if pg else None,
            "Game total": totals.get(r["team"]),
            "Roster": ("Active" if on_roster else "not on Active roster") if on_roster is not None else "—",
        })
    table = pd.DataFrame(rows).sort_values("Proj", ascending=False).reset_index(drop=True)

    top = table.iloc[0]
    st.markdown(f"### Start: **{top['Player']}** — {top['Proj']} proj ({SCORING_LABELS[scoring]})")
    reasons = [f"top projection of the {len(picks)}"]
    if top["Matchup"] != "—":
        rank = int(top["Matchup"].lstrip("#"))
        if rank >= 22:
            reasons.append(f"soft matchup (D ranks #{rank} vs {top['Pos']}, {top['Pts allowed/gm']}/gm)")
        elif rank <= 10:
            reasons.append(f"tough on paper (D ranks #{rank} vs {top['Pos']})")
    if pd.notna(top["Game total"]):
        reasons.append(f"game total {top['Game total']}")
    if top["Roster"] not in ("Active", "—"):
        reasons.append(f"⚠️ {top['Roster']}")
    st.caption(" · ".join(reasons))

    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Ranked by projection. Matchup rank runs 1 (toughest defense vs this position) "
               "to 32 (softest). Use it to break ties and spot streamers, not to override a "
               "clear projection gap.")


# ------------------------------------------------------------------- matchups

def _render_matchups(scoring: str):
    m = load_csv_if_exists("fantasy_matchups.csv")
    if m is None:
        st.warning("`fantasy_matchups.csv` not built yet — run **Fantasy matchups** on Run Data Pulls.")
        return
    pg_col = f"fp_{scoring}_pg" if f"fp_{scoring}_pg" in m.columns else "fp_ppr_pg"

    grid = m.pivot(index="def_team", columns="position", values=pg_col)[POSITIONS]
    st.caption(f"{SCORING_LABELS[scoring]} fantasy points allowed **per game**, by defense and "
               f"position ({m['derived_from_seasons'].iloc[0]}). Higher = softer matchup for "
               f"that position's offense. Green = exploit, red = fade.")
    st.dataframe(grid.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.1f}"),
                 width="stretch", height=600)

    sched = load_csv_if_exists("schedules.csv")
    if sched is None:
        return
    unplayed = sched[sched["home_score"].isna()].sort_values(["season", "week"])
    if unplayed.empty:
        return
    wk = int(unplayed["week"].iloc[0])
    games = unplayed[unplayed["week"] == wk]
    rank_col = f"rank_{scoring}" if f"rank_{scoring}" in m.columns else "rank_ppr"
    rank_lookup = m.set_index(["def_team", "position"])[rank_col].to_dict()

    lines = []
    for _, g in games.iterrows():
        for off, dfn in ((g["away_team"], g["home_team"]), (g["home_team"], g["away_team"])):
            lines.append({"Offense": off, "vs D": dfn,
                          **{p: f"#{rank_lookup.get((dfn, p), '?')}" for p in POSITIONS}})
    st.markdown(f"##### Week {wk} — defensive rank each offense faces per position")
    st.caption("#1 = toughest defense vs that position … #32 = softest (best to target).")
    st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch")


# ------------------------------------------------------------------ boom/bust

@st.cache_data(show_spinner=False)
def _boom_bust(weekly_mtime: float, scoring: str, seasons: tuple) -> pd.DataFrame:
    w = load_csv_if_exists("weekly_stats.csv")
    if w is None:
        return pd.DataFrame()
    w = w[(w["season_type"] == "REG") & w["position"].isin(POSITIONS)].copy()
    if seasons:
        w = w[w["season"].isin(seasons)]
    w["fp"] = fantasy_points_from_weekly(w, scoring)

    rows = []
    for (pid, pos), sub in w.groupby(["player_id", "position"]):
        if len(sub) < 4:
            continue
        fp = sub["fp"]
        rows.append({
            "player_id": pid, "Player": sub["player_display_name"].iloc[-1], "Pos": pos,
            "G": len(sub), "PPG": round(fp.mean(), 1),
            "Floor": round(fp.quantile(0.25), 1), "Median": round(fp.median(), 1),
            "Ceiling": round(fp.quantile(0.85), 1),
            "Boom%": round(100 * (fp >= BOOM[pos]).mean()),
            "Bust%": round(100 * (fp <= BUST[pos]).mean()),
        })
    return pd.DataFrame(rows).sort_values("PPG", ascending=False)


def _render_boom_bust(scoring: str):
    w = load_csv_if_exists("weekly_stats.csv")
    if w is None:
        st.warning("`weekly_stats.csv` not pulled yet.")
        return
    seasons_all = sorted(w.loc[w["season_type"] == "REG", "season"].dropna().unique(), reverse=True)
    default = [s for s in seasons_all if s >= seasons_all[0] - 1] if seasons_all else []
    picks = st.multiselect("Seasons", seasons_all, default=default, key="ff_bb_seasons")
    bb = _boom_bust(_weekly_mtime(), scoring, tuple(sorted(picks)))
    if bb.empty:
        st.caption("Not enough games in the selected seasons.")
        return

    pos = st.radio("Position", POSITIONS, horizontal=True, key="ff_bb_pos")
    view = bb[bb["Pos"] == pos].drop(columns=["player_id", "Pos"])
    st.caption(f"{SCORING_LABELS[scoring]} per-game outcomes across the selected seasons. "
               f"Floor / Ceiling = 25th / 85th-percentile game. Boom ≥ {BOOM[pos]}, "
               f"Bust ≤ {BUST[pos]}. Min 4 games.")
    st.dataframe(view, hide_index=True, width="stretch", height=560, column_config={
        "Boom%": st.column_config.ProgressColumn("Boom%", min_value=0, max_value=100, format="%d%%"),
        "Bust%": st.column_config.ProgressColumn("Bust%", min_value=0, max_value=100, format="%d%%"),
    })


# ----------------------------------------------------------------------- page

def page_fantasy():
    st.title("🏆 Fantasy")
    st.caption(
        "Weekly projections, start/sit, defense-vs-position matchups, and boom/bust. "
        "Projections roll up the prop model's per-stat trailing averages; matchups and "
        "boom/bust come from weekly box scores."
    )
    label = st.radio("Scoring", ["PPR", "Half-PPR"], horizontal=True, key="ff_scoring")
    scoring = "ppr" if label == "PPR" else "half"

    tabs = st.tabs(["Projections", "Start / Sit", "Matchups", "Boom / Bust"])
    with tabs[0]:
        _render_projections(scoring)
    with tabs[1]:
        _render_start_sit(scoring)
    with tabs[2]:
        _render_matchups(scoring)
    with tabs[3]:
        _render_boom_bust(scoring)
