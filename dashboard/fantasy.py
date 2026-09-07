"""
Fantasy football page: weekly projections & rankings, start/sit, defense-vs-position
matchups, and boom/bust profiles. Own module (like compare.py) -- app.py imports
page_fantasy() and registers it.

Projections roll up the prop model's own proxy_line per stat (its trailing rolling
average for that player) into fantasy points -- see dashboard/fantasy_scoring.py. The
"lean" column is the model's average P(over) across that player's props: a real
signal for which way the model expects them to break from that baseline, kept
separate from the projection itself rather than baked in.

Known gap: the prop model has no RB-receptions market, so RB projections carry
receiving yards + TDs but no per-reception points. Flagged in the UI.
"""
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    CURRENT_PREDICTIONS_PATH, RAW_DIR, load_csv_if_exists, load_current_predictions,
    load_player_photos, load_player_jersey_numbers, normalize_name,
)
from fantasy_scoring import (
    SCORING_LABELS, project_points, project_breakdown, fantasy_points_from_weekly,
    project_season_points,
)

try:
    import yahoo_fantasy as yf
except Exception:  # noqa: BLE001 -- the page must still load if the optional module can't import
    yf = None

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

# projection drop (fantasy points) that starts a new tier -- weekly scale for the
# in-season Projections tab, full-season scale for the draft board.
_WEEKLY_TIER_GAP = {"QB": 3.0, "RB": 2.5, "WR": 2.5, "TE": 2.0}
_SEASON_TIER_GAP = {"QB": 18, "RB": 22, "WR": 20, "TE": 14}

_STATUS_LABELS = {"IR": "Injured Reserve", "PUP": "Reserve/PUP", "NFI": "Reserve/NFI",
                  "SUS": "Suspended", "NR": "Reserve/Did not report", "RET": "Reserve/Retired",
                  "EXE": "Commissioner exempt", "PS": "Practice squad"}


def _spotrac_row(name: str, team: str):
    """This player's row from spotrac_contracts.csv (contract + reserve status),
    matched on normalized name within the team. None if not pulled / not found."""
    sc = load_csv_if_exists("spotrac_contracts.csv")
    if sc is None:
        return None
    key = normalize_name(name)
    m = sc[(sc["team_abbr"] == team) & (sc["player"].apply(normalize_name) == key)]
    if m.empty:
        return None
    # a player can appear on both Active and a reserve list -- prefer the reserve one
    non_active = m[m["roster_status"] != "Active"]
    return (non_active if not non_active.empty else m).iloc[0]


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

    out = out[out["proj"] > 0].sort_values("proj", ascending=False).reset_index()  # keep player_id
    out["pos_rank"] = out.groupby("position")["proj"].rank(method="first", ascending=False).astype(int)
    out["overall_rank"] = range(1, len(out) + 1)   # across all QB/RB/WR/TE
    out["tier"] = 0
    for pos in POSITIONS:
        m = out["position"] == pos
        out.loc[m, "tier"] = _tier(
            out.loc[m].sort_values("proj", ascending=False)["proj"], _WEEKLY_TIER_GAP[pos])
    return out


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
    view["Tier"] = view["tier"]
    view["Model"] = pd.array([_lean_score(v) for v in view["lean"]], dtype="Int64")

    wk = int(df["week"].mode().iloc[0]) if not df["week"].mode().empty else "?"
    st.caption(f"{SCORING_LABELS[scoring]} · projected points for week {wk} · rolled up from "
               f"the prop model's per-stat trailing averages. **Model** = its avg "
               f"over-probability on this player's yardage/reception props (50 = neutral, "
               f"higher = expects them above recent form). RB receptions estimated from RB "
               f"receiving yards (no RB-receptions market in the model).")

    cols = ["pos_rank", "Tier", "player", "position", "team", "opp", "proj", "Model",
            "pass_yds", "rush_yds", "rec_yds", "rec", "td"]
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


# ---------------------------------------------------------------- player card

def _player_weekly(weekly: pd.DataFrame, player_id: str, scoring: str) -> pd.DataFrame:
    sub = weekly[(weekly["player_id"] == player_id) & (weekly["season_type"] == "REG")].copy()
    sub = sub.sort_values(["season", "week"])
    sub["fp"] = fantasy_points_from_weekly(sub, scoring)
    return sub


def _usage_shares(weekly: pd.DataFrame, player_rows: pd.DataFrame, season) -> dict:
    """Target share, air-yards share and WOPR for a receiver's season. Recomputed
    from raw targets / receiving_air_yards vs the player's team totals, because the
    pbp-derived seasons (2025) leave weekly_stats' own air_yards_share / wopr / racr
    columns NaN even though the underlying counting stats are there."""
    cur = player_rows[player_rows["season"] == season]
    if cur.empty:
        return {}
    team = cur["recent_team"].mode().iloc[0]
    team_wk = (weekly[(weekly["season"] == season) & (weekly["recent_team"] == team)
                      & (weekly["season_type"] == "REG")]
               .groupby("week")[["targets", "receiving_air_yards"]].sum()
               .rename(columns={"targets": "team_tgt", "receiving_air_yards": "team_ay"}))
    j = cur.merge(team_wk, left_on="week", right_index=True, how="left")
    tgt_share = (j["targets"] / j["team_tgt"]).replace([float("inf")], pd.NA).mean()
    ay_share = (j["receiving_air_yards"] / j["team_ay"]).replace([float("inf")], pd.NA).mean()
    wopr = (1.5 * tgt_share + 0.7 * ay_share) if pd.notna(tgt_share) and pd.notna(ay_share) else None
    return {"target_share": tgt_share, "air_yards_share": ay_share, "wopr": wopr}


def _consistency_label(fp: pd.Series) -> str:
    if len(fp) < 4 or fp.mean() <= 0:
        return "—"
    cv = fp.std() / fp.mean()
    return "Steady" if cv < 0.5 else "Boom-or-bust" if cv > 0.85 else "Average variance"


def _implied_team_total(sched: pd.DataFrame, team: str):
    """Half the game total, shifted by half the spread toward the favorite."""
    if sched is None or "total_line" not in sched.columns:
        return None, None
    unplayed = sched[sched["home_score"].isna()].sort_values(["season", "week"])
    game = unplayed[(unplayed["home_team"] == team) | (unplayed["away_team"] == team)].head(1)
    if game.empty:
        return None, None
    g = game.iloc[0]
    total = g.get("total_line")
    spread = g.get("spread_line")  # nflverse: home-team spread, negative = home favored
    if pd.isna(total):
        return None, None
    if pd.isna(spread):
        return round(total / 2, 1), total
    home = g["home_team"] == team
    team_spread = spread if home else -spread
    return round(total / 2 - team_spread / 2, 1), total


def _render_player_card(scoring: str):
    proj = _projections(_pred_mtime(), scoring)
    if proj.empty:
        st.warning("No current predictions — run **Run Data Pulls** and regenerate predictions.")
        return
    weekly = load_csv_if_exists("weekly_stats.csv")
    matchups = load_csv_if_exists("fantasy_matchups.csv")
    snaps = load_csv_if_exists("snap_counts.csv")
    sched = load_csv_if_exists("schedules.csv")
    photos = load_player_photos()
    jerseys = load_player_jersey_numbers()

    proj = proj.sort_values("proj", ascending=False)
    label_map = {f"{r.player}  ·  {r.position}{r.pos_rank} ({r.team})": r.player_id
                 for r in proj.itertuples()}
    pick = st.selectbox("Player", list(label_map), key="ff_card_player")
    p = proj[proj["player_id"] == label_map[pick]].iloc[0]
    pid, name, position, team = p["player_id"], p["player"], p["position"], p["team"]
    n_pos = int((proj["position"] == position).sum())

    # ---- header
    head = st.columns([1, 5])
    with head[0]:
        photo = photos.get(normalize_name(name))
        if isinstance(photo, str) and photo:
            st.markdown(f'<img src="{photo}" class="pm-dialog-photo">', unsafe_allow_html=True)
        else:
            initials = "".join(w[0] for w in name.split()[:2]).upper()
            st.markdown(f'<div class="pm-dialog-photo-placeholder">{initials}</div>',
                        unsafe_allow_html=True)
    with head[1]:
        jersey = jerseys.get(normalize_name(name))
        st.subheader(name)
        st.markdown(f"**{position} · {team}**" + (f" · #{int(jersey)}" if jersey else "")
                    + f"  ·  Week {int(p['week'])} vs **{p['opp']}**")
        st.markdown(f"**{position}{int(p['pos_rank'])}** of {n_pos}  ·  Tier {int(p['tier'])}  "
                    f"·  #{int(p['overall_rank'])} overall (QB/RB/WR/TE)")
        sp = _spotrac_row(name, team)
        if sp is not None:
            if sp["roster_status"] != "Active":
                st.warning(f"Spotrac: **{_STATUS_LABELS.get(sp['roster_status'], sp['roster_status'])}**"
                           + (f" — {sp['reason'].title()}" if isinstance(sp.get("reason"), str) else ""))
            bits = []
            if pd.notna(sp.get("cap_hit")):
                bits.append(f"Cap hit **${sp['cap_hit'] / 1e6:.1f}M**"
                            + (f" ({sp['cap_pct']:.1f}% of cap)" if pd.notna(sp.get("cap_pct")) else ""))
            if pd.notna(sp.get("fa_year")):
                bits.append(f"free agent **{int(sp['fa_year'])}**")
            if pd.notna(sp.get("age")):
                bits.append(f"age {int(sp['age'])}")
            if bits:
                st.caption("Contract (Spotrac): " + "  ·  ".join(bits))

    st.divider()

    # ---- this week's projection + fantasy point breakdown
    st.subheader("This week's projection")
    c = st.columns([1, 2])
    c[0].metric(f"Projected points ({SCORING_LABELS[scoring]})", f"{p['proj']:.1f}")
    c[0].metric("Model lean (0–100)", _lean_score(p["lean"]),
                help="The prop model's average over-probability on this player's "
                     "yardage/reception props. 50 = neutral; higher = the model expects "
                     "them above their recent form. Not baked into the projection.")
    bd = pd.DataFrame(project_breakdown(p.to_dict(), scoring))
    if not bd.empty:
        bd = bd.rename(columns={"category": "Category", "projected": "Projected stat",
                                "points": "Fantasy points", "rule": "Scoring rule"})
        bd.loc[len(bd)] = ["Total projection", pd.NA, round(bd["Fantasy points"].sum(), 1), pd.NA]
        bd["Projected stat"] = bd["Projected stat"].astype("Float64")
        bd["Scoring rule"] = bd["Scoring rule"].astype("string")
        c[1].dataframe(bd, hide_index=True, width="stretch")

    st.divider()

    # ---- matchup
    st.subheader("Matchup")
    mrank, mpg = _matchup_rank(matchups, p["opp"], position, scoring)
    team_total, game_total = _implied_team_total(sched, team)
    mc = st.columns(4)
    if mrank:
        verdict = "great" if mrank >= 27 else "good" if mrank >= 20 else \
                  "tough" if mrank <= 6 else "average"
        mc[0].metric(f"{p['opp']} defense vs {position}", f"#{mrank} of 32",
                     help="1 = toughest defense against this position, 32 = softest.")
        mc[1].metric(f"{p['opp']} allows to {position}", f"{mpg:.1f} / game")
    mc[2].metric("Game total", f"{game_total:g}" if game_total else "—")
    mc[3].metric(f"{team} implied total", f"{team_total:g}" if team_total else "—",
                 help="Half the game total, adjusted for the point spread — how many "
                      "points Vegas expects this offense to score.")
    if mrank:
        st.caption(f"**{verdict.title()} matchup** — {p['opp']} ranks #{mrank} of 32 against "
                   f"{position}s this season ({mpg:.1f} {SCORING_LABELS[scoring]} pts/game allowed).")

    st.divider()

    # ---- recent form + season profile
    if weekly is not None:
        wk = _player_weekly(weekly, pid, scoring)
        if not wk.empty:
            st.subheader("Recent form")
            last = wk.tail(6)
            chart = last.assign(Game=last["season"].astype(str).str[-2:] + " wk" + last["week"].astype(str))
            st.bar_chart(chart.set_index("Game")["fp"], height=200, y_label=f"{SCORING_LABELS[scoring]} pts")
            l5 = wk.tail(5)["fp"]
            st.caption("Last 5: " + " · ".join(f"{v:.1f}" for v in l5)
                       + f"  →  {l5.mean():.1f} avg" + (
                           f" (season avg {wk[wk['season'] == wk['season'].max()]['fp'].mean():.1f})"
                           if (wk["season"] == wk["season"].max()).any() else ""))

            st.subheader("Season profile")
            cur, prev = wk["season"].max(), wk["season"].max() - 1
            cur_fp, prev_fp = wk[wk["season"] == cur]["fp"], wk[wk["season"] == prev]["fp"]
            hist = wk[wk["season"] >= prev]["fp"]  # last two seasons for the distribution
            sp = st.columns(4)
            sp[0].metric(f"{int(cur)} pts/game", f"{cur_fp.mean():.1f}" if len(cur_fp) else "—",
                         help=f"{len(cur_fp)} games played in {int(cur)}")
            sp[1].metric(f"{int(prev)} pts/game", f"{prev_fp.mean():.1f}" if len(prev_fp) else "—",
                         help=f"{len(prev_fp)} games played in {int(prev)}")
            sp[2].metric("Consistency", _consistency_label(hist),
                         help="Game-to-game variation in fantasy points (coefficient of "
                              "variation): Steady < 0.5, Boom-or-bust > 0.85.")
            sp[3].metric("Games (last 2 yr)", int(len(hist)))
            if len(hist) >= 4:
                fl = st.columns(4)
                fl[0].metric("Floor", f"{hist.quantile(0.25):.1f}", help="25th-percentile game")
                fl[1].metric("Median game", f"{hist.median():.1f}")
                fl[2].metric("Ceiling", f"{hist.quantile(0.85):.1f}", help="85th-percentile game")
                boom = 100 * (hist >= BOOM.get(position, 20)).mean()
                bust = 100 * (hist <= BUST.get(position, 6)).mean()
                fl[3].metric("Boom / Bust rate", f"{boom:.0f}% / {bust:.0f}%",
                             help=f"Share of games ≥ {BOOM.get(position, 20)} pts / ≤ "
                                  f"{BUST.get(position, 6)} pts, over the last two seasons.")

            st.divider()

            # ---- usage & role
            st.subheader("Usage & role")
            cs = wk[wk["season"] == cur]
            u = st.columns(4)
            if snaps is not None:
                sn = snaps[(snaps["player"].apply(normalize_name) == normalize_name(name))
                           & (snaps["season"] == cur) & (snaps["game_type"] == "REG")]
                u[0].metric("Snap share", f"{100 * sn['offense_pct'].mean():.0f}%"
                            if len(sn) else "—", help=f"Offensive snaps, {int(cur)} avg")
            if position in ("WR", "TE"):
                sh = _usage_shares(weekly, wk, cur)
                ts, ays, wopr = sh.get("target_share"), sh.get("air_yards_share"), sh.get("wopr")
                u[1].metric("Target share", f"{100 * ts:.0f}%" if pd.notna(ts) else "—",
                            help="Share of the team's targets while he was active.")
                u[2].metric("Air-yards share", f"{100 * ays:.0f}%" if pd.notna(ays) else "—",
                            help="Share of the team's downfield passing volume.")
                u[3].metric("WOPR", f"{wopr:.2f}" if wopr is not None else "—",
                            help="Weighted Opportunity Rating — 1.5·target share + 0.7·air-yards "
                                 "share. ~0.7+ is a clear WR1 workload.")
            elif position == "RB":
                u[1].metric("Carries / game", f"{cs['carries'].mean():.1f}" if len(cs) else "—")
                u[2].metric("Targets / game", f"{cs['targets'].mean():.1f}" if len(cs) else "—")
                u[3].metric("Touches / game",
                            f"{(cs['carries'].fillna(0) + cs['targets'].fillna(0)).mean():.1f}"
                            if len(cs) else "—")
            elif position == "QB":
                u[1].metric("Pass att / game", f"{cs['attempts'].mean():.1f}" if len(cs) else "—")
                u[2].metric("Rush att / game", f"{cs['carries'].mean():.1f}" if len(cs) else "—")
                u[3].metric("Pass yд / game", f"{cs['passing_yards'].mean():.0f}" if len(cs) else "—")

    # ---- rest-of-season schedule
    if sched is not None and matchups is not None:
        st.divider()
        st.subheader("Next 4 weeks — schedule strength")
        unplayed = sched[sched["home_score"].isna()].sort_values(["season", "week"])
        mine = unplayed[(unplayed["home_team"] == team) | (unplayed["away_team"] == team)].head(4)
        rank_lookup = matchups.set_index(["def_team", "position"])[
            f"rank_{scoring}" if f"rank_{scoring}" in matchups.columns else "rank_ppr"].to_dict()
        rows = []
        for _, g in mine.iterrows():
            opp = g["away_team"] if g["home_team"] == team else g["home_team"]
            r = rank_lookup.get((opp, position))
            grade = "—" if r is None else "Great" if r >= 27 else "Good" if r >= 20 \
                else "Tough" if r <= 6 else "Neutral"
            rows.append({"Week": int(g["week"]), "Opponent": opp,
                         f"Opp D vs {position}": f"#{r} of 32" if r else "—", "Grade": grade})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Opponent defensive rank against this position (1 = toughest, "
                       "32 = softest), from this season's fantasy points allowed.")


# -------------------------------------------------------------- draft: season

SEASON_STATS = ["season_pass_yards", "season_pass_tds", "season_rush_yards",
                "season_rush_tds", "season_receiving_yards", "season_rec_tds"]
# starters per position for a default 12-team league; FLEX split 45/45/10 RB/WR/TE
DEFAULT_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}

# Underdog only prices season-long O/U on ~150 notable players. Past that the prop
# model's own season projections (its trailing per-game rate carried to a full
# season) fill the board so a 12-team draft has enough depth. The model runs a bit
# hot vs the market, so its points are scaled by the per-position median ratio of
# (market projection / model projection) over players priced by both. QB is left
# market-only: the model has no season passing line, so a QB's model season row is
# rushing-only and useless.
_MODEL_SEASON_PROPS = ["season_rush_yards", "season_rush_tds",
                       "season_receiving_yards", "season_rec_tds"]
_MODEL_SCALE = {"RB": 0.79, "WR": 1.00, "TE": 0.88}


def _clean_status(s) -> str:
    s = str(s).strip()
    # blank out the "he's fine" states -- only reserve / PS / suspended etc. is worth
    # flagging on a draft board. "ACT"/"DEV" are raw codes a couple of team sites emit.
    return "" if not s or s in ("nan", "ACT", "DEV") or s.startswith("Active") else s


def _roster_lookup() -> tuple[dict, dict]:
    """normalized name -> team_abbr / roster_status from nfl_rosters.csv, keeping
    the Active row when a name collides across teams (e.g. a star and a practice-
    squad player who share a name)."""
    ros = load_csv_if_exists("nfl_rosters.csv")
    if ros is None:
        return {}, {}
    rn = ros.assign(
        _k=ros["player"].apply(normalize_name),
        _act=ros["roster_status"].fillna("").str.startswith("Active"),
    ).sort_values("_act", ascending=False).drop_duplicates("_k")
    return (rn.set_index("_k")["team_abbr"].to_dict(),
            rn.set_index("_k")["roster_status"].to_dict())


def _bye_weeks() -> dict:
    """team_abbr -> bye week, from the upcoming season's schedule (the week 1-18
    the team has no game)."""
    s = load_csv_if_exists("schedules.csv")
    if s is None:
        return {}
    s = s[s["season"] == s["season"].max()]
    played = {}
    for t in set(s["home_team"]) | set(s["away_team"]):
        wks = set(s[(s["home_team"] == t) | (s["away_team"] == t)]["week"])
        bye = [w for w in range(1, 19) if w not in wks]
        played[t] = bye[0] if bye else None
    return played


def _num(v) -> float:
    return float(v) if v is not None and pd.notna(v) else float("nan")


@st.cache_data(show_spinner=False)
def _draft_pool(_ud_mtime: float, _ros_mtime: float, _pred_mtime: float, scoring: str) -> pd.DataFrame:
    ud = load_csv_if_exists("underdog_props.csv")
    if ud is None:
        return pd.DataFrame()
    s = ud[ud["stat_name"].isin(SEASON_STATS) & (ud["choice"].str.lower() == "over")]
    if s.empty:
        return pd.DataFrame()
    wide = s.pivot_table(index="full_name", columns="stat_name", values="stat_value", aggfunc="first")
    ud_pos = (s.sort_values("stat_name").groupby("full_name")["position_name"].first()
              if "position_name" in s.columns
              else s.groupby("full_name")["position_display_name"].first())

    team_by, status_by = _roster_lookup()
    byes = _bye_weeks()

    def _row(name, get, position, source, team):
        d = {k: _num(get(k)) for k in SEASON_STATS}
        if source == "model":
            # pull the model's (slightly hot) season line onto the market scale so
            # its points, reception estimate and shown yardage are all consistent.
            f = _MODEL_SCALE.get(position, 1.0)
            d = {k: (v * f if pd.notna(v) else v) for k, v in d.items()}
        pr = project_season_points(d, position, scoring)
        return {
            "player": name, "position": position, "team": team or "",
            "bye": byes.get(team or ""), "proj": round(pr["points"], 1), "source": source,
            "status": _clean_status(status_by.get(normalize_name(name), "")),
            "pass_yd": d["season_pass_yards"], "rush_yd": d["season_rush_yards"],
            "rec_yd": d["season_receiving_yards"], "rec": pr["rec_est"],
            "pass_td": d["season_pass_tds"], "rush_td": d["season_rush_tds"],
            "rec_td": d["season_rec_tds"],
        }

    rows, seen = [], set()
    for name, r in wide.iterrows():
        position = str(ud_pos.get(name, "")).upper()
        if position not in POSITIONS:
            continue
        seen.add(normalize_name(name))
        rows.append(_row(name, r.get, position, "market", team_by.get(normalize_name(name), "")))

    # depth past Underdog's board, from the prop model's own season projections
    preds = load_current_predictions()
    if preds is not None:
        ms = preds[preds["prop_type"].isin(_MODEL_SEASON_PROPS)]
        if not ms.empty:
            mw = ms.pivot_table(index="player_display_name", columns="prop_type",
                                values="proxy_line", aggfunc="first")
            mmeta = ms.groupby("player_display_name").agg(
                position=("position", "first"), team=("recent_team", "first"))
            for name, r in mw.iterrows():
                if normalize_name(name) in seen:
                    continue
                position = str(mmeta.loc[name, "position"]).upper()
                if position not in _MODEL_SCALE:            # QB model season = rushing only
                    continue
                seen.add(normalize_name(name))
                team = mmeta.loc[name, "team"] or team_by.get(normalize_name(name), "")
                rows.append(_row(name, r.get, position, "model", team))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("pass_yd", "rush_yd", "rec_yd", "pass_td", "rush_td", "rec_td"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep every market-priced player; drop model-only depth below ~3 pts/game so the
    # board isn't padded with camp bodies the model gave a token projection.
    keep = (df["source"] == "market") | (df["proj"] >= 50)
    df = df[keep & (df["proj"] > 0)].sort_values("proj", ascending=False).reset_index(drop=True)
    df["pos_rank"] = df.groupby("position")["proj"].rank(method="first", ascending=False).astype(int)
    return df


def _add_vbd(df: pd.DataFrame, teams: int, starters: dict) -> pd.DataFrame:
    """Value over replacement: replacement = the projected points of the player at
    the last-startable slot for their position (starters + a share of FLEX)."""
    flex_split = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
    df = df.copy()
    repl = {}
    for pos in POSITIONS:
        pool = df[df["position"] == pos]["proj"].tolist()
        n_start = starters.get(pos, 0) + starters.get("FLEX", 0) * flex_split.get(pos, 0)
        idx = max(0, min(len(pool) - 1, round(teams * n_start) - 1)) if pool else 0
        repl[pos] = pool[idx] if pool else 0.0
    df["vbd"] = (df["proj"] - df["position"].map(repl)).round(1)
    return df.sort_values("vbd", ascending=False).reset_index(drop=True)


def _tier(sub: pd.Series, gap: float) -> list:
    """Tier the (descending) projections: a new tier starts either on a single
    cliff between adjacent players >= gap, or once the running drop from the
    current tier's top player exceeds gap (so a smooth bleed still forms tiers)."""
    tiers, t, prev, top = [], 1, None, None
    for v in sub:
        if prev is not None and (prev - v >= gap or top - v >= gap):
            t += 1
            top = v
        if top is None:
            top = v
        tiers.append(t)
        prev = v
    return tiers


def _draft_table(scoring: str, teams: int, starters: dict) -> pd.DataFrame:
    df = _draft_pool(_mtime(RAW_DIR, "underdog_props.csv"), _mtime(RAW_DIR, "nfl_rosters.csv"),
                     _pred_mtime(), scoring)
    if df.empty:
        return df
    df = _add_vbd(df, teams, starters)
    df["tier"] = 0
    for pos in POSITIONS:
        m = df["position"] == pos
        df.loc[m, "tier"] = _tier(df.loc[m].sort_values("proj", ascending=False)["proj"],
                                  _SEASON_TIER_GAP[pos])
    df["overall"] = range(1, len(df) + 1)
    return df


def _mtime(d: str, name: str) -> float:
    p = os.path.join(d, name)
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def _league_settings(lsettings: dict | None = None):
    """Draft-board league shape. Pre-filled from a connected Yahoo league when
    there is one (still editable), otherwise a standard 12-team default."""
    if lsettings and st.session_state.get("_ff_lg_seeded") != lsettings.get("name"):
        s = lsettings["starters"]
        st.session_state.update({
            "ff_lg_teams": int(lsettings.get("num_teams") or 12),
            "ff_lg_qb": s["QB"], "ff_lg_rb": s["RB"], "ff_lg_wr": s["WR"],
            "ff_lg_te": s["TE"], "ff_lg_flex": s["FLEX"],
            "_ff_lg_seeded": lsettings.get("name"),
        })
    src = " · pre-filled from your Yahoo league" if lsettings else ""
    with st.expander("League settings" + src):
        c = st.columns(6)
        teams = c[0].number_input("Teams", 4, 20, 12, key="ff_lg_teams")
        starters = {
            "QB": c[1].number_input("QB", 0, 3, 1, key="ff_lg_qb"),
            "RB": c[2].number_input("RB", 0, 5, 2, key="ff_lg_rb"),
            "WR": c[3].number_input("WR", 0, 6, 2, key="ff_lg_wr"),
            "TE": c[4].number_input("TE", 0, 3, 1, key="ff_lg_te"),
            "FLEX": c[5].number_input("FLEX", 0, 4, 1, key="ff_lg_flex"),
        }
    return int(teams), {k: int(v) for k, v in starters.items()}


def _render_rankings(scoring: str, teams: int, starters: dict):
    df = _draft_table(scoring, teams, starters)
    if df.empty:
        st.warning("No Underdog season-long O/U lines pulled yet — run **Underdog pick'em props** "
                   "in Run Data Pulls.")
        return
    pos = st.radio("Position", ["All"] + POSITIONS, horizontal=True, key="ff_rank_pos")
    view = df if pos == "All" else df[df["position"] == pos]
    view = view.sort_values("vbd", ascending=False)

    n_model = int((df["source"] == "model").sum())
    st.caption(f"{SCORING_LABELS[scoring]} · projected **full-season** points. **VBD** = points "
               f"above the last startable player at the position in a {teams}-team league — sort "
               f"by this, not raw points. **Src**: *mkt* = Underdog's season O/U line (the market's "
               f"implied total); *mdl* = the prop model's own season projection, scaled to the "
               f"market, for the {n_model} skill players Underdog doesn't price. Receptions & QB "
               f"INTs are estimated (no Underdog market).")

    show = view.assign(
        Pos=lambda d: d["position"] + d["pos_rank"].astype(str),
        Bye=lambda d: d["bye"].map(lambda b: "" if pd.isna(b) else str(int(b))),
        Src=lambda d: d["source"].map({"market": "mkt", "model": "mdl"}),
    )[["overall", "Pos", "tier", "player", "team", "Bye", "proj", "vbd", "Src", "status",
       "pass_yd", "rush_yd", "rec_yd", "rec", "rush_td", "rec_td"]]
    show.columns = ["#", "Pos", "Tier", "Player", "Team", "Bye", "Proj", "VBD", "Src", "Status",
                    "PaYd", "RuYd", "ReYd", "Rec", "RuTD", "ReTD"]
    st.dataframe(show, hide_index=True, width="stretch", height=620,
                 column_config={"VBD": st.column_config.NumberColumn("VBD", format="%.0f")})


def _sync_yahoo_draft(league_key: str, board: pd.DataFrame) -> None:
    """Pull the Yahoo draft results and flip matching board rows to 'off the board'
    (and 'my picks' for the connected team). Silently skips players not on our board
    (kickers, DST, deep picks Underdog never priced)."""
    try:
        dr = yf.draft_results(league_key)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't read draft results: {e}")
        return
    if dr.empty:
        st.warning("This Yahoo league hasn't drafted yet (no draft results).")
        return

    dr["yahoo_id"] = dr["player_key"].str.split(".").str[-1]
    bridge = load_csv_if_exists("player_ids.csv")
    name_by_yid: dict[str, str] = {}
    if bridge is not None and "yahoo_id" in bridge.columns:
        b = bridge.dropna(subset=["yahoo_id"]).copy()
        b["yahoo_id"] = b["yahoo_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        name_by_yid = dict(zip(b["yahoo_id"], b["name"]))

    board_by_norm = {normalize_name(p): p for p in board["player"]}
    try:
        my_tk = yf.my_team_key(league_key)
    except Exception:  # noqa: BLE001
        my_tk = None

    taken, mine, unmatched = [], [], 0
    for _, r in dr.iterrows():
        nm = name_by_yid.get(r["yahoo_id"])
        hit = board_by_norm.get(normalize_name(nm)) if nm else None
        if not hit:
            unmatched += 1
            continue
        taken.append(hit)
        if my_tk and r["team_key"] == my_tk:
            mine.append(hit)

    st.session_state["draft_taken"] = sorted(set(taken))
    st.session_state["draft_mine"] = sorted(set(mine))
    st.toast(f"Synced {len(set(taken))} drafted players from Yahoo"
             + (f" · {len(set(mine))} yours" if mine else "")
             + (f" · {unmatched} off-board picks skipped" if unmatched else ""))
    st.rerun()


def _render_draft_board(scoring: str, teams: int, starters: dict, league_key: str | None = None):
    df = _draft_table(scoring, teams, starters)
    if df.empty:
        st.warning("No Underdog season-long O/U lines pulled yet.")
        return
    names = df["player"].tolist()

    if league_key and yf is not None:
        if st.button("↻ Sync drafted players from Yahoo", key="ff_draft_sync"):
            _sync_yahoo_draft(league_key, df)

    c1, c2, c3 = st.columns([4, 4, 1])
    taken = c1.multiselect("Off the board (drafted by anyone)", names, key="draft_taken")
    # keep My picks a subset of what's off the board -- set before that widget renders
    st.session_state["draft_mine"] = [m for m in st.session_state.get("draft_mine", []) if m in taken]
    mine = c2.multiselect("My picks (from the off-the-board list)", taken, key="draft_mine")
    if c3.button("Reset", width="stretch"):
        st.session_state["draft_taken"] = []
        st.session_state["draft_mine"] = []
        st.rerun()

    avail = df[~df["player"].isin(taken)]

    st.subheader("Best available")
    top = avail.head(12).assign(
        Pos=lambda d: d["position"] + d["pos_rank"].astype(str),
        Src=lambda d: d["source"].map({"market": "mkt", "model": "mdl"}))
    st.dataframe(
        top[["overall", "Pos", "tier", "player", "team", "bye", "proj", "vbd", "Src"]].rename(
            columns={"overall": "#", "tier": "Tier", "player": "Player", "team": "Team",
                     "bye": "Bye", "proj": "Proj", "vbd": "VBD"}),
        hide_index=True, width="stretch")

    st.subheader("Best available by position")
    st.caption("Top 5 left at each spot. The header counts how many are still on the "
               "board in that position's current-best tier — a low number means the next "
               "pick there is a real step down. *mdl* = model projection (past Underdog's board).")
    cols = st.columns(4)
    for col, p in zip(cols, POSITIONS):
        pos_avail = avail[avail["position"] == p]
        sub = pos_avail.head(5)
        with col:
            if sub.empty:
                st.markdown(f"**{p}** — none left")
                continue
            best_tier = int(sub.iloc[0]["tier"])
            in_tier = int((pos_avail["tier"] == best_tier).sum())
            flag = " ⚠️" if in_tier <= 2 else ""
            st.markdown(f"**{p}** · {in_tier} left in T{best_tier}{flag}")
            for _, r in sub.iterrows():
                src = "" if r.source == "market" else " ·mdl"
                st.caption(f"{r.player} · {r.proj:.0f} · T{int(r.tier)}{src}")

    if mine:
        st.subheader("My roster")
        roster = df[df["player"].isin(mine)]
        counts = roster["position"].value_counts().to_dict()
        need = [p for p in POSITIONS
                if counts.get(p, 0) < starters.get(p, 0)]
        rc = st.columns(4)
        for col, p in zip(rc, POSITIONS):
            col.metric(p, f"{counts.get(p, 0)} / {starters.get(p, 0)}")
        st.caption(("Still need a starter at: **" + ", ".join(need) + "**") if need
                   else "Starting lineup filled — draft upside / depth.")
        st.dataframe(
            roster.sort_values("overall").assign(Pos=lambda d: d["position"] + d["pos_rank"].astype(str))[
                ["overall", "Pos", "player", "team", "bye", "proj", "vbd"]].rename(
                columns={"overall": "#", "player": "Player", "team": "Team", "bye": "Bye",
                         "proj": "Proj", "vbd": "VBD"}),
            hide_index=True, width="stretch")


# ------------------------------------------------------------------ yahoo league

_STARTER_SLOTS = {"QB", "RB", "WR", "TE", "W/R/T", "W/R", "R/W/T", "Q/W/R/T", "FLEX"}
_FLEX_ELIGIBLE = {"RB", "WR", "TE"}


def _next_week() -> int | None:
    df = load_current_predictions()
    if df is None or "next_week" not in df.columns or df["next_week"].dropna().empty:
        return None
    return int(df["next_week"].dropna().mode().iloc[0])


def _proj_lookup(proj: pd.DataFrame) -> dict:
    """Our weekly projection keyed by both nflverse player_id and normalized name,
    so an external roster joins by id first, name second."""
    d: dict = {}
    for r in proj.itertuples():
        rec = {"proj": r.proj, "opp": r.opp, "pos_rank": int(r.pos_rank),
               "tier": int(r.tier), "lean": r.lean, "our_pos": r.position}
        d[("id", r.player_id)] = rec
        d.setdefault(("nm", normalize_name(r.player)), rec)
    return d


def _attach_proj(ext: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    look = _proj_lookup(proj)
    recs = []
    for row in ext.itertuples():
        rec = look.get(("id", getattr(row, "gsis_id", None))) or \
            look.get(("nm", getattr(row, "norm_name", ""))) or {}
        recs.append(rec)
    add = pd.DataFrame(recs, index=ext.index)
    for c in ("proj", "opp", "pos_rank", "tier", "lean", "our_pos"):
        if c not in add.columns:
            add[c] = pd.NA
    return pd.concat([ext, add], axis=1)


def _optimal_lineup(m: pd.DataFrame, starters: dict) -> set:
    """Greedy best lineup: fill each dedicated slot then FLEX with the highest-
    projected eligible players. Returns the set of chosen row indices."""
    chosen: set = set()
    pool = m.dropna(subset=["proj"]).sort_values("proj", ascending=False)
    for pos in ("QB", "RB", "WR", "TE"):
        picks = pool[(pool["position"] == pos) & (~pool.index.isin(chosen))].head(starters.get(pos, 0))
        chosen.update(picks.index)
    flex_n = starters.get("FLEX", 0)
    if flex_n:
        flex = pool[(pool["position"].isin(_FLEX_ELIGIBLE)) & (~pool.index.isin(chosen))].head(flex_n)
        chosen.update(flex.index)
    return chosen


def _yahoo_panel() -> tuple[str | None, dict | None]:
    """Connect / league-picker UI. Returns (league_key, league_settings) once a
    league is chosen, else (None, None). Silent when Yahoo isn't configured."""
    if yf is None or not yf.configured():
        return None, None

    yf.handle_oauth_redirect()
    err = st.session_state.get("yahoo_auth_error")

    with st.expander("🟣 Yahoo league" + ("" if yf.connected() else " — not connected"),
                     expanded=not yf.connected()):
        if err:
            st.error(err)
        if not yf.connected():
            url = yf.authorize_url()
            # A plain link/link_button opens a NEW tab, and Streamlit scopes
            # session_state per tab -- the token would land where the user can't
            # see it. Navigate the current tab via JS instead.
            if st.button("Authorize with Yahoo →", type="primary", key="ff_yh_go"):
                components.html(
                    f"<script>window.top.location.href = {json.dumps(url)};</script>",
                    height=0,
                )
            st.caption("Sends you to Yahoo **in this tab**, then back here connected. "
                       "If nothing happens, copy this and paste it into this tab's address bar:")
            st.code(url, language=None)
            with st.expander("Connection details"):
                st.json(yf.diagnostics())
            return None, None

        st.link_button("Yahoo account", "https://football.fantasysports.yahoo.com/", disabled=True)
        try:
            lg = yf.leagues()
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't list your leagues: {e}")
            return None, None
        if lg.empty:
            st.warning("Connected, but no NFL leagues found on this Yahoo account for the current season.")
            _disconnect_button()
            return None, None

        names = {f'{r["name"]}  ·  {r["num_teams"]}-team  ·  {r["season"]}': r["league_key"]
                 for _, r in lg.iterrows()}
        pick = st.selectbox("League", list(names), key="ff_yh_league_pick")
        league_key = names[pick]
        st.session_state["yahoo_league_key"] = league_key

        try:
            settings = yf.league_settings(league_key)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read league settings: {e}")
            _disconnect_button()
            return league_key, None

        s = settings["starters"]
        st.caption(
            f'**{settings["name"]}** · {settings["num_teams"]} teams · '
            f'{SCORING_LABELS.get(settings["scoring"], settings["scoring"])} '
            f'({settings["reception_point"]:g} pt/rec) · starters '
            f'QB {s["QB"]} / RB {s["RB"]} / WR {s["WR"]} / TE {s["TE"]} / FLEX {s["FLEX"]}'
        )

        if not yf._secret("yahoo_refresh_token"):
            tok = st.session_state.get("yahoo_token", {})
            if tok.get("refresh_token"):
                st.info("To stay connected across restarts, add this to **App settings → Secrets**:")
                st.code(f'yahoo_refresh_token = "{tok["refresh_token"]}"', language="toml")
        _disconnect_button()
        return league_key, settings


def _disconnect_button() -> None:
    if st.button("Disconnect Yahoo", key="ff_yh_disconnect"):
        yf.disconnect()
        st.rerun()


def _starter_slots(lsettings: dict | None) -> dict:
    """Starting-lineup shape for My Team. Pre-filled from a connected Yahoo league,
    else a standard 1QB / 2RB / 2WR / 1TE / 1FLEX."""
    if lsettings and st.session_state.get("_ff_mt_seeded") != lsettings.get("name"):
        s = lsettings["starters"]
        st.session_state.update({
            "ff_mt_qb": s["QB"], "ff_mt_rb": s["RB"], "ff_mt_wr": s["WR"],
            "ff_mt_te": s["TE"], "ff_mt_flex": s["FLEX"],
            "_ff_mt_seeded": lsettings.get("name"),
        })
    with st.expander("Starting slots" + (" · from your Yahoo league" if lsettings else "")):
        c = st.columns(5)
        return {
            "QB": c[0].number_input("QB", 0, 3, 1, key="ff_mt_qb"),
            "RB": c[1].number_input("RB", 0, 5, 2, key="ff_mt_rb"),
            "WR": c[2].number_input("WR", 0, 6, 2, key="ff_mt_wr"),
            "TE": c[3].number_input("TE", 0, 3, 1, key="ff_mt_te"),
            "FLEX": c[4].number_input("FLEX", 0, 4, 1, key="ff_mt_flex"),
        }


def _ros_lookup(scoring: str) -> dict:
    """normalized name -> rest-of-season projection (draft-board full-season scale)."""
    ros = _draft_table(scoring, 12, DEFAULT_STARTERS)
    if ros.empty:
        return {}
    return {normalize_name(p): v for p, v in zip(ros["player"], ros["proj"])}


def _my_roster_df(scoring: str, league_key: str | None) -> tuple[pd.DataFrame, str]:
    """The user's roster joined to our weekly projections. From Yahoo if connected,
    otherwise from a manual multiselect saved in session. Returns (df, source)."""
    proj = _projections(_pred_mtime(), scoring)
    if proj.empty:
        return pd.DataFrame(), "none"

    if league_key and yf is not None:
        try:
            roster = yf.my_roster(league_key, _next_week())
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't load your Yahoo roster: {e}")
            return pd.DataFrame(), "yahoo"
        m = _attach_proj(roster, proj)
        m["position"] = m["our_pos"].where(m["our_pos"].notna(), m["position"].str.upper())
        m["is_starter_now"] = ~m["slot"].isin(["BN", "IR", "NA"])
        return m, "yahoo"

    options = proj.sort_values("proj", ascending=False)["player"].tolist()
    picked = st.multiselect(
        "Your roster — pick every player you own", options, key="ff_my_roster",
        help="Type to search. Saved for this browser session.",
    )
    if not picked:
        return pd.DataFrame(), "manual-empty"
    m = proj[proj["player"].isin(picked)].copy().reset_index(drop=True)
    m["name"] = m["player"]
    m["status"] = ""
    m["is_starter_now"] = pd.NA
    return m, "manual"


def _render_my_team(scoring: str, league_key: str | None, lsettings: dict | None):
    m, source = _my_roster_df(scoring, league_key)
    if source == "none":
        st.warning("No weekly projections available yet — run the data pulls.")
        return
    if source == "manual-empty":
        st.info("Pick your roster above to get your best lineup and start/sit calls.")
        return
    if m.empty:
        st.warning("Couldn't match any of those players to our projections.")
        return

    starters = _starter_slots(lsettings)
    best = _optimal_lineup(m, starters)
    m = m.copy()
    m["best_xi"] = m.index.isin(best)
    _ros = _ros_lookup(scoring)
    m["ReadOfSeason"] = m["name"].map(lambda n: _ros.get(normalize_name(n)))

    week = _next_week()
    wk_txt = f"Week {week}" if week else "this week"
    st.caption(f"{SCORING_LABELS[scoring]} · {wk_txt}. **Start** = the highest-projected legal "
               f"lineup from your roster. Weekly projections are the prop model's roll-up; "
               f"**ROS** is the rest-of-season (full-season) projection.")

    show = m.assign(
        Slot=m["best_xi"].map({True: "✅ START", False: "bench"}),
        Proj=m["proj"].round(1),
        Matchup=m["opp"].where(m["opp"].notna(), "—"),
    ).sort_values(["best_xi", "proj"], ascending=[False, False])

    cols = ["Slot", "name", "position", "team", "Proj", "Matchup", "ReadOfSeason"]
    if source == "yahoo":
        show["Yahoo now"] = m["slot"].replace({"W/R/T": "FLEX"})
        cols.insert(1, "Yahoo now")
    st.dataframe(
        show[cols].rename(columns={"name": "Player", "position": "Pos", "team": "Team",
                                   "ReadOfSeason": "ROS"}),
        hide_index=True, width="stretch", height=520)

    start = show[show["best_xi"]]
    bench = show[~show["best_xi"] & show["proj"].notna()]

    if source == "yahoo":
        # moves vs the lineup actually set on Yahoo
        wrong = show[show["best_xi"] & show["Yahoo now"].isin(["BN", "IR", "NA"])]
        if not wrong.empty:
            st.markdown("**Change on Yahoo — start:** " + ", ".join(
                f"{r['name']} ({r['proj']:.1f})" for _, r in wrong.iterrows()))
        else:
            st.success("Your Yahoo lineup already matches the best projected XI.")
    else:
        # manual mode: the best XI is the call; flag the tight ones + first off the bench
        if not start.empty and not bench.empty:
            floor = start["proj"].min()
            tight = bench[bench["proj"] >= floor - 1.5].head(3)
            if not tight.empty:
                st.markdown("**Tight calls** (within ~1.5 of a starter — let matchup decide): "
                            + " · ".join(f"{r['name']} {r['proj']:.1f}" for _, r in tight.iterrows()))
            top_bench = bench.iloc[0]
            st.caption(f"First off the bench if a starter sits: **{top_bench['name']}** "
                       f"({top_bench['proj']:.1f}, {top_bench['position']}).")


def _render_waivers(scoring: str, league_key: str | None):
    proj = _projections(_pred_mtime(), scoring)
    if proj.empty:
        st.warning("No weekly projections available yet.")
        return
    ros_by = _ros_lookup(scoring)
    week = _next_week()
    pos = st.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True, key="ff_wv_pos")

    if league_key and yf is not None:
        try:
            fa = yf.free_agents(league_key, None if pos == "All" else pos, count=75)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't load free agents: {e}")
            return
        m = _attach_proj(fa, proj)
        m["name"] = m.get("name", m.get("player"))
        src_note = "free agents in your league"
    else:
        mine = set(st.session_state.get("ff_my_roster", []))
        m = proj[~proj["player"].isin(mine)].copy()
        m["name"] = m["player"]
        m["status"] = ""
        if pos != "All":
            m = m[m["position"] == pos]
        src_note = "every projected player not on your roster (set it on **My Team**)"

    m = m[m["proj"].notna()].copy()
    m["ROS"] = m["name"].map(lambda n: ros_by.get(normalize_name(n)))
    m = m.sort_values("proj", ascending=False)

    st.caption(f"{SCORING_LABELS[scoring]} · {src_note}, ranked by our "
               f"{'Week ' + str(week) if week else 'weekly'} projection. **ROS** = rest-of-season.")
    show = m.head(40).assign(
        Proj=m["proj"].round(1), ROS=m["ROS"].round(0),
        Matchup=m["opp"].where(m["opp"].notna(), "—"),
    )[["name", "position", "team", "Proj", "Matchup", "ROS", "status"]]
    st.dataframe(
        show.rename(columns={"name": "Player", "position": "Pos", "team": "Team", "status": "Inj"}),
        hide_index=True, width="stretch", height=520)


# ----------------------------------------------------------------------- page

def page_fantasy():
    st.title("🏆 Fantasy")

    league_key, lsettings = _yahoo_panel()
    if lsettings and st.session_state.get("_ff_scoring_seeded") != league_key:
        st.session_state["ff_scoring"] = "PPR" if lsettings["scoring"] == "ppr" else "Half-PPR"
        st.session_state["_ff_scoring_seeded"] = league_key

    label = st.radio("Scoring", ["PPR", "Half-PPR"], horizontal=True, key="ff_scoring")
    scoring = "ppr" if label == "PPR" else "half"

    mode = st.radio("Mode", ["In-season", "Draft prep"], horizontal=True, key="ff_mode",
                    label_visibility="collapsed")

    if mode == "Draft prep":
        st.caption(
            "Full-season value-based rankings and a live draft board. Projections come from "
            "Underdog's season-long O/U lines (the market's implied totals) where they exist, "
            "and from the prop model's own scaled season projection for skill players past "
            "Underdog's board."
        )
        teams, starters = _league_settings(lsettings)
        d_rank, d_board = st.tabs(["Rankings / cheat sheet", "Draft board"])
        with d_rank:
            _render_rankings(scoring, teams, starters)
        with d_board:
            _render_draft_board(scoring, teams, starters, league_key)
        return

    st.caption(
        "Weekly projections, start/sit, defense-vs-position matchups, and boom/bust. "
        "Projections roll up the prop model's per-stat trailing averages; matchups and "
        "boom/bust come from weekly box scores."
    )
    names = ["My Team", "Waivers", "Player Card", "Projections",
             "Start / Sit", "Matchups", "Boom / Bust"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["My Team"]:
        _render_my_team(scoring, league_key, lsettings)
    with tabs["Waivers"]:
        _render_waivers(scoring, league_key)
    with tabs["Player Card"]:
        _render_player_card(scoring)
    with tabs["Projections"]:
        _render_projections(scoring)
    with tabs["Start / Sit"]:
        _render_start_sit(scoring)
    with tabs["Matchups"]:
        _render_matchups(scoring)
    with tabs["Boom / Bust"]:
        _render_boom_bust(scoring)
