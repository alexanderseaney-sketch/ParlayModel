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
    load_player_photos, load_player_jersey_numbers, normalize_name,
)
from fantasy_scoring import (
    SCORING_LABELS, project_points, project_breakdown, fantasy_points_from_weekly,
)

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

    out = out.sort_values("proj", ascending=False).reset_index()  # keep player_id as a column
    out["pos_rank"] = out.groupby("position")["proj"].rank(method="first", ascending=False).astype(int)
    out["overall_rank"] = range(1, len(out) + 1)   # across all QB/RB/WR/TE
    out["tier"] = out.groupby("position", group_keys=False)["proj"].apply(
        lambda s: pd.Series(_tiers(s), index=s.index))
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
    view["Tier"] = view["tier"]
    view["Model"] = view["lean"].map(_lean_score)

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

    tabs = st.tabs(["Player Card", "Projections", "Start / Sit", "Matchups", "Boom / Bust"])
    with tabs[0]:
        _render_player_card(scoring)
    with tabs[1]:
        _render_projections(scoring)
    with tabs[2]:
        _render_start_sit(scoring)
    with tabs[3]:
        _render_matchups(scoring)
    with tabs[4]:
        _render_boom_bust(scoring)
