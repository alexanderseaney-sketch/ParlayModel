"""
Compare page: side-by-side comparison of players, teams, and coaches -- stats
and history, no betting math. Three tabs, each backed entirely by data already
pulled locally:

  Players -- weekly_stats.csv (2014+, REG + POST, per-game box scores)
  Teams   -- schedules.csv (2014+, scores) + team_scheme_tendencies.csv (current
             HC/OC/DC + scheme identity)
  Coaches -- coach_history.csv (derived from pbp by models/build_coach_history.py)
             + coaching_staff.csv for each coach's current job

Kept in its own module rather than growing app.py further -- app.py imports
page_compare() and registers it as a page (same split precedent as theme.py).
"""
import pandas as pd
import streamlit as st

from utils import load_csv_if_exists, normalize_name

# (raw column, label) stat sets; a player's position picks which sets apply.
PASSING_STATS = [
    ("completions", "Completions"), ("attempts", "Attempts"),
    ("passing_yards", "Pass Yds"), ("passing_tds", "Pass TD"),
    ("interceptions", "INT"), ("sacks", "Sacked"), ("passing_epa", "Pass EPA"),
]
RUSHING_STATS = [
    ("carries", "Carries"), ("rushing_yards", "Rush Yds"),
    ("rushing_tds", "Rush TD"), ("rushing_epa", "Rush EPA"),
]
RECEIVING_STATS = [
    ("targets", "Targets"), ("receptions", "Receptions"),
    ("receiving_yards", "Rec Yds"), ("receiving_tds", "Rec TD"),
    ("receiving_epa", "Rec EPA"),
]
MISC_STATS = [("fantasy_points_ppr", "Fantasy Pts (PPR)")]

POSITION_STAT_SETS = {
    "QB": PASSING_STATS + RUSHING_STATS + MISC_STATS,
    "RB": RUSHING_STATS + RECEIVING_STATS + MISC_STATS,
    "FB": RUSHING_STATS + RECEIVING_STATS + MISC_STATS,
    "WR": RECEIVING_STATS + RUSHING_STATS + MISC_STATS,
    "TE": RECEIVING_STATS + MISC_STATS,
}
DEFAULT_STAT_SET = PASSING_STATS + RUSHING_STATS + RECEIVING_STATS + MISC_STATS


def _stat_set_for(positions: set) -> list[tuple[str, str]]:
    """Union of both players' position stat sets, first-seen order preserved --
    a QB-vs-WR comparison shows passing AND receiving rather than forcing one."""
    seen, out = set(), []
    for pos in positions:
        for col, label in POSITION_STAT_SETS.get(pos, DEFAULT_STAT_SET):
            if col not in seen:
                seen.add(col)
                out.append((col, label))
    return out or DEFAULT_STAT_SET


def _fmt(v, digits=0):
    if v is None or pd.isna(v):
        return "—"
    return f"{v:,.{digits}f}"


def _compare_grid(cols: dict, index: list):
    """A small labelled comparison table. Explicit 'string' dtype so pyarrow
    doesn't see '579', '831' ... and try to coerce the column to int64 (then
    choke on a thousands-separated value like '6,670') -- it renders either way
    but spews a traceback into the server log every time."""
    return pd.DataFrame(cols, index=index).astype("string")


# ---------------------------------------------------------------- Players tab

def _player_header(latest: pd.Series):
    photo = latest.get("headshot_url")
    cols = st.columns([1, 3])
    with cols[0]:
        if isinstance(photo, str) and photo:
            st.markdown(f'<img src="{photo}" class="pm-dialog-photo">', unsafe_allow_html=True)
        else:
            initials = "".join(p[0] for p in str(latest["player_display_name"]).split()[:2]).upper()
            st.markdown(f'<div class="pm-dialog-photo-placeholder">{initials}</div>',
                        unsafe_allow_html=True)
    with cols[1]:
        st.subheader(latest["player_display_name"])
        st.caption(f"{latest.get('position', '?')} · {latest.get('recent_team', '?')} · "
                   f"last seen {int(latest['season'])} wk {int(latest['week'])}")


def _render_players_tab():
    weekly = load_csv_if_exists("weekly_stats.csv")
    if weekly is None:
        st.warning("`weekly_stats.csv` hasn't been pulled yet — run **Run Data Pulls**.")
        return

    latest_rows = (weekly.sort_values(["season", "week"])
                   .groupby("player_id").tail(1).set_index("player_id"))

    def label_for(pid):
        r = latest_rows.loc[pid]
        return f"{r['player_display_name']} ({r.get('position', '?')}, {r.get('recent_team', '?')})"

    # Default to the two top total-yardage players of the most recent season -- an
    # arbitrary alphabetical default would land on someone nobody is comparing.
    # Total yards, NOT fantasy_points_ppr: the pbp-derived rows that fill recent
    # seasons (see pull_nflverse's gap-year derivation) leave fantasy points NaN,
    # which made a PPR sort return garbage.
    last_season = int(weekly["season"].max())
    recent = weekly[weekly["season"] == last_season]
    season_totals = (recent["passing_yards"].fillna(0) + recent["rushing_yards"].fillna(0)
                     + recent["receiving_yards"].fillna(0)).groupby(recent["player_id"]).sum()
    default_ids = list(season_totals.sort_values(ascending=False).index[:2])

    ordered_ids = sorted(latest_rows.index, key=lambda pid: latest_rows.loc[pid, "player_display_name"])

    pick = st.columns(2)
    with pick[0]:
        pid_a = st.selectbox("Player A", ordered_ids, format_func=label_for,
                             index=ordered_ids.index(default_ids[0]) if default_ids else 0,
                             key="cmp_player_a")
    with pick[1]:
        pid_b = st.selectbox("Player B", ordered_ids, format_func=label_for,
                             index=ordered_ids.index(default_ids[1]) if len(default_ids) > 1 else 0,
                             key="cmp_player_b")
    if pid_a == pid_b:
        st.info("Pick two different players.")
        return

    filt = st.columns([2, 1])
    with filt[0]:
        seasons_all = sorted(weekly["season"].unique(), reverse=True)
        seasons_sel = st.multiselect("Seasons (empty = all)", seasons_all, default=[], key="cmp_player_seasons")
    with filt[1]:
        reg_only = st.checkbox("Regular season only", value=True, key="cmp_player_reg")

    df = weekly[weekly["player_id"].isin([pid_a, pid_b])].copy()
    if seasons_sel:
        df = df[df["season"].isin(seasons_sel)]
    if reg_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    a_latest, b_latest = latest_rows.loc[pid_a], latest_rows.loc[pid_b]
    head = st.columns(2)
    with head[0]:
        _player_header(a_latest)
    with head[1]:
        _player_header(b_latest)

    stats = _stat_set_for({a_latest.get("position"), b_latest.get("position")})
    name_a, name_b = a_latest["player_display_name"], b_latest["player_display_name"]

    def agg_for(pid):
        sub = df[df["player_id"] == pid]
        games = len(sub)
        totals = {label: sub[col].sum() if col in sub else None for col, label in stats}
        return games, totals

    games_a, tot_a = agg_for(pid_a)
    games_b, tot_b = agg_for(pid_b)

    scope = f"{len(seasons_sel)} selected season(s)" if seasons_sel else "career (all pulled seasons)"
    st.markdown(f"##### Totals — {scope}")
    total_rows = {"Games": [games_a, games_b]}
    for _, label in stats:
        total_rows[label] = [_fmt(tot_a[label], 1 if "EPA" in label or "Fantasy" in label else 0),
                             _fmt(tot_b[label], 1 if "EPA" in label or "Fantasy" in label else 0)]
    st.table(_compare_grid(total_rows, [name_a, name_b]).T)

    st.markdown("##### Per game")
    pg_rows = {}
    for _, label in stats:
        pg_rows[label] = [
            _fmt(tot_a[label] / games_a if games_a else None, 1),
            _fmt(tot_b[label] / games_b if games_b else None, 1),
        ]
    st.table(_compare_grid(pg_rows, [name_a, name_b]).T)

    st.markdown("##### Season by season")
    chart_options = [label for _, label in stats]
    chart_stat = st.selectbox("Stat", chart_options, key="cmp_player_chart_stat")
    chart_col = next(col for col, label in stats if label == chart_stat)
    by_season = (df.groupby(["season", "player_id"])[chart_col].sum().reset_index())
    pivot = by_season.pivot(index="season", columns="player_id", values=chart_col)
    pivot = pivot.rename(columns={pid_a: name_a, pid_b: name_b}).sort_index()
    st.line_chart(pivot)

    per_season = (df.groupby(["player_id", "season"])
                  .agg(G=("week", "count"),
                       **{label: (col, "sum") for col, label in stats if col in df.columns})
                  .round(1).reset_index())
    detail = st.columns(2)
    for c, pid, nm in ((detail[0], pid_a, name_a), (detail[1], pid_b, name_b)):
        with c:
            st.caption(nm)
            sub = (per_season[per_season["player_id"] == pid]
                   .drop(columns="player_id").sort_values("season", ascending=False))
            st.dataframe(sub, hide_index=True, width="stretch")


# ------------------------------------------------------------------ Teams tab

def _team_games_long(schedules: pd.DataFrame) -> pd.DataFrame:
    played = schedules[schedules["home_score"].notna()]
    home = played.rename(columns={"home_team": "team", "home_score": "points_for",
                                  "away_team": "opponent", "away_score": "points_against"})
    away = played.rename(columns={"away_team": "team", "away_score": "points_for",
                                  "home_team": "opponent", "home_score": "points_against"})
    cols = ["game_id", "season", "week", "game_type", "team", "opponent",
            "points_for", "points_against"] if "game_type" in schedules.columns else \
           ["game_id", "season", "week", "team", "opponent", "points_for", "points_against"]
    out = pd.concat([home[cols], away[cols]], ignore_index=True)
    out["won"] = (out.points_for > out.points_against).astype(int)
    out["tied"] = (out.points_for == out.points_against).astype(int)
    return out


def _record_line(sub: pd.DataFrame) -> str:
    w = int(sub.won.sum())
    t = int(sub.tied.sum())
    l = len(sub) - w - t
    return f"{w}-{l}" + (f"-{t}" if t else "")


def _render_teams_tab():
    schedules = load_csv_if_exists("schedules.csv")
    if schedules is None:
        st.warning("`schedules.csv` hasn't been pulled yet — run **Run Data Pulls**.")
        return
    ident = load_csv_if_exists("team_scheme_tendencies.csv")
    weekly = load_csv_if_exists("weekly_stats.csv")

    games = _team_games_long(schedules)
    teams = sorted(games["team"].unique())

    pick = st.columns(2)
    with pick[0]:
        team_a = st.selectbox("Team A", teams, index=teams.index("KC") if "KC" in teams else 0,
                              key="cmp_team_a")
    with pick[1]:
        team_b = st.selectbox("Team B", teams, index=teams.index("BUF") if "BUF" in teams else 1,
                              key="cmp_team_b")
    if team_a == team_b:
        st.info("Pick two different teams.")
        return

    seasons_all = sorted(games["season"].unique(), reverse=True)
    seasons_sel = st.multiselect("Seasons (empty = all)", seasons_all,
                                 default=seasons_all[:3], key="cmp_team_seasons")
    reg_only = st.checkbox("Regular season only", value=True, key="cmp_team_reg")

    sel = games if not seasons_sel else games[games["season"].isin(seasons_sel)]
    if reg_only and "game_type" in sel.columns:
        sel = sel[sel["game_type"] == "REG"]

    # Current identity (coaches + scheme), when built
    if ident is not None:
        id_cols = st.columns(2)
        for c, t in ((id_cols[0], team_a), (id_cols[1], team_b)):
            with c:
                row = ident[ident["team_abbr"] == t]
                if row.empty:
                    continue
                r = row.iloc[0]
                st.markdown(f"**{t}** — HC: {r.get('hc_name') or '?'} · "
                            f"OC: {r.get('oc_name') or '?'} · DC: {r.get('dc_name') or '?'}")
                st.caption(f"Off: {r.get('off_identity') or '—'}")
                st.caption(f"Def: {r.get('def_identity') or '—'}")

    def team_stats(t):
        sub = sel[sel["team"] == t]
        g = len(sub)
        out = {
            "Record": _record_line(sub),
            "Win %": _fmt(100 * sub.won.mean() if g else None, 1),
            "PF / game": _fmt(sub.points_for.mean() if g else None, 1),
            "PA / game": _fmt(sub.points_against.mean() if g else None, 1),
            "Point diff / game": _fmt((sub.points_for - sub.points_against).mean() if g else None, 1),
        }
        if weekly is not None:
            wsub = weekly[weekly["recent_team"] == t]
            if seasons_sel:
                wsub = wsub[wsub["season"].isin(seasons_sel)]
            if reg_only and "season_type" in wsub.columns:
                wsub = wsub[wsub["season_type"] == "REG"]
            team_games = wsub.groupby(["season", "week"]).ngroups
            if team_games:
                out["Pass yds / game"] = _fmt(wsub.passing_yards.sum() / team_games, 1)
                out["Rush yds / game"] = _fmt(wsub.rushing_yards.sum() / team_games, 1)
        return out

    st.markdown(f"##### {'Selected seasons' if seasons_sel else 'All pulled seasons'}")
    stats_a, stats_b = team_stats(team_a), team_stats(team_b)
    all_keys = list(dict.fromkeys(list(stats_a) + list(stats_b)))
    table = _compare_grid({team_a: [stats_a.get(k, "—") for k in all_keys],
                           team_b: [stats_b.get(k, "—") for k in all_keys]}, all_keys)
    st.table(table)

    st.markdown("##### Season by season")
    by_season = (sel[sel["team"].isin([team_a, team_b])]
                 .groupby(["season", "team"])
                 .agg(G=("won", "count"), W=("won", "sum"),
                      PFpg=("points_for", "mean"), PApg=("points_against", "mean"))
                 .round(1).reset_index())
    by_season["Record"] = by_season.apply(lambda r: f"{int(r.W)}-{int(r.G - r.W)}", axis=1)
    pivot = by_season.pivot(index="season", columns="team", values="Record").sort_index(ascending=False)
    st.dataframe(pivot, width="stretch")

    st.markdown("##### Head to head")
    h2h = sel[(sel["team"] == team_a) & (sel["opponent"] == team_b)]
    if h2h.empty:
        st.caption("No meetings in the selected span.")
    else:
        st.markdown(f"**{team_a} {_record_line(h2h)}** vs {team_b} "
                    f"({len(h2h)} meeting{'s' if len(h2h) != 1 else ''})")
        show = h2h.sort_values(["season", "week"], ascending=False)[
            ["season", "week", "points_for", "points_against"]]
        show.columns = ["Season", "Week", f"{team_a} pts", f"{team_b} pts"]
        st.dataframe(show, hide_index=True, width="stretch")


# ---------------------------------------------------------------- Coaches tab

def _render_coaches_tab():
    hist = load_csv_if_exists("coach_history.csv")
    if hist is None:
        st.warning("`coach_history.csv` hasn't been built yet — run **Coach history "
                   "(from pbp)** on the Run Data Pulls page (needs `pbp.csv`).")
        return
    staff = load_csv_if_exists("coaching_staff.csv")

    span = hist.groupby("coach").agg(first=("season", "min"), last=("season", "max"),
                                     games=("won", "count"))
    coaches = sorted(span.index)

    def label_for(c):
        s = span.loc[c]
        return f"{c} ({int(s['first'])}–{int(s['last'])}, {int(s['games'])} gms)"

    most_games = span.sort_values("games", ascending=False).index[:2].tolist()
    pick = st.columns(2)
    with pick[0]:
        coach_a = st.selectbox("Coach A", coaches, format_func=label_for,
                               index=coaches.index(most_games[0]), key="cmp_coach_a")
    with pick[1]:
        coach_b = st.selectbox("Coach B", coaches, format_func=label_for,
                               index=coaches.index(most_games[1]), key="cmp_coach_b")
    if coach_a == coach_b:
        st.info("Pick two different coaches.")
        return

    def current_job(name):
        if staff is None:
            return None
        m = staff[staff["name"].apply(normalize_name) == normalize_name(name)]
        if m.empty:
            return None
        r = m.iloc[0]
        return f"{r['title']}, {r['team_abbr']}"

    def coach_stats(name):
        sub = hist[hist["coach"] == name]
        reg = sub[sub["season_type"] == "REG"]
        post = sub[sub["season_type"] != "REG"]
        return {
            "Regular season": _record_line(reg),
            "Win % (REG)": _fmt(100 * reg.won.mean() if len(reg) else None, 1),
            "Playoffs": _record_line(post) if len(post) else "—",
            "PF / game": _fmt(sub.points_for.mean(), 1),
            "PA / game": _fmt(sub.points_against.mean(), 1),
            "Teams": ", ".join(sub.drop_duplicates("team").team.tolist()),
            "Current role": current_job(name) or "not on a 2026 staff pull",
        }

    st.caption("History from play-by-play (2014 on) — earlier seasons aren't in the local pull.")
    stats_a, stats_b = coach_stats(coach_a), coach_stats(coach_b)
    keys = list(stats_a)
    table = _compare_grid({coach_a: [stats_a[k] for k in keys],
                           coach_b: [stats_b[k] for k in keys]}, keys)
    st.table(table)

    st.markdown("##### Season by season")
    both = hist[hist["coach"].isin([coach_a, coach_b])]
    reg = both[both["season_type"] == "REG"]
    # Column is "Ties", not "T" -- r.T on a row Series is pandas' transpose
    # property (the whole row), so `r.G - r.W - r.T` was int minus a str-bearing
    # Series. Bracket access everywhere in the lambda for the same reason.
    by_season = (reg.groupby(["season", "coach"])
                 .agg(team=("team", "first"), G=("won", "count"), W=("won", "sum"),
                      Ties=("tied", "sum"))
                 .reset_index())
    by_season["Record"] = by_season.apply(
        lambda r: f"{int(r['W'])}-{int(r['G'] - r['W'] - r['Ties'])}"
                  + (f"-{int(r['Ties'])}" if r["Ties"] else "") + f" ({r['team']})",
        axis=1)
    pivot = by_season.pivot(index="season", columns="coach", values="Record").sort_index(ascending=False)
    st.dataframe(pivot, width="stretch")

    st.markdown("##### Head to head")
    h2h = hist[(hist["coach"] == coach_a) & (hist["opp_coach"] == coach_b)]
    if h2h.empty:
        st.caption("These two have never faced each other (in the pulled seasons).")
    else:
        st.markdown(f"**{coach_a} {_record_line(h2h)}** vs {coach_b} "
                    f"({len(h2h)} meeting{'s' if len(h2h) != 1 else ''})")
        show = h2h.sort_values(["season", "week"], ascending=False)[
            ["season", "week", "season_type", "team", "opponent", "points_for", "points_against"]]
        show.columns = ["Season", "Week", "Type", f"{coach_a} team", f"{coach_b} team",
                        f"{coach_a} pts", f"{coach_b} pts"]
        st.dataframe(show, hide_index=True, width="stretch")


# ----------------------------------------------------------------------- page

def page_compare():
    st.title("⚖️ Compare")
    st.caption(
        "Side-by-side stats and history. Players from weekly box scores (2014 on), "
        "teams from schedules + the current scheme/staff pulls, coaches from "
        "play-by-play game records."
    )
    tabs = st.tabs(["Players", "Teams", "Coaches"])
    with tabs[0]:
        _render_players_tab()
    with tabs[1]:
        _render_teams_tab()
    with tabs[2]:
        _render_coaches_tab()
