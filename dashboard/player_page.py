"""
Player Pages (2026-10 redesign): one player in depth -- header with status, team, depth
slot and this week's opponent; last-8 game log vs this week's line; Underdog line
history; snap/target share trend; every current prop; news. A full-page version of the
player dialog (app._player_detail_dialog stays as-is for the quick pop-up).
st.session_state.player_sel selects the player (Usage Trends links here).
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from model_performance import _layout
from research_data import (PRIMARY_STAT, STAT_WEEKLY_COL, injury_report, line_history, live_over_lines,
                           week_games, current_week, priced_candidates, _mtime)
from utils import load_csv_if_exists, normalize_name, load_player_photos, get_player_news, pretty_stat_name

BAR_OVER, BAR_UNDER = ui.ACCENT, ui.BORDER


@st.cache_data(show_spinner=False)
def _players(ws_m: float, ud_key: str) -> pd.DataFrame:
    """Selectable players: skill players with 2026-season games, plus anyone on the live board."""
    w = load_csv_if_exists("weekly_stats.csv")
    season = int(w["season"].max())
    cur = w[(w["season"] == season) & w["position"].isin(["QB", "RB", "WR", "TE"])]
    p = (cur.sort_values("week").groupby("player_display_name")
         .agg(position=("position", "last"), team=("recent_team", "last"), games=("week", "nunique"))
         .reset_index().rename(columns={"player_display_name": "name"}))
    p["_k"] = p["name"].map(normalize_name)
    return p.sort_values(["games", "name"], ascending=[False, True])


def _game_log(name: str, stat: str) -> pd.DataFrame:
    w = load_csv_if_exists("weekly_stats.csv")
    col = STAT_WEEKLY_COL.get(stat, "receiving_yards")
    g = w[w["player_display_name"].map(normalize_name) == normalize_name(name)]
    g = g[g["season_type"].fillna("REG") == "REG"].sort_values(["season", "week"]).tail(8)
    g = g.assign(value=g[col].fillna(0), label=[f"{'' if s == g['season'].max() else str(s)[2:] + ' '}W{w_}"
                                                 for s, w_ in zip(g["season"], g["week"])])
    return g


def _snap_shares(name: str, team: str) -> pd.DataFrame:
    s = load_csv_if_exists("snap_counts.csv")
    if s is None:
        return pd.DataFrame()
    s = s[(s["player"].map(normalize_name) == normalize_name(name)) & (s["game_type"].fillna("REG") == "REG")]
    return s.sort_values(["season", "week"]).tail(8)


def _bars(values, labels, line=None, colors=None, height=230, fmt="{:g}"):
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=colors or ui.ACCENT, cornerradius=4),
                           text=[fmt.format(v) for v in values], textposition="outside",
                           textfont=dict(family="JetBrains Mono", size=11, color=ui.N1),
                           hovertemplate="%{x}: %{y}<extra></extra>"))
    if line is not None:
        fig.add_hline(y=line, line=dict(color=ui.A_TEXT, width=1, dash="dash"),
                      annotation_text=f"line {line:g}", annotation_position="top right",
                      annotation_font=dict(color=ui.A_TEXT, family="JetBrains Mono", size=11))
    _layout(fig, height=height, showlegend=False, bargap=0.35,
            yaxis=dict(showticklabels=False, showgrid=False), xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _spark(values, fmt, height=90):
    colors = [ui.BORDER] * (len(values) - 1) + [ui.ACCENT] if values else []
    fig = go.Figure(go.Bar(y=values, marker=dict(color=colors, cornerradius=3),
                           hovertemplate="%{y:" + fmt + "}<extra></extra>"))
    _layout(fig, height=height, showlegend=False, bargap=0.25,
            yaxis=dict(visible=False, range=[0, max(values + [0.01]) * 1.1]), xaxis=dict(visible=False))
    fig.update_layout(margin=dict(l=0, r=0, t=4, b=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_player():
    st.title("👤 Player Pages")
    players = _players(_mtime("weekly_stats.csv"), "")
    names = list(players["name"])
    sel = st.session_state.get("player_sel")
    if sel not in names:
        # Links from other pages may carry another source's spelling ("Jr.", periods).
        by_key = dict(zip(players["_k"], players["name"]))
        st.session_state["player_sel"] = by_key.get(normalize_name(sel or ""), names[0] if names else None)
    name = st.selectbox("Player", names, key="player_sel", placeholder="Search a player")
    if not name:
        return
    info = players[players["name"] == name].iloc[0]
    k, pos, team = info["_k"], info["position"], info["team"]
    stat = PRIMARY_STAT.get(pos, "receiving_yds")

    # ---- header
    photo = load_player_photos().get(k)
    inj = injury_report()
    status = inj.loc[inj["full_name"].map(normalize_name) == k, "now"]
    status = status.iloc[0] if len(status) and status.iloc[0] not in ("—", "TBD") else None
    depth = load_csv_if_exists("footballguys_depth.csv")
    slot = ""
    if depth is not None:
        d = depth[depth["player_name"].map(normalize_name) == k]
        if len(d):
            slot = f"{d.iloc[0]['position']}{int(d.iloc[0]['depth_rank'])}"
    games = week_games()
    opp = ""
    if not games.empty:
        gm = games[(games["home_team"] == team) | (games["away_team"] == team)]
        if len(gm):
            r = gm.iloc[0]
            opp = f"vs {r['away_team']}" if r["home_team"] == team else f"@ {r['home_team']}"
        elif current_week():
            opp = "bye"
    log = _game_log(name, stat)
    live = live_over_lines()
    my_lines = live[live["_k"] == k] if not live.empty else live
    line_now = my_lines.loc[my_lines["stat_name"] == stat, "line"]
    line_now = float(line_now.iloc[0]) if len(line_now) else None
    hits = int((log["value"] > line_now).sum()) if line_now is not None else None

    avatar = (f"<img src='{photo}' style='width:64px;height:64px;border-radius:50%;object-fit:cover'>" if photo else
              f"<span style='width:64px;height:64px;border-radius:50%;background:{ui.A_TINT};color:{ui.A_TEXT};display:flex;"
              f"align-items:center;justify-content:center;font-family:{ui.MONO};font-size:20px'>"
              f"{ui.esc(''.join(p[0] for p in name.split()[:2]))}</span>")
    meta = " · ".join(x for x in [team, pos, slot and f"depth {slot}", opp and f"{opp} this week"] if x)
    stats = [("LAST 8 AVG", f"{log['value'].mean():.1f}" if len(log) else "—"),
             ("THIS WEEK LINE", f"{line_now:g}" if line_now is not None else "—"),
             ("HIT RATE", f"{hits}/{len(log)}" if hits is not None else "—")]
    st.markdown(
        f"<section style='display:flex;flex-wrap:wrap;align-items:center;gap:28px;padding:18px 22px;border-radius:14px;"
        f"background:{ui.SURFACE};box-shadow:0 0 0 1px {ui.EDGE};margin-bottom:8px'>"
        f"<div style='display:flex;align-items:center;gap:14px;flex:1;min-width:260px'>{avatar}"
        f"<div style='display:flex;flex-direction:column;gap:4px'><div style='display:flex;align-items:center;gap:8px'>"
        f"<span style='font-size:20px;font-weight:500'>{ui.esc(name)}</span>"
        + (ui.badge(status, *ui.INJ_BADGE.get(status, (ui.EDGE, ui.N2))) if status else "")
        + f"</div><span style='font-size:12px;color:{ui.N3};font-family:{ui.MONO}'>{ui.esc(meta)}</span></div></div>"
        + "".join(f"<div style='display:flex;flex-direction:column;gap:3px'><span style='font-size:9.5px;letter-spacing:.14em;"
                  f"color:{ui.N4};font-family:{ui.MONO}'>{a}</span><span style='font-size:18px;font-family:{ui.MONO}'>{b}</span></div>"
                  for a, b in stats) + "</section>", unsafe_allow_html=True)

    # ---- game log + line history
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section(f"Game log · {pretty_stat_name(stat)}")
        if log.empty:
            st.caption("No games yet.")
        else:
            colors = [BAR_OVER if line_now is not None and v > line_now else BAR_UNDER for v in log["value"]]
            _bars(list(log["value"]), list(log["label"]), line=line_now, colors=colors)
            if line_now is not None:
                st.caption(f"Cleared this week's line in **{hits} of {len(log)}** games.")
            else:
                st.caption("No Underdog line for this stat this week yet.")
    with c2:
        ui.section("Underdog line history")
        h = line_history()
        h = h[(h["_k"] == k) & (h["stat_name"] == stat) & (h["event"] != "pulled")] if not h.empty else h
        if h.empty:
            st.caption("No archived lines for this player yet.")
        else:
            h = h.assign(day=h["seen_at"].dt.tz_convert("US/Eastern").dt.strftime("%m/%d"))
            daily = h.sort_values("seen_at").groupby("day", sort=False).tail(1).tail(8)
            _bars(list(daily["line"]), list(daily["day"]), colors=[ui.BORDER] * (len(daily) - 1) + [ui.ACCENT])
            first, last = float(h.iloc[0]["line"]), float(h.iloc[-1]["line"])
            now_txt = f"{line_now:g}" if line_now is not None else f"{last:g} (not posted right now)"
            mv = (line_now if line_now is not None else last) - first
            st.caption(f"Opened at {first:g}, now {now_txt} ({mv:+g}).")

    # ---- usage + props
    c3, c4 = st.columns(2, gap="large")
    with c3:
        ui.section("Usage trend")
        snaps = _snap_shares(name, team)
        if len(snaps):
            vals = list((snaps["offense_pct"] * 100).round(0))
            st.markdown(f"<span style='font-size:11px;color:{ui.N3}'>Snap share · now <b>{vals[-1]:.0f}%</b></span>",
                        unsafe_allow_html=True)
            _spark(vals, ".0f")
        if pos != "QB" and len(log) and "target_share" in log.columns:
            vals = list((log["target_share"].fillna(0) * 100).round(0))
            st.markdown(f"<span style='font-size:11px;color:{ui.N3}'>Target share · now <b>{vals[-1]:.0f}%</b></span>",
                        unsafe_allow_html=True)
            _spark(vals, ".0f")
        if not len(snaps) and (pos == "QB" or not len(log)):
            st.caption("No usage data yet.")
    with c4:
        ui.section("Current props")
        cands = priced_candidates("Blend")
        mine = cands[(cands["_k"] == k) & (cands["choice"].astype(str).str.lower() == "over")] if not cands.empty else cands
        pk = dict(zip(zip(mine["stat_name"], mine["line"]), mine["prob"])) if not mine.empty else {}
        if my_lines.empty:
            st.caption("No Underdog lines for this player right now.")
        for r in my_lines.sort_values("stat_name").itertuples():
            prices = f"o {r.over_price:.2f} / u {r.under_price:.2f}" if r.under_price == r.under_price else f"o {r.over_price:.2f}"
            p = pk.get((r.stat_name, r.line))
            prob = f"P(over) {p * 100:.0f}%" if p is not None else "no model"
            st.markdown(ui.row_card(f"{ui.esc(pretty_stat_name(r.stat_name))} {ui.mono(f'{r.line:g}', ui.A_TEXT, '12px')}",
                                    f"{prices} · {prob}"), unsafe_allow_html=True)
        st.caption("P(over) = blended model + market estimate for this exact line (same as the +EV Finder).")

    ui.section("News")
    if st.button("Pull recent news", key=f"pp_news_{k}"):
        st.session_state[f"pp_news_items_{k}"] = get_player_news(name)
    for it in st.session_state.get(f"pp_news_items_{k}", []):
        st.markdown(f"- [{ui.esc(it.get('headline'))}]({it.get('link')}) · {ui.esc(it.get('source') or '')}")
