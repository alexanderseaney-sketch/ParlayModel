"""
Game Center page (2026-10 redesign): one game at a time -- matchup header with implied
points, odds (spread / total / moneylines from schedules.csv), weather, both teams'
injury report, and the best model-backed Underdog props in the game.
Home's game cards set st.session_state.game_id and switch here.
"""
import pandas as pd
import streamlit as st

import ui
from research_data import week_games, injury_report, priced_candidates, TEAM_NAMES


def _ml(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{v:+.0f}" if v == v else "—"


def _tile(label: str, value: str) -> str:
    return (f"<div style='display:flex;flex-direction:column;gap:4px;padding:12px 14px;border-radius:8px;"
            f"background:{ui.ROW};box-shadow:0 0 0 1px {ui.EDGE}'><span style='font-size:9.5px;letter-spacing:.14em;"
            f"color:{ui.N4};font-family:{ui.MONO}'>{ui.esc(label)}</span><span style='font-size:14px;font-weight:500;"
            f"font-family:{ui.MONO}'>{ui.esc(value)}</span></div>")


def _team_block(abbr: str, implied, align: str) -> str:
    imp = f"implied {implied:.1f}" if implied == implied else "implied —"
    return (f"<div style='display:flex;flex-direction:column;gap:4px;align-items:{align}'>"
            f"<span style='font-size:34px;font-weight:500;letter-spacing:-.02em;font-family:{ui.MONO}'>{ui.esc(abbr)}</span>"
            f"<span style='font-size:12px;color:{ui.N3}'>{ui.esc(TEAM_NAMES.get(abbr, abbr))}</span>"
            f"<span style='font-size:11px;color:{ui.A_TEXT};font-family:{ui.MONO}'>{imp}</span></div>")


@st.cache_data(ttl=60, show_spinner=False)
def _game_props(away: str, home: str) -> pd.DataFrame:
    """Best side of every model-backed single-game Underdog prop for either team."""
    c = priced_candidates("Blend")
    if c.empty:
        return c
    c = c[c["team"].isin([away, home])]
    c = c.sort_values("prob", ascending=False).drop_duplicates(["player", "stat_name", "line"])
    c = c[c["prob"] >= 0.5]  # the favored side only (Underdog sometimes prices just one side)
    c["conf"] = (c["prob"] - 0.5).abs()
    return c.sort_values("conf", ascending=False)


def page_game_center():
    st.title("🏟️ Game Center")
    games = week_games()
    if games.empty:
        st.info("No upcoming games found in schedules.csv — run the nflverse pull.")
        return
    ids = list(games["game_id"])
    labels = dict(zip(games["game_id"], games["matchup"]))
    if st.session_state.get("game_id") not in ids:
        st.session_state["game_id"] = ids[0]
    picked = st.pills("Game", ids, format_func=labels.get, key="game_id", label_visibility="collapsed")
    g = games[games["game_id"] == (picked or ids[0])].iloc[0]

    venue = g["stadium"] if isinstance(g["stadium"], str) else ""
    st.markdown(
        f"<section style='display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:center;padding:20px 24px;"
        f"border-radius:14px;background:{ui.SURFACE};box-shadow:0 0 0 1px {ui.EDGE};margin:6px 0 10px'>"
        + _team_block(g["away_team"], g["away_implied"], "flex-start")
        + f"<div style='display:flex;flex-direction:column;gap:4px;align-items:center;text-align:center'>"
          f"<span style='font-size:10px;letter-spacing:.16em;color:{ui.N4};font-family:{ui.MONO}'>AT</span>"
          f"<span style='font-size:13px;font-family:{ui.MONO}'>{ui.esc(g['kick_label'])}</span>"
          f"<span style='font-size:11px;color:{ui.N3}'>{ui.esc(venue)}</span></div>"
        + _team_block(g["home_team"], g["home_implied"], "flex-end") + "</section>",
        unsafe_allow_html=True)

    total = f"{g['total_line']:g}" if g["total_line"] == g["total_line"] else "—"
    tiles = [("SPREAD", g["spread_txt"]), ("TOTAL", total), (f"{g['away_team']} ML", _ml(g["away_moneyline"])),
             (f"{g['home_team']} ML", _ml(g["home_moneyline"])), ("WEATHER", g["weather_txt"])]
    st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px'>"
                + "".join(_tile(a, b) for a, b in tiles) + "</div>", unsafe_allow_html=True)
    st.caption("Lines from nflverse schedules (consensus close/current); implied points = (total ± spread) / 2.")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Injury report")
        inj = injury_report()
        inj = inj[inj["team"].isin([g["away_team"], g["home_team"]])] if not inj.empty else inj
        if inj.empty:
            st.caption("No one on either injury report.")
        for _, r in inj.iterrows():
            bg, fg = ui.INJ_BADGE.get(r["now"], (ui.EDGE, ui.N2))
            prac = r["practice"] if isinstance(r["practice"], str) else "—"
            st.markdown(ui.row_card(
                f"{ui.mono(r['team'], ui.N4)} {ui.esc(r['full_name'])} {ui.mono(r['position'], ui.N4)}",
                f"{ui.esc(r['injury'])} · practice {ui.esc(prac)}", ui.badge(r["now"], bg, fg)), unsafe_allow_html=True)
    with right:
        ui.section("Best props in this game")
        props = _game_props(g["away_team"], g["home_team"])
        if props.empty:
            st.caption("No model-backed props posted for this game yet.")
        slip_keys = {(l["player"], l["stat"], str(l["choice"]).lower(), l["line"]) for l in st.session_state.get("slip", [])}
        for i, p in enumerate(props.head(12).to_dict("records")):
            c1, c2 = st.columns([6, 1])
            pick = f"{str(p['choice']).upper()} {p['line']:g} {p['stat_name'].replace('_', ' ')}"
            right_html = (f"<span style='display:flex;flex-direction:column;align-items:flex-end;gap:2px'>"
                          + ui.mono(f"{p['prob'] * 100:.0f}%", ui.A_TEXT, "12px")
                          + ui.mono(f"{p['decimal']:.2f}x" if p["decimal"] == p["decimal"] else "—", ui.N4) + "</span>")
            c1.markdown(ui.row_card(f"{ui.esc(p['player'])} {ui.mono(p['team'], ui.N4)}", ui.esc(pick), right_html),
                        unsafe_allow_html=True)
            in_slip = (p["player"], p["stat_name"], str(p["choice"]).lower(), p["line"]) in slip_keys
            if c2.button("✓" if in_slip else "➕", key=f"gc_add_{g['game_id']}_{i}", disabled=in_slip,
                         help="In your slip" if in_slip else "Add to Parlay Builder slip"):
                from ev_finder import _send_to_builder
                _send_to_builder([p])
                st.toast(f"Added {p['player']} {pick}")
                st.rerun()
        st.caption("Probability is the blended model + market estimate (same as the +EV Finder's default).")
