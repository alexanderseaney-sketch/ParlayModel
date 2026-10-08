"""
Home page (2026-10 redesign, the default page): a one-screen summary -- bankroll stat
band, this week's games (each opens Game Center), open entries, the best edges right
now (add to slip), and injury movers. Everything comes from the same loaders the
detail pages use.
"""
import pandas as pd
import streamlit as st

import nav_registry
import ui
from bankroll import bankroll_now
from bet_entries import _with_entries, _entry_payout, _RESULT_TO_STATUS
from research_data import week_games, injury_report, priced_candidates
from theme import stat_band
from utils import load_bet_log, pretty_stat_name

STATE_BADGE = {"live": ("LIVE", ui.A_TINT, ui.A_TEXT, ui.A_MID), "pending": ("PENDING", ui.EDGE, ui.N2, ui.BORDER),
               "won": ("WON", ui.A_TINT, ui.A_TEXT, ui.A_MID), "lost": ("LOST", ui.ROSE_FILL, ui.ROSE, "#5c3a44"),
               "push": ("PUSH", ui.EDGE, ui.N2, ui.BORDER)}


def _open_entries() -> list[dict]:
    from live_tracker import entry_status
    try:
        bets = load_bet_log()
    except Exception:  # noqa: BLE001
        return []
    if bets.empty:
        return []
    b = _with_entries(bets)
    out = []
    for eid, legs in b.groupby("entry_id", sort=False):
        if not (legs["result"] == "pending").any():
            continue
        state = entry_status([_RESULT_TO_STATUS.get(x, "pending") for x in legs["result"]])
        stake = pd.to_numeric(legs["stake"], errors="coerce").max()
        payout = _entry_payout(legs)
        n = len(legs)
        out.append({"title": "Single pick" if n == 1 else f"{n}-pick entry",
                    "sub": ", ".join(legs["player"].astype(str).head(4)) + (" …" if n > 4 else ""),
                    "stake": (f"${stake:,.2f}" if stake == stake else "—")
                    + (f" → ${stake * payout:,.2f}" if payout and stake == stake else ""),
                    "state": state, "logged": str(legs["logged_at"].astype(str).max())})
    return sorted(out, key=lambda e: e["logged"], reverse=True)


def page_home():
    st.title("🏈 Home")
    # ---- stat band
    bank, last_week = bankroll_now()
    opens = _open_entries()
    cands = priced_candidates("Blend")
    best = cands.sort_values("edge", ascending=False).head(1) if not cands.empty else cands
    best_txt = (f"{best.iloc[0]['edge'] * 100:+.1f} pts" if not best.empty else "—")
    stat_band([("BANKROLL", f"${bank:,.0f}"),
               ("LAST WEEK P/L", "—" if last_week is None else f"{'-' if last_week < 0 else '+'}${abs(last_week):,.2f}"),
               ("OPEN ENTRIES", str(len(opens))), ("BEST EDGE", best_txt)])

    # ---- this week's games
    ui.section("This week's games")
    games = week_games()
    if games.empty:
        st.caption("No upcoming games in schedules.csv.")
    per_row = 5
    for start in range(0, len(games), per_row):
        cols = st.columns(per_row)
        for col, g in zip(cols, games.iloc[start:start + per_row].itertuples()):
            total = f"{g.total_line:g}" if g.total_line == g.total_line else "—"
            col.markdown(
                f"<div style='display:flex;flex-direction:column;gap:5px;padding:12px 14px;border-radius:8px;"
                f"background:{ui.ROW};box-shadow:0 0 0 1px {ui.EDGE}'>"
                f"<span style='font-size:9.5px;letter-spacing:.1em;color:{ui.N4};font-family:{ui.MONO};white-space:nowrap'>{ui.esc(g.kick_label)}</span>"
                f"<span style='font-size:14px;font-weight:500;font-family:{ui.MONO};white-space:nowrap'>{ui.esc(g.matchup)}</span>"
                f"<span style='font-size:11px;color:{ui.N3};font-family:{ui.MONO}'>{ui.esc(g.spread_txt)} · O/U {total}</span>"
                f"<span style='font-size:10.5px;color:{ui.N4}'>{ui.esc(g.weather_txt)}</span></div>",
                unsafe_allow_html=True)
            if col.button("Open ›", key=f"home_game_{g.game_id}", use_container_width=True, help="Open in Game Center"):
                st.session_state["game_id"] = g.game_id
                nav_registry.go("game_center")

    c1, c2, c3 = st.columns(3, gap="large")
    # ---- open entries
    with c1:
        ui.section("Open entries")
        if not opens:
            st.caption("No open entries.")
        for e in opens[:6]:
            label, bg, fg, border = STATE_BADGE.get(e["state"], STATE_BADGE["pending"])
            st.markdown(ui.row_card(ui.esc(e["title"]), ui.esc(e["sub"]), ui.badge(label, bg, fg, border, "9.5px"),
                                    extra=ui.mono(e["stake"], ui.N4, "10.5px")), unsafe_allow_html=True)
        if st.button("Bet Log →", key="home_go_log", type="tertiary"):
            nav_registry.go("bet_log")
    # ---- best edges
    with c2:
        ui.section("Best edges right now")
        top = cands.sort_values("edge", ascending=False).head(5) if not cands.empty else cands
        if top.empty:
            st.caption("No priced props right now.")
        slip_keys = {(l["player"], l["stat"], str(l["choice"]).lower(), l["line"]) for l in st.session_state.get("slip", [])}
        for i, p in enumerate(top.to_dict("records")):
            a, b = st.columns([6, 1])
            pick = f"{str(p['choice']).upper()} {p['line']:g} {pretty_stat_name(p['stat_name'])}"
            a.markdown(ui.row_card(ui.esc(p["player"]), ui.esc(pick), ui.badge(f"{p['edge'] * 100:+.1f}")),
                       unsafe_allow_html=True)
            in_slip = (p["player"], p["stat_name"], str(p["choice"]).lower(), p["line"]) in slip_keys
            if b.button("✓" if in_slip else "➕", key=f"home_add_{i}", disabled=in_slip,
                        help="In your slip" if in_slip else "Add to Parlay Builder slip"):
                from ev_finder import _send_to_builder
                _send_to_builder([p])
                st.toast(f"Added {p['player']} {pick}")
                st.rerun()
        if st.button("+EV Finder →", key="home_go_ev", type="tertiary"):
            nav_registry.go("ev_finder")
    # ---- injury movers
    with c3:
        ui.section("Injury movers")
        inj = injury_report()
        movers = inj[inj["abs_move"] > 0].head(4) if not inj.empty else inj
        if movers.empty:
            st.caption("No injury-related line moves this week yet.")
        for r in movers.itertuples():
            icon, icolor = {"up": ("↗", ui.A_TEXT), "down": ("↘", ui.ROSE)}.get(r.dir, ("→", ui.N4))
            stat = (r.stat or "").replace("_", " ")
            line = (f"{stat} {r.line_open:g} → pulled" if r.line_pulled else
                    f"{stat} {r.line_open:g} → {r.line_now:g}")
            bg, fg = ui.INJ_BADGE.get(r.now, (ui.EDGE, ui.N2))
            st.markdown(ui.row_card(f"<span style='color:{icolor}'>{icon}</span> {ui.esc(r.full_name)} {ui.mono(r.team)}",
                                    ui.mono(line, ui.N3, "11px"), ui.badge(r.now, bg, fg, size="9.5px")),
                        unsafe_allow_html=True)
        if st.button("Injury Tracker →", key="home_go_inj", type="tertiary"):
            nav_registry.go("injuries")
