"""
Injury Tracker page (2026-10 redesign): this week's injury report for skill players --
status change since last week, latest practice participation, and how the player's
main Underdog line has moved this week. Data: research_data.injury_report().
"""
import streamlit as st

import ui
from research_data import injury_report, current_week

DIR_ICON = {"up": ("↗", ui.A_TEXT), "down": ("↘", ui.ROSE), "same": ("→", ui.N4)}


def _line_cell(r) -> str:
    stat = (r["stat"] or "").replace("_", " ")
    if r["line_open"] is None or r["line_open"] != r["line_open"]:
        return ui.mono("no line this week", ui.N4, "11px")
    if r["line_pulled"]:
        return f"<span style='color:{ui.N3}'>{ui.esc(stat)} {r['line_open']:g} → </span>" + ui.mono("pulled", ui.ROSE, "11px")
    mv = r["line_move"]
    txt = f"<span style='color:{ui.N3}'>{ui.esc(stat)} {r['line_open']:g} → {r['line_now']:g}</span>"
    if mv and mv == mv:
        txt += " " + ui.mono(f"({'+' if mv > 0 else '−'}{abs(mv):g})", ui.A_TEXT if mv > 0 else ui.ROSE, "11px")
    return txt


def page_injuries():
    st.title("🩹 Injury Tracker")
    cw = current_week()
    st.caption(f"Week {cw[1]} injury report for QB / RB / WR / TE — status change since last week, latest "
               "practice participation, and how each player's main Underdog line has moved." if cw else "")
    rows = injury_report()
    if rows.empty:
        st.info("No injury report for this week yet — run the nflverse pull from **Run Data Pulls**.")
        return

    view = st.segmented_control("Show", ["All", "Downgraded", "Upgraded", "Line moved"], default="All",
                                key="inj_filter", label_visibility="collapsed") or "All"
    team_opts = sorted(rows["team"].dropna().unique())
    teams = st.multiselect("Teams", team_opts, key="inj_teams", placeholder="All teams")
    if view == "Downgraded":
        rows = rows[rows["dir"] == "down"]
    elif view == "Upgraded":
        rows = rows[rows["dir"] == "up"]
    elif view == "Line moved":
        rows = rows[rows["abs_move"] > 0]
    if teams:
        rows = rows[rows["team"].isin(teams)]

    out = []
    for _, r in rows.iterrows():
        icon, icolor = DIR_ICON[r["dir"]]
        bg, fg = ui.INJ_BADGE.get(r["now"], (ui.EDGE, ui.N2))
        status = (ui.mono(r["prev"], ui.N4) + f" <span style='color:{icolor};font-size:13px'>{icon}</span> "
                  + ui.badge(r["now"], bg, fg))
        prac = (ui.badge(r["practice"], "#161826", ui.PRACTICE_COLOR.get(r["practice"], ui.N2), size="9.5px")
                if isinstance(r["practice"], str) else ui.mono("not on report", ui.N4))
        out.append([
            f"<span style='font-weight:500'>{ui.esc(r['full_name'])}</span> " + ui.mono(r["team"]),
            ui.mono(r["position"], ui.N4, "12px"),
            f"<span style='color:{ui.N3}'>{ui.esc(r['injury'])}</span>",
            status, prac, _line_cell(r),
        ])
    if not out:
        st.info("No players in this view.")
    else:
        ui.table(["PLAYER", "POS", "INJURY", "STATUS", "PRACTICE", "LINE MOVE"], out,
                 ["1.6fr", ".5fr", "1fr", "1.1fr", "1fr", "1.7fr"], min_width=780)
    st.caption("Practice: DNP did not practice · LP limited · FP full (latest practice report this week). "
               "Status is last week's game designation → this week's; TBD = the team hasn't posted game "
               "designations yet (usually the day before its game). Line move is the player's first Underdog "
               "line this week vs. now.")
