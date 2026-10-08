"""
Matchup Heatmap page (2026-10 redesign): every defense's rank against QB / RB / WR / TE
(fantasy points allowed per game, current season; #1 toughest, #32 softest) and who it
plays this week. Data: data/raw/fantasy_matchups.csv (models/build_fantasy_matchups.py,
the same table the Fantasy page's Matchups tab uses).
"""
import pandas as pd
import streamlit as st

import ui
from research_data import week_games, current_week

POSITIONS = ["QB", "RB", "WR", "TE"]
SCORING = {"Half-PPR": ("rank_half", "fp_half_pg"), "PPR": ("rank_ppr", "fp_ppr_pg"), "Standard": ("rank_std", "fp_std_pg")}
STOPS = [(1, (0x3d, 0x25, 0x30)), (16, (0x29, 0x2b, 0x31)), (32, (0x5d, 0x52, 0x94))]


def _cell_color(rank: float) -> str:
    """3-stop scale: rose-dark (1, toughest) -> neutral (16) -> accent (32, softest)."""
    r = max(1.0, min(32.0, float(rank)))
    (r0, c0), (r1, c1) = (STOPS[0], STOPS[1]) if r <= 16 else (STOPS[1], STOPS[2])
    t = (r - r0) / (r1 - r0)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(c0, c1))


def _opponents() -> dict[str, str]:
    g = week_games()
    out = {}
    for r in g.itertuples():
        out[r.home_team] = f"vs {r.away_team}"
        out[r.away_team] = f"@ {r.home_team}"
    return out


def page_matchups():
    st.title("🔥 Matchup Heatmap")
    from utils import load_csv_if_exists
    m = load_csv_if_exists("fantasy_matchups.csv")
    if m is None or m.empty:
        st.info("No matchup table yet — run **Fantasy matchups** from Run Data Pulls.")
        return
    c1, c2 = st.columns([3, 2])
    sort_pos = c1.segmented_control("Sort by", POSITIONS, default="WR", key="heat_sort") or "WR"
    scoring = c2.segmented_control("Scoring", list(SCORING), default="Half-PPR", key="heat_scoring") or "Half-PPR"
    rank_col, pts_col = SCORING[scoring]
    st.caption("#1 = toughest defense against that position, #32 = softest (best to target). "
               "Pick a position above to sort by it.")

    wide_rank = m.pivot_table(index="def_team", columns="position", values=rank_col, aggfunc="first")
    wide_pts = m.pivot_table(index="def_team", columns="position", values=pts_col, aggfunc="first")
    wide_rank = wide_rank.sort_values(sort_pos, ascending=False)
    opp = _opponents()
    has_week = current_week() is not None

    legend = (f"<div style='display:flex;align-items:center;gap:10px;margin:2px 0 10px;font-size:9.5px;"
              f"letter-spacing:.14em;color:{ui.N4};font-family:{ui.MONO}'><span>TOUGHEST</span>"
              f"<div style='width:180px;height:8px;border-radius:4px;background:linear-gradient(90deg,"
              f"{_cell_color(1)},{_cell_color(16)},{_cell_color(32)})'></div><span>SOFTEST</span></div>")
    head = (f"<div style='display:grid;grid-template-columns:.8fr 1fr repeat(4,1fr);gap:6px;padding:8px 12px;"
            f"font-size:9.5px;letter-spacing:.14em;color:{ui.N4};font-family:{ui.MONO}'><span>DEF</span>"
            f"<span>THIS WEEK</span>" + "".join(
                f"<span style='text-align:center;color:{ui.A_TEXT if p == sort_pos else ui.N4}'>{p}"
                f"{' ▾' if p == sort_pos else ''}</span>" for p in POSITIONS) + "</div>")
    rows = []
    for team, ranks in wide_rank.iterrows():
        cells = []
        for p in POSITIONS:
            rk = ranks.get(p)
            if pd.isna(rk):
                cells.append(f"<div style='text-align:center;color:{ui.N5}'>—</div>")
                continue
            pts = wide_pts.loc[team, p]
            fg = "#f5f4ff" if rk >= 22 else ui.N1
            tip = f"{team} allows {pts:.1f} {scoring} pts/g to {p}s (#{int(rk)})"
            cells.append(f"<div title='{ui.esc(tip)}' style='text-align:center;padding:7px 0;border-radius:4px;"
                         f"background:{_cell_color(rk)};color:{fg};font-family:{ui.MONO};font-size:12px'>{int(rk)}</div>")
        this_week = opp.get(team, "bye" if has_week else "—")
        rows.append(f"<div style='display:grid;grid-template-columns:.8fr 1fr repeat(4,1fr);gap:6px;padding:4px 12px;"
                    f"align-items:center'><span style='font-family:{ui.MONO};font-weight:500'>{ui.esc(team)}</span>"
                    f"<span style='font-family:{ui.MONO};font-size:11.5px;color:{ui.N3}'>{ui.esc(this_week)}</span>"
                    + "".join(cells) + "</div>")
    seasons = m["derived_from_seasons"].iloc[0] if "derived_from_seasons" in m.columns else ""
    st.markdown(legend + f"<div style='border-radius:14px;padding:6px 0;background:{ui.TABLE_ROW};"
                f"box-shadow:0 0 0 1px {ui.EDGE};max-width:820px'>" + head + "".join(rows) + "</div>",
                unsafe_allow_html=True)
    st.caption(f"Fantasy points allowed per game by position ({scoring}), seasons {seasons}. Hover a cell for the "
               "points allowed.")
