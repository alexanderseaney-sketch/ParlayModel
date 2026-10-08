"""
Usage Trends page (2026-10 redesign): who is gaining or losing role -- snap share and
target share over each player's last 8 games, sorted by the 3-game snap-share change
so risers come first. Click a row to open that player's page.
Data: nflverse snap_counts.csv (offense_pct) + weekly_stats.csv (target_share).
"""
import pandas as pd
import streamlit as st

import nav_registry
from research_data import _mtime
from utils import load_csv_if_exists, normalize_name


@st.cache_data(show_spinner=False)
def _usage(snap_m: float, ws_m: float) -> pd.DataFrame:
    s = load_csv_if_exists("snap_counts.csv")
    w = load_csv_if_exists("weekly_stats.csv")
    if s is None or w is None:
        return pd.DataFrame()
    s = s[(s["game_type"].fillna("REG") == "REG") & s["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    season = int(s["season"].max())
    latest_wk = int(s.loc[s["season"] == season, "week"].max())
    s["_k"] = s["player"].map(normalize_name)
    s = s.sort_values(["season", "week"])
    # Players who played in the current season, ranked on their last 8 games.
    active = set(s.loc[(s["season"] == season) & (s["week"] >= latest_wk - 1), "_k"])
    s = s[s["_k"].isin(active)]
    w = w[w["season_type"].fillna("REG") == "REG"].copy()
    w["_k"] = w["player_display_name"].map(normalize_name)
    # dict, not an indexed Series: a mid-season trade / duplicate row would make .get()
    # return a Series. Keep the larger share for a duplicated player-game.
    ts = w.groupby(["_k", "season", "week"])["target_share"].max().to_dict()

    rows = []
    for k, g in s.groupby("_k"):
        g = g.tail(8)
        snaps = [round(v * 100) for v in g["offense_pct"].fillna(0)]
        tgts = [round(float(v) * 100) if pd.notna(v := ts.get((k, se, wk))) else 0
                for se, wk in zip(g["season"], g["week"])]
        last = g.iloc[-1]
        pos = last["position"]
        change = snaps[-1] - snaps[-4] if len(snaps) >= 4 else None
        rows.append({"Player": last["player"], "Team": last["team"], "Pos": pos,
                     "Snap share · 8 games": snaps, "Snap now": snaps[-1] / 100,
                     "Target share · 8 games": None if pos == "QB" else tgts,
                     "Target now": None if pos == "QB" else tgts[-1] / 100,
                     "3-game snap change": change,
                     "Trend": "—" if change is None else ("↗ +" if change > 0 else "↘ " if change < 0 else "– ")
                     + ("" if change is None else f"{change} pts")})
    df = pd.DataFrame(rows)
    return df.sort_values("3-game snap change", ascending=False, na_position="last").reset_index(drop=True)


def page_usage():
    st.title("📈 Usage Trends")
    df = _usage(_mtime("snap_counts.csv"), _mtime("weekly_stats.csv"))
    if df.empty:
        st.info("No snap-count data yet — run the nflverse pull.")
        return
    c1, c2 = st.columns([3, 2])
    pos = c1.segmented_control("Position", ["All", "QB", "RB", "WR", "TE"], default="All", key="usage_pos",
                               label_visibility="collapsed") or "All"
    min_snap = c2.slider("Min snap share now", 0, 100, 20, 5, format="%d%%", key="usage_min")
    view = df if pos == "All" else df[df["Pos"] == pos]
    view = view[view["Snap now"] * 100 >= min_snap].reset_index(drop=True)

    event = st.dataframe(
        view[["Player", "Team", "Pos", "Snap share · 8 games", "Snap now", "Target share · 8 games", "Target now", "Trend"]],
        hide_index=True, use_container_width=True, height=620, on_select="rerun", selection_mode="single-row",
        key=f"usage_table_{st.session_state.get('usage_nonce', 0)}",
        column_config={
            "Snap share · 8 games": st.column_config.BarChartColumn(y_min=0, y_max=100, color="#9184d9"),
            "Target share · 8 games": st.column_config.BarChartColumn(y_min=0, y_max=40, color="#b5abfc"),
            "Snap now": st.column_config.NumberColumn(format="percent"),
            "Target now": st.column_config.NumberColumn(format="percent"),
        })
    st.caption("Share of team offensive snaps and of team targets over each player's last 8 games (target share "
               "'—' for QBs). Sorted by 3-game snap-share change, risers first. Select a row to open the player page.")
    sel = event.selection.rows if event and event.selection else []
    if sel:
        st.session_state["player_sel"] = view.iloc[sel[0]]["Player"]
        # New table key next time, so coming back here doesn't re-fire the old selection.
        st.session_state["usage_nonce"] = st.session_state.get("usage_nonce", 0) + 1
        nav_registry.go("player")
