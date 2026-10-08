"""
Bankroll page (2026-10 redesign): how the bankroll has moved, from the bet log's
SETTLED entries (an entry is won only if every non-push pick won; P/L uses the entry's
payout multiple -- won entries with no payout logged are left out of P/L).

Starting bankroll is a saved setting (bet_store.read_settings / write_settings: the
private bets repo when configured, else bet_logs/settings.json).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from bet_entries import _with_entries, _entry_payout, _RESULT_TO_STATUS
from model_performance import _layout
from theme import stat_band
from utils import load_bet_log

LAVENDER, ROSE = "#b5abfc", "#d1798a"
POS_BAR, NEG_BAR = "#9184d9", "#a8566a"
DEFAULT_START = 500.0


def starting_bankroll() -> float:
    if "bankroll_start" not in st.session_state:
        try:
            import bet_store
            st.session_state["bankroll_start"] = float(bet_store.read_settings().get("starting_bankroll", DEFAULT_START))
        except Exception:  # noqa: BLE001
            st.session_state["bankroll_start"] = DEFAULT_START
    return float(st.session_state["bankroll_start"])


def settled_entries() -> pd.DataFrame:
    """One row per settled entry: date, week label, legs, stats, stake, payout, state, pnl."""
    from live_tracker import entry_status
    try:
        bets = load_bet_log()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if bets.empty:
        return pd.DataFrame()
    b = _with_entries(bets)
    rows = []
    for eid, legs in b.groupby("entry_id"):
        state = entry_status([_RESULT_TO_STATUS.get(x, "pending") for x in legs["result"]])
        if state not in ("won", "lost", "push"):
            continue
        stake = pd.to_numeric(legs["stake"], errors="coerce").max()
        payout = _entry_payout(legs)
        if pd.isna(stake) or (state == "won" and not payout):
            continue
        pnl = stake * (payout - 1) if state == "won" else (-stake if state == "lost" else 0.0)
        d = pd.to_datetime(legs["date"].iloc[0], errors="coerce")
        rows.append({"entry_id": eid, "date": d, "legs": len(legs), "stats": list(legs["stat"].astype(str)),
                     "stake": float(stake), "payout": payout, "state": state, "pnl": float(pnl)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    iso = df["date"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    return df.sort_values("date")


def bankroll_now() -> tuple[float, float | None]:
    """(current bankroll, last ISO-week P/L) for Home."""
    s = settled_entries()
    start = starting_bankroll()
    if s.empty:
        return start, None
    weekly = s.groupby("week")["pnl"].sum()
    return start + s["pnl"].sum(), float(weekly.iloc[-1])


def _roi_bars(title: str, groups: pd.DataFrame) -> None:
    ui.section(title)
    if groups.empty:
        st.caption("No settled entries yet.")
        return
    g = groups.sort_values("roi")
    labels = [f"{a}  ·  {n} entr{'y' if n == 1 else 'ies'}" for a, n in zip(g["label"], g["n"])]
    fig = go.Figure(go.Bar(x=g["roi"], y=labels, orientation="h",
                           marker=dict(color=[POS_BAR if v >= 0 else NEG_BAR for v in g["roi"]], cornerradius=4),
                           hovertemplate="%{y}: ROI %{x:+.0%}<extra></extra>"))
    _layout(fig, height=60 + 34 * len(g), showlegend=False,
            xaxis=dict(tickformat="+.0%", zeroline=True, zerolinecolor=ui.BORDER))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_bankroll():
    st.title("💰 Bankroll")
    with st.sidebar:
        st.subheader("Bankroll")
        start = st.number_input("Starting bankroll ($)", min_value=0.0, step=50.0, value=starting_bankroll(),
                                key="bankroll_start_input")
        if start != st.session_state.get("bankroll_start"):
            st.session_state["bankroll_start"] = start
            try:
                import bet_store
                settings = bet_store.read_settings()
                settings["starting_bankroll"] = start
                bet_store.write_settings(settings)
            except Exception as e:  # noqa: BLE001
                st.caption(f"Couldn't save the setting: {e}")

    rng = st.segmented_control("Range", ["Season", "Last 4 weeks"], default="Season", key="bank_range",
                               label_visibility="collapsed") or "Season"
    s = settled_entries()
    if s.empty:
        stat_band([("BANKROLL", f"${start:,.0f}"), ("PROFIT/LOSS", "$0"), ("ROI", "—"), ("WIN RATE", "—"),
                   ("MAX DRAWDOWN", "—")])
        st.info("No settled entries yet. Results appear here as entries are graded on the Bet Log "
                "(✔ Save final results).")
        return

    weekly = s.groupby("week").agg(pnl=("pnl", "sum"), staked=("stake", "sum")).reset_index()
    weekly["balance"] = start + weekly["pnl"].cumsum()
    weekly["peak"] = np.maximum.accumulate(np.r_[start, weekly["balance"].to_numpy()])[1:]
    weekly["drawdown"] = weekly["balance"] - weekly["peak"]
    if rng == "Last 4 weeks":
        keep = set(weekly["week"].tail(4))
        weekly, s = weekly[weekly["week"].isin(keep)], s[s["week"].isin(keep)]

    pnl, staked = s["pnl"].sum(), s["stake"].sum()
    wins = (s["state"] == "won").sum()
    decided = s["state"].isin(["won", "lost"]).sum()
    stat_band([("BANKROLL", f"${start + s['pnl'].sum() if rng == 'Season' else weekly['balance'].iloc[-1]:,.0f}"),
               ("PROFIT/LOSS", f"{'-' if pnl < 0 else '+'}${abs(pnl):,.2f}"),
               ("ROI", f"{pnl / staked:+.1%}" if staked else "—"),
               ("WIN RATE", f"{wins / decided:.0%} ({wins}/{decided})" if decided else "—"),
               ("MAX DRAWDOWN", f"-${abs(weekly['drawdown'].min()):,.2f}")])

    ui.section("Balance")
    fig = go.Figure(go.Bar(
        x=weekly["week"], y=weekly["balance"],
        marker=dict(color=ui.A_TINT, line=dict(color=[LAVENDER if v >= 0 else ROSE for v in weekly["pnl"]], width=2),
                    cornerradius=4),
        customdata=weekly["pnl"], hovertemplate="%{x}: $%{y:,.2f} (%{customdata:+,.2f})<extra></extra>"))
    lo = min(weekly["balance"].min(), start)
    _layout(fig, height=260, showlegend=False, yaxis=dict(tickprefix="$", range=[lo * 0.95, weekly["balance"].max() * 1.03]),
            xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Each bar is the week-end balance; lavender edges are winning weeks, rose are losing weeks. "
               "Hover for the number.")

    ui.section("Drawdown from peak")
    fig = go.Figure(go.Bar(x=weekly["week"], y=weekly["drawdown"], marker=dict(color=ROSE, cornerradius=4),
                           hovertemplate="%{x}: %{y:,.2f} from peak<extra></extra>"))
    _layout(fig, height=170, showlegend=False, yaxis=dict(tickprefix="$"), xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ROI by stat: each entry's stake and P/L split evenly across its picks' stats.
    per_stat = []
    for r in s.itertuples():
        for stat in r.stats:
            per_stat.append({"label": stat.replace("_", " "), "stake": r.stake / r.legs, "pnl": r.pnl / r.legs,
                             "entry": r.entry_id})
    ps = pd.DataFrame(per_stat).groupby("label").agg(stake=("stake", "sum"), pnl=("pnl", "sum"),
                                                     n=("entry", "nunique")).reset_index()
    ps["roi"] = ps["pnl"] / ps["stake"]
    size = s.assign(label=np.select([s["legs"] == 1, s["legs"] == 2, s["legs"] == 3], ["Single", "2-pick", "3-pick"],
                                    "4+ picks")) \
        .groupby("label").agg(stake=("stake", "sum"), pnl=("pnl", "sum"), n=("entry_id", "count")).reset_index()
    size["roi"] = size["pnl"] / size["stake"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        _roi_bars("ROI by stat", ps)
    with c2:
        _roi_bars("ROI by entry size", size)
    st.caption("Settled entries only. Won entries count once a payout multiple is logged; a multi-pick entry's "
               "stake and result are split evenly across its picks for ROI by stat.")
