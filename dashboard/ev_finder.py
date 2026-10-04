"""
+EV Finder page (added 2026-10-02): ranks every model-backed Underdog option by its
edge over the de-vigged market, searches automatically for the highest-EV 2-6 pick
entries, and lets you build your own entry with model-vs-market odds, joint
probability, EV, the riskiest leg and an AI read.

Where it differs from the Parlay Builder: the builder is a manual slip editor that
compares each leg to Underdog's raw price; this page prices every leg against the
MARKET's fair probability (Underdog's two-sided prices with the margin removed, plus
an optional sportsbook consensus from The Odds API) and does the combination search
for you. Entries found here can be sent straight to the Parlay Builder's slip.

All math lives in models/ (odds_utils, parlay_calculator, interpreter); this module
is presentation only.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    load_csv_if_exists, load_current_predictions, normalize_name, score_underdog_board,
    is_low_noise_line, line_matches_proxy, load_leg_correlations, pretty_stat_name, ROOT_DIR,
    underdog_freshness_bar,
)
from odds_utils import add_underdog_market_probs, decimal_to_american, blend_with_market

_DATA_DIR = os.path.join(ROOT_DIR, "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)
from parlay_calculator import evaluate_entry, find_best_entries, riskiest_leg

UNDERDOG_PICKEM_URL = "https://underdogfantasy.com/pick-em/higher-lower/all/nfl"
BOOK_URLS = {
    "draftkings": "https://sportsbook.draftkings.com/leagues/football/nfl",
    "fanduel": "https://sportsbook.fanduel.com/navigation/nfl",
    "betmgm": "https://sports.betmgm.com/en/sports/football-11/betting/usa-9/nfl-35",
    "williamhill_us": "https://www.caesars.com/sportsbook-and-casino",
    "espnbet": "https://espnbet.com/sport/football/organization/us/competition/nfl",
}
# Nocturne dark series steps (dataviz reference palette, dark column).
C_MODEL, C_MARKET, C_BLEND = "#3987e5", "#d95926", "#199e70"
TEXT, MUTED, GRID = "#e9e9ed", "#9a9aa6", "rgba(233,233,237,0.08)"


def _key_status() -> dict[str, bool]:
    def has(name):
        if os.environ.get(name) and os.environ.get(name) != "your_odds_api_key_here":
            return True
        try:
            return bool(st.secrets.get(name))
        except Exception:
            return False
    return {"ODDS_API_KEY": has("ODDS_API_KEY"), "ANTHROPIC_API_KEY": has("ANTHROPIC_API_KEY")}


@st.cache_data(show_spinner=False)
def _team_lookup(mtime: float) -> dict[str, str]:
    """Current team per normalized player name from the 32 club team sites (fresher
    than weekly_stats' recent_team for anyone traded/signed since their last game)."""
    df = load_csv_if_exists("nfl_rosters.csv")
    if df is None:
        return {}
    df = df.assign(_k=df["player"].apply(normalize_name))
    df = df.sort_values("roster_status", key=lambda s: ~s.fillna("").str.startswith("Active"))
    return df.drop_duplicates("_k").set_index("_k")["team_abbr"].to_dict()


@st.cache_data(ttl=300, show_spinner=False)
def _sportsbook_consensus(mtime: float) -> pd.DataFrame | None:
    """Saved Odds API pull (data/raw/odds_api_props.csv) collapsed to a per-prop
    consensus. Cached 5 minutes; re-pulling (credits) only happens on the button."""
    df = load_csv_if_exists("odds_api_props.csv")
    if df is None or df.empty:
        return None
    from pull_odds_api import consensus
    c = consensus(df)
    c["_k"] = c["player"].apply(normalize_name)
    return c


def build_candidates(min_edge_floor: float = -1.0) -> tuple[pd.DataFrame, dict]:
    """Every model-backed, comparable single-game Underdog option with model, market
    and blended probabilities. Returns (candidates, info)."""
    props = load_csv_if_exists("underdog_props.csv")
    preds = load_current_predictions()
    weekly = load_csv_if_exists("weekly_stats.csv")
    info = {"pulled_at": None, "n_props": 0}
    if props is None or preds is None:
        return pd.DataFrame(), info
    info["pulled_at"] = props["pulled_at"].iloc[0] if "pulled_at" in props.columns else None
    info["n_props"] = len(props)

    board = score_underdog_board(props, preds, weekly)
    board = add_underdog_market_probs(board)
    board = board[board["has_model"] & (board["scope"] == "game")].copy()
    board = board[~board.apply(lambda r: is_low_noise_line(r["stat_name"], r["stat_value"]), axis=1)]
    # Same comparability gate the Parlay Builder / Weekly Bet Slip use: a prediction
    # made against a proxy far from the real line isn't trustworthy there.
    model_stat = board["stat_name"].replace({"receiving_rec": "receptions"})
    board = board[[line_matches_proxy(v, p, s) for v, p, s in
                   zip(board["stat_value"], board["proxy_line"], model_stat)]]
    if board.empty:
        return pd.DataFrame(), info

    roster_path = os.path.join(ROOT_DIR, "data", "raw", "nfl_rosters.csv")
    teams = _team_lookup(os.path.getmtime(roster_path) if os.path.exists(roster_path) else 0.0)
    board["team"] = board["_k"].map(teams).fillna(board["recent_team"])

    odds_path = os.path.join(ROOT_DIR, "data", "raw", "odds_api_props.csv")
    cons = _sportsbook_consensus(os.path.getmtime(odds_path) if os.path.exists(odds_path) else 0.0)
    board["book_fair_over"], board["n_books"] = np.nan, 0
    if cons is not None:
        m = board[["_k", "stat_name", "stat_value"]].merge(
            cons.rename(columns={"line": "stat_value"})[["_k", "stat_name", "stat_value", "consensus_fair_over", "n_books"]],
            on=["_k", "stat_name", "stat_value"], how="left")
        board["book_fair_over"] = m["consensus_fair_over"].to_numpy()
        board["n_books"] = m["n_books"].fillna(0).astype(int).to_numpy()

    is_over = board["choice"].astype(str).str.lower().isin(["over", "higher"])
    # Market fair prob for this side: Underdog's de-vigged two-way price, else the
    # sportsbook consensus when Underdog only offers one side.
    book_side = np.where(is_over, board["book_fair_over"], 1 - board["book_fair_over"])
    board["market_prob"] = board["market_fair_prob"].fillna(pd.Series(book_side, index=board.index))
    board["model_prob"] = board["side_prob"]
    board["blend_prob"] = blend_with_market(board["model_prob"], board["market_prob"])

    c = pd.DataFrame({
        "player": board["full_name"], "team": board["team"], "position": board["position"],
        "stat_name": board["stat_name"], "line": board["stat_value"], "choice": board["choice"],
        "decimal": board["decimal"], "payout_multiplier": board.get("payout_multiplier"),
        "model_prob": board["model_prob"], "market_prob": board["market_prob"],
        "underdog_fair": board["market_fair_prob"], "book_fair_over": board["book_fair_over"],
        "n_books": board["n_books"], "blend_prob": board["blend_prob"],
        "proxy_line": board["proxy_line"], "prop_type": board["prop_type"],
        "real_line_used": board["real_line_used"],
    })
    c["position_prop"] = c["position"].astype(str) + " " + c["prop_type"].astype(str)
    return c.reset_index(drop=True), info


def _apply_prob_source(c: pd.DataFrame, source: str) -> pd.DataFrame:
    c = c.copy()
    c["prob"] = c["blend_prob"] if source == "Blend" else c["model_prob"]
    c["edge"] = c["prob"] - c["market_prob"]
    c["ev"] = c["prob"] * c["decimal"] - 1
    return c


def _fmt_pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:.{digits}f}%"


def _odds_label(p):
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return "—"
    a = decimal_to_american(1 / p)
    return f"{a:+.0f}"


def _prob_bar_chart(legs: list[dict]):
    """Model vs market probability per leg, horizontal bars (one axis, two series)."""
    labels = [f"{l['player']} {str(l['choice']).upper()} {l['line']} {pretty_stat_name(l['stat_name'])}" for l in legs]
    fig = go.Figure()
    fig.add_bar(y=labels, x=[l["prob"] for l in legs], orientation="h", name="Ours",
                marker=dict(color=C_MODEL, cornerradius=4), hovertemplate="%{y}<br>Ours %{x:.1%}<extra></extra>")
    fig.add_bar(y=labels, x=[l.get("market_prob") if l.get("market_prob") == l.get("market_prob") else None for l in legs],
                orientation="h", name="Market (fair)",
                marker=dict(color=C_MARKET, cornerradius=4), hovertemplate="%{y}<br>Market %{x:.1%}<extra></extra>")
    fig.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, height=90 + 46 * len(legs),
                      margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT, family="Inter, sans-serif", size=12),
                      xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _send_to_builder(legs: list[dict]):
    for l in legs:
        st.session_state.slip.append({
            "player": l["player"], "stat": l["stat_name"], "choice": l["choice"], "line": l["line"],
            "underdog_multiplier": float(l["decimal"]) if pd.notna(l["decimal"]) else None,
            "my_prob": float(l["prob"]), "has_model": True,
            "team": l.get("team"), "position_prop": l.get("position_prop"),
        })


def _recent_games(player: str, stat_name: str, weekly: pd.DataFrame | None, n: int = 5):
    col = {"receiving_yds": "receiving_yards", "rushing_yds": "rushing_yards", "passing_yds": "passing_yards",
           "receiving_rec": "receptions", "passing_tds": "passing_tds", "passing_ints": "interceptions"}.get(stat_name)
    if weekly is None:
        return None
    w = weekly[weekly["player_display_name"].apply(normalize_name) == normalize_name(player)]
    w = w.sort_values(["season", "week"]).tail(n)
    if stat_name == "rush_rec_tds":
        vals = (w["rushing_tds"].fillna(0) + w["receiving_tds"].fillna(0)).tolist()
    elif col:
        vals = w[col].tolist()
    else:
        return None
    return [{"season": int(s), "week": int(wk), "value": (None if pd.isna(v) else float(v))}
            for s, wk, v in zip(w["season"], w["week"], vals)]


@st.cache_data(show_spinner=False)
def _top_features(mtime: float) -> dict[str, list]:
    path = os.path.join(ROOT_DIR, "models", "feature_importance.csv")
    if not os.path.exists(path):
        return {}
    fi = pd.read_csv(path)
    return {pt: [[r.feature, round(r.importance, 3)] for r in g.head(4).itertuples()]
            for pt, g in fi.sort_values("importance", ascending=False).groupby("prop_type")}


def _entry_panel(legs: list[dict], payout_mode: str, key: str):
    res = evaluate_entry(legs, payout_mode, load_leg_correlations())
    if not res["valid"]:
        st.error(f"Underdog wouldn't accept this entry: {res['validity_note']}.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Our joint prob", _fmt_pct(res["joint_prob"]),
              help="Correlation-adjusted (same-team legs use measured phi).")
    m2.metric("Market joint prob", _fmt_pct(res["market_joint_prob"]))
    m3.metric("Payout", f"{res['payout']:.2f}x" if res["payout"] else "—",
              help=f"Model fair payout {res['fair_payout']:.2f}x" if res["fair_payout"] else None)
    m4.metric("EV per $1", f"{res['ev']*100:+.1f}%" if res["ev"] is not None else "—")
    if res["adjustments"]:
        st.caption("Correlated pairs: " + "; ".join(f"{a} + {b} (phi {phi:+.2f})" for a, b, phi in res["adjustments"]))
    _prob_bar_chart(legs)
    risk = riskiest_leg(legs)
    if risk:
        st.markdown(f"⚠️ **Riskiest leg:** {risk['player']} {str(risk['choice']).upper()} {risk['line']} "
                    f"{pretty_stat_name(risk['stat_name'])} — {_fmt_pct(risk['prob'], 0)} to hit")

    b1, b2, b3 = st.columns([1.2, 1.2, 2])
    with b1:
        if st.button("➕ Send to Parlay Builder", key=f"send_{key}"):
            _send_to_builder(legs)
            st.toast(f"Added {len(legs)} legs to the slip.")
    with b2:
        st.link_button("Open Underdog pick'em ↗", UNDERDOG_PICKEM_URL)
    with b3:
        if st.button("🧠 AI read", key=f"ai_{key}"):
            from interpreter import explain_entry
            weekly = load_csv_if_exists("weekly_stats.csv")
            fi_path = os.path.join(ROOT_DIR, "models", "feature_importance.csv")
            feats = _top_features(os.path.getmtime(fi_path) if os.path.exists(fi_path) else 0.0)
            enriched = [{**l, "recent_games": _recent_games(l["player"], l["stat_name"], weekly),
                         "top_features": feats.get(l.get("prop_type"))} for l in legs]
            with st.spinner("Writing it up…"):
                text, source = explain_entry(res, enriched)
            st.session_state[f"ai_text_{key}"] = (text, source)
    if f"ai_text_{key}" in st.session_state:
        text, source = st.session_state[f"ai_text_{key}"]
        with st.container(border=True):
            st.markdown(text)
            st.caption("Written by Claude from the numbers above." if source == "claude"
                       else "Template summary — set ANTHROPIC_API_KEY for the AI write-up.")
    st.caption("Underdog has no public pre-filled slip link — the button opens the NFL pick'em board; "
               "add the picks there and check the payout it shows before entering.")


def page_ev_finder():
    st.title("📈 +EV Finder")
    st.caption("Every model-backed Underdog pick priced against the market's fair odds, plus automatic "
               "2–6 pick entry search. You place every entry yourself.")

    with st.sidebar:
        st.subheader("+EV Finder")
        prob_source = st.radio("Probability", ["Blend", "Model"], index=0, horizontal=True,
                               help="Blend = 35% model + 65% de-vigged market. "
                                    "Default because it scored best against real Underdog lines (Model "
                                    "Performance → Real-line backtest); the model alone is worse than "
                                    "the market on receiving and rushing yards.")
        min_edge = st.slider("Min edge over market", 0.0, 0.20, 0.02, 0.01, format="%.2f",
                             help="Model probability minus the market's fair probability for that side.")
        legs_range = st.slider("Entry size (picks)", 2, 6, (2, 4))
        payout_mode = st.radio("Entry payout", ["product", "standard"], horizontal=True,
                               format_func=lambda m: "Pick prices" if m == "product" else "Legacy table",
                               help="Pick prices: entry pays the product of each pick's decimal price "
                                    "(Underdog's odds-based pick'em). Legacy table: 3x/6x/10x/20x/37.5x.")
        st.markdown("**Data & keys**")
        keys = _key_status()
        for name, ok in keys.items():
            st.markdown(f"{'🟢' if ok else '⚪'} `{name}` {'set' if ok else 'not set'}")

    cands, info = build_candidates()
    if cands.empty:
        st.warning("No comparable model-backed props right now. Pull Underdog props and run "
                   "`python models/current_predictions.py` (Run Data Pulls page).")
        return
    cands = _apply_prob_source(cands, prob_source)
    underdog_freshness_bar(load_csv_if_exists("underdog_props.csv"), key="ev")
    st.caption(f"{len(cands)} comparable model-backed options"
               f" · {int((cands['n_books'] > 0).sum())} with sportsbook consensus")

    tab_legs, tab_auto, tab_build = st.tabs(["🎯 Best single picks", "🤖 Best entries", "🛠️ Build your own"])

    with tab_legs:
        view = cands[cands["edge"] >= min_edge].sort_values("edge", ascending=False)
        st.markdown(f"**{len(view)}** picks clear a {min_edge:.0%} edge.")
        show = pd.DataFrame({
            "Player": view["player"], "Team": view["team"],
            "Pick": view["choice"].str.upper() + " " + view["line"].astype(str) + " " + view["stat_name"].map(pretty_stat_name),
            prob_source: view["prob"], "Market": view["market_prob"], "Edge": view["edge"],
            "Fair odds": view["prob"].map(_odds_label), "Price": view["decimal"], "EV/$1": view["ev"],
            "Books": view["n_books"],
        })
        st.dataframe(show, hide_index=True, use_container_width=True, height=520, column_config={
            prob_source: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Market": st.column_config.NumberColumn(format="percent"),
            "Edge": st.column_config.NumberColumn(format="percent"),
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "EV/$1": st.column_config.NumberColumn(format="percent"),
        })

    with tab_auto:
        st.caption("Exhaustive search over the 18 best-edge picks (one per player, ≥2 teams per entry), "
                   "ranked by correlation-adjusted EV.")
        st.warning("Entry EV multiplies every leg's edge, so any overstatement compounds: a 4-pick "
                   "entry at +90% EV is four ~+17% legs. On real Underdog lines so far the model's "
                   "edges have been only partly real (Model Performance → Real-line backtest) — "
                   "treat big entry EVs as a ranking, not a forecast, and size stakes small.", icon="⚠️")
        with st.spinner("Searching combinations…"):
            best = find_best_entries(cands, n_legs_range=legs_range, top_k=8, pool_size=18,
                                     min_leg_edge=min_edge, payout_mode=payout_mode,
                                     correlations=load_leg_correlations())
        if best.empty:
            st.info("No valid entries from picks clearing this edge. Lower the edge floor.")
        else:
            best = best[best["valid"]]
            for i, row in best.head(12).iterrows():
                title = (f"{row['n_legs']} picks · EV {row['ev']*100:+.1f}% · hit {row['joint_prob']*100:.1f}% "
                         f"· {row['payout']:.2f}x")
                with st.expander(title, expanded=(i == 0)):
                    for l in row["legs"]:
                        st.markdown(f"- **{l['player']}** ({l['team']}) {str(l['choice']).upper()} {l['line']} "
                                    f"{pretty_stat_name(l['stat_name'])} — ours {_fmt_pct(l['prob'], 0)}, "
                                    f"market {_fmt_pct(l['market_prob'], 0)}, price {l['decimal']:.2f}")
                    _entry_panel(row["legs"], payout_mode, key=f"auto{i}")

    with tab_build:
        opts = cands.sort_values("edge", ascending=False).reset_index(drop=True)
        labels = {i: (f"{r.player} · {str(r.choice).upper()} {r.line} {pretty_stat_name(r.stat_name)} "
                      f"· edge {r.edge*100:+.1f}%") for i, r in opts.iterrows()}
        picked = st.multiselect("Pick 2–6 legs", list(labels), format_func=labels.get, max_selections=6)
        if len(picked) >= 2:
            _entry_panel(opts.loc[picked].to_dict("records"), payout_mode, key="custom")
        else:
            st.info("Select at least two legs.")

    with st.expander("Sportsbook lines (The Odds API)"):
        if not keys["ODDS_API_KEY"]:
            st.markdown("Add `ODDS_API_KEY` to `.env` to compare against DraftKings/FanDuel/etc. "
                        "(free tier: 500 credits/month; a full slate of 6 markets ≈ 84 credits).")
        else:
            credits = st.number_input("Max credits to spend", 6, 200, 36, 6)
            if st.button("Pull current sportsbook props"):
                from pull_odds_api import pull, OUT_PATH, DEFAULT_MARKETS
                with st.spinner("Pulling…"):
                    try:
                        df = pull(int(credits), DEFAULT_MARKETS)
                        if not df.empty:
                            df.to_csv(OUT_PATH, index=False)
                            _sportsbook_consensus.clear()
                            st.success(f"{len(df)} rows from {df['book'].nunique()} books.")
                    except Exception as e:
                        st.error(f"Pull failed: {e}")
            for book, url in BOOK_URLS.items():
                st.markdown(f"- [{book}]({url})")
