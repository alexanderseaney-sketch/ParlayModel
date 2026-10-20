"""
The three main screens of the simplified app (2026-10-20 redesign): Picks, Build and
My Bets. Each is the first view of its tab; the older pages are the other views (nav_registry).

Design rules for these pages: one decision per screen, plain-English labels instead of
model jargon, every number shown is one the user acts on (chance it hits, payout, is
it worth it), and nothing is shown for stat types the model hasn't earned trust on
against real Underdog lines (backtesting/line_history_track_record.json).

Probabilities are the same ones the +EV Finder uses: the model's real-line probability
blended with the de-vigged market (odds_utils.blend_with_market).
"""
import json
import os
import sys
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import recent_form
from utils import (
    ROOT_DIR, append_bets, load_bet_log, save_bet_results, bet_log_location, pretty_stat_name,
)

_MODELS_DIR = os.path.join(ROOT_DIR, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)
from parlay_calculator import (  # noqa: E402
    evaluate_entry, find_best_entries, games_in, leg_count_note, MAX_LEGS, RECOMMENDED_LEGS,
)

TRACK_RECORD_PATH = os.path.join(ROOT_DIR, "backtesting", "line_history_track_record.json")

# Edge = our chance minus the market's fair chance. On the real-line backtest, picks
# with an 8%+ edge were the only band with a clearly positive ROI.
STRONG_EDGE = 0.08
GOOD_EDGE = 0.03

VERDICT_COPY = {
    "trusted": ("Proven", "Beats the market baseline on real lines"),
    "unproven": ("Unproven", "No better than always taking the commoner side yet"),
    "avoid": ("Avoid", "Worse than a coin flip on real lines"),
}

CSS = """
<style>
.ph-card {background: var(--pm-surface, #232532); border: 1px solid var(--pm-n800, #3f424d);
  border-radius: 12px; padding: 14px 16px; margin-bottom: 6px;}
.ph-row {display: flex; justify-content: space-between; align-items: center; gap: 12px;}
.ph-player {font-size: 17px; font-weight: 600; color: var(--pm-text, #e9e9ed);}
.ph-sub {font-size: 13px; color: var(--pm-n400, #b2b6ca); margin-top: 2px;}
.ph-pick {font-size: 15px; font-weight: 600; margin-top: 8px; color: var(--pm-text, #e9e9ed);}
.ph-big {font-size: 26px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
  color: var(--pm-text, #e9e9ed); line-height: 1.1; text-align: right;}
.ph-small {font-size: 12px; color: var(--pm-n500, #9397ab); text-align: right;}
.ph-tag {display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
  border-radius: 999px; margin-right: 6px; letter-spacing: .02em;}
.ph-strong {background: #1f3d2c; color: #7ee2a8;}
.ph-good {background: #2b2741; color: #b5abfc;}
.ph-lean {background: #292b31; color: #b2b6ca;}
.ph-warn {background: #3d2a1f; color: #f0b27a;}
.ph-stat {background: var(--pm-surface, #232532); border: 1px solid var(--pm-n800, #3f424d);
  border-radius: 12px; padding: 14px 16px; text-align: center;}
.ph-stat .v {font-size: 28px; font-weight: 600; font-family: 'JetBrains Mono', monospace;}
.ph-stat .l {font-size: 12px; color: var(--pm-n400, #b2b6ca); margin-top: 2px;}
.ph-verdict {border-radius: 12px; padding: 12px 16px; font-size: 15px; font-weight: 600; margin: 8px 0 4px;}
.ph-yes {background: #1f3d2c; color: #7ee2a8;}
.ph-no {background: #3d1f24; color: #f08a96;}
.ph-meh {background: #292b31; color: #cfd3e5;}
/* Slip rows: keep the remove / player card buttons beside a long pick name on phones. */
[class*="st-key-ph_leg_"] {flex-wrap: nowrap !important;}
[class*="st-key-ph_leg_"] > div:first-child {min-width: 0; flex: 1 1 0;}
</style>
"""


def _css():
    st.markdown(CSS, unsafe_allow_html=True)


def _slip() -> list[dict]:
    if "helper_slip" not in st.session_state:
        st.session_state.helper_slip = []
    return st.session_state.helper_slip


def _pct(p, digits=0) -> str:
    return "—" if p is None or not np.isfinite(p) else f"{p * 100:.{digits}f}%"


def load_track_record() -> dict:
    try:
        with open(TRACK_RECORD_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _strength(edge: float) -> tuple[str, str]:
    if edge >= STRONG_EDGE:
        return "Strong", "ph-strong"
    if edge >= GOOD_EDGE:
        return "Good", "ph-good"
    return "Lean", "ph-lean"


@st.cache_data(ttl=600, show_spinner=False)
def _candidates() -> tuple[pd.DataFrame, dict]:
    from ev_finder import build_candidates
    return build_candidates()


def get_picks() -> tuple[pd.DataFrame, dict]:
    """One row per (player, stat): the side we like, with chance, market chance, edge
    and the stat's real-line verdict."""
    c, info = _candidates()
    if c.empty:
        return c, info
    c = c.copy()
    c["chance"] = c["blend_prob"].astype(float)
    # No market price (one-sided board rows): compare against a coin flip.
    c["market"] = c["market_prob"].astype(float).fillna(0.5)
    c["edge"] = c["chance"] - c["market"]
    c = c[c["chance"] > 0.5]
    c = (c.sort_values("edge", ascending=False)
         .drop_duplicates(["player", "stat_name"], keep="first"))
    rec = load_track_record().get("by_stat", {})
    c["verdict"] = c["stat_name"].map(lambda s: rec.get(s, {}).get("verdict", "unproven"))
    c["record"] = c["stat_name"].map(lambda s: rec.get(s, {}).get("accuracy"))
    return c.reset_index(drop=True), info


def game_labels(picks: pd.DataFrame) -> dict:
    """Underdog match_id -> "BUF vs KC", from the teams on the board in that game."""
    if picks.empty or "game" not in picks.columns:
        return {}
    g = picks.dropna(subset=["game", "team"]).groupby("game")["team"]
    return {k: " vs ".join(sorted(set(v))) for k, v in g}


def _leg_from_pick(r: pd.Series) -> dict:
    return {
        "player": r["player"], "team": r.get("team"), "stat_name": r["stat_name"],
        "position": r.get("position"), "game": r.get("game"), "leg_key": r.get("leg_key"),
        "line": float(r["line"]), "choice": str(r["choice"]).lower(),
        "prob": float(r["chance"]), "market_prob": float(r["market"]),
        "decimal": float(r["decimal"]) if pd.notna(r["decimal"]) else None,
        "payout_multiplier": r.get("payout_multiplier"), "position_prop": r.get("position_prop"),
        "verdict": r.get("verdict"),
    }


STAT_LABELS = {
    "receiving_yds": "receiving yards", "rushing_yds": "rushing yards", "passing_yds": "passing yards",
    "receiving_rec": "receptions", "rush_rec_tds": "rush + rec TDs", "passing_tds": "passing TDs",
    "passing_ints": "interceptions thrown",
    "passing_comps": "pass completions", "passing_att": "pass attempts", "passing_long": "longest completion",
    "passing_and_rushing_yds": "pass + rush yards", "rushing_att": "rush attempts",
    "rushing_long": "longest rush", "receiving_tgts": "targets", "receiving_long": "longest reception",
    "rush_rec_yds": "rush + rec yards", "fantasy_points": "fantasy points", "fumbles_lost": "fumbles lost",
    "last_touchdown_scored": "last TD scorer", "period_first_touchdown_scored": "first TD scorer",
    "tackles": "solo tackles", "assists": "tackle assists", "tackles_and_assists": "tackles + assists",
    "sacks": "sacks", "defensive_ints": "interceptions (defense)", "field_goals_made": "field goals made",
    "extra_points_made": "extra points made", "kicking_points": "kicking points",
    "period_1_receiving_rec": "1Q receptions", "period_1_2_receiving_rec": "1H receptions",
    "period_1_receiving_yds": "1Q receiving yards", "period_1_2_receiving_yds": "1H receiving yards",
    "period_1_rushing_yds": "1Q rushing yards", "period_1_2_rushing_yds": "1H rushing yards",
    "period_1_passing_yds": "1Q passing yards", "period_1_2_passing_yds": "1H passing yards",
    "period_1_passing_tds": "1Q passing TDs", "period_1_2_passing_tds": "1H passing TDs",
    "period_1_rush_rec_tds": "1Q rush + rec TDs", "period_1_2_rush_rec_tds": "1H rush + rec TDs",
    "25_rec_yds_each_half": "25+ rec yards each half", "50_rec_yds_each_half": "50+ rec yards each half",
    "25_rush_yds_each_half": "25+ rush yards each half", "50_rush_yds_each_half": "50+ rush yards each half",
    "100_pass_yds_each_half": "100+ pass yards each half", "150_pass_yds_each_half": "150+ pass yards each half",
    "25_pass_yds_each_quarter": "25+ pass yards each quarter", "50_pass_yds_each_quarter": "50+ pass yards each quarter",
}


def _stat_label(stat_name: str) -> str:
    return STAT_LABELS.get(stat_name, pretty_stat_name(stat_name).lower())


def _pick_label(choice: str, line, stat_name: str) -> str:
    side = "Over" if str(choice).lower() in ("over", "higher") else "Under"
    return f"{side} {line:g} {_stat_label(stat_name)}"


# ---------------------------------------------------------------------------- Picks

def page_picks():
    _css()
    recent_form.css()
    st.title("Today's picks")
    st.caption("The best-value props on Underdog right now, ranked by how much better our "
               "chance is than the price. Tap **Add** to put a pick on your parlay.")

    picks, info = get_picks()
    if picks.empty:
        st.info("No picks yet: the board or the predictions haven't been pulled. "
                "They refresh automatically every 6 hours.")
        return

    f1, f2, f3 = st.columns([2, 2, 3])
    min_label = f1.segmented_control("Show", ["Strong", "Good+", "All"], default="Good+",
                                     key="ph_strength")
    proven_only = f2.toggle("Proven stat types only", value=True, key="ph_proven",
                            help="Only stat types where the model has beaten the market "
                                 "baseline on real Underdog lines.")
    search = f3.text_input("Search player", key="ph_search", placeholder="e.g. Puka Nacua",
                           label_visibility="collapsed")

    floor = {"Strong": STRONG_EDGE, "Good+": GOOD_EDGE}.get(min_label or "Good+", -1.0)
    view = picks[picks["edge"] >= floor]
    if proven_only:
        view = view[view["verdict"] == "trusted"]
    else:
        view = view[view["verdict"] != "avoid"]
    if search:
        view = view[view["player"].str.contains(search, case=False, na=False)]

    games = game_labels(picks)
    # Underdog allows one pick per player per entry.
    slip_players = {l["player"] for l in _slip()}
    st.caption(f"{len(view)} pick(s) · board pulled {str(info.get('pulled_at') or '')[:16].replace('T', ' ')} UTC")
    if info.get("held_back"):
        with st.expander(f"{len(info['held_back'])} pick(s) held back by the safety check"):
            st.caption("The model's starting estimate for these is far from what the player did in "
                       "his last 4 games, which usually means a team or role change the data hasn't "
                       "caught up with. Hidden until it lines up again.")
            st.write(", ".join(info["held_back"]))
    if view.empty:
        st.info("Nothing clears this bar right now. Try **All**, or turn off **Proven stat types only**.")

    for i, r in view.head(40).iterrows():
        label, cls = _strength(r["edge"])
        tags = f'<span class="ph-tag {cls}">{label}</span>'
        if r["verdict"] != "trusted":
            tags += ('<span class="ph-tag ph-warn">Unproven stat</span>' if pd.notna(r["record"])
                     else '<span class="ph-tag ph-warn">Not graded yet</span>')
        is_over = str(r["choice"]).lower() in ("over", "higher")
        if r.get("last5_n", 0) == 5 and pd.notna(r.get("last5_over")):
            against = r["last5_over"] <= 1 if is_over else r["last5_over"] >= 4
            if against:
                hit5 = int(r["last5_over"]) if not is_over else 5 - int(r["last5_over"])
                tags += f'<span class="ph-tag ph-lean">Against his last 5 ({hit5}/5 went the other way)</span>'
        price = f"pays {r['decimal']:.2f}x" if pd.notna(r["decimal"]) else ""
        record = (f" · this prop type: {r['record'] * 100:.0f}% right on real lines"
                  if pd.notna(r["record"]) else "")
        card, btn = st.columns([6, 1], vertical_alignment="center")
        card.markdown(f"""
<div class="ph-card"><div class="ph-row">
  <div>
    <div class="ph-player">{r['player']}</div>
    <div class="ph-sub">{r.get('team') or ''} {r.get('position') or ''}{' · ' + games[r['game']] if r.get('game') in games else ''}{record}</div>
    <div class="ph-pick">{_pick_label(r['choice'], r['line'], r['stat_name'])}</div>
    <div style="margin-top:6px">{tags}</div>
    {recent_form.chart_html(recent_form.recent_games(r['player'], r['stat_name']), r['line'], r['choice'])}
  </div>
  <div>
    <div class="ph-big">{_pct(r['chance'])}</div>
    <div class="ph-small">chance it hits</div>
    <div class="ph-small">market {_pct(r['market'])} · {price}</div>
  </div>
</div></div>""", unsafe_allow_html=True)
        if r["player"] in slip_players:
            btn.button("In parlay", key=f"ph_add_{i}", disabled=True, width="stretch")
        elif btn.button("Add", key=f"ph_add_{i}", type="primary", width="stretch",
                        disabled=len(_slip()) >= MAX_LEGS):
            _slip().append(_leg_from_pick(r))
            st.toast(f"Added {r['player']} to your parlay")
            st.rerun()

    if _slip():
        st.divider()
        if st.button(f"Go to my parlay ({len(_slip())} picks) →", type="primary", width="stretch"):
            import nav_registry
            nav_registry.go("build")

    with st.expander("How we rate stat types"):
        rec = load_track_record()
        order = {"trusted": 0, "unproven": 1, "avoid": 2}
        rows = [{"Stat": _stat_label(s).capitalize(), "Graded props": v["n"],
                 "Model right": _pct(v["accuracy"], 1), "Baseline": _pct(v["baseline"], 1),
                 "Overs hit": _pct(v.get("over_rate"), 0),
                 "Rating": VERDICT_COPY[v["verdict"]][0]}
                for s, v in sorted(rec.get("by_stat", {}).items(),
                                   key=lambda kv: (order[kv[1]["verdict"]], -kv[1]["accuracy"]))]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Graded against real Underdog closing lines, weeks {', '.join(rec.get('weeks', []))}. "
                       "Baseline = always taking whichever side hit more often. Overs hit = how often "
                       "the real result went over Underdog's line (under 50% means Unders won more). "
                       "1st quarter / 1st half props aren't graded yet: their lines are logged from now on. "
                       "Updated by "
                       "`backtesting/backtest_line_history.py`.")
        else:
            st.caption("No track record file yet: run `python backtesting/backtest_line_history.py`.")


# ---------------------------------------------------------------------------- Build

def _auto_build(picks: pd.DataFrame, games: dict):
    """The "Build it for me" panel: pick a parlay type and size, and fill the slip with the
    highest expected-return entry the model can find."""
    with st.container(border=True):
        st.markdown("**Build it for me**")
        kind = st.segmented_control("Parlay type", ["Across games", "Same game"], default="Across games",
                                    key="ph_kind", label_visibility="collapsed")
        kind = kind or "Across games"
        game = None
        if kind == "Same game":
            counts = picks[picks["game"].isin(games)].groupby("game").size().sort_values(ascending=False)
            if counts.empty:
                st.caption("No game on the board has picks yet.")
                return
            game = st.selectbox("Game", counts.index.tolist(), key="ph_game",
                                format_func=lambda g: f"{games[g]} ({counts[g]} picks)")
            st.caption("Picks from both teams in one game. Underdog pays less when picks tend to "
                       "hit together, and we price that in.")
        else:
            st.caption("One pick per game, so one bad game can only sink one pick.")
        n = st.segmented_control("Picks", [2, 3, 4, 5, 6], default=RECOMMENDED_LEGS, key="ph_n_legs")
        n = int(n or RECOMMENDED_LEGS)
        note = leg_count_note(n)
        if note:
            st.caption(note)
        if st.button("Build my parlay", type="primary", width="stretch", key="ph_auto"):
            pool = picks[picks["verdict"] == "trusted"].assign(prob=picks["chance"], market_prob=picks["market"])
            mode = "same_game" if kind == "Same game" else "multi_game"
            with st.spinner("Trying every combination..."):
                best = find_best_entries(pool, n_legs_range=(n, n), top_k=1, pool_size=18 if n <= 4 else 13,
                                         mode=mode, game=game)
            best = best[best["valid"] & (best["ev"] > 0)] if not best.empty else best
            if best.empty:
                where = f"in {games.get(game, 'this game')}" if game is not None else "on the board"
                st.warning(f"No {n}-pick parlay {where} is worth it right now with proven stat types. "
                           "Try fewer picks or another game.")
                return
            _slip().clear()
            _slip().extend(_leg_from_pick(pd.Series(l)) for l in best.iloc[0]["legs"])
            st.rerun()


def _link_lines(links: list[dict]) -> list[str]:
    out = []
    for k in links:
        if abs(k["phi"]) < 0.05:
            continue
        who = f"{k['a']} and {k['b']}"
        where = "teammates" if k["relation"] == "same_team" else "opponents"
        if k["phi"] > 0:
            out.append(f"{who} ({where}) tend to hit together.")
        else:
            out.append(f"{who} ({where}) tend to pull against each other.")
    return out


def page_build():
    _css()
    recent_form.css()
    st.title("My parlay")
    picks, _ = get_picks()
    games = game_labels(picks)
    legs = _slip()
    if not picks.empty:
        _auto_build(picks, games)
    if not legs:
        st.info("Your parlay is empty. Build one above, or add picks from **Picks**.")
        if st.button("Find picks →"):
            import nav_registry
            nav_registry.go("picks")
        return

    n_games = games_in(legs)
    if n_games == 1:
        st.markdown(f"**Same game parlay** · {games.get(legs[0].get('game'), '')}")
    elif n_games > 1:
        st.markdown(f"**{len(legs)} picks across {n_games} games**")
    for i, l in enumerate(legs):
        sub = " · ".join(x for x in [games.get(l.get("game")), f"{_pct(l['prob'])} chance"] if x)
        form = recent_form.chart_html(recent_form.recent_games(l["player"], l["stat_name"]),
                                      l["line"], l["choice"])
        # A horizontal container (not columns) so the buttons stay beside the pick on phones.
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"ph_leg_{i}"):
            st.markdown(f"**{l['player']}** · {_pick_label(l['choice'], l['line'], l['stat_name'])}"
                        f"  \n<span style='font-size:12px;color:#9397ab'>{sub}</span>" + form,
                        unsafe_allow_html=True, width="stretch")
            with st.container(width="content", gap="small"):
                if st.button("✕", key=f"ph_rm_{i}", help="Remove"):
                    legs.pop(i)
                    st.rerun()
                if st.button("", key=f"ph_card_{i}", icon=":material/bar_chart:", help="Player card: every game"):
                    import nav_registry
                    st.session_state["player_sel"] = l["player"]
                    nav_registry.go("player")

    ev = evaluate_entry(legs)
    payout = ev["payout"]
    st.write("")
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="ph-stat"><div class="v">{_pct(ev["joint_prob"], 1)}</div>'
                f'<div class="l">chance all hit</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="ph-stat"><div class="v">{f"{payout:.1f}x" if payout else "—"}</div>'
                f'<div class="l">pays</div></div>', unsafe_allow_html=True)
    ev_txt = f"{ev['ev'] * 100:+.0f}%" if ev["ev"] is not None else "—"
    s3.markdown(f'<div class="ph-stat"><div class="v">{ev_txt}</div>'
                f'<div class="l">expected return</div></div>', unsafe_allow_html=True)

    if not ev["valid"]:
        st.markdown(f'<div class="ph-verdict ph-meh">Not a valid entry yet: {ev["validity_note"]}.</div>',
                    unsafe_allow_html=True)
    elif ev["ev"] is None:
        st.markdown('<div class="ph-verdict ph-meh">Missing a price on one pick, so we can\'t '
                    'tell if it\'s worth it.</div>', unsafe_allow_html=True)
    elif ev["ev"] > 0.05:
        st.markdown('<div class="ph-verdict ph-yes">Worth it: you get paid more than the risk.</div>',
                    unsafe_allow_html=True)
    elif ev["ev"] > -0.05:
        st.markdown('<div class="ph-verdict ph-meh">Roughly a fair bet: no real edge either way.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="ph-verdict ph-no">Not worth it: the payout is too small for '
                    'the risk. Drop the weakest pick.</div>', unsafe_allow_html=True)
    note = leg_count_note(len(legs))
    if note:
        st.caption(note)
    for line in _link_lines(ev.get("links") or []):
        st.caption(line)
    base = ev.get("base_payout")
    if payout and base and abs(payout - base) / base > 0.01:
        direction = "lowers" if payout < base else "raises"
        st.caption(f"Underdog {direction} the payout on linked picks in the same game: about "
                   f"{payout:.1f}x instead of {base:.1f}x. Underdog shows the exact number when you "
                   "build the entry.")
    if any(l.get("verdict") != "trusted" for l in legs):
        st.caption("⚠️ Includes a stat type the model hasn't proven on real lines yet.")

    st.divider()
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    stake = c1.number_input("Stake ($)", min_value=0.0, value=5.0, step=1.0, key="ph_stake")
    if payout:
        c2.markdown(f"Win **\\${stake * payout:,.2f}** if all hit")
    if c3.button("Clear", width="stretch"):
        legs.clear()
        st.rerun()
    if st.button("I placed this bet: save it to My Bets", type="primary", width="stretch",
                 disabled=not ev["valid"]):
        entry_id, now = uuid.uuid4().hex[:10], datetime.now().isoformat()
        rows = [{
            "date": pd.Timestamp.now(tz="US/Eastern").date().isoformat(), "sport": "NFL", "player": l["player"],
            "stat": l["stat_name"], "choice": l["choice"], "line": l["line"],
            "multiplier_or_odds": l.get("decimal"), "stake": stake, "result": "pending",
            "notes": "Saved from My parlay", "logged_at": now, "entry_id": entry_id,
            "entry_payout": round(payout, 2) if payout else "",
        } for l in legs]
        err = append_bets(rows)
        if err:
            st.error(err)
        else:
            legs.clear()
            st.success("Saved. **My Bets** settles it automatically once the games finish.")
    st.caption("This app never places bets: place the entry on Underdog yourself, then save it here.")


# -------------------------------------------------------------------------- My Bets

_LEG_ICON = {"won": "✅", "lost": "❌", "push": "⚪", "pending": "⏳"}


def _entries(log: pd.DataFrame) -> pd.DataFrame:
    from bet_entries import _with_entries, settle_entry
    b = _with_entries(log)
    out = []
    for eid, legs in b.groupby("entry_id", sort=False):
        status, payout = settle_entry(legs)
        stake = pd.to_numeric(legs["stake"], errors="coerce").iloc[0]
        profit = (stake * payout - stake if status == "won" and payout else
                  -stake if status == "lost" else 0.0)
        out.append({"entry_id": eid, "date": legs["date"].iloc[0], "legs": legs, "status": status,
                    "stake": stake, "payout": payout, "profit": profit})
    return pd.DataFrame(out)


def _auto_grade(log: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Settle every pending leg whose game is final (bet_grading), save any change, and
    return (log, live tracker output for legs still pending)."""
    from bet_grading import grade_pending
    if not log["result"].fillna("pending").astype(str).str.lower().isin(["pending", ""]).any():
        return log, {}
    with st.spinner("Checking box scores…"):
        graded, n, live = grade_pending(log)
    if n:
        try:
            save_bet_results(graded, message=f"Auto-grade {n} pick(s) from final box scores")
            st.toast(f"Settled {n} pick{'s' if n != 1 else ''} from final box scores.")
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Graded {n} pick(s) but couldn't save them: {ex}")
    return graded, live


def _leg_line(l: pd.Series, t: dict | None) -> str:
    from bet_grading import can_grade
    res = str(l.get("result") or "pending").lower()
    text = f"{l['player']}: {_pick_label(l['choice'], float(l['line']), l['stat'])}"
    actual = pd.to_numeric(l.get("actual"), errors="coerce")
    if res != "pending":
        detail = ("voided, didn't play or landed on the line" if res == "push" and pd.isna(actual)
                  else "landed on the line" if res == "push"
                  else f"finished with {actual:g}" if pd.notna(actual) else "")
        return f"{_LEG_ICON.get(res, '⏳')} {text}" + (f" · {detail}" if detail else "")
    if not can_grade(l["stat"]):
        return f"⏳ {text} · can't be graded automatically, settle it by hand below"
    if t and t.get("state") == "live":
        now = f"has {t['value']:g}" if t.get("value") is not None else "in progress"
        return f"🟡 {text} · {now}" + (f", {t['text']}" if t.get("text") else "")
    if t and t.get("state") == "upcoming":
        return f"⏳ {text} · {t['text']}"
    return f"⏳ {text} · waiting on the final box score"


def _settle_by_hand(log: pd.DataFrame, legs: pd.DataFrame, result: str) -> None:
    """Fallback for legs the box score can't grade: set this entry's still-pending legs."""
    pending = legs[legs["result"].astype(str).str.lower() == "pending"]
    keys = set(pending["logged_at"].astype(str) + "|" + pending["player"].astype(str)
               + "|" + pending["stat"].astype(str))
    upd = log.copy()
    hit = (upd["logged_at"].astype(str) + "|" + upd["player"].astype(str)
           + "|" + upd["stat"].astype(str)).isin(keys)
    upd.loc[hit, "result"] = result
    try:
        save_bet_results(upd)
        st.rerun()
    except Exception as ex:  # noqa: BLE001
        st.error(f"Couldn't save: {ex}")


def page_my_bets():
    _css()
    st.title("My bets")
    log = load_bet_log()
    if log.empty:
        st.info("No bets saved yet. Build a parlay and tap **I placed this bet**.")
        st.caption(bet_log_location())
        return
    log, live = _auto_grade(log)
    e = _entries(log)
    settled = e[e["status"].isin(["won", "lost"])]
    won = int((settled["status"] == "won").sum())
    profit = float(pd.to_numeric(e["profit"], errors="coerce").fillna(0).sum())
    staked = float(pd.to_numeric(settled["stake"], errors="coerce").fillna(0).sum())
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="ph-stat"><div class="v">{won}-{len(settled) - won}</div>'
                f'<div class="l">won-lost</div></div>', unsafe_allow_html=True)
    color = "#7ee2a8" if profit >= 0 else "#f08a96"
    s2.markdown(f'<div class="ph-stat"><div class="v" style="color:{color}">'
                f'{"+" if profit >= 0 else "-"}${abs(profit):,.2f}</div><div class="l">profit</div></div>',
                unsafe_allow_html=True)
    roi = profit / staked if staked else None
    s3.markdown(f'<div class="ph-stat"><div class="v">{_pct(roi, 0) if roi is not None else "—"}</div>'
                f'<div class="l">return on settled bets</div></div>', unsafe_allow_html=True)

    pending = e[e["status"] == "pending"]
    st.subheader(f"Waiting on results ({len(pending)})")
    if pending.empty:
        st.caption("Nothing pending.")
    else:
        st.caption("Each pick settles on its own once its game is final, from the official box "
                   "score. A pick whose player doesn't play is voided and the entry pays at the "
                   "smaller multiplier, like on Underdog.")
    for _, row in pending.iterrows():
        with st.container(border=True):
            to_win = f"\\${row['stake'] * row['payout']:,.2f}" if row["payout"] else "—"
            st.markdown(f"**{len(row['legs'])}-pick entry** · {row['date']} · "
                        f"\\${row['stake']:,.2f} to win {to_win}")
            for _, l in row["legs"].iterrows():
                t = live.get((str(l["logged_at"]), str(l["player"]), str(l["stat"])))
                st.markdown("- " + _leg_line(l, t))
            with st.expander("Settle by hand"):
                st.caption("Only needed for a pick the box score can't grade (quarter props, "
                           "first TD scorer…). Sets this entry's unsettled picks.")
                b1, b2, b3 = st.columns(3)
                for btn, result, label in ((b1, "won", "They hit"), (b2, "lost", "One missed"),
                                           (b3, "push", "Voided")):
                    if btn.button(label, key=f"ph_{result}_{row['entry_id']}", width="stretch"):
                        _settle_by_hand(log, row["legs"], result)

    done = e[e["status"] != "pending"]
    if not done.empty:
        st.subheader("History")
        st.dataframe(pd.DataFrame({
            "Date": done["date"],
            "Picks": done["legs"].map(lambda legs: "  ".join(
                f"{_LEG_ICON.get(str(r).lower(), '⏳')} {p}" for p, r in zip(legs["player"], legs["result"]))),
            "Result": done["status"].map({"won": "Won", "lost": "Lost", "push": "Refunded"}),
            "Paid": done.apply(lambda r: f"{r['payout']:g}x" if r["status"] == "won" and r["payout"] else "—",
                               axis=1),
            "Stake": done["stake"].map(lambda v: f"${v:,.2f}"),
            "Profit": done["profit"].map(lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"),
        }), hide_index=True, width="stretch")
    st.caption(bet_log_location() + " To correct a settled result, use My bets → Full log.")
