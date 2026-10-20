"""
The three main screens of the simplified app (2026-10-20 redesign): Picks, Build and
My Bets. Everything else in app.py still exists, under "More".

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
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    ROOT_DIR, append_bets, load_bet_log, save_bet_results, bet_log_location, pretty_stat_name,
)

_MODELS_DIR = os.path.join(ROOT_DIR, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)
from parlay_calculator import evaluate_entry, MAX_LEGS  # noqa: E402

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


def _leg_from_pick(r: pd.Series) -> dict:
    return {
        "player": r["player"], "team": r.get("team"), "stat_name": r["stat_name"],
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
}


def _stat_label(stat_name: str) -> str:
    return STAT_LABELS.get(stat_name, pretty_stat_name(stat_name).lower())


def _pick_label(choice: str, line, stat_name: str) -> str:
    side = "Over" if str(choice).lower() in ("over", "higher") else "Under"
    return f"{side} {line:g} {_stat_label(stat_name)}"


# ---------------------------------------------------------------------------- Picks

def page_picks():
    _css()
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

    # Underdog allows one pick per player per entry.
    slip_players = {l["player"] for l in _slip()}
    st.caption(f"{len(view)} pick(s) · board pulled {str(info.get('pulled_at') or '')[:16].replace('T', ' ')} UTC")
    if view.empty:
        st.info("Nothing clears this bar right now. Try **All**, or turn off **Proven stat types only**.")

    for i, r in view.head(40).iterrows():
        label, cls = _strength(r["edge"])
        tags = f'<span class="ph-tag {cls}">{label}</span>'
        if r["verdict"] != "trusted":
            tags += '<span class="ph-tag ph-warn">Unproven stat</span>'
        price = f"pays {r['decimal']:.2f}x" if pd.notna(r["decimal"]) else ""
        record = (f" · this prop type: {r['record'] * 100:.0f}% right on real lines"
                  if pd.notna(r["record"]) else "")
        card, btn = st.columns([6, 1], vertical_alignment="center")
        card.markdown(f"""
<div class="ph-card"><div class="ph-row">
  <div>
    <div class="ph-player">{r['player']}</div>
    <div class="ph-sub">{r.get('team') or ''} {r.get('position') or ''}{record}</div>
    <div class="ph-pick">{_pick_label(r['choice'], r['line'], r['stat_name'])}</div>
    <div style="margin-top:6px">{tags}</div>
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
        rows = [{"Stat": _stat_label(s).capitalize(), "Graded props": v["n"],
                 "Model right": _pct(v["accuracy"], 1), "Baseline": _pct(v["baseline"], 1),
                 "Rating": VERDICT_COPY[v["verdict"]][0]}
                for s, v in rec.get("by_stat", {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Graded against real Underdog closing lines, weeks {', '.join(rec.get('weeks', []))}. "
                       "Baseline = always taking whichever side hit more often. Updated by "
                       "`backtesting/backtest_line_history.py`.")
        else:
            st.caption("No track record file yet: run `python backtesting/backtest_line_history.py`.")


# ---------------------------------------------------------------------------- Build

def page_build():
    _css()
    st.title("My parlay")
    legs = _slip()
    if not legs:
        st.info("Your parlay is empty. Add picks from **Picks**.")
        if st.button("Find picks →", type="primary"):
            import nav_registry
            nav_registry.go("picks")
        return

    for i, l in enumerate(legs):
        c1, c2, c3 = st.columns([5, 2, 1], vertical_alignment="center")
        c1.markdown(f"**{l['player']}** · {_pick_label(l['choice'], l['line'], l['stat_name'])}")
        c2.markdown(f"{_pct(l['prob'])} chance")
        if c3.button("✕", key=f"ph_rm_{i}", help="Remove"):
            legs.pop(i)
            st.rerun()

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
    if ev["adjustments"]:
        st.caption("Adjusted for picks on the same team that tend to hit (or miss) together.")
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
            "date": date.today().isoformat(), "sport": "NFL", "player": l["player"],
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
            st.success("Saved. Mark it won or lost on **My Bets** once the games finish.")
    st.caption("This app never places bets: place the entry on Underdog yourself, then save it here.")


# -------------------------------------------------------------------------- My Bets

def _entries(log: pd.DataFrame) -> pd.DataFrame:
    from bet_entries import _with_entries, _entry_payout
    b = _with_entries(log)
    out = []
    for eid, legs in b.groupby("entry_id", sort=False):
        res = legs["result"].astype(str).str.lower()
        status = "lost" if (res == "lost").any() else "won" if (res == "won").all() else \
            "push" if (res.isin(["won", "push"])).all() else "pending"
        stake = pd.to_numeric(legs["stake"], errors="coerce").iloc[0]
        payout = _entry_payout(legs)
        profit = (stake * payout - stake if status == "won" and payout else
                  -stake if status == "lost" else 0.0)
        out.append({"entry_id": eid, "date": legs["date"].iloc[0], "legs": legs, "status": status,
                    "stake": stake, "payout": payout, "profit": profit})
    return pd.DataFrame(out)


def page_my_bets():
    _css()
    st.title("My bets")
    log = load_bet_log()
    if log.empty:
        st.info("No bets saved yet. Build a parlay and tap **I placed this bet**.")
        st.caption(bet_log_location())
        return
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
    for _, row in pending.iterrows():
        with st.container(border=True):
            to_win = f"\\${row['stake'] * row['payout']:,.2f}" if row["payout"] else "—"
            st.markdown(f"**{len(row['legs'])}-pick entry** · {row['date']} · "
                        f"\\${row['stake']:,.2f} to win {to_win}")
            for _, l in row["legs"].iterrows():
                st.markdown(f"- {l['player']}: {_pick_label(l['choice'], float(l['line']), l['stat'])}")
            b1, b2, _ = st.columns([1, 1, 3])
            for btn, result in ((b1, "won"), (b2, "lost")):
                if btn.button("Won" if result == "won" else "Lost", key=f"ph_{result}_{row['entry_id']}",
                              width="stretch"):
                    keys = set(row["legs"]["logged_at"].astype(str) + "|" + row["legs"]["player"].astype(str))
                    upd = log.copy()
                    hit = (upd["logged_at"].astype(str) + "|" + upd["player"].astype(str)).isin(keys)
                    upd.loc[hit, "result"] = result
                    try:
                        save_bet_results(upd)
                        st.rerun()
                    except Exception as ex:  # noqa: BLE001
                        st.error(f"Couldn't save: {ex}")

    done = e[e["status"] != "pending"]
    if not done.empty:
        st.subheader("History")
        st.dataframe(pd.DataFrame({
            "Date": done["date"], "Picks": done["legs"].map(len),
            "Result": done["status"].str.title(), "Stake": done["stake"].map(lambda v: f"${v:,.2f}"),
            "Profit": done["profit"].map(lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"),
        }), hide_index=True, width="stretch")
    st.caption(bet_log_location() + " Full editing lives under More → Bet Log.")
