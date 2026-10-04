"""
AI Assistant page (added 2026-10-04, from IDEAS.md): a chat with Claude that answers
questions about tonight's board by calling the project's own functions as tools --
live Underdog lines, model predictions, the +EV Finder's pricing, the entry search,
game logs, depth charts, news and the real-line backtest -- instead of guessing.

Rules it runs under (same as models/interpreter.py): every number it states must come
from a tool result; it never places bets. The one write action, logging a bet, is
only STAGED by the tool -- nothing is written until you click Confirm on the page.

Needs ANTHROPIC_API_KEY (env / .env locally, Streamlit secrets when hosted).
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from anthropic import beta_tool

from utils import (
    load_csv_if_exists, load_current_predictions, normalize_name,
    load_bet_log, append_bet, get_player_news, ROOT_DIR, load_leg_correlations,
)

MODEL = "claude-opus-5-5"
MAX_ROWS = 25  # cap per tool result -- keeps context (and cost) bounded

SYSTEM_PROMPT = """You are the betting analyst inside ParlayModel, an NFL player-prop \
dashboard for Underdog Fantasy pick'em. The user is a bettor who knows the basics.

How you work:
- Call tools for every fact. Every number you state (lines, probabilities, edges, stats, \
EV) must come from a tool result in this conversation. Never invent stats, injuries or news.
- "ours"/"model" probabilities are the project's; by default the dashboard uses a blend \
(35% model, 65% de-vigged market), which scored best against real Underdog lines. Say \
which one you're quoting. The model alone has only one graded week against real lines \
and was worse than the market on receiving and rushing yards -- be honest about that \
when it matters, and don't oversell edges.
- Entry EV multiplies leg edges; treat big entry EVs as a ranking, not a forecast.
- You cannot place bets. To record a bet the user says they placed, use stage_bet_log; \
it only stages the bet and the user confirms it on the page.
- Be concise: short paragraphs or a compact list. Use markdown tables only for 3+ rows."""


def _api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _records(df: pd.DataFrame, limit: int = MAX_ROWS) -> str:
    if df is None or df.empty:
        return json.dumps({"rows": [], "note": "no matching rows"})
    out = df.head(limit).copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(3)
    return json.dumps({"rows": json.loads(out.to_json(orient="records")),
                       "total_matching": int(len(df)), "shown": int(min(len(df), limit))})


def _match_player(names: pd.Series, query: str) -> pd.Series:
    """Exact normalized match, else substring match on the normalized name."""
    q = normalize_name(query)
    norm = names.fillna("").map(normalize_name)
    exact = norm == q
    return exact if exact.any() else norm.str.contains(q, regex=False)


@st.cache_data(ttl=60, show_spinner=False)
def _candidates(prob_source: str) -> pd.DataFrame:
    from ev_finder import build_candidates, _apply_prob_source
    cands, _ = build_candidates()
    return _apply_prob_source(cands, prob_source) if not cands.empty else cands


def _cand_view(c: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player": c["player"], "team": c["team"], "stat": c["stat_name"], "line": c["line"],
        "side": c["choice"].astype(str).str.upper(), "prob": c["prob"], "model_prob": c["model_prob"],
        "market_fair_prob": c["market_prob"], "edge": c["edge"], "decimal_price": c["decimal"],
        "ev_per_dollar": c["ev"],
    })


def build_tools(prob_source: str) -> list:
    """Tools are rebuilt per request so they close over the page's probability source."""

    @beta_tool
    def get_best_picks(min_edge: float = 0.02, stat: str = "", team: str = "", limit: int = 15) -> str:
        """Best single Underdog picks right now, ranked by edge over the de-vigged market.
        Uses live Underdog lines (refreshed every 60 s) and only props where the model's
        baseline is comparable to the real line.

        Args:
            min_edge: Minimum probability edge over the market's fair probability (0.02 = 2 points).
            stat: Optional Underdog stat filter, e.g. receiving_yds, rushing_yds, passing_yds, receiving_rec, rush_rec_tds, passing_tds.
            team: Optional team abbreviation filter, e.g. KC, PHI.
            limit: Max rows to return (<= 25).
        """
        c = _candidates(prob_source)
        if c.empty:
            return json.dumps({"rows": [], "note": "no comparable model-backed props right now"})
        c = c[c["edge"] >= min_edge]
        if stat:
            c = c[c["stat_name"] == stat]
        if team:
            c = c[c["team"].astype(str).str.upper() == team.upper()]
        return _records(_cand_view(c.sort_values("edge", ascending=False)), min(limit, MAX_ROWS))

    @beta_tool
    def find_best_entries(min_legs: int = 2, max_legs: int = 3, min_leg_edge: float = 0.02, limit: int = 5) -> str:
        """Search for the highest-EV valid Underdog pick'em entries (one pick per player, at
        least 2 teams), using correlation-adjusted joint probability and the product of the
        picks' decimal prices as payout.

        Args:
            min_legs: Smallest entry size (>= 2).
            max_legs: Largest entry size (<= 5; 4-5 is slower).
            min_leg_edge: Each leg's minimum edge over the market.
            limit: Number of entries to return (<= 10).
        """
        from parlay_calculator import find_best_entries as _search
        c = _candidates(prob_source)
        if c.empty:
            return json.dumps({"entries": [], "note": "no candidates"})
        best = _search(c, n_legs_range=(max(2, min_legs), min(5, max(min_legs, max_legs))), top_k=limit,
                       pool_size=16, min_leg_edge=min_leg_edge, correlations=load_leg_correlations())
        if best.empty:
            return json.dumps({"entries": [], "note": "no valid entries at this edge"})
        best = best[best["valid"]].head(min(limit, 10))
        return json.dumps({"entries": [{
            "legs": [{"player": l["player"], "team": l["team"], "stat": l["stat_name"], "line": l["line"],
                      "side": str(l["choice"]).upper(), "prob": round(l["prob"], 3),
                      "market_fair_prob": round(l["market_prob"], 3), "price": l["decimal"]} for l in r.legs],
            "joint_prob": round(r.joint_prob, 4), "naive_joint_prob": round(r.naive_prob, 4),
            "payout_multiple": round(r.payout, 2), "ev_per_dollar": round(r.ev, 3),
        } for r in best.itertuples()]})

    @beta_tool
    def player_props(player: str) -> str:
        """Every current Underdog line for a player (all stats, both sides), with our
        probability and the market's fair probability where the model covers it.

        Args:
            player: Player name, e.g. "Ja'Marr Chase".
        """
        c = _candidates(prob_source)
        props = load_csv_if_exists("underdog_props.csv")
        if props is None:
            return json.dumps({"rows": [], "note": "no Underdog board loaded"})
        props = props[_match_player(props["full_name"], player)]
        if props.empty:
            return json.dumps({"rows": [], "note": f"no Underdog lines for '{player}' right now"})
        view = props[["full_name", "stat_name", "stat_value", "choice", "decimal_price"]].rename(
            columns={"full_name": "player", "stat_name": "stat", "stat_value": "line", "choice": "side",
                     "decimal_price": "price"})
        if not c.empty:
            cv = c[["player", "stat_name", "line", "choice", "prob", "market_prob", "edge"]].rename(
                columns={"stat_name": "stat", "choice": "side", "market_prob": "market_fair_prob"})
            view = view.merge(cv, on=["player", "stat", "line", "side"], how="left")
        view["side"] = view["side"].astype(str).str.upper()
        return _records(view.drop_duplicates(), 40)

    @beta_tool
    def model_prediction(player: str) -> str:
        """The model's raw predictions for a player: each prop type's baseline (the player's
        trailing average, called the proxy line), probability of beating it, which game the
        stats run through, the next opponent, and the model's top features for that prop.

        Args:
            player: Player name.
        """
        preds = load_current_predictions()
        if preds is None:
            return json.dumps({"rows": [], "note": "no predictions file"})
        p = preds[_match_player(preds["player_display_name"], player)]
        cols = ["player_display_name", "position", "recent_team", "prop_type", "proxy_line",
                "predicted_prob_over", "stats_as_of_season", "stats_as_of_week", "next_week", "next_opponent"]
        out = p[[c for c in cols if c in p.columns]].copy()
        fi_path = os.path.join(ROOT_DIR, "models", "feature_importance.csv")
        if os.path.exists(fi_path) and not out.empty:
            fi = pd.read_csv(fi_path).sort_values("importance", ascending=False)
            top = fi.groupby("prop_type").head(3).groupby("prop_type")["feature"].apply(list)
            out["top_features"] = out["prop_type"].map(top)
        return _records(out, 30)

    @beta_tool
    def player_game_log(player: str, games: int = 8) -> str:
        """A player's most recent games from weekly box scores (passing, rushing, receiving).

        Args:
            player: Player name.
            games: How many recent games (<= 17).
        """
        w = load_csv_if_exists("weekly_stats.csv")
        if w is None:
            return json.dumps({"rows": [], "note": "no weekly stats"})
        g = w[_match_player(w["player_display_name"], player)].sort_values(["season", "week"]).tail(min(games, 17))
        cols = ["player_display_name", "recent_team", "season", "week", "opponent_team", "completions", "attempts",
                "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards", "rushing_tds",
                "targets", "receptions", "receiving_yards", "receiving_tds"]
        g = g[[c for c in cols if c in g.columns]]
        g = g.loc[:, (g.notna() & (g != 0)).any(axis=0) | g.columns.isin(cols[:5])]
        return _records(g, 17)

    @beta_tool
    def depth_chart(team: str) -> str:
        """A team's current depth chart with injury/status tags (Q, O, IR, PUP, ...).

        Args:
            team: Team abbreviation, e.g. BAL, LA (Rams), LAC.
        """
        d = load_csv_if_exists("footballguys_depth.csv")
        if d is None:
            return json.dumps({"rows": [], "note": "no depth chart pulled"})
        t = {"LA": "LAR"}.get(team.upper(), team.upper())
        d = d[d["team_abbr"].astype(str).str.upper() == t]
        d = d[d["position"].isin(["QB", "RB", "FB", "WR", "TE", "K"])]  # prop-relevant spots only
        return _records(d[["position", "depth_rank", "player_name", "status"]].sort_values(["position", "depth_rank"]), 40)

    @beta_tool
    def player_news(player: str) -> str:
        """Latest news headlines for a player (Google News search, live).

        Args:
            player: Player name.
        """
        try:
            items = get_player_news(player, max_items=6)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"items": [], "note": f"news lookup failed: {e}"})
        return json.dumps({"items": [{k: it.get(k) for k in ("headline", "source", "published")} for it in items]})

    @beta_tool
    def model_track_record() -> str:
        """How the model has scored against REAL Underdog closing lines (log loss / Brier for
        model, market and blend, by stat) and flat-bet ROI by edge threshold."""
        path = os.path.join(ROOT_DIR, "backtesting", "underdog_line_backtest_summary.json")
        if not os.path.exists(path):
            return json.dumps({"note": "no backtest summary yet"})
        with open(path) as f:
            bt = json.load(f)
        bt.pop("blend_curve", None)
        return json.dumps(bt)

    @beta_tool
    def my_bet_log(limit: int = 20) -> str:
        """The user's logged bets (most recent first).

        Args:
            limit: Max rows.
        """
        log = load_bet_log()
        return _records(log.iloc[::-1], min(limit, MAX_ROWS))

    @beta_tool
    def stage_bet_log(player: str, stat: str, side: str, line: float, stake: float,
                      price_or_multiplier: str = "", notes: str = "") -> str:
        """Stage a bet the user says they PLACED, for logging in their bet log. Does not write
        anything: the page shows a Confirm button and the bet is saved only if the user clicks it.
        Call once per leg.

        Args:
            player: Player name.
            stat: Stat, e.g. receiving_yds.
            side: OVER or UNDER.
            line: The line, e.g. 64.5.
            stake: Dollars staked on the entry (repeat on each leg of the same entry).
            price_or_multiplier: Price or payout multiple, e.g. 1.92 or "3x".
            notes: Optional note, e.g. "2-pick with Kelce".
        """
        row = {"date": datetime.now().date().isoformat(), "sport": "NFL", "player": player, "stat": stat,
               "choice": side.lower(), "line": line, "multiplier_or_odds": price_or_multiplier,
               "stake": stake, "result": "", "notes": notes,
               "logged_at": datetime.now(timezone.utc).isoformat()}
        st.session_state.setdefault("assistant_pending_bets", []).append(row)
        return json.dumps({"status": "staged -- waiting for the user to click Confirm on the page", "bet": row})

    return [get_best_picks, find_best_entries, player_props, model_prediction, player_game_log,
            depth_chart, player_news, model_track_record, my_bet_log, stage_bet_log]


def _run_turn(client, history: list, prob_source: str, status) -> tuple[list, str | None]:
    """Runs one user turn through the tool runner. Returns (new messages to append to
    history -- assistant turns + tool results, mirrored because the runner keeps its own
    copy -- and the final stop_reason)."""
    new, stop_reason = [], None
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=build_tools(prob_source),
        messages=history,
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},
        # Server-side refusal fallback: if this model declines, the API re-runs the
        # request on a fallback model chosen by refusal category.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        max_iterations=12,
    )
    for message in runner:
        stop_reason = message.stop_reason
        new.append({"role": "assistant", "content": message.content})
        for b in message.content:
            if b.type == "tool_use":
                status.write(f"🔧 `{b.name}` {json.dumps(b.input)[:120]}")
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            new.append(tool_response)
    return new, stop_reason


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text").strip()


def _render_pending_bets():
    pending = st.session_state.get("assistant_pending_bets") or []
    if not pending:
        return
    with st.container(border=True):
        st.markdown(f"**Log {len(pending)} bet leg(s)?** Nothing is saved until you confirm.")
        st.dataframe(pd.DataFrame(pending)[["date", "player", "stat", "choice", "line", "multiplier_or_odds",
                                            "stake", "notes"]], hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        if c1.button("✅ Confirm and log", type="primary"):
            for row in pending:
                append_bet(row)
            st.session_state["assistant_pending_bets"] = []
            st.success(f"Logged {len(pending)} leg(s) to the Bet Log.")
        if c2.button("✖ Discard"):
            st.session_state["assistant_pending_bets"] = []
            st.rerun()


def page_assistant():
    st.title("💬 AI Assistant")
    st.caption("Ask about tonight's board — it looks things up with the app's own data "
               "(live Underdog lines, model predictions, +EV pricing, game logs, depth charts, news) "
               "and never places bets.")

    key = _api_key()
    if not key:
        st.info("Add an `ANTHROPIC_API_KEY` to use the assistant — locally in `.env`, or on Streamlit "
                "Cloud under **Manage app → Settings → Secrets** as `ANTHROPIC_API_KEY = \"sk-ant-...\"`. "
                "Get a key at console.anthropic.com.")
        return

    with st.sidebar:
        st.subheader("Assistant")
        prob_source = st.radio("Probability", ["Blend", "Model"], horizontal=True, key="assistant_prob",
                               help="Which probability the tools rank with (same as the +EV Finder).")
        if st.button("🗑 New conversation"):
            st.session_state["assistant_history"] = []
            st.session_state["assistant_pending_bets"] = []
            st.rerun()
        st.caption("Each message is a Claude Opus 5.5 call with a few tool lookups — "
                   "typically a few cents to ~$0.20.")

    history = st.session_state.setdefault("assistant_history", [])

    if not history:
        st.markdown("**Try:** “Best 3-pick entry for the 1pm games?” · “Why is the model on the under "
                    "for Ja'Marr Chase?” · “Compare Puka Nacua and Davante Adams receiving yards” · "
                    "“How has the model done against real lines?” · “Log my bet: Chase over 6.5 "
                    "receptions, $10 2-pick with Kelce”")

    for m in history:
        if m["role"] == "user" and isinstance(m["content"], str):
            with st.chat_message("user"):
                st.markdown(m["content"])
        elif m["role"] == "assistant":
            text = _text_of(m["content"])
            if text:
                with st.chat_message("assistant"):
                    st.markdown(text)

    _render_pending_bets()

    prompt = st.chat_input("Ask about picks, players, entries…")
    if not prompt:
        return
    with st.chat_message("user"):
        st.markdown(prompt)
    history.append({"role": "user", "content": prompt})

    import anthropic
    client = anthropic.Anthropic(api_key=key)
    with st.chat_message("assistant"):
        status = st.status("Looking things up…", expanded=False)
        try:
            new, stop_reason = _run_turn(client, list(history), prob_source, status)
        except anthropic.AuthenticationError:
            status.update(label="Invalid ANTHROPIC_API_KEY", state="error")
            history.pop()
            return
        except anthropic.RateLimitError:
            status.update(label="Rate limited — try again in a moment", state="error")
            history.pop()
            return
        except anthropic.APIStatusError as e:
            status.update(label=f"Claude API error {e.status_code}", state="error")
            history.pop()
            return
        except anthropic.APIConnectionError:
            status.update(label="Couldn't reach the Claude API", state="error")
            history.pop()
            return
        status.update(label="Done", state="complete")
        text = _text_of(new[-1]["content"]) if new else ""
        if stop_reason == "refusal":
            text = text or "_Claude declined to answer that one — try rephrasing._"
        elif stop_reason == "tool_use":
            note = "_Stopped after too many lookups — ask a narrower question._"
            text = f"{text}\n\n{note}" if text else note
        st.markdown(text or "_(no answer — try rephrasing)_")
    history.extend(new)
    if st.session_state.get("assistant_pending_bets"):
        st.rerun()
