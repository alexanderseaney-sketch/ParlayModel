"""
AI Assistant page (added 2026-10-04, from IDEAS.md): a chat with Claude that answers
questions about tonight's board by calling the project's own functions as tools --
live Underdog lines, model predictions, the +EV Finder's pricing, the entry search and
entry checker, game logs, usage trends, injuries, line movement, depth charts, news,
this week's games, defense-vs-position ranks, fantasy projections, the real-line
backtest, data freshness, and the user's bet log (live tracking + bankroll) -- instead
of guessing.

Rules it runs under (same as models/interpreter.py): every number it states must come
from a tool result; it never places bets. The two write actions -- logging a bet and
running a data pull -- are only STAGED by the tools; nothing is written or run until
you click Confirm on the page.

Needs ANTHROPIC_API_KEY (env / .env locally, Streamlit secrets when hosted).
"""
import json
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import ui
from anthropic import beta_tool

from utils import (
    load_csv_if_exists, load_current_predictions, normalize_name,
    load_bet_log, append_bets, get_player_news, ROOT_DIR, load_leg_correlations,
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
- For every entry you suggest, state its chance of winning in plain terms from the tool's joint_prob ("hits about 1 in 11"). When the user asks for several 4+ pick entries, say plainly that most of them will lose even when the picks are good -- e.g. ten 4-pick entries at ~8% each all lose about 40% of the time -- and offer 2-3 pick entries as the lower-variance option. Don't stack the same side by default: mix overs and unders when the edges support it.
- For a time slot ("the 1pm games", "Sunday night"), get the teams from this_weeks_games \
and pass them to get_best_picks / find_best_entries. When the user proposes their own picks \
or entry, price it with check_entry rather than estimating. Before recommending a player, \
check injury_report (and line_movement if their line moved or was pulled); usage_trends and \
defense_vs_position explain role and matchup.
- You cannot place bets. To record a bet the user says they placed, use stage_bet_log; \
it only stages the bet and the user confirms it on the page. Data pulls work the same way \
(stage_data_pull). Never say a bet was logged or a pull ran -- say it's waiting for their Confirm.
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


def _side_of(choice: pd.Series) -> pd.Series:
    """Underdog's higher/lower (or over/under) -> OVER/UNDER."""
    c = choice.astype(str).str.lower()
    return pd.Series(["OVER" if v in ("over", "higher") else "UNDER" if v in ("under", "lower") else v.upper()
                      for v in c], index=choice.index)


def _side_str(choice) -> str:
    return _side_of(pd.Series([choice])).iloc[0]


def _filter_teams(c: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    if not teams or c.empty:
        return c
    want = {t.upper() for t in teams}
    return c[c["team"].astype(str).str.upper().isin(want)]


def _cand_view(c: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "player": c["player"], "team": c["team"], "stat": c["stat_name"], "line": c["line"],
        "side": _side_of(c["choice"]), "prob": c["prob"], "model_prob": c["model_prob"],
        "market_fair_prob": c["market_prob"], "edge": c["edge"], "decimal_price": c["decimal"],
        "ev_per_dollar": c["ev"],
    })


def build_tools(prob_source: str) -> list:
    """Tools are rebuilt per request so they close over the page's probability source."""

    @beta_tool
    def get_best_picks(min_edge: float = 0.02, stat: str = "", teams: list[str] = [], side: str = "",
                       limit: int = 15) -> str:
        """Best single Underdog picks right now, ranked by edge over the de-vigged market.
        Uses live Underdog lines (refreshed every 60 s) and only props where the model's
        baseline is comparable to the real line. To restrict to a time slot (e.g. "the 1pm
        games"), get the teams from this_weeks_games first and pass them in `teams`.

        Args:
            min_edge: Minimum probability edge over the market's fair probability (0.02 = 2 points).
            stat: Optional Underdog stat filter, e.g. receiving_yds, rushing_yds, passing_yds, receiving_rec, rush_rec_tds, passing_tds.
            teams: Optional team abbreviations to keep, e.g. ["KC", "PHI"].
            side: Optional "OVER" or "UNDER" to keep one side only.
            limit: Max rows to return (<= 25).
        """
        c = _candidates(prob_source)
        if c.empty:
            return json.dumps({"rows": [], "note": "no comparable model-backed props right now"})
        c = _filter_teams(c[c["edge"] >= min_edge], teams)
        if stat:
            c = c[c["stat_name"] == stat]
        if side:
            c = c[_side_of(c["choice"]) == side.upper()]
        return _records(_cand_view(c.sort_values("edge", ascending=False)), min(limit, MAX_ROWS))

    @beta_tool
    def find_best_entries(min_legs: int = 2, max_legs: int = 3, min_leg_edge: float = 0.02, limit: int = 5,
                          teams: list[str] = [], exclude_players: list[str] = []) -> str:
        """Search for the highest-EV valid Underdog pick'em entries (one pick per player, at
        least 2 teams), using correlation-adjusted joint probability and the product of the
        picks' decimal prices as payout.

        Args:
            min_legs: Smallest entry size (>= 2).
            max_legs: Largest entry size (<= 5; 4-5 is slower).
            min_leg_edge: Each leg's minimum edge over the market.
            limit: Number of entries to return (<= 10).
            teams: Optional team abbreviations to build from, e.g. the teams in the 1pm games.
            exclude_players: Players to leave out (e.g. ones the user already has or dislikes).
        """
        from parlay_calculator import find_best_entries as _search
        c = _filter_teams(_candidates(prob_source), teams)
        for name in exclude_players:
            c = c[~_match_player(c["player"], name)]
        if c.empty:
            return json.dumps({"entries": [], "note": "no candidates"})
        best = _search(c, n_legs_range=(max(2, min_legs), min(5, max(min_legs, max_legs))), top_k=limit,
                       pool_size=16, min_leg_edge=min_leg_edge, correlations=load_leg_correlations())
        if best.empty:
            return json.dumps({"entries": [], "note": "no valid entries at this edge"})
        best = best[best["valid"]].head(min(limit, 10))
        return json.dumps({"entries": [{
            "legs": [{"player": l["player"], "team": l["team"], "stat": l["stat_name"], "line": l["line"],
                      "side": _side_str(l["choice"]), "prob": round(l["prob"], 3),
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
        view["side"] = _side_of(view["side"])
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
                      entry: str = "A", payout_multiple: str = "", notes: str = "") -> str:
        """Stage a bet the user says they PLACED, for logging in their bet log. Does not write
        anything: the page shows a Confirm button and the bet is saved only if the user clicks it.
        Call once per pick; picks of the same entry share the same `entry` label.

        Args:
            player: Player name.
            stat: Stat, e.g. receiving_yds.
            side: OVER or UNDER.
            line: The line, e.g. 64.5.
            stake: Dollars staked on the whole entry (same value on every pick of the entry).
            entry: Label grouping picks into one entry, e.g. "A" for all picks of the first
                entry and "B" for a separate entry in the same message.
            payout_multiple: The entry's payout multiple if the user gave it, e.g. "6" or "3.2x".
            notes: Optional note, e.g. "2-pick with Kelce".
        """
        row = {"date": datetime.now().date().isoformat(), "sport": "NFL", "player": player, "stat": stat,
               "choice": side.lower(), "line": line, "multiplier_or_odds": "",
               "stake": stake, "result": "pending", "notes": notes,
               "logged_at": datetime.now(timezone.utc).isoformat(),
               "_entry_label": entry or "A", "entry_payout": payout_multiple}
        st.session_state.setdefault("assistant_pending_bets", []).append(row)
        return json.dumps({"status": "staged -- waiting for the user to click Confirm on the page", "bet": row})

    @beta_tool
    def check_entry(picks: list[dict], payout_mode: str = "product") -> str:
        """Evaluate an entry the USER proposes: each pick's probability and market price, the
        correlation-adjusted chance all picks hit, the payout, EV per $1, whether it is a valid
        Underdog entry (2-8 picks, one per player, at least 2 teams) and the riskiest pick.

        Args:
            picks: The picks, e.g. [{"player": "Ja'Marr Chase", "stat": "receiving_yds", "side": "OVER", "line": 74.5}]. "line" is optional (the live line is used if omitted).
            payout_mode: "product" (Underdog's odds-based payout: product of the picks' prices) or "standard" (legacy fixed multipliers).
        """
        from parlay_calculator import evaluate_entry, riskiest_leg
        c = _candidates(prob_source)
        legs, missing = [], []
        for pk in picks:
            name, stat, side = str(pk.get("player", "")), str(pk.get("stat", "")), str(pk.get("side", "")).upper()
            m = c
            if not m.empty:
                m = m[_match_player(m["player"], name) & (m["stat_name"] == stat) & (_side_of(m["choice"]) == side)]
            if pk.get("line") not in (None, "") and not m.empty:
                m = m[pd.to_numeric(m["line"], errors="coerce") == float(pk["line"])]
            if m.empty:
                missing.append(f"{name} {side} {pk.get('line', '')} {stat}".replace("  ", " "))
            else:
                legs.append(m.iloc[0].to_dict())
        if missing:
            return json.dumps({
                "not_found": missing,
                "found": [f"{l['player']} {_side_str(l['choice'])} {l['line']} {l['stat_name']}" for l in legs],
                "note": "These picks aren't on the live board with a comparable model price (wrong stat name or "
                        "line, not offered, or not model-backed). Check player_props for the exact stat and line."})
        res = evaluate_entry(legs, payout_mode, load_leg_correlations())
        risk = riskiest_leg(legs)
        return json.dumps({
            "picks": [{"player": l["player"], "team": l["team"], "stat": l["stat_name"], "line": l["line"],
                       "side": _side_str(l["choice"]), "prob": round(l["prob"], 3),
                       "market_fair_prob": round(l["market_prob"], 3), "price": l["decimal"]} for l in legs],
            "valid": res["valid"], "validity_note": res["validity_note"],
            "joint_prob": round(res["joint_prob"], 4), "naive_joint_prob": round(res["naive_prob"], 4),
            "market_joint_prob": None if res["market_joint_prob"] is None else round(res["market_joint_prob"], 4),
            "payout_multiple": None if res["payout"] is None else round(res["payout"], 2),
            "ev_per_dollar": None if res["ev"] is None else round(res["ev"], 3),
            "correlation_adjustments": [list(a) for a in res["adjustments"]],
            "riskiest_pick": risk["player"] if risk else None,
        })

    @beta_tool
    def this_weeks_games(teams: list[str] = []) -> str:
        """This week's games: kickoff (ET), matchup, spread, total, each team's implied points
        and the weather forecast. Use it for schedule questions and to turn a time slot
        ("1pm games", "Sunday night") into team abbreviations for the pick and entry tools.

        Args:
            teams: Optional team abbreviations to keep.
        """
        from research_data import week_games
        g = week_games()
        if g.empty:
            return json.dumps({"rows": [], "note": "no upcoming games in schedules.csv"})
        if teams:
            want = {t.upper() for t in teams}
            g = g[g["home_team"].isin(want) | g["away_team"].isin(want)]
            if g.empty:
                return json.dumps({"rows": [], "note": f"{', '.join(sorted(want))}: no game this week (bye)"})
        view = g[["kick_label", "gameday", "gametime", "away_team", "home_team", "spread_txt", "total_line",
                  "away_implied", "home_implied", "weather_txt"]].rename(
            columns={"kick_label": "kickoff", "spread_txt": "spread", "weather_txt": "weather"})
        return _records(view, 20)

    @beta_tool
    def injury_report(team: str = "", player: str = "", changes_only: bool = False, limit: int = 25) -> str:
        """This week's injury report for QB/RB/WR/TE: last week's and this week's game status
        (Q questionable, D doubtful, O out, IR; "TBD" = the team hasn't posted game statuses
        yet, which happens the day before its game; "—" = no designation), latest practice
        participation (DNP/LP/FP), and how the player's main Underdog line moved this week
        (line_pulled = the line came off the board). Biggest line moves and most serious
        statuses first.

        Args:
            team: Optional team abbreviation.
            player: Optional player name.
            changes_only: Only players whose status got worse or better since last week.
            limit: Max rows (<= 25).
        """
        from research_data import injury_report as _report
        r = _report()
        if r.empty:
            return json.dumps({"rows": [], "note": "no injury report for this week yet"})
        if team:
            r = r[r["team"].astype(str).str.upper() == team.upper()]
        if player:
            r = r[_match_player(r["full_name"], player)]
        if changes_only:
            r = r[r["dir"] != "same"]
        view = r.drop(columns=["abs_move", "sev_now"]).rename(columns={
            "full_name": "player", "prev": "status_last_week", "now": "status_now", "dir": "direction",
            "practice": "latest_practice", "stat": "main_stat"})
        return _records(view, min(limit, MAX_ROWS))

    @beta_tool
    def usage_trends(player: str = "", team: str = "", position: str = "", fallers_first: bool = False,
                     limit: int = 15) -> str:
        """Role trends from each player's last 8 games: offensive snap share and target share
        per game (oldest -> newest, in %), the latest values, and the snap-share change over
        the last 3 games. Default order puts the biggest risers first.

        Args:
            player: Optional player name.
            team: Optional team abbreviation.
            position: Optional QB, RB, WR or TE.
            fallers_first: Put the biggest snap-share drops first instead.
            limit: Max rows (<= 25).
        """
        from research_data import _mtime
        from usage import _usage
        u = _usage(_mtime("snap_counts.csv"), _mtime("weekly_stats.csv"))
        if u.empty:
            return json.dumps({"rows": [], "note": "no snap counts pulled"})
        if player:
            u = u[_match_player(u["Player"], player)]
        if team:
            u = u[u["Team"].astype(str).str.upper() == team.upper()]
        if position:
            u = u[u["Pos"] == position.upper()]
        if fallers_first:
            u = u.sort_values("3-game snap change", na_position="last")
        view = u.drop(columns=["Trend"]).rename(columns={
            "Player": "player", "Team": "team", "Pos": "position", "Snap share · 8 games": "snap_pct_last8",
            "Snap now": "snap_share_now", "Target share · 8 games": "target_pct_last8",
            "Target now": "target_share_now", "3-game snap change": "snap_change_3g_pts"})
        return _records(view, min(limit, MAX_ROWS))

    @beta_tool
    def defense_vs_position(defense: str = "", position: str = "", scoring: str = "half",
                            softest_first: bool = True, limit: int = 12) -> str:
        """How defenses rank against a position this season by fantasy points allowed per
        game: rank 1 = toughest, 32 = softest (best to target), plus who each plays this week.

        Args:
            defense: Optional defense team abbreviation.
            position: Optional QB, RB, WR or TE.
            scoring: "half" (Half-PPR), "ppr" or "std".
            softest_first: Softest defenses first (else toughest first).
            limit: Max rows (<= 32).
        """
        m = load_csv_if_exists("fantasy_matchups.csv")
        if m is None or m.empty:
            return json.dumps({"rows": [], "note": "no matchup table built yet"})
        sc = scoring.lower() if scoring.lower() in ("half", "ppr", "std") else "half"
        if defense:
            m = m[m["def_team"].astype(str).str.upper() == defense.upper()]
        if position:
            m = m[m["position"] == position.upper()]
        from research_data import week_games
        opp = {}
        for r in week_games().itertuples():
            opp[r.home_team], opp[r.away_team] = f"vs {r.away_team}", f"@ {r.home_team}"
        view = pd.DataFrame({"defense": m["def_team"], "position": m["position"], "rank": m[f"rank_{sc}"],
                             "fantasy_pts_allowed_pg": m[f"fp_{sc}_pg"], "games": m["team_games"],
                             "this_week": m["def_team"].map(opp).fillna("bye")})
        return _records(view.sort_values("rank", ascending=not softest_first), min(limit, 32))

    @beta_tool
    def line_movement(player: str, stat: str = "") -> str:
        """How a player's Underdog lines have moved: each recorded open / move / pulled event
        (UTC) from the line-history archive, plus the live line and prices now. A pulled line
        often means injury news or a role change.

        Args:
            player: Player name.
            stat: Optional Underdog stat, e.g. receiving_yds.
        """
        from research_data import line_history, live_over_lines
        h = line_history()
        h = h[_match_player(h["full_name"], player)]
        live = live_over_lines()
        live = live[_match_player(live["full_name"], player)]
        if stat:
            h, live = h[h["stat_name"] == stat], live[live["stat_name"] == stat]
        h = h.sort_values("seen_at").tail(40).assign(seen_at=lambda d: d["seen_at"].astype(str))
        return json.dumps({
            "history": json.loads(h[["stat_name", "line", "event", "seen_at"]].to_json(orient="records")),
            "live_now": json.loads(live[["stat_name", "line", "over_price", "under_price"]].to_json(orient="records")),
        })

    @beta_tool
    def fantasy_projections(player: str = "", position: str = "", team: str = "", scoring: str = "half",
                            limit: int = 15) -> str:
        """This week's fantasy projections (the Fantasy page's numbers): projected points,
        position rank, tier, opponent and the projected stat line, plus the prop model's lean
        (0-100, 50 = neutral). Use for start/sit and other fantasy questions.

        Args:
            player: Optional player name.
            position: Optional QB, RB, WR or TE.
            team: Optional team abbreviation.
            scoring: "half" (Half-PPR) or "ppr".
            limit: Max rows (<= 25).
        """
        from fantasy import _projections, _pred_mtime
        p = _projections(_pred_mtime(), "ppr" if scoring.lower() == "ppr" else "half")
        if p.empty:
            return json.dumps({"rows": [], "note": "no current predictions"})
        if player:
            p = p[_match_player(p["player"], player)]
        if position:
            p = p[p["position"] == position.upper()]
        if team:
            p = p[p["team"].astype(str).str.upper() == team.upper()]
        cols = ["player", "position", "team", "opp", "week", "proj", "pos_rank", "tier", "overall_rank", "lean",
                "pass_yds", "passing_tds", "rush_yds", "rec", "rec_yds", "td"]
        view = p[[c for c in cols if c in p.columns]].rename(columns={"proj": "projected_points", "opp": "opponent"})
        view["lean"] = (pd.to_numeric(view["lean"], errors="coerce") * 100).round()
        return _records(view, min(limit, MAX_ROWS))

    @beta_tool
    def track_open_entries() -> str:
        """Live status of the user's OPEN logged entries from ESPN box scores: each pick's
        current stat vs its line, pace, hit / miss / alive / pending, the game score and
        clock, and whether each entry is still alive. Use for "how's my parlay doing?"."""
        from bet_entries import _with_entries, _entry_payout, _RESULT_TO_STATUS
        from live_tracker import track_leg, entry_status
        log = load_bet_log()
        if log.empty:
            return json.dumps({"entries": [], "note": "the bet log is empty"})
        out = []
        for _, legs in _with_entries(log).groupby("entry_id", sort=False):
            if not (legs["result"] == "pending").any():
                continue
            rows = []
            for _, leg in legs.iterrows():
                base = {"player": leg["player"], "stat": leg["stat"], "side": _side_str(leg["choice"]),
                        "line": leg["line"]}
                if leg["result"] == "pending":
                    t = track_leg(leg["player"], leg["stat"], leg["choice"], leg["line"])
                    rows.append({**base, "value": t["value"], "pace": t["pace"], "status": t["status"],
                                 "note": t["text"], "game": t["game"]})
                else:
                    rows.append({**base, "status": _RESULT_TO_STATUS.get(leg["result"], "pending"),
                                 "note": "already graded in the log"})
            stake = pd.to_numeric(legs["stake"], errors="coerce").max()
            out.append({"date": str(legs["date"].iloc[0]), "notes": str(legs["notes"].iloc[0] or ""),
                        "stake": None if pd.isna(stake) else float(stake), "payout_multiple": _entry_payout(legs),
                        "entry_status": entry_status([r["status"] for r in rows]), "picks": rows})
        if not out:
            return json.dumps({"entries": [], "note": "no open entries"})
        return json.dumps({"open_entries": len(out), "entries": out[:10]}, default=str)

    @beta_tool
    def bankroll_summary() -> str:
        """The user's results from SETTLED entries in the bet log: bankroll now (starting
        bankroll + P/L), total P/L, amount staked, ROI, entries won / lost, the latest week's
        P/L and results by entry size. Won entries count only once a payout multiple is logged."""
        from bankroll import settled_entries, starting_bankroll
        s, start = settled_entries(), starting_bankroll()
        if s.empty:
            return json.dumps({"starting_bankroll": start, "note": "no settled entries yet"})
        weekly = s.groupby("week")["pnl"].sum()
        size = (s.assign(picks=s["legs"].clip(upper=4).map(lambda n: "4+" if n >= 4 else str(n)))
                .groupby("picks").agg(entries=("entry_id", "count"), won=("state", lambda x: int((x == "won").sum())),
                                      staked=("stake", "sum"), pnl=("pnl", "sum")).reset_index())
        staked = s["stake"].sum()
        return json.dumps({
            "starting_bankroll": start, "bankroll_now": round(start + s["pnl"].sum(), 2),
            "total_pnl": round(s["pnl"].sum(), 2), "total_staked": round(staked, 2),
            "roi": round(s["pnl"].sum() / staked, 3) if staked else None,
            "entries_won": int((s["state"] == "won").sum()), "entries_lost": int((s["state"] == "lost").sum()),
            "latest_week": weekly.index[-1], "latest_week_pnl": round(float(weekly.iloc[-1]), 2),
            "by_entry_size": json.loads(size.round(2).to_json(orient="records")),
        })

    @beta_tool
    def data_freshness() -> str:
        """When the app's data was last updated: the live Underdog board's pull time, the
        predictions file, and each core data file's age (stats, injuries, rosters). Check this
        before leaning on injuries or stats close to kickoff."""
        from utils import data_freshness_check, file_status
        f = data_freshness_check()
        props = load_csv_if_exists("underdog_props.csv")
        pulled = props["pulled_at"].iloc[0] if props is not None and len(props) and "pulled_at" in props.columns else None
        preds = os.path.join(ROOT_DIR, "models", "current_player_predictions.csv")
        return json.dumps({
            "underdog_board_pulled_at": pulled,
            "predictions_updated": datetime.fromtimestamp(os.path.getmtime(preds)).isoformat(timespec="minutes")
            if os.path.exists(preds) else None,
            "files_updated": {fn: file_status(fn)["modified"].isoformat(timespec="minutes") for fn in f["ok"]},
            "files_stale_hours": {fn: round(h, 1) for fn, h in f["stale"]}, "files_missing": f["missing"],
            "note": "Data refreshes automatically every 6 hours; Underdog lines are fetched live.",
        }, default=str)

    @beta_tool
    def stage_data_pull(pull: str) -> str:
        """Stage one of the app's data pulls. Runs nothing: the page shows a Confirm button and
        the pull runs only if the user clicks it (it can take a minute). Only use when the user
        asks to refresh data -- Underdog lines are already live.

        Args:
            pull: The pull's exact name; pass "list" to get the available names.
        """
        from utils import PULL_SCRIPTS
        if pull not in PULL_SCRIPTS:
            return json.dumps({"available_pulls": list(PULL_SCRIPTS),
                               "note": "" if pull == "list" else f"unknown pull '{pull}' -- use one of these names"})
        staged = st.session_state.setdefault("assistant_pending_pulls", [])
        if pull not in staged:
            staged.append(pull)
        return json.dumps({"status": "staged -- waiting for the user to click Confirm on the page", "pull": pull})

    return [get_best_picks, find_best_entries, check_entry, player_props, model_prediction, player_game_log,
            usage_trends, injury_report, line_movement, depth_chart, player_news, this_weeks_games,
            defense_vs_position, fantasy_projections, model_track_record, data_freshness, my_bet_log,
            track_open_entries, bankroll_summary, stage_bet_log, stage_data_pull]


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
        max_iterations=16,
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
        st.markdown(ui.badge("NOT SAVED YET", ui.ROSE_FILL, ui.ROSE, ui.ROSE_BAR) + "&nbsp; "
                    f"**Log {len(pending)} bet leg(s)?** Nothing is saved until you confirm.",
                    unsafe_allow_html=True)
        view = pd.DataFrame(pending).rename(columns={"_entry_label": "entry"})
        st.dataframe(view[["entry", "date", "player", "stat", "choice", "line", "stake", "entry_payout", "notes"]],
                     hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        if c1.button("✅ Confirm and log", type="primary"):
            # One entry_id per staged entry label (picks the assistant grouped together).
            ids = {lbl: uuid.uuid4().hex[:10] for lbl in {r.get("_entry_label", "A") for r in pending}}
            rows = [{**{k: v for k, v in r.items() if k != "_entry_label"},
                     "entry_id": ids[r.get("_entry_label", "A")]} for r in pending]
            err = append_bets(rows)
            if err:
                st.error(err)
            else:
                st.session_state["assistant_pending_bets"] = []
                st.success(f"Logged {len(pending)} leg(s) to the Bet Log.")
        if c2.button("✖ Discard"):
            st.session_state["assistant_pending_bets"] = []
            st.rerun()


def _render_pending_pulls():
    """Data pulls the assistant staged -- run only after the user confirms."""
    pending = st.session_state.get("assistant_pending_pulls") or []
    if not pending:
        return
    from utils import PULL_SCRIPTS, run_pull_script
    with st.container(border=True):
        st.markdown(ui.badge("NOT RUN YET", ui.ROSE_FILL, ui.ROSE, ui.ROSE_BAR) + "&nbsp; "
                    f"**Run {len(pending)} data pull(s)?** " + " · ".join(f"`{p}`" for p in pending),
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("▶️ Confirm and run", type="primary", key="assistant_pull_confirm"):
            st.session_state["assistant_pending_pulls"] = []
            for label in pending:
                with st.spinner(f"Running {label}..."):
                    ok, output = run_pull_script(PULL_SCRIPTS[label])
                (st.success if ok else st.error)(f"{label} {'completed' if ok else 'failed or had errors'}.")
                with st.expander(f"Output — {label}"):
                    st.code(output or "(no output)")
            st.cache_data.clear()  # new files -> fresh tool results on the next question
        if c2.button("✖ Discard", key="assistant_pull_discard"):
            st.session_state["assistant_pending_pulls"] = []
            st.rerun()


def page_assistant():
    st.title("💬 AI Assistant")
    st.caption("Ask about tonight's board — it looks things up with the app's own data "
               "(live Underdog lines, model predictions, +EV pricing, injuries, usage, line moves, matchups, "
               "game logs, depth charts, news, your bet log) and never places bets.")

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
            st.session_state["assistant_pending_pulls"] = []
            st.rerun()
        st.caption("Each message is a Claude Opus 5.5 call with a few tool lookups — "
                   "typically a few cents to ~$0.20.")

    history = st.session_state.setdefault("assistant_history", [])

    if not history:
        examples = ["Best 3-pick entry for the 1pm games?", "Why is the model on the under for Ja'Marr Chase?",
                    "Compare Puka Nacua and Davante Adams receiving yards",
                    "Check my entry: Chase over 74.5 rec yds + Kelce over 5.5 receptions",
                    "Who's trending up in snaps this week?", "How's my parlay doing?",
                    "How has the model done against real lines?",
                    "Log my bet: Chase over 6.5 receptions, $10 2-pick with Kelce"]
        with st.container(border=True):
            ui.section("Try asking")
            st.html("<div style='display:flex;flex-wrap:wrap;gap:8px'>" + "".join(
                f"<span style='font-size:12px;padding:6px 10px;border-radius:8px;background:{ui.ROW};"
                f"box-shadow:0 0 0 1px {ui.EDGE};color:{ui.N1}'>“{ui.esc(e)}”</span>" for e in examples) + "</div>")

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
    _render_pending_pulls()

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
    if st.session_state.get("assistant_pending_bets") or st.session_state.get("assistant_pending_pulls"):
        st.rerun()
