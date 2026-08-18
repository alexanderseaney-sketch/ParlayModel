"""
ParlayModel Dashboard — local UI for managing parlays and project data.

Run with:
    streamlit run dashboard/app.py
"""
import os
import sys
from datetime import datetime, date

import pandas as pd
import streamlit as st

from utils import (
    EXPECTED_FILES, PULL_SCRIPTS, max_line_divergence_for, BET_LOG_PATH,
    file_status, load_csv_if_exists, load_bet_log, append_bet, run_pull_script,
    find_column, load_current_predictions, normalize_name, get_player_detail,
    load_player_photos, load_player_jersey_numbers, get_player_news,
    correlation_adjusted_parlay_probability, data_freshness_check, run_all_pulls,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from generate_weekly_bet_slip import (  # noqa: E402
    load_matched_props, build_single_leg_candidates, build_parlay_candidates, allocate_budget,
)

st.set_page_config(page_title="ParlayModel", page_icon="🏈", layout="wide")

# Applied on every page (including the password gate itself) rather than per-page --
# a consistent look shouldn't depend on which page happened to inject it first.
# Selectors below were verified against the actual rendered DOM (Streamlit 1.61.1),
# not guessed from memory -- data-testid/class names are internal and do shift
# between versions. div[data-testid="stVerticalBlock"] in particular is intentionally
# broad: Streamlit only applies a visible border/radius to it when a container was
# actually created with border=True, so styling every instance is a safe no-op on
# the many plain (unbordered) ones used just for layout.
#
# Design direction: a film-room / scouting-terminal feel grounded in NFL broadcast
# graphics, not a generic dark SaaS dashboard reskin. Three deliberate type roles
# (condensed broadcast display, clean body, mono for anything numeric -- odds,
# confidence scores, stat tables) instead of one font on headers only. A gold "yard
# marker" divider as the one signature element, used sparingly. Sharper 4px corners
# throughout instead of the rounded-everywhere SaaS-card default.
_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --pm-radius: 4px;
    --pm-border: rgba(212, 169, 74, 0.18);
    --pm-surface: #141920;
    --pm-text: #E8ECEF;
    --pm-muted: #7C8894;
    --pm-gold: #D4A94A;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Oswald', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    text-transform: uppercase;
}

h1 { border-bottom: 2px solid var(--pm-gold); padding-bottom: 0.4rem; }

/* Numeric/data surfaces read like a real stat sheet, not decorative UI chrome */
div[data-testid="stMetricValue"],
div[data-testid="stDataFrame"],
code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
}

div[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.75rem !important;
    color: var(--pm-muted) !important;
}

.block-container {
    padding-top: 2.5rem !important;
}

div[data-testid="stVerticalBlock"] {
    border-radius: var(--pm-radius) !important;
    border-color: var(--pm-border) !important;
}

div[data-testid^="stBaseButton-"] {
    border-radius: var(--pm-radius) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid var(--pm-border);
}

/* Signature element: a "yard marker" divider -- gold hash ticks instead of a plain
   hairline, used between major sections instead of st.divider()'s generic rule. */
.pm-yard-divider {
    display: flex; align-items: center; gap: 6px; margin: 1.75rem 0 1.25rem 0;
}
.pm-yard-divider::before, .pm-yard-divider::after {
    content: ""; flex: 1; height: 1px; background: var(--pm-border);
}
.pm-yard-divider span {
    color: var(--pm-gold); font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
    opacity: 0.75;
}

.pm-photo { width: 64px; height: 64px; border-radius: 50%; object-fit: cover;
            display: block; margin: 4px auto; border: 2px solid var(--pm-border); }
.pm-photo-placeholder { width: 64px; height: 64px; border-radius: 50%; background: var(--pm-surface);
            color: var(--pm-text); display: flex; align-items: center; justify-content: center;
            margin: 4px auto; font-weight: 600; font-size: 20px; border: 2px solid var(--pm-border); }
.pm-pos-pill { text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
            font-weight: 600; opacity: 0.75; color: var(--pm-gold);
            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.pm-dialog-photo { width: 96px; height: 96px; border-radius: 50%; object-fit: cover;
            display: block; border: 2px solid var(--pm-border); }
.pm-dialog-photo-placeholder { width: 96px; height: 96px; border-radius: 50%; background: var(--pm-surface);
            color: var(--pm-text); display: flex; align-items: center; justify-content: center;
            font-weight: 600; font-size: 28px; border: 2px solid var(--pm-border); }
</style>
"""
st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def yard_divider(label: str):
    """The signature section divider -- a gold hash-marked rule with a small caps
    label, instead of Streamlit's plain st.divider(). Use between major sections."""
    st.markdown(f'<div class="pm-yard-divider"><span>{label}</span></div>', unsafe_allow_html=True)


def check_password() -> bool:
    """Simple password gate for hosted deployment. Locally (no secrets.toml file at
    all), this is skipped entirely — auth only matters once this is reachable from the
    open internet via Streamlit Community Cloud."""
    try:
        has_secret = "dashboard_password" in st.secrets
    except Exception:
        has_secret = False  # no secrets.toml file present at all — local dev, skip the gate

    if not has_secret:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("ParlayModel")
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pw == st.secrets["dashboard_password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


if not check_password():
    st.stop()


def show_freshness_banner():
    """Checks data freshness on every load and surfaces it -- instead of silently
    rendering pages against empty or day(s)-old data. Matters most right after a fresh
    Streamlit Community Cloud deploy: data/raw/ is gitignored on purpose (it's
    regenerated data, not source code), so it starts completely empty and never ships
    with the deployment itself -- nothing in it until something actually runs a pull."""
    freshness = data_freshness_check()
    total = len(freshness["missing"]) + len(freshness["stale"]) + len(freshness["ok"])

    if total > 0 and len(freshness["missing"]) == total:
        st.error(
            "No data has been pulled yet in this deployment — everything below will be "
            "empty until you pull it. Takes a minute or two."
        )
        if st.button("🔄 Pull all data now", type="primary"):
            with st.spinner("Pulling all data sources — this can take a couple minutes..."):
                results = run_all_pulls()
            failed = [label for label, success, _ in results if not success]
            if failed:
                st.warning(f"Some pulls had issues: {', '.join(failed)}. See Run Data Pulls for details.")
            else:
                st.success("All data pulled.")
            st.cache_data.clear()
            st.rerun()
        st.stop()

    elif freshness["missing"] or freshness["stale"]:
        parts = []
        if freshness["missing"]:
            parts.append(f"{len(freshness['missing'])} source(s) never pulled")
        if freshness["stale"]:
            oldest = max(freshness["stale"], key=lambda x: x[1])
            parts.append(f"{len(freshness['stale'])} source(s) stale (oldest: {oldest[0]}, {oldest[1]:.0f}h old)")
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.warning(f"⚠️ Data freshness: {' · '.join(parts)}. Predictions/props may be out of date.")
            if c2.button("🔄 Refresh"):
                with st.spinner("Refreshing all data sources..."):
                    run_all_pulls()
                st.cache_data.clear()
                st.rerun()


show_freshness_banner()

if "slip" not in st.session_state:
    st.session_state.slip = []


def _confidence_badge(model_prob: float, conf: float):
    """One consistent visual language for model confidence, used everywhere a
    prediction is shown instead of ad hoc emoji-in-text strings."""
    pct = f"{model_prob*100:.0f}%"
    if conf >= 0.4:
        st.badge(f"{pct} confident", icon="🟢", color="green")
    elif conf >= 0.2:
        st.badge(f"{pct} confident", icon="🟡", color="orange")
    else:
        st.badge(f"{pct} confident", icon="⚪", color="gray")


def _pretty_stat(stat_name) -> str:
    """receiving_yds -> Receiving Yds -- a readable label instead of a raw column
    name, used anywhere a prop's stat type is shown to a human rather than matched
    against other data."""
    if not isinstance(stat_name, str) or not stat_name:
        return "?"
    return stat_name.replace("_", " ").title()


def _team_next_week_lookup(schedules: pd.DataFrame) -> dict:
    """team abbreviation -> that team's next unplayed game's week number. Computed
    independently of model predictions (same idea as models/current_predictions.py's
    _next_game_lookup, duplicated in miniature here) so a week filter can cover EVERY
    prop card, not just the ones that happened to match a trained model -- Underdog
    posts plenty of props for players/stat types nothing here predicts yet, and those
    shouldn't just disappear from a week filter."""
    home = schedules[["season", "week", "home_team", "away_team", "home_score"]].rename(
        columns={"home_team": "team", "away_team": "opponent"})
    away = schedules[["season", "week", "home_team", "away_team", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
    all_games = pd.concat([home, away], ignore_index=True)
    unplayed = all_games[all_games["home_score"].isna()].sort_values(["team", "season", "week"])
    next_game = unplayed.groupby("team").head(1)
    return next_game.set_index("team")["week"].to_dict()


def _leg_photo(player_name: str, photo_map: dict):
    """The player photo (or initials placeholder) used on every leg/bet card --
    Weekly Bet Slip, Parlay Builder, Bet Log -- reusing the exact same .pm-photo /
    .pm-photo-placeholder visual language already established on the Depth Charts
    formation cards rather than inventing a second style."""
    if not isinstance(player_name, str) or not player_name.strip():
        st.markdown('<div class="pm-photo-placeholder">?</div>', unsafe_allow_html=True)
        return
    key = normalize_name(player_name)
    photo_url = photo_map.get(key)
    if isinstance(photo_url, str) and photo_url:
        st.markdown(f'<img src="{photo_url}" class="pm-photo">', unsafe_allow_html=True)
    else:
        initials = "".join(p[0] for p in player_name.split()[:2] if p).upper()
        st.markdown(f'<div class="pm-photo-placeholder">{initials}</div>', unsafe_allow_html=True)


def _add_leg_details_to_slip(leg_details: list[dict]) -> None:
    """Shared by the Weekly Bet Slip 'Add to Parlay Builder' buttons and anything
    else that wants to push a suggestion into the slip -- one leg or a whole
    correlated pair at once, using the exact structure the Parlay Builder already
    reads (so its existing correlation math picks up multi-leg additions for free)."""
    for ld in leg_details:
        st.session_state.slip.append({
            "player": ld["player"], "stat": ld["stat"], "choice": ld["choice"], "line": ld["line"],
            "underdog_multiplier": ld["decimal_price"], "my_prob": ld["my_prob"],
            "team": ld["team"], "position_prop": ld["position_prop"],
        })


# ==================================================================== Betting

def page_weekly_bet_slip():
    st.title("🎯 Weekly Bet Slip")
    st.caption(
        "Generates real, sized bet suggestions from live model predictions and current "
        "Underdog prices, on demand — run it whenever you're actually about to bet. "
        "You place every bet yourself; this only suggests and sizes."
    )
    with st.expander("How this works, and what to trust"):
        st.write(
            "Model confidence is validated against each player's own rolling average "
            "(a proxy line), **not** a real historical Underdog line — that archive is "
            "still accumulating. Real accuracy against actual market prices is "
            "unvalidated. Sizing uses Kelly-fraction edge ranking against REAL live "
            "prices (not assumed odds), with a strict line-divergence check so a "
            "prediction is never used against a materially different number than what "
            "it was actually computed against. Only wager what you can afford to lose."
        )

    budget = st.number_input("Weekly budget ($)", min_value=1.0, value=10.0, step=1.0)

    if st.button("Generate this week's bets", type="primary", width="content"):
        try:
            with st.spinner("Pulling live predictions and current Underdog prices..."):
                matched = load_matched_props()
                candidates = build_single_leg_candidates(matched) + build_parlay_candidates(matched)
                allocated = allocate_budget(candidates, budget)
            st.session_state["bet_slip_result"] = {
                "matched_count": len(matched), "candidate_count": len(candidates),
                "allocated": allocated, "budget": budget,
            }
        except FileNotFoundError as e:
            st.error(f"Missing data: {e}. Run the data pulls first from **Run Data Pulls**.")

    result = st.session_state.get("bet_slip_result")
    if not result:
        return

    yard_divider("MATCHED PROPS")
    st.caption(
        f"{result['matched_count']} live props matched to a model prediction and passed "
        f"the line-divergence check · {result['candidate_count']} genuinely +EV at real prices."
    )

    if not result["allocated"]:
        st.info("No +EV opportunities clear the bar right now. Suggestion: skip this week rather than force a weak bet.")
        return

    for i, c in enumerate(result["allocated"]):
        edge = c["model_prob"] * c["decimal_odds"] - 1
        with st.container(border=True):
            head, badge_col, action = st.columns([5, 1, 1])
            with head:
                st.markdown(f"**${c['suggested_stake']:.2f}** — {c['description']}")
                st.caption(f"{c['type']} · model {c['model_prob']*100:.1f}% · price {c['decimal_odds']:.2f}x")
            with badge_col:
                st.badge(f"{edge*100:+.1f}% edge", color="green" if edge > 0 else "red")
            with action:
                if st.button("➕ Add to slip", key=f"wbs_add_{i}", width="stretch"):
                    _add_leg_details_to_slip(c["leg_details"])
                    st.toast(f"Added to Parlay Builder ({len(c['leg_details'])} leg(s))", icon="✅")

    total = sum(c["suggested_stake"] for c in result["allocated"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total suggested", f"${total:.2f}")
    col2.metric("Of budget", f"${result['budget']:.2f}")
    col3.metric("Legs in your slip now", len(st.session_state.slip))
    if total < result["budget"] - 0.01:
        st.caption(f"${result['budget'] - total:.2f} intentionally unallocated — not enough qualifying "
                   "+EV opportunities to use the full budget this week. That's fine; forcing it isn't.")

    if st.session_state.slip and st.button("Go build/review your parlay →", type="secondary"):
        st.switch_page(PAGE_PARLAY_BUILDER)


def page_parlay_builder():
    st.title("🧩 Parlay Builder")
    st.caption(
        "Add legs, filter by real model confidence, and see combined parlay math "
        "before anything gets placed."
    )

    df = load_csv_if_exists("underdog_props.csv")
    predictions = load_current_predictions()

    if df is None:
        st.warning("`underdog_props.csv` hasn't been pulled yet. Run it from **Run Data Pulls** first.")
        st.stop()

    if predictions is not None:
        predictions = predictions.copy()
        predictions["_match_key"] = predictions["player_display_name"].apply(normalize_name)
        # "Freshest available" rather than a hardcoded season -- stays correct as new
        # data gets pulled without needing to update this each year.
        freshest_season = predictions["stats_as_of_season"].max()
    else:
        st.info(
            "No model predictions found yet — run `python3 models/current_predictions.py` "
            "to generate them (covers receiving yards, rushing yards, passing yards, and "
            "receptions). Falling back to manual probability entry until then."
        )

    # Live depth-chart status (Q/PUP/IR/SUS/NFI/O), refreshed daily -- separate from
    # the trained models (no historical archive exists to validate it as a feature
    # yet, see README), but genuinely useful as a live warning next to a prediction
    # since it updates same-day, unlike the official injury report it supplements.
    depth = load_csv_if_exists("footballguys_depth.csv")
    if depth is not None:
        depth_status = (
            depth[depth["status"].notna()]
            .assign(_match_key=lambda d: d["player_name"].apply(normalize_name))
            .drop_duplicates(subset="_match_key")
            .set_index("_match_key")["status"]
        )
        # Current team, from the same live pull -- NOT predictions' own recent_team,
        # which comes from weekly_stats.csv and is only as fresh as that player's most
        # recent GAME (currently frozen at 2024, see utils._default_pull_years -- real
        # bug found 2026-08-16: a since-traded player still showed his old team,
        # because the prediction was generated from his last known game stats, not
        # his current roster spot). Footballguys' depth chart is pulled fresh and
        # reflects actual current rosters regardless of how stale weekly_stats is.
        depth_team = (
            depth.assign(_match_key=lambda d: d["player_name"].apply(normalize_name))
            .drop_duplicates(subset="_match_key")
            .set_index("_match_key")["team_abbr"]
        )
    else:
        depth_status = None
        depth_team = None

    schedules_df = load_csv_if_exists("schedules.csv")
    team_next_week = _team_next_week_lookup(schedules_df) if schedules_df is not None else None

    photo_map = load_player_photos()
    tab_slip, tab_add = st.tabs([f"📋 Current slip ({len(st.session_state.slip)})", "➕ Add legs"])

    with tab_add:
        st.subheader("Confidence filter")
        st.caption(
            "There's nothing wrong with being confident — validated testing showed roughly "
            "78% accuracy at the 0.4 threshold, pooled across 5 real seasons. Being confident "
            "here means the model has a real, tested reason, not a guess."
        )
        min_confidence = st.slider(
            "Minimum confidence to show a prop", 0.0, 1.0, 0.4, 0.05,
            help="0 = show everything (coinflips included). 0.4 historically ~78% accurate. Higher = fewer, stronger picks.",
        )

        if predictions is not None:
            qualifying = predictions[predictions["confidence"] >= min_confidence]
            st.metric("Players clearing this bar right now", f"{len(qualifying)} of {len(predictions)}")

        name_col = find_column(df, ["full_name", "player_name", "name"])
        stat_col = find_column(df, ["stat_name", "stat"])
        line_col = find_column(df, ["stat_value", "line", "value"])
        choice_col = find_column(df, ["choice"])
        # decimal_price first, not payout_multiplier -- a real bug found 2026-08-16:
        # payout_multiplier is Underdog's own internal boost ratio (hovers around
        # 1.00, e.g. 1.06/0.86 for a boosted/reduced leg), not the actual payout.
        # decimal_price is the real "multiply your stake by this" number (1.90x,
        # 2.08x, etc.) and is what the rest of this codebase already treats as the
        # real price (see generate_weekly_bet_slip.py). Priority-list matching had
        # been silently preferring the wrong one since payout_multiplier happens to
        # exist in the pulled data and was listed first.
        mult_col = find_column(df, ["decimal_price", "payout_multiplier", "multiplier", "odds", "american_price"])

        if not name_col or not stat_col:
            st.error(
                "Couldn't find the expected player/stat columns in underdog_props.csv. "
                f"Actual columns: {list(df.columns)}"
            )
        else:
            # Team comes from the live depth-chart pull (see depth_team above), not
            # from underdog_props.csv itself -- that file has no plain team-abbrev
            # column, only an internal team_id UUID. Computed here (not just at
            # render time) so the team filter's own option list and the actual
            # filtering both use the same source.
            df = df.copy()
            if depth_team is not None:
                df["_team"] = df[name_col].apply(normalize_name).map(depth_team)
            else:
                df["_team"] = None
            # Week comes from each player's TEAM's next unplayed game (see
            # _team_next_week_lookup) rather than predictions' own next_week --
            # that column only exists for props that matched a trained model, and a
            # week filter should cover every card, matched or not.
            df["_week"] = df["_team"].map(team_next_week) if team_next_week is not None else None

            search_col, type_col, team_col_ui, week_col_ui = st.columns([2, 2, 1, 1])
            with search_col:
                search = st.text_input("Search player", placeholder="Type a name and press Enter…")
            with type_col:
                prop_type_options = sorted(df[stat_col].dropna().unique())
                prop_type_filter = st.multiselect(
                    "Parlay type", options=prop_type_options, format_func=_pretty_stat,
                    placeholder="All types",
                )
            with team_col_ui:
                if depth_team is not None:
                    team_options = sorted(df["_team"].dropna().unique())
                    team_filter = st.multiselect("Team", options=team_options, placeholder="All teams")
                else:
                    team_filter = []
                    st.caption("Team filter needs Footballguys depth chart data.")
            with week_col_ui:
                if team_next_week is not None:
                    week_options = sorted(int(w) for w in df["_week"].dropna().unique())
                    week_filter = st.multiselect("Week", options=week_options, placeholder="All weeks")
                else:
                    week_filter = []
                    st.caption("Week filter needs schedules.csv.")

            # No .head(50) here anymore -- it used to truncate BEFORE the predictions
            # merge below, so the default (unfiltered) view was just whatever order
            # the raw CSV happens to be in. Underdog's pull is dominated by season-
            # long novelty props (win totals, "games started") the model was never
            # built to cover, so real model data could end up completely absent from
            # the visible list even though it was working correctly (confirmed
            # 2026-08-16: 1483 real predictions existed the whole time, just buried
            # past the default cutoff). Truncation now happens after sorting further
            # down, once matched props are guaranteed to be at the front.
            options_df = df[df[name_col].astype(str).str.contains(search, case=False, na=False)] if search else df
            if prop_type_filter:
                options_df = options_df[options_df[stat_col].isin(prop_type_filter)]
            if team_filter:
                options_df = options_df[options_df["_team"].isin(team_filter)]
            if week_filter:
                options_df = options_df[options_df["_week"].isin(week_filter)]
            narrowed = bool(search) or bool(prop_type_filter) or bool(team_filter) or bool(week_filter)

            if predictions is not None:
                options_df = options_df.copy()
                options_df["_match_key"] = options_df[name_col].apply(normalize_name)
                # Match on player AND stat type -- name-only matching would attach e.g.
                # a receiving-yards prediction to that same player's rushing-yards prop
                # now that current_predictions.py covers four different prop types.
                options_df = options_df.merge(
                    predictions[["_match_key", "stat_name", "predicted_prob_over", "confidence",
                                  "stats_as_of_season", "stats_as_of_week", "recent_team", "position",
                                  "prop_type", "proxy_line"]],
                    left_on=["_match_key", stat_col], right_on=["_match_key", "stat_name"],
                    how="left",
                )
                below_bar = options_df["confidence"] < min_confidence
                if below_bar.any() and not narrowed:
                    st.caption(f"{below_bar.sum()} props below the confidence bar are hidden. Search or lower the bar to see them.")
                options_df = options_df[~below_bar | options_df["confidence"].isna() | narrowed]
                options_df = options_df.sort_values("confidence", ascending=False, na_position="last")

            if not narrowed:
                options_df = options_df.head(50)

            if not mult_col:
                st.info(
                    "Note: no obvious odds/payout column found in the pulled data yet — "
                    "re-check this once `pull_underdog.py` has run against live data."
                )

            # One card per (player, stat, line) -- Underdog's own data has separate
            # rows per side (over/under), which used to mean two near-identical rows
            # with no way to compare them. The whole point of pulling live prices is
            # to show what the model thinks EACH side is worth, so both sides render
            # together: their real (possibly asymmetric -- confirmed 2026-08-16 some
            # props price over/under differently, e.g. 2.08x/1.75x) decimal_price
            # multiplier and the model's predicted hit probability for that specific
            # side, side by side.
            group_cols = [name_col, stat_col, line_col]
            for (player_name, stat_name, line_value), group in options_df.groupby(group_cols, dropna=False, sort=False):
                over_rows = group[group[choice_col].astype(str).str.lower() == "over"]
                under_rows = group[group[choice_col].astype(str).str.lower() == "under"]
                over_row = over_rows.iloc[0] if not over_rows.empty else None
                under_row = under_rows.iloc[0] if not under_rows.empty else None
                any_row = over_row if over_row is not None else under_row

                # Two real bugs fixed here 2026-08-15 (found while building the
                # weekly bet-slip generator, see README):
                # 1. predicted_prob_over was used directly regardless of which side
                #    (over/under) this specific row actually is -- a confident UNDER
                #    prediction was shown as if it endorsed the OVER row it happened
                #    to be attached to. Now computed separately for each side below.
                # 2. predicted_prob_over answers "beats OUR proxy line", not "beats
                #    Underdog's posted line" -- only valid when those two numbers are
                #    close (see max_line_divergence_for in utils.py, grain-specific
                #    since season/period-scoped proxies are structurally noisier
                #    than weekly ones).
                raw_prob_over = any_row.get("predicted_prob_over") if predictions is not None else None
                proxy_line = any_row.get("proxy_line")
                line_ok = True
                if pd.notna(raw_prob_over) and pd.notna(proxy_line) and pd.notna(line_value) and proxy_line:
                    divergence_limit = max_line_divergence_for(stat_name)
                    line_ok = abs(float(line_value) - float(proxy_line)) / abs(float(proxy_line)) <= divergence_limit
                has_model = pd.notna(raw_prob_over) and line_ok

                fbg_status = depth_status.get(normalize_name(player_name)) if depth_status is not None else None

                with st.container(border=True):
                    photo_col, info_col = st.columns([1, 5])
                    with photo_col:
                        _leg_photo(player_name, photo_map)
                    with info_col:
                        # Current team from the live depth chart, not predictions'
                        # own (currently 2024-frozen) recent_team -- see depth_team
                        # comment above. Shown regardless of has_model now, since it's
                        # an independent source: a player with no model match can
                        # still show their real current team.
                        team = depth_team.get(normalize_name(player_name)) if depth_team is not None else None
                        position = any_row.get("position") if has_model else None
                        meta = " · ".join(str(b) for b in [team, position] if pd.notna(b) and b)
                        # Same player-profile dialog as Depth Charts, not a second
                        # one -- name_col/stat_col/line_col already make this key
                        # unique per card the same way the Add buttons below are keyed.
                        if st.button(player_name, key=f"pname_{player_name}_{stat_name}_{line_value}", type="tertiary"):
                            _player_detail_dialog({
                                "player_name": player_name,
                                "team_abbr": team,
                                "position": position,
                                "status": fbg_status,
                            })
                        if meta:
                            st.caption(meta)
                        st.markdown(f"{_pretty_stat(stat_name)} — **{line_value}**")
                        badges = st.columns(3)
                        b_i = 0
                        if has_model:
                            stats_season = any_row.get("stats_as_of_season")
                            if pd.notna(stats_season):
                                stats_week = int(any_row["stats_as_of_week"])
                                if stats_season < freshest_season:
                                    with badges[b_i]:
                                        st.badge(f"stale: {int(stats_season)} wk{stats_week}", icon="⚠️", color="red")
                                    b_i += 1
                        elif pd.notna(raw_prob_over) and not line_ok:
                            with badges[b_i]:
                                st.badge(f"line mismatch (proxy {proxy_line:.1f})", icon="⚠️", color="red",
                                         help=f"Our proxy line ({proxy_line:.1f}) is too far from Underdog's real line ({line_value}) to trust this prediction for this specific bet.")
                            b_i += 1
                        if fbg_status:
                            with badges[b_i]:
                                st.badge(fbg_status, icon="🚑", color="red", help="Footballguys depth chart status, pulled today")
                            b_i += 1

                    side_specs = [
                        ("OVER", "over", over_row, raw_prob_over if pd.notna(raw_prob_over) else None),
                        ("UNDER", "under", under_row, 1 - raw_prob_over if pd.notna(raw_prob_over) else None),
                    ]
                    side_cols = st.columns(2)
                    for side_col, (label, choice, side_row, side_prob) in zip(side_cols, side_specs):
                        with side_col:
                            if side_row is None:
                                st.caption(f"{label} not offered")
                                continue
                            st.markdown(f"**{label}**")
                            if has_model and side_prob is not None:
                                _confidence_badge(side_prob, any_row["confidence"])
                            else:
                                st.caption("No model prediction")
                            price = side_row.get(mult_col) if mult_col else None
                            st.caption(f"Pays {float(price):.2f}x" if pd.notna(price) else "Price unknown")
                            if st.button(f"➕ Add {label}", key=f"add_{player_name}_{stat_name}_{line_value}_{choice}", width="stretch"):
                                position_prop = None
                                if has_model and pd.notna(any_row.get("position")) and pd.notna(any_row.get("prop_type")):
                                    position_prop = f"{any_row['position']} {any_row['prop_type']}"
                                st.session_state.slip.append({
                                    "player": player_name,
                                    "stat": stat_name,
                                    "choice": choice,
                                    "line": line_value,
                                    "underdog_multiplier": float(price) if pd.notna(price) else None,
                                    "my_prob": float(side_prob) if (has_model and side_prob is not None) else 0.55,
                                    # Live depth-chart team, not predictions' stale
                                    # recent_team -- this feeds the slip's same-team
                                    # correlation detection, so a wrong team here
                                    # doesn't just mislabel a card, it can silently
                                    # miss or misapply a real correlation adjustment.
                                    "team": team,
                                    "position_prop": position_prop,
                                })
                                st.rerun()

    with tab_slip:
        if not st.session_state.slip:
            st.info("No legs added yet — head to the **➕ Add legs** tab, or generate suggestions on the **Weekly Bet Slip** page and send them here.")
            return

        for i, leg in enumerate(st.session_state.slip):
            with st.container(border=True):
                photo_col, info_col, prob_col, action_col = st.columns([1, 3, 2, 1])
                with photo_col:
                    _leg_photo(leg["player"], photo_map)
                with info_col:
                    st.markdown(f"**{leg['player']}**")
                    st.markdown(f"{_pretty_stat(leg['stat'])} — **{str(leg['choice']).upper()} {leg['line']}**")
                with prob_col:
                    # Collapsed label + a caption above it instead -- smaller and less
                    # visually heavy than Streamlit's default slider label, but (unlike
                    # a fully hidden label) still says up front that this is YOUR
                    # estimate feeding the fair-odds math, not the real betting odds --
                    # a real user confusion this fixes: it looked like an editable
                    # "odds" field with no clarifying label at all.
                    st.caption("Your win % estimate")
                    leg["my_prob"] = st.slider(
                        "Your win % estimate", 0.0, 1.0, leg["my_prob"], 0.01,
                        key=f"prob_{i}", label_visibility="collapsed",
                    )
                    fair_mult = 1 / leg["my_prob"] if leg["my_prob"] > 0 else float("inf")
                    ud_mult = leg["underdog_multiplier"]
                    if ud_mult:
                        st.caption(f"Underdog pays: {ud_mult}x (fixed)")
                        st.caption(f"Your fair odds: {fair_mult:.2f}x")
                        if fair_mult < float(ud_mult):
                            st.badge("+EV", color="green")
                        else:
                            st.badge("-EV", color="red")
                    else:
                        st.caption(f"Your fair odds: {fair_mult:.2f}x (Underdog odds unknown)")
                if action_col.button("🗑️", key=f"rm_{i}", help="Remove this leg", width="stretch"):
                    st.session_state.slip.pop(i)
                    st.rerun()

        yard_divider("CORRELATION CHECK")

        legs_for_corr = [
            {"team": leg.get("team"), "position_prop": leg.get("position_prop"), "prob": leg["my_prob"]}
            for leg in st.session_state.slip
        ]
        corr_result = correlation_adjusted_parlay_probability(legs_for_corr)
        naive_prob = corr_result["naive_prob"]
        adjusted_prob = corr_result["adjusted_prob"]
        adjustments = corr_result["adjustments"]
        combined_fair_mult = 1 / adjusted_prob if adjusted_prob > 0 else float("inf")

        weak_legs = [leg for leg in st.session_state.slip if leg["underdog_multiplier"] and 1 / leg["my_prob"] >= float(leg["underdog_multiplier"])]

        if adjustments:
            st.info(
                f"⚠️ **{len(adjustments)} correlated leg pair(s) detected** (same team, same "
                "game) — the naive independence math would be wrong for this slip. Using "
                "real measured correlations instead:"
            )
            for pos_prop_a, pos_prop_b, phi in adjustments:
                direction = "raises" if phi > 0 else "lowers"
                st.caption(f"　　{pos_prop_a} + {pos_prop_b}: phi={phi:+.3f} — {direction} the true combined hit rate vs. treating them as independent")
            col1, col2, col3 = st.columns(3)
            col1.metric("Naive (independent) probability", f"{naive_prob * 100:.1f}%")
            col2.metric("Correlation-adjusted probability", f"{adjusted_prob * 100:.1f}%",
                        delta=f"{(adjusted_prob - naive_prob) * 100:+.1f}pt")
            col3.metric("Legs without individual edge", len(weak_legs))
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Combined true probability", f"{adjusted_prob * 100:.1f}%")
            col2.metric("Your fair combined payout", f"{combined_fair_mult:.2f}x")
            col3.metric("Legs without individual edge", len(weak_legs))

        if weak_legs:
            st.warning(
                f"{len(weak_legs)} leg(s) don't show positive edge on their own. Parlays "
                "should only combine legs that are already independently +EV."
            )
        else:
            st.success("Every leg shows positive edge individually, based on the probabilities above.")

        stake = st.number_input("Stake ($)", min_value=0.0, value=10.0, step=1.0)
        st.write(f"Potential payout at your fair (correlation-adjusted) odds: **${stake * combined_fair_mult:.2f}**")

        yard_divider("STAKE")
        st.caption(
            "Placement isn't automated yet (Phase 5 — Claude in Chrome, home only, "
            "human-approved each time). For now this gives you a clean slip to place manually."
        )
        b1, b2 = st.columns(2)
        if b1.button("📒 Send slip to Bet Log (as pending)", width="stretch"):
            for leg in st.session_state.slip:
                append_bet({
                    "date": date.today().isoformat(),
                    "sport": "NFL",
                    "player": leg["player"],
                    "stat": leg["stat"],
                    "choice": leg["choice"],
                    "line": leg["line"],
                    "multiplier_or_odds": leg["underdog_multiplier"],
                    "stake": round(stake / len(st.session_state.slip), 2),
                    "result": "pending",
                    "notes": "Sent from Parlay Builder slip",
                    "logged_at": datetime.now().isoformat(),
                })
            st.success(f"Logged {len(st.session_state.slip)} leg(s) as pending bets.")
        if b2.button("Clear slip", width="stretch"):
            st.session_state.slip = []
            st.rerun()


def page_bet_log():
    st.title("📒 Bet Log")
    st.caption("Manual tracking for now — will connect to the automated flow once Phase 4/5 are built.")

    with st.expander("➕ Log a new bet", expanded=False):
        with st.form("new_bet_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                bet_date = st.date_input("Date", value=date.today())
                sport = st.text_input("Sport/League", value="NFL")
                player = st.text_input("Player")
            with c2:
                stat = st.text_input("Stat (e.g. rushing_yards)")
                choice = st.selectbox("Choice", ["over", "under"])
                line = st.number_input("Line", step=0.5)
            with c3:
                multiplier = st.text_input("Multiplier / odds")
                stake = st.number_input("Stake ($)", min_value=0.0, step=1.0)
                result = st.selectbox("Result", ["pending", "won", "lost", "push"])
            notes = st.text_area("Notes")

            if st.form_submit_button("Save bet"):
                append_bet({
                    "date": bet_date.isoformat(),
                    "sport": sport,
                    "player": player,
                    "stat": stat,
                    "choice": choice,
                    "line": line,
                    "multiplier_or_odds": multiplier,
                    "stake": stake,
                    "result": result,
                    "notes": notes,
                    "logged_at": datetime.now().isoformat(),
                })
                st.success("Bet logged.")
                st.rerun()

    bets = load_bet_log()
    if bets.empty:
        st.info("No bets logged yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    n_pending = (bets["result"] == "pending").sum()
    n_won = (bets["result"] == "won").sum()
    n_lost = (bets["result"] == "lost").sum()
    staked = pd.to_numeric(bets["stake"], errors="coerce").sum()
    c1.metric("Pending", n_pending)
    c2.metric("Won", n_won)
    c3.metric("Lost", n_lost)
    c4.metric("Total staked", f"${staked:,.2f}")

    RESULT_OPTIONS = ["pending", "won", "lost", "push"]
    RESULT_COLOR = {"won": "green", "lost": "red", "push": "gray", "pending": "orange"}

    photo_map = load_player_photos()
    editable = bets.sort_values("date", ascending=False).reset_index(drop=True)
    changed = False

    for i, bet in editable.iterrows():
        with st.container(border=True):
            photo_col, info_col, result_col = st.columns([1, 4, 2])
            with photo_col:
                _leg_photo(bet.get("player"), photo_map)
            with info_col:
                st.markdown(f"**{bet.get('player') or '?'}**")
                st.markdown(f"{_pretty_stat(bet.get('stat'))} — **{str(bet.get('choice', '')).upper()} {bet.get('line', '')}**")
                stake_val = pd.to_numeric(bet.get("stake"), errors="coerce")
                meta_bits = [
                    bet.get("date"),
                    f"${stake_val:.2f}" if pd.notna(stake_val) else None,
                    bet.get("multiplier_or_odds"),
                ]
                st.caption(" · ".join(str(b) for b in meta_bits if pd.notna(b) and str(b).strip()))
                if isinstance(bet.get("notes"), str) and bet["notes"].strip():
                    st.caption(f"📝 {bet['notes']}")
            with result_col:
                current_result = bet.get("result") if bet.get("result") in RESULT_OPTIONS else "pending"
                st.badge(current_result.upper(), color=RESULT_COLOR[current_result])
                new_result = st.selectbox(
                    "Result", RESULT_OPTIONS, index=RESULT_OPTIONS.index(current_result),
                    key=f"bet_result_{i}", label_visibility="collapsed",
                )
                if new_result != current_result:
                    editable.at[i, "result"] = new_result
                    changed = True

    if changed:
        editable.to_csv(BET_LOG_PATH, index=False)
        st.rerun()


# ==================================================================== Research

def page_overview():
    st.title("🗂️ Data Status")
    st.caption("What's been pulled, when, and how much of it there is.")

    rows = []
    for filename, source in EXPECTED_FILES.items():
        status = file_status(filename)
        if status["exists"]:
            rows.append({
                "File": filename,
                "Status": "✅ pulled",
                "Rows": f"{status['rows']:,}" if status["rows"] is not None else "?",
                "Size": f"{status['size_kb']:.0f} KB",
                "Last pulled": status["modified"].strftime("%Y-%m-%d %H:%M"),
                "Source script": source,
            })
        else:
            rows.append({
                "File": filename,
                "Status": "⬜ not pulled",
                "Rows": "—",
                "Size": "—",
                "Last pulled": "—",
                "Source script": source,
            })

    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    n_missing = sum(1 for r in rows if r["Status"].startswith("⬜"))
    if n_missing:
        st.info(f"{n_missing} file(s) not pulled yet — head to **Run Data Pulls** to fetch them.")
    else:
        st.success("All expected data files are present.")


def page_nfl_stats():
    st.title("📊 NFL Stats (nflverse)")

    dataset = st.selectbox(
        "Dataset",
        ["schedules.csv", "weekly_stats.csv", "ngs_passing.csv", "ngs_rushing.csv",
         "ngs_receiving.csv", "injuries.csv", "snap_counts.csv"],
    )
    df = load_csv_if_exists(dataset)

    if df is None:
        st.warning(f"`{dataset}` hasn't been pulled yet. Run it from **Run Data Pulls**.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if "season" in df.columns:
                seasons = sorted(df["season"].dropna().unique(), reverse=True)
                season_filter = st.multiselect("Season", seasons, default=seasons[:1] if seasons else [])
            else:
                season_filter = None
        with col2:
            team_col = next((c for c in ["team", "recent_team", "home_team", "club_code"] if c in df.columns), None)
            if team_col:
                teams = sorted(df[team_col].dropna().unique())
                team_filter = st.multiselect("Team", teams)
            else:
                team_filter = None
                team_col = None
        with col3:
            search = st.text_input("Search (any column, text match)")

        filtered = df.copy()
        if season_filter:
            filtered = filtered[filtered["season"].isin(season_filter)]
        if team_filter and team_col:
            filtered = filtered[filtered[team_col].isin(team_filter)]
        if search:
            mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            filtered = filtered[mask]

        st.caption(f"{len(filtered):,} of {len(df):,} rows")
        st.dataframe(filtered, width='stretch', height=500)


STATUS_LABELS = {
    "Q": ("Questionable", "orange"),
    "O": ("Out", "red"),
    "IR": ("Injured Reserve", "red"),
    "PUP": ("PUP", "gray"),
    "SUS": ("Suspended", "violet"),
    "NFI": ("Non-Football Injury", "gray"),
}

# The offensive line + TE render as one straight "line of scrimmage" row -- deliberately
# a superset of what any one team carries (e.g. most teams have no listed Center in this
# data source, a known upstream gap -- see pull_footballguys_depth.py) so a team only
# shows the slots it actually has a starter for.
OFFENSE_LINE_ORDER = ["LT", "LG", "C", "RG", "RT", "TE"]

# Canonical left-to-right order within each defensive/special-teams group -- a superset
# of any one team's actual scheme (e.g. a 4-3 team has LDT/RDT, a 3-4 team has NT
# instead) so both render correctly: a team only shows the codes it actually has data
# for, in this order, rather than every team being forced into one scheme's slots. This
# is also *why* the front row genuinely reflects each team's real looks: a 3-4 team's
# data only ever has an NT entry, never LDT/RDT, so its line row is 3 players wide
# instead of 4 -- driven by their real depth chart, not a guessed formation.
DEFENSE_GROUPS = [
    ("Line", ["LDE", "LDT", "NT", "RDT", "RDE"]),
    ("Linebackers", ["LOLB", "LILB", "MLB", "RILB", "ROLB", "WLB", "SLB"]),
    ("Secondary", ["LCB", "RCB", "SCB", "FS", "SS"]),
]
SPECIAL_TEAMS_GROUPS = [
    ("Specialists", ["PK", "P", "LS", "H"]),
    ("Return Game", ["KR", "PR"]),
]
POSITION_GROUPS = {"defense": DEFENSE_GROUPS, "special_teams": SPECIAL_TEAMS_GROUPS}

def _status_badge(status):
    if not isinstance(status, str) or not status.strip():
        return
    label, color = STATUS_LABELS.get(status, (status, "gray"))
    st.badge(label, color=color)


@st.dialog("Player", width="large")
def _player_detail_dialog(row: dict):
    """row only strictly needs "player_name" -- everything else is optional so this
    same dialog works from both Depth Charts (full depth-chart context: position,
    depth_rank, team_name, status) and Parlay Builder (a prop leg, which has no
    depth-chart slot at all -- just whatever team/position happen to be known)."""
    player_name = row["player_name"]
    name_key = normalize_name(player_name)
    photo_url = load_player_photos().get(name_key)
    jersey = load_player_jersey_numbers().get(name_key)

    header_cols = st.columns([1, 3])
    with header_cols[0]:
        if isinstance(photo_url, str) and photo_url:
            st.markdown(f'<img src="{photo_url}" class="pm-dialog-photo">', unsafe_allow_html=True)
        else:
            initials = "".join(p[0] for p in player_name.split()[:2] if p).upper()
            st.markdown(f'<div class="pm-dialog-photo-placeholder">{initials}</div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.subheader(player_name)
        depth_rank = row.get("depth_rank")
        meta_bits = [
            row.get("team_name") or row.get("team_abbr"),
            f"#{int(jersey)}" if jersey else None,
            row.get("position"),
            f"Depth #{int(depth_rank)}" if depth_rank is not None else None,
        ]
        st.caption(" · ".join(str(b) for b in meta_bits if b))
        _status_badge(row.get("status"))

    detail = get_player_detail(player_name)

    if "injury_report" in detail:
        inj = detail["injury_report"]
        with st.container(border=True):
            st.markdown(f"**Latest injury report** — Week {int(inj['week'])}, {int(inj['season'])}")
            status_val = inj.get("report_status")
            st.write(status_val if isinstance(status_val, str) and status_val.strip() else "No designation")
            primary = inj.get("report_primary_injury")
            if isinstance(primary, str) and primary.strip():
                st.caption(primary)

    if "predictions" in detail:
        with st.container(border=True):
            st.markdown("**Model predictions** (vs. our own proxy line, not necessarily today's market line)")
            for _, p in detail["predictions"].iterrows():
                side = "over" if p["predicted_prob_over"] >= 0.5 else "under"
                prob = p["predicted_prob_over"] if side == "over" else 1 - p["predicted_prob_over"]
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"{p['stat_name']} {side} {p['proxy_line']}")
                with c2:
                    _confidence_badge(prob, p["confidence"])

    if "props" in detail:
        with st.container(border=True):
            st.markdown("**Current Underdog lines**")
            cols = [c for c in ["stat_name", "choice", "stat_value", "american_price"] if c in detail["props"].columns]
            st.dataframe(detail["props"][cols].sort_values("stat_name"), hide_index=True, width="stretch")

    if "recent_games" in detail:
        with st.container(border=True):
            st.markdown("**Recent games**")
            games = detail["recent_games"]
            stat_cols = ["passing_yards", "rushing_yards", "receptions", "receiving_yards",
                         "passing_tds", "rushing_tds", "receiving_tds", "fantasy_points_ppr"]
            cols = ["season", "week", "opponent_team"] + [c for c in stat_cols if c in games.columns and games[c].fillna(0).ne(0).any()]
            st.dataframe(games[cols], hide_index=True, width="stretch")

    if "snap_trend" in detail:
        with st.container(border=True):
            st.markdown("**Snap share, recent games**")
            snaps = detail["snap_trend"]
            pct_cols = [c for c in ["offense_pct", "defense_pct", "st_pct"]
                        if c in snaps.columns and snaps[c].fillna(0).ne(0).any()]
            cols = [c for c in ["season", "week", "opponent"] if c in snaps.columns] + pct_cols
            st.dataframe(snaps[cols], hide_index=True, width="stretch")

    if not detail:
        st.caption("No model/stats/market data matched for this player yet.")

    with st.container(border=True):
        st.markdown("**Recent news**")
        # name_key alone is enough to be unique per player; position/depth_rank are
        # appended when present (Depth Charts) purely so the same real player showing
        # up at two depth-chart slots (e.g. a WR who's also the punt returner) gets
        # separate cached results per slot, matching how their card buttons are keyed.
        news_key = f"news_{name_key}_{row.get('position', '')}_{depth_rank if depth_rank is not None else ''}"
        if st.button("📰 Pull recent news", key=f"pull_{news_key}"):
            with st.spinner("Searching Google News..."):
                st.session_state[news_key] = get_player_news(player_name)
        if news_key in st.session_state:
            news_items = st.session_state[news_key]
            if not news_items:
                st.caption("No recent news found.")
            for n in news_items:
                headline = n.get("headline") or "(untitled)"
                st.markdown(f"[{headline}]({n['link']})" if n.get("link") else headline)
                meta = " · ".join(str(b) for b in [n.get("source"), n.get("published")] if b)
                if meta:
                    st.caption(meta)


def _card_button(p: pd.Series, is_starter_card: bool):
    label = p["player_name"] if is_starter_card else f"{p['depth_rank']}. {p['player_name']}"
    if isinstance(p.get("status"), str) and p["status"].strip():
        label += f"  ({p['status']})"
    # team_abbr+position+depth_rank, not footballguys_player_id -- the same real player
    # often appears twice in the source data (e.g. a WR who's also the punt returner
    # shows up under both WR and PR), which makes their player_id a legitimate
    # duplicate here even though each occurrence needs its own widget key.
    key = f"depth_card_{p['team_abbr']}_{p['position']}_{p['depth_rank']}"
    if st.button(label, key=key, type="primary" if is_starter_card else "tertiary", width="stretch"):
        _player_detail_dialog(p.to_dict())


def _player_card(starter: pd.Series, cat_df: pd.DataFrame, photo_map: dict, show_backups: bool = True):
    """One formation slot: starter's photo (or an initials placeholder when they have
    no Underdog prop to source a photo from -- common for non-skill positions), then
    up to 2 backups inline and any deeper depth in a '+N more' expander, so real depth
    stays reachable without the starter row turning into a wall of names.

    show_backups=False for every starter at a position except the deepest-ranked one
    (see _formation_row) -- a position can have more than one starter (every current
    team starts 3 WRs), and those co-starters aren't "backups" of each other, so only
    the last one in the group should list who's actually behind them on the depth
    chart."""
    with st.container(border=True):
        key = normalize_name(starter["player_name"])
        photo_url = photo_map.get(key)
        if isinstance(photo_url, str) and photo_url:
            st.markdown(f'<img src="{photo_url}" class="pm-photo">', unsafe_allow_html=True)
        else:
            initials = "".join(p[0] for p in starter["player_name"].split()[:2] if p).upper()
            st.markdown(f'<div class="pm-photo-placeholder">{initials}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pm-pos-pill">{starter["position"]}</div>', unsafe_allow_html=True)

        _card_button(starter, is_starter_card=True)

        if not show_backups:
            return
        same_position = cat_df[cat_df["position"] == starter["position"]].sort_values("depth_rank")
        backups = same_position[same_position["depth_rank"] > starter["depth_rank"]]
        for _, b in backups.head(2).iterrows():
            _card_button(b, is_starter_card=False)
        rest = backups.iloc[2:]
        if len(rest):
            with st.expander(f"+{len(rest)} more"):
                for _, b in rest.iterrows():
                    _card_button(b, is_starter_card=False)


def _formation_row(starters: list[pd.Series], cat_df: pd.DataFrame, photo_map: dict):
    if not starters:
        return
    max_rank_by_position: dict[str, int] = {}
    for s in starters:
        pos = s["position"]
        max_rank_by_position[pos] = max(max_rank_by_position.get(pos, -1), s["depth_rank"])

    cols = st.columns(len(starters))
    for col, starter in zip(cols, starters):
        show_backups = starter["depth_rank"] == max_rank_by_position[starter["position"]]
        with col:
            _player_card(starter, cat_df, photo_map, show_backups=show_backups)


def _offense_formation(cat_df: pd.DataFrame, photo_map: dict):
    starters = cat_df[cat_df["is_starter"] == True]  # noqa: E712

    st.markdown("##### Offensive Line")
    line = [starters[starters["position"] == p].iloc[0] for p in OFFENSE_LINE_ORDER
            if p in starters["position"].values]
    _formation_row(line, cat_df, photo_map)

    # Skill positions arranged like a real backfield/receiver split rather than a
    # generic list: RB and FB (if the team carries one -- plenty don't anymore) flank
    # the QB, WRs at the outside edges. This uses each team's REAL starters, not a
    # fixed template -- e.g. every current team happens to start 3 WRs, but a team
    # that only started 2 would only show 2, because "starter" here comes straight
    # from Footballguys' own depth chart rather than an assumed personnel package.
    st.markdown("##### Skill Positions")
    wrs = starters[starters["position"] == "WR"].sort_values("depth_rank")
    rbs = starters[starters["position"] == "RB"].sort_values("depth_rank")
    qbs = starters[starters["position"] == "QB"]
    fbs = starters[starters["position"] == "FB"]
    skill = []
    if len(wrs) >= 1:
        skill.append(wrs.iloc[0])
    if len(rbs) >= 1:
        skill.append(rbs.iloc[0])
    if len(qbs) >= 1:
        skill.append(qbs.iloc[0])
    if len(fbs) >= 1:
        skill.append(fbs.iloc[0])
    if len(wrs) >= 2:
        skill.append(wrs.iloc[1])
    if len(wrs) >= 3:
        skill.append(wrs.iloc[2])
    _formation_row(skill, cat_df, photo_map)


def _grouped_formation(cat_df: pd.DataFrame, groups: list, photo_map: dict):
    starters = cat_df[cat_df["is_starter"] == True]  # noqa: E712
    present = set(cat_df["position"].unique())
    covered = {p for _, ps in groups for p in ps}
    uncovered = sorted(present - covered)
    for group_name, canonical_positions in groups + [("Other", uncovered)]:
        positions_here = [p for p in canonical_positions
                           if p in present and p in starters["position"].values]
        if not positions_here:
            continue
        st.markdown(f"##### {group_name}")
        row = [starters[starters["position"] == p].iloc[0] for p in positions_here]
        _formation_row(row, cat_df, photo_map)


def page_depth_charts():
    st.title("🏈 Team Depth Charts")
    st.caption(
        "Each team's real starters, laid out like a formation rather than a flat "
        "table -- a 3-4 team's line is 3 wide because that's what's actually on "
        "their depth chart, not a template. Click any player for their injury "
        "status, model predictions, current Underdog lines, and recent games. "
        "Source: Footballguys, with structured status tags (Q/PUP/IR/SUS/NFI/O) "
        "that update faster than the official injury report."
    )
    df = load_csv_if_exists("footballguys_depth.csv")

    if df is None:
        st.warning("`footballguys_depth.csv` hasn't been pulled yet. Run it from **Run Data Pulls**.")
        return

    photo_map = load_player_photos()

    team_names = df.drop_duplicates("team_abbr").set_index("team_name")["team_abbr"].sort_index()
    chosen_team_name = st.selectbox("Team", team_names.index)
    team_abbr = team_names[chosen_team_name]
    team_df = df[df["team_abbr"] == team_abbr]

    tabs = st.tabs(["Offense", "Defense", "Special Teams"])
    with tabs[0]:
        cat_df = team_df[team_df["category"] == "offense"]
        if cat_df.empty:
            st.caption("No data pulled for this unit.")
        else:
            _offense_formation(cat_df, photo_map)
    with tabs[1]:
        cat_df = team_df[team_df["category"] == "defense"]
        if cat_df.empty:
            st.caption("No data pulled for this unit.")
        else:
            _grouped_formation(cat_df, DEFENSE_GROUPS, photo_map)
    with tabs[2]:
        cat_df = team_df[team_df["category"] == "special_teams"]
        if cat_df.empty:
            st.caption("No data pulled for this unit.")
        else:
            _grouped_formation(cat_df, SPECIAL_TEAMS_GROUPS, photo_map)


def page_underdog_props():
    st.title("💰 Underdog Pick'em Props")
    df = load_csv_if_exists("underdog_props.csv")

    if df is None:
        st.warning("`underdog_props.csv` hasn't been pulled yet. Run it from **Run Data Pulls**.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            stat_col = "stat_name" if "stat_name" in df.columns else None
            stats = sorted(df[stat_col].dropna().unique()) if stat_col else []
            stat_filter = st.multiselect("Stat type", stats)
        with col2:
            search = st.text_input("Search player name")

        filtered = df.copy()
        if stat_filter and stat_col:
            filtered = filtered[filtered[stat_col].isin(stat_filter)]
        if search and "full_name" in filtered.columns:
            filtered = filtered[filtered["full_name"].astype(str).str.contains(search, case=False, na=False)]

        st.caption(f"{len(filtered):,} of {len(df):,} prop options")
        st.dataframe(filtered, width='stretch', height=500)


def _news_page(title: str, icon: str, caption: str, filename: str, group_col: str, group_label: str,
                caption_cols: tuple[str, ...] = ("author", "published")):
    """Shared layout for the three news feed pages -- same filter/card pattern, only
    the source file, the column used to group by (team vs. feed), and which extra
    columns are meaningful to show in each card's caption differ. caption_cols is
    explicit per page rather than guessed generically: sbnation_news.csv's "source"
    column is a raw blog URL (not meant for display), while espn_news.csv's "source"
    is a real feed-name label meant to be shown -- same column name, different
    meaning, so this can't be inferred safely from column presence alone."""
    st.title(f"{icon} {title}")
    st.caption(caption)
    df = load_csv_if_exists(filename)

    if df is None:
        st.warning(f"`{filename}` hasn't been pulled yet. Run it from **Run Data Pulls**.")
        return

    col1, col2 = st.columns(2)
    with col1:
        groups = sorted(df[group_col].dropna().unique()) if group_col in df.columns else []
        group_filter = st.multiselect(group_label, groups)
    with col2:
        search = st.text_input("Search headlines/summary", key=f"search_{filename}")

    filtered = df.copy()
    if group_filter:
        filtered = filtered[filtered[group_col].isin(group_filter)]
    if search:
        text_cols = [c for c in ("headline", "description", "summary") if c in filtered.columns]
        mask = pd.Series(False, index=filtered.index)
        for c in text_cols:
            mask |= filtered[c].astype(str).str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    if "published" in filtered.columns:
        filtered = filtered.sort_values("published", ascending=False)
    st.caption(f"{len(filtered):,} of {len(df):,} items")
    for _, row in filtered.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row.get('headline', '(no headline)')}**")
            parts = [str(row[c]) for c in (group_col, *caption_cols) if c in row.index and pd.notna(row.get(c))]
            st.caption(" · ".join(dict.fromkeys(parts)))  # dedupe while preserving order
            body = row.get("summary") if pd.notna(row.get("summary")) else row.get("description")
            if pd.notna(body):
                st.write(body)
            if pd.notna(row.get("athletes_tagged")) and row.get("athletes_tagged"):
                st.caption(f"Athletes: {row['athletes_tagged']}")
            if pd.notna(row.get("link")):
                st.markdown(f"[Read more]({row['link']})")


def page_espn_news():
    _news_page("ESPN News Feed", "📰",
               "League-wide news from ESPN's public RSS feed (their JSON API is blocked by "
               "ESPN's own edge/bot protection, unrelated to per-team breakdowns — see "
               "pull_espn_news.py for details).",
               "espn_news.csv", "source", "Source", caption_cols=("author", "published"))


def page_sbnation_news():
    _news_page("SB Nation Team News", "📰",
               "Daily pull from all 32 teams' SB Nation blogs — per-team breakdown ESPN's feed doesn't provide.",
               "sbnation_news.csv", "team", "Team", caption_cols=("author", "published"))


def page_nbc_news():
    _news_page(
        "NBC Sports / ProFootballTalk Rumor Mill", "📰",
        "Short, atomic insider-sourced roster/injury items (Mike Florio, Charean "
        "Williams, etc.) -- different in kind from SB Nation's longer team recaps, "
        "closer to \"ahead of the official injury report.\" Only ~4 items per feed per "
        "pull, so this accumulates over time rather than being a deep archive.",
        "nbcsports_news.csv", "feed", "Feed",
    )


# ==================================================================== Admin

def page_run_pulls():
    st.title("🔄 Run Data Pulls")
    st.caption("Runs the actual pull scripts. Output streams below once finished (can take a minute).")

    if st.button("▶️ Run All", type="primary", width="stretch"):
        n = len(PULL_SCRIPTS)
        progress = st.progress(0.0, text="Starting...")
        results = []
        for i, (label, cmd) in enumerate(PULL_SCRIPTS.items()):
            progress.progress(i / n, text=f"Running {label}... ({i + 1}/{n})")
            success, output = run_pull_script(cmd)
            results.append((label, success, output))
        progress.progress(1.0, text="Done.")

        n_ok = sum(1 for _, success, _ in results if success)
        if n_ok == n:
            st.success(f"All {n} pulls completed.")
        else:
            st.warning(f"{n_ok} of {n} pulls completed; {n - n_ok} had errors.")
        for label, success, output in results:
            with st.expander(f"{'✅' if success else '❌'} {label}"):
                st.code(output or "(no output)")

    yard_divider("MANUAL PULLS")

    for label, cmd in PULL_SCRIPTS.items():
        if st.button(f"Run: {label}"):
            with st.spinner(f"Running {label}..."):
                success, output = run_pull_script(cmd)
            if success:
                st.success(f"{label} completed.")
            else:
                st.error(f"{label} failed or had errors.")
            st.code(output or "(no output)")


# ==================================================================== Navigation
# Named individually (not inline in the dict below) so page functions can reference
# a specific Page object for st.switch_page -- e.g. the Weekly Bet Slip page's
# "go build your parlay" button needs PAGE_PARLAY_BUILDER by name. These assignments
# run before nav.run() executes any page body, so the reference is always valid by
# the time a page function actually uses it.

PAGE_WEEKLY_BET_SLIP = st.Page(page_weekly_bet_slip, title="Weekly Bet Slip", icon="🎯", default=True)
PAGE_PARLAY_BUILDER = st.Page(page_parlay_builder, title="Parlay Builder", icon="🧩")
PAGE_BET_LOG = st.Page(page_bet_log, title="Bet Log", icon="📒")
PAGE_UNDERDOG_PROPS = st.Page(page_underdog_props, title="Underdog Props", icon="💰")
PAGE_DEPTH_CHARTS = st.Page(page_depth_charts, title="Depth Charts", icon="🏈")
PAGE_NFL_STATS = st.Page(page_nfl_stats, title="NFL Stats", icon="📊")
PAGE_SBNATION_NEWS = st.Page(page_sbnation_news, title="SB Nation News", icon="📰")
PAGE_NBC_NEWS = st.Page(page_nbc_news, title="NBC/PFT Rumor Mill", icon="📰")
PAGE_ESPN_NEWS = st.Page(page_espn_news, title="ESPN News", icon="📰")
PAGE_OVERVIEW = st.Page(page_overview, title="Data Status", icon="🗂️")
PAGE_RUN_PULLS = st.Page(page_run_pulls, title="Run Data Pulls", icon="🔄")

nav = st.navigation({
    "Betting": [PAGE_WEEKLY_BET_SLIP, PAGE_PARLAY_BUILDER, PAGE_BET_LOG],
    "Research": [PAGE_UNDERDOG_PROPS, PAGE_DEPTH_CHARTS, PAGE_NFL_STATS,
                 PAGE_SBNATION_NEWS, PAGE_NBC_NEWS, PAGE_ESPN_NEWS],
    "Admin": [PAGE_OVERVIEW, PAGE_RUN_PULLS],
})

st.sidebar.divider()
st.sidebar.caption("ParlayModel")

nav.run()
