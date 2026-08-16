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
    EXPECTED_FILES, PULL_SCRIPTS, MAX_LINE_DIVERGENCE, BET_LOG_PATH,
    file_status, load_csv_if_exists, load_bet_log, append_bet, run_pull_script,
    find_column, load_current_predictions, normalize_name,
    correlation_adjusted_parlay_probability,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from generate_weekly_bet_slip import (  # noqa: E402
    load_matched_props, build_single_leg_candidates, build_parlay_candidates, allocate_budget,
)

st.set_page_config(page_title="ParlayModel", page_icon="🏈", layout="wide")


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

    st.divider()
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
            head, action = st.columns([5, 1])
            with head:
                st.markdown(f"**${c['suggested_stake']:.2f}** — {c['description']}")
                b1, b2, b3 = st.columns(3)
                b1.badge(f"{c['type']}", color="violet")
                b2.badge(f"model {c['model_prob']*100:.1f}%", color="blue")
                b3.badge(f"edge {edge*100:+.1f}%", color="green" if edge > 0 else "red")
                st.caption(f"real price: {c['decimal_odds']:.2f}x")
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
    else:
        depth_status = None

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
        mult_col = find_column(df, ["payout_multiplier", "multiplier", "odds", "american_price"])

        if not name_col or not stat_col:
            st.error(
                "Couldn't find the expected player/stat columns in underdog_props.csv. "
                f"Actual columns: {list(df.columns)}"
            )
        else:
            search = st.text_input("Search player", placeholder="Type a name and press Enter…")
            options_df = df[df[name_col].astype(str).str.contains(search, case=False, na=False)] if search else df.head(50)

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
                if below_bar.any() and not search:
                    st.caption(f"{below_bar.sum()} props below the confidence bar are hidden. Search or lower the bar to see them.")
                options_df = options_df[~below_bar | options_df["confidence"].isna() | (search != "")]

            if not mult_col:
                st.info(
                    "Note: no obvious odds/payout column found in the pulled data yet — "
                    "re-check this once `pull_underdog.py` has run against live data."
                )

            for idx, row in options_df.iterrows():
                # Two real bugs fixed here 2026-08-15 (found while building the
                # weekly bet-slip generator, see README):
                # 1. predicted_prob_over was used directly regardless of which side
                #    (over/under) this specific row actually is -- a confident UNDER
                #    prediction was shown as if it endorsed the OVER row it happened
                #    to be attached to. Now flipped (1 - p) when this row's choice is
                #    "under".
                # 2. predicted_prob_over answers "beats OUR proxy line", not "beats
                #    Underdog's posted line" -- only valid when those two numbers are
                #    close (see MAX_LINE_DIVERGENCE in utils.py). Now gated.
                raw_prob_over = row.get("predicted_prob_over") if predictions is not None else None
                choice_lower = str(row.get(choice_col, "")).lower()
                if pd.notna(raw_prob_over) and choice_lower in ("over", "under"):
                    model_prob = raw_prob_over if choice_lower == "over" else 1 - raw_prob_over
                else:
                    model_prob = None

                proxy_line, line_value = row.get("proxy_line"), row.get(line_col)
                line_ok = True
                if model_prob is not None and pd.notna(proxy_line) and pd.notna(line_value) and proxy_line:
                    line_ok = abs(float(line_value) - float(proxy_line)) / abs(float(proxy_line)) <= MAX_LINE_DIVERGENCE

                has_model = model_prob is not None and line_ok
                fbg_status = depth_status.get(normalize_name(row[name_col])) if depth_status is not None else None

                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    with left:
                        st.markdown(f"**{row[name_col]}** — {row.get(stat_col, '?')} {row.get(choice_col, '')} {row.get(line_col, '')}")
                        badges = st.columns(4)
                        b_i = 0
                        if has_model:
                            with badges[b_i]:
                                _confidence_badge(model_prob, row["confidence"])
                            b_i += 1
                            stats_season = row.get("stats_as_of_season")
                            if pd.notna(stats_season):
                                stats_week = int(row["stats_as_of_week"])
                                if stats_season < freshest_season:
                                    with badges[b_i]:
                                        st.badge(f"stale: {int(stats_season)} wk{stats_week}", icon="⚠️", color="red")
                                    b_i += 1
                        elif model_prob is not None and not line_ok:
                            with badges[b_i]:
                                st.badge(f"line mismatch (proxy {proxy_line:.1f})", icon="⚠️", color="red",
                                         help=f"Our proxy line ({proxy_line:.1f}) is too far from Underdog's real line ({line_value}) to trust this prediction for this specific bet.")
                            b_i += 1
                        if fbg_status:
                            with badges[b_i]:
                                st.badge(fbg_status, icon="🚑", color="red", help="Footballguys depth chart status, pulled today")
                            b_i += 1
                    with right:
                        if st.button("➕ Add", key=f"add_{idx}", width="stretch"):
                            position_prop = None
                            if has_model and pd.notna(row.get("position")) and pd.notna(row.get("prop_type")):
                                position_prop = f"{row['position']} {row['prop_type']}"
                            st.session_state.slip.append({
                                "player": row[name_col],
                                "stat": row.get(stat_col),
                                "choice": row.get(choice_col, ""),
                                "line": row.get(line_col),
                                "underdog_multiplier": row.get(mult_col) if mult_col else None,
                                "my_prob": float(model_prob) if has_model else 0.55,
                                "team": row.get("recent_team") if has_model else None,
                                "position_prop": position_prop,
                            })
                            st.rerun()

    with tab_slip:
        if not st.session_state.slip:
            st.info("No legs added yet — head to the **➕ Add legs** tab, or generate suggestions on the **Weekly Bet Slip** page and send them here.")
            return

        for i, leg in enumerate(st.session_state.slip):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                c1.markdown(f"**{leg['player']}**  \n{leg['stat']} {leg['choice']} {leg['line']}")
                leg["my_prob"] = c2.slider(
                    "Win prob.", 0.0, 1.0, leg["my_prob"], 0.01, key=f"prob_{i}", label_visibility="collapsed",
                )
                fair_mult = 1 / leg["my_prob"] if leg["my_prob"] > 0 else float("inf")
                ud_mult = leg["underdog_multiplier"]
                with c3:
                    if ud_mult:
                        st.caption(f"UD: {ud_mult}x · fair: {fair_mult:.2f}x")
                        if fair_mult < float(ud_mult):
                            st.badge("+EV", color="green")
                        else:
                            st.badge("-EV", color="red")
                    else:
                        st.caption(f"fair: {fair_mult:.2f}x (UD odds unknown)")
                if c4.button("🗑️", key=f"rm_{i}", help="Remove this leg", width="stretch"):
                    st.session_state.slip.pop(i)
                    st.rerun()

        st.divider()

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

        st.divider()
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

    editable = bets.sort_values("date", ascending=False).reset_index(drop=True)
    st.caption("Update results directly in the table below (double-click a Result cell).")
    edited = st.data_editor(
        editable, width="stretch", height=400, hide_index=True, key="bet_log_editor",
        column_config={"result": st.column_config.SelectboxColumn(options=["pending", "won", "lost", "push"])},
    )
    if not edited.equals(editable):
        edited.to_csv(BET_LOG_PATH, index=False)
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


def page_depth_charts():
    st.title("🏈 Depth Charts (Footballguys)")
    st.caption(
        "All 32 teams, offense + defense + special teams, with structured per-player "
        "status tags (Q/PUP/IR/SUS/NFI/O) -- freer and faster-updating than the "
        "official injury report, which is often incomplete this early in a season."
    )
    df = load_csv_if_exists("footballguys_depth.csv")

    if df is None:
        st.warning("`footballguys_depth.csv` hasn't been pulled yet. Run it from **Run Data Pulls**.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            teams = sorted(df["team_abbr"].dropna().unique())
            team_filter = st.multiselect("Team", teams)
        with col2:
            categories = sorted(df["category"].dropna().unique())
            category_filter = st.multiselect("Category", categories, default=["offense"])
        with col3:
            only_flagged = st.checkbox("Only players with a status tag", value=False)

        filtered = df.copy()
        if team_filter:
            filtered = filtered[filtered["team_abbr"].isin(team_filter)]
        if category_filter:
            filtered = filtered[filtered["category"].isin(category_filter)]
        if only_flagged:
            filtered = filtered[filtered["status"].notna()]

        st.caption(f"{len(filtered):,} of {len(df):,} rows")
        st.dataframe(
            filtered[["team_abbr", "position", "depth_rank", "is_starter", "player_name", "status"]]
            .sort_values(["team_abbr", "position", "depth_rank"]),
            width='stretch', height=500, hide_index=True,
        )


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
