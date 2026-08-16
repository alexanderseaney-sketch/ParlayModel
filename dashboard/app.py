"""
ParlayModel Dashboard — local UI for managing project data.

Run with:
    streamlit run dashboard/app.py
"""
from datetime import datetime, date

import pandas as pd
import streamlit as st

from utils import (
    EXPECTED_FILES, PULL_SCRIPTS,
    file_status, load_csv_if_exists, load_bet_log, append_bet, run_pull_script,
    find_column, load_current_predictions, normalize_name,
    correlation_adjusted_parlay_probability,
)

st.set_page_config(page_title="ParlayModel Dashboard", layout="wide")


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

PAGE = st.sidebar.radio(
    "Navigate",
    ["Overview", "Parlay Builder", "NFL Stats", "Depth Charts", "ESPN News", "SB Nation News", "NBC/PFT Rumor Mill", "Underdog Props", "Bet Log", "Run Data Pulls"],
)

st.sidebar.markdown("---")
st.sidebar.caption("ParlayModel — local data management UI")


# ---------------------------------------------------------------- Overview
if PAGE == "Overview":
    st.title("Data Status")
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


# ---------------------------------------------------------------- Parlay Builder
elif PAGE == "Parlay Builder":
    st.title("Parlay Builder")
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

    if "slip" not in st.session_state:
        st.session_state.slip = []

    # --- Confidence filter ---
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

    st.markdown("---")

    # --- Add a leg ---
    with st.expander("➕ Add a leg", expanded=len(st.session_state.slip) == 0):
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
            search = st.text_input("Search player")
            options_df = df[df[name_col].astype(str).str.contains(search, case=False, na=False)] if search else df.head(50)

            if predictions is not None:
                options_df = options_df.copy()
                options_df["_match_key"] = options_df[name_col].apply(normalize_name)
                # Match on player AND stat type -- name-only matching would attach e.g.
                # a receiving-yards prediction to that same player's rushing-yards prop
                # now that current_predictions.py covers four different prop types.
                options_df = options_df.merge(
                    predictions[["_match_key", "stat_name", "predicted_prob_over", "confidence",
                                  "stats_as_of_season", "stats_as_of_week", "recent_team", "position", "prop_type"]],
                    left_on=["_match_key", stat_col], right_on=["_match_key", "stat_name"],
                    how="left",
                )
                below_bar = options_df["confidence"] < min_confidence
                if below_bar.any() and not search:
                    st.caption(f"{below_bar.sum()} props below the confidence bar are hidden. Search or lower the bar to see them.")
                options_df = options_df[~below_bar | options_df["confidence"].isna() | (search != "")]

            for idx, row in options_df.iterrows():
                model_prob = row.get("predicted_prob_over") if predictions is not None else None
                has_model = pd.notna(model_prob) if model_prob is not None else False

                label = f"{row[name_col]} — {row.get(stat_col, '?')} {row.get(choice_col, '')} {row.get(line_col, '')}"
                if has_model:
                    conf = row["confidence"]
                    tag = "🟢" if conf >= 0.4 else ("🟡" if conf >= 0.2 else "⚪")
                    label += f"  {tag} model: {model_prob*100:.0f}% (confidence {conf:.2f})"
                    stats_season = row.get("stats_as_of_season")
                    if pd.notna(stats_season):
                        stats_week = int(row["stats_as_of_week"])
                        if stats_season < freshest_season:
                            label += f"  ⚠️ form as of {int(stats_season)} wk{stats_week} (stale — no games since)"
                        else:
                            label += f"  · form as of {int(stats_season)} wk{stats_week}"

                if depth_status is not None:
                    fbg_status = depth_status.get(normalize_name(row[name_col]))
                    if fbg_status:
                        label += f"  🚑 {fbg_status} (Footballguys, today)"

                c1, c2 = st.columns([4, 1])
                c1.write(label)
                if c2.button("Add", key=f"add_{idx}"):
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

        if not mult_col:
            st.info(
                "Note: no obvious odds/payout column found in the pulled data yet — "
                "re-check this once `pull_underdog.py` has run against live data."
            )

    st.markdown("---")

    # --- Current slip ---
    if not st.session_state.slip:
        st.info("No legs added yet.")
    else:
        st.subheader(f"Current slip ({len(st.session_state.slip)} legs)")

        for i, leg in enumerate(st.session_state.slip):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            c1.write(f"**{leg['player']}** — {leg['stat']} {leg['choice']} {leg['line']}")
            leg["my_prob"] = c2.slider(
                "Win prob.", 0.0, 1.0, leg["my_prob"], 0.01, key=f"prob_{i}", label_visibility="collapsed",
            )
            fair_mult = 1 / leg["my_prob"] if leg["my_prob"] > 0 else float("inf")
            ud_mult = leg["underdog_multiplier"]
            if ud_mult:
                edge = "✅ +EV" if fair_mult < float(ud_mult) else "⚠️ -EV"
                c3.write(f"UD: {ud_mult}x · fair: {fair_mult:.2f}x · {edge}")
            else:
                c3.write(f"fair: {fair_mult:.2f}x (UD odds unknown)")
            if c4.button("Remove", key=f"rm_{i}"):
                st.session_state.slip.pop(i)
                st.rerun()

        st.markdown("---")

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
                "game) -- the naive independence math below is wrong for this slip. Using "
                "real measured correlations instead (see models/analyze_parlay_correlations.py):"
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

        st.markdown("---")
        st.caption(
            "Placement isn't automated yet (Phase 5 — Claude in Chrome, home only, "
            "human-approved each time). For now this gives you a clean slip to place manually."
        )
        if st.button("Clear slip"):
            st.session_state.slip = []
            st.rerun()



elif PAGE == "NFL Stats":
    st.title("NFL Stats (nflverse)")

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


# ---------------------------------------------------------------- Depth Charts
elif PAGE == "Depth Charts":
    st.title("Depth Charts (Footballguys)")
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


# ---------------------------------------------------------------- ESPN News
elif PAGE == "ESPN News":
    st.title("ESPN News Feed")
    df = load_csv_if_exists("espn_news.csv")

    if df is None:
        st.warning("`espn_news.csv` hasn't been pulled yet. Run it from **Run Data Pulls**.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sources = sorted(df["source"].dropna().unique()) if "source" in df.columns else []
            source_filter = st.multiselect("Source (league / team:XXX)", sources)
        with col2:
            search = st.text_input("Search headlines/description")

        filtered = df.copy()
        if source_filter:
            filtered = filtered[filtered["source"].isin(source_filter)]
        if search:
            mask = (
                filtered["headline"].astype(str).str.contains(search, case=False, na=False)
                | filtered["description"].astype(str).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]

        st.caption(f"{len(filtered):,} of {len(df):,} articles")
        for _, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row.get('headline', '(no headline)')}**")
                st.caption(f"{row.get('source', '')} · {row.get('published', '')}")
                if pd.notna(row.get("description")):
                    st.write(row["description"])
                if pd.notna(row.get("athletes_tagged")) and row.get("athletes_tagged"):
                    st.caption(f"Athletes: {row['athletes_tagged']}")
                if pd.notna(row.get("link")):
                    st.markdown(f"[Read more]({row['link']})")


# ---------------------------------------------------------------- SB Nation News
elif PAGE == "SB Nation News":
    st.title("SB Nation Team News")
    st.caption("Daily pull from all 32 teams' SB Nation blogs — replaces ESPN news, which is blocked from this environment.")
    df = load_csv_if_exists("sbnation_news.csv")

    if df is None:
        st.warning("`sbnation_news.csv` hasn't been pulled yet. Run it from **Run Data Pulls**.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            teams = sorted(df["team"].dropna().unique()) if "team" in df.columns else []
            team_filter = st.multiselect("Team", teams)
        with col2:
            search = st.text_input("Search headlines/summary")

        filtered = df.copy()
        if team_filter:
            filtered = filtered[filtered["team"].isin(team_filter)]
        if search:
            mask = (
                filtered["headline"].astype(str).str.contains(search, case=False, na=False)
                | filtered["summary"].astype(str).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]

        filtered = filtered.sort_values("published", ascending=False)
        st.caption(f"{len(filtered):,} of {len(df):,} articles")
        for _, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row.get('headline', '(no headline)')}**")
                st.caption(f"{row.get('team', '')} · {row.get('author', '')} · {row.get('published', '')}")
                if pd.notna(row.get("summary")):
                    st.write(row["summary"])
                if pd.notna(row.get("link")):
                    st.markdown(f"[Read more]({row['link']})")


# ---------------------------------------------------------------- NBC/PFT Rumor Mill
elif PAGE == "NBC/PFT Rumor Mill":
    st.title("NBC Sports / ProFootballTalk Rumor Mill")
    st.caption(
        "Short, atomic insider-sourced roster/injury items (Mike Florio, Charean "
        "Williams, etc.) -- different in kind from SB Nation's longer team recaps, "
        "closer to \"ahead of the official injury report.\" Only ~4 items per feed per "
        "pull, so this accumulates over time rather than being a deep archive."
    )
    df = load_csv_if_exists("nbcsports_news.csv")

    if df is None:
        st.warning("`nbcsports_news.csv` hasn't been pulled yet. Run it from **Run Data Pulls**.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            feeds = sorted(df["feed"].dropna().unique()) if "feed" in df.columns else []
            feed_filter = st.multiselect("Feed", feeds)
        with col2:
            search = st.text_input("Search headlines/summary")

        filtered = df.copy()
        if feed_filter:
            filtered = filtered[filtered["feed"].isin(feed_filter)]
        if search:
            mask = (
                filtered["headline"].astype(str).str.contains(search, case=False, na=False)
                | filtered["summary"].astype(str).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]

        filtered = filtered.sort_values("published", ascending=False)
        st.caption(f"{len(filtered):,} of {len(df):,} items")
        for _, row in filtered.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row.get('headline', '(no headline)')}**")
                parts = [str(row[c]) for c in ("feed", "author", "published") if pd.notna(row.get(c))]
                st.caption(" · ".join(parts))
                if pd.notna(row.get("summary")):
                    st.write(row["summary"])
                if pd.notna(row.get("link")):
                    st.markdown(f"[Read more]({row['link']})")


# ---------------------------------------------------------------- Underdog Props
elif PAGE == "Underdog Props":
    st.title("Underdog Pick'em Props")
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


# ---------------------------------------------------------------- Bet Log
elif PAGE == "Bet Log":
    st.title("Bet Log")
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
    else:
        c1, c2, c3 = st.columns(3)
        n_pending = (bets["result"] == "pending").sum()
        n_won = (bets["result"] == "won").sum()
        n_lost = (bets["result"] == "lost").sum()
        c1.metric("Pending", n_pending)
        c2.metric("Won", n_won)
        c3.metric("Lost", n_lost)
        st.dataframe(bets.sort_values("date", ascending=False), width='stretch', height=400)


# ---------------------------------------------------------------- Run Data Pulls
elif PAGE == "Run Data Pulls":
    st.title("Run Data Pulls")
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
