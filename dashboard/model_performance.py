"""
Model Performance page (added 2026-10-02): what drives each prop model, how well
calibrated its probabilities are, and how it scores against REAL Underdog lines.

Reads only files produced offline -- no training at render time:
  models/calibrators/*.pkl          -- train_*.py via calibration.fit_and_save()
  models/feature_importance.csv     -- current_predictions.py
  backtesting/underdog_line_backtest_summary.json -- backtest_underdog_lines.py
"""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ROOT_DIR, pretty_stat_name
from calibration import load_all_metrics

C_RAW, C_CAL, C_MARKET = "#3987e5", "#199e70", "#d95926"
TEXT, MUTED, GRID = "#e9e9ed", "#9a9aa6", "rgba(233,233,237,0.08)"
BACKTEST_JSON = os.path.join(ROOT_DIR, "backtesting", "underdog_line_backtest_summary.json")
IMPORTANCE_CSV = os.path.join(ROOT_DIR, "models", "feature_importance.csv")


def _layout(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=28, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT, family="Inter, sans-serif", size=12),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _feature_chart(fi: pd.DataFrame, prop: str):
    d = fi[fi["prop_type"] == prop].sort_values("importance").tail(12)
    fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h",
                           marker=dict(color=C_RAW, cornerradius=4),
                           hovertemplate="%{y}<br>%{x:.1%} of total importance<extra></extra>"))
    _layout(fig, height=60 + 28 * len(d), xaxis=dict(tickformat=".0%"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _reliability_chart(m: dict):
    curve = pd.DataFrame(m.get("reliability") or [])
    if curve.empty:
        st.caption("No reliability points saved.")
        return
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dot"),
                    name="Perfect", hoverinfo="skip")
    fig.add_scatter(x=curve["raw_mean"], y=curve["actual"], mode="lines+markers", name="Raw model",
                    line=dict(color=C_RAW, width=2), marker=dict(size=8),
                    customdata=curve["n"], hovertemplate="Predicted %{x:.1%} → actual %{y:.1%}<br>n=%{customdata}<extra>Raw</extra>")
    if m.get("method") != "none":
        fig.add_scatter(x=curve["cal_mean"], y=curve["actual"], mode="lines+markers", name="Calibrated",
                        line=dict(color=C_CAL, width=2), marker=dict(size=8),
                        customdata=curve["n"], hovertemplate="Predicted %{x:.1%} → actual %{y:.1%}<br>n=%{customdata}<extra>Calibrated</extra>")
    _layout(fig, xaxis=dict(range=[0, 1], tickformat=".0%", title="Predicted P(over proxy)"),
            yaxis=dict(range=[0, 1], tickformat=".0%", title="Actual hit rate"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_model_performance():
    st.title("🔬 Model Performance")
    st.caption("Calibration on leave-one-season-out holdouts, what drives each model, "
               "and a point-in-time backtest against real Underdog lines.")

    metrics = load_all_metrics()
    fi = pd.read_csv(IMPORTANCE_CSV) if os.path.exists(IMPORTANCE_CSV) else pd.DataFrame()
    props = sorted(set(metrics) | set(fi["prop_type"].unique() if not fi.empty else []))

    tab_bt, tab_cal, tab_fi = st.tabs(["📉 Real-line backtest", "🎯 Calibration", "🧩 Feature drivers"])

    with tab_bt:
        if not os.path.exists(BACKTEST_JSON):
            st.info("Run `python backtesting/backtest_underdog_lines.py` to grade the model against "
                    "archived Underdog lines.")
        else:
            with open(BACKTEST_JSON) as f:
                bt = json.load(f)
            o = bt.get("overall", {})
            st.caption(f"Graded weeks: {', '.join(bt.get('weeks', []))} · {o.get('n', 0)} props "
                       f"(over side, model + market both available) · generated {bt.get('generated_at', '')[:16]}")
            if o.get("n"):
                c = st.columns(3)
                for col, name in zip(c, ("model", "market", "blend")):
                    col.metric(f"{name.title()} log loss", f"{o[name + '_log_loss']:.4f}",
                               help=f"Brier {o[name + '_brier']:.4f} · lower is better")
                rows = [{"Stat": pretty_stat_name(s), "n": m["n"], "Model LL": m["model_log_loss"],
                         "Market LL": m["market_log_loss"], "Blend LL": m["blend_log_loss"],
                         "Model Brier": m["model_brier"], "Market Brier": m["market_brier"]}
                        for s, m in bt.get("by_stat", {}).items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                             column_config={k: st.column_config.NumberColumn(format="%.4f")
                                            for k in ("Model LL", "Market LL", "Blend LL", "Model Brier", "Market Brier")})
            bets = pd.DataFrame([b for b in bt.get("betting", []) if b.get("n")])
            if not bets.empty:
                st.markdown("**Flat $1 singles at Underdog prices, by minimum model edge over market**")
                fig = go.Figure(go.Bar(x=[f">{b:.0%}" for b in bets["min_edge"]], y=bets["roi"],
                                       marker=dict(color=[C_CAL if r > 0 else C_MARKET for r in bets["roi"]], cornerradius=4),
                                       customdata=bets[["n", "hit_rate"]].to_numpy(),
                                       hovertemplate="Edge %{x}<br>ROI %{y:+.1%}<br>n=%{customdata[0]} · hit %{customdata[1]:.1%}<extra></extra>"))
                _layout(fig, height=280, yaxis=dict(tickformat="+.0%", title="ROI"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption("Small sample — the line archive only covers the weeks listed above. "
                           "Treat as a sanity check, not proof of edge.")

    with tab_cal:
        if not metrics:
            st.info("No calibrators yet — re-run the train_*.py scripts.")
        else:
            rows = [{"Prop": p, "Method": m["method"], "Holdout n": m["n"],
                     "Log loss (raw)": m["raw_log_loss"], "Log loss (cal)": m["cal_log_loss"],
                     "Brier (raw)": m["raw_brier"], "Brier (cal)": m["cal_brier"]} for p, m in metrics.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(format="%.4f") for k in
                                        ("Log loss (raw)", "Log loss (cal)", "Brier (raw)", "Brier (cal)")})
            st.caption("Calibrated scores are cross-fitted (fit on other folds, scored on the held-out fold). "
                       "'none' = no map beat the raw output, so it's left untouched.")
            pick = st.selectbox("Reliability curve", list(metrics), key="cal_pick")
            _reliability_chart(metrics[pick])

    with tab_fi:
        if fi.empty:
            st.info("Run `python models/current_predictions.py` to write feature_importance.csv.")
        else:
            pick = st.selectbox("Prop model", sorted(fi["prop_type"].unique()), key="fi_pick")
            _feature_chart(fi, pick)
            st.caption("XGBoost: mean gain importance over the 100-model bootstrap. Logistic models: "
                       "|coefficient| × the feature's spread across currently scored players.")
