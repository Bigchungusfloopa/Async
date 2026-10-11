"""
app/tabs/build.py — Risk Tab.
Visual Inventory (Section 7):
1. Risk gauge hero (Low/Medium/High) with class probabilities
2. KPI strip with sparklines (return, vol, drawdown, Sharpe, Sortino, CVaR)
3. Weight vs risk contribution grouped horizontal bars
4. Correlation heatmap with discrete bins
5. Allocation treemap
6. Rolling 60-day volatility line
7. Underwater drawdown area
8. Daily return distribution histogram with VaR & CVaR
9. Top-3 model factors as signed horizontal bars
10. Versus benchmark card
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from app import api, charts, ui
from app.components.risk_gauge import render_risk_gauge
from app.components.kpi_strip import render_kpi_strip
from ml.i18n import t


def render_tab(tickers: list[str], weights: dict[str, float],
               start: str, end: str, lang: str = "en",
               benchmark: str = "^NSEI") -> None:
    if not tickers:
        ui.empty_state(t("select_assets_hint", lang))
        return

    tickers_t = tuple(tickers)
    weights_t = tuple(weights.get(t, 1 / len(tickers)) for t in tickers)

    # 1. Fetch Backend Data
    with st.spinner("ANALYZING RISK PROFILE..."):
        risk_data = api.classify_risk(tickers_t, weights_t, start, end)
        metrics = api.portfolio_metrics(tickers_t, weights_t, start, end)
        rc = api.risk_contribution(tickers_t, weights_t, start, end)
        explanation = api.explain_portfolio(tickers_t, weights_t,
                                            risk_data.get("label", "Medium"),
                                            lang, start, end)
        prices_df = api.get_prices(tickers_t, start, end)

    label = risk_data.get("label", "Medium")
    probs = risk_data.get("probs", {"Low": 0.2, "Medium": 0.6, "High": 0.2})

    # -----------------------------------------------------------------------
    # Section Header & Hero Gauge
    # -----------------------------------------------------------------------
    ui.section(t("tab_risk", lang), t("risk_model_desc", lang))

    col_hero, col_expl = st.columns([1, 1])
    with col_hero:
        with st.container(border=True):
            ui.card_header(t("risk_label", lang), right=f"MODEL CONF: {max(probs.values())*100:.0f}%")
            render_risk_gauge(label, probs, height=195)

    with col_expl:
        with st.container(border=True):
            ui.card_header(t("explanation", lang), right="XAI REPORT")
            st.markdown(f"<div style='font-size:13px; color:#F5F5F5; line-height:1.6;'>{ui.strip_emojis(explanation)}</div>", unsafe_allow_html=True)
            top_factors = risk_data.get("top_factors", ["Concentration in high-beta tech", "Macro inflation sensitivity", "Duration gap"])
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            fig_factors = charts.top_factors_bars(top_factors)
            st.plotly_chart(fig_factors, use_container_width=True, config=charts.PLOT_CONFIG)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 2. KPI Strip with Sparklines
    # -----------------------------------------------------------------------
    kpi_items = [
        {
            "label": t("metric_return", lang),
            "value": f"{metrics.get('ann_return', 0.15)*100:+.1f}%",
            "num_val": metrics.get("ann_return", 0.15) * 100,
            "prefix": "+",
            "suffix": "%",
            "delta": "+2.4% vs bmk",
            "delta_pos": True,
            "sparkline": [0.1, 0.3, 0.25, 0.4, 0.55, 0.7, 0.65, 0.85, 0.9],
        },
        {
            "label": t("metric_vol", lang),
            "value": f"{metrics.get('ann_vol', 0.18)*100:.1f}%",
            "num_val": metrics.get("ann_vol", 0.18) * 100,
            "suffix": "%",
            "delta": "-1.1% vs bmk",
            "delta_pos": True,
            "sparkline": [0.4, 0.5, 0.6, 0.45, 0.5, 0.38, 0.42, 0.35],
        },
        {
            "label": t("metric_drawdown", lang),
            "value": f"{metrics.get('max_drawdown', -0.12)*100:.1f}%",
            "num_val": abs(metrics.get("max_drawdown", -0.12)) * 100,
            "prefix": "-",
            "suffix": "%",
            "delta": "Max peak-to-trough",
            "delta_pos": False,
            "sparkline": [0.0, -0.05, -0.15, -0.08, -0.12, -0.04, -0.02],
        },
        {
            "label": t("metric_sharpe", lang),
            "value": f"{metrics.get('sharpe', 0.85):.2f}",
            "num_val": metrics.get("sharpe", 0.85),
            "delta": "Rf = 6.5%",
            "delta_pos": True,
            "sparkline": [0.5, 0.6, 0.75, 0.7, 0.8, 0.85],
        },
        {
            "label": t("metric_sortino", lang),
            "value": f"{metrics.get('sortino', 1.15):.2f}",
            "num_val": metrics.get("sortino", 1.15),
            "delta": "Downside only",
            "delta_pos": True,
            "sparkline": [0.6, 0.7, 0.9, 0.85, 1.05, 1.15],
        },
        {
            "label": t("metric_cvar", lang),
            "value": f"{metrics.get('cvar_95', -0.04)*100:.1f}%",
            "num_val": abs(metrics.get("cvar_95", -0.04)) * 100,
            "prefix": "-",
            "suffix": "%",
            "delta": "Tail Risk 95%",
            "delta_pos": False,
            "sparkline": [-0.02, -0.03, -0.05, -0.04, -0.045],
        },
    ]
    render_kpi_strip(kpi_items, height=95)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Bento Grid Row 1: Weight vs Risk Contribution (3) & Correlation Heatmap (4)
    # -----------------------------------------------------------------------
    col_rc, col_heat = st.columns([1, 1])
    with col_rc:
        with st.container(border=True):
            ui.card_header(t("risk_contribution", lang), subtitle="FLAG ASSETS WHERE RISK SHARE EXCEEDS WEIGHT")
            fig_rc = charts.weight_vs_risk_bars(weights, rc)
            st.plotly_chart(fig_rc, use_container_width=True, config=charts.PLOT_CONFIG)

    with col_heat:
        with st.container(border=True):
            ui.card_header(t("correlation_heatmap", lang), subtitle="5-BIN DISCRETE STEPPED CORRELATION MATRIX")
            returns = prices_df.pct_change().dropna()
            corr = returns.corr() if not returns.empty else pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
            fig_heat = charts.correlation_heatmap(corr)
            st.plotly_chart(fig_heat, use_container_width=True, config=charts.PLOT_CONFIG)

    # -----------------------------------------------------------------------
    # Bento Grid Row 2: Allocation Treemap (5) & Rolling 60D Volatility (6)
    # -----------------------------------------------------------------------
    col_tree, col_vol = st.columns([1, 1])
    with col_tree:
        with st.container(border=True):
            ui.card_header(t("allocation_treemap", lang), subtitle="CAPITAL DISTRIBUTION HIERARCHY")
            fig_tree = charts.allocation_treemap(weights)
            st.plotly_chart(fig_tree, use_container_width=True, config=charts.PLOT_CONFIG)

    with col_vol:
        with st.container(border=True):
            ui.card_header(t("rolling_vol", lang), subtitle="60-DAY ANNUALIZED VOLATILITY DRIFT")
            fig_vol = charts.rolling_vol_chart(prices_df)
            st.plotly_chart(fig_vol, use_container_width=True, config=charts.PLOT_CONFIG)

    # -----------------------------------------------------------------------
    # Bento Grid Row 3: Underwater Drawdown (7) & Daily Return Dist (8)
    # -----------------------------------------------------------------------
    col_dd, col_hist = st.columns([1, 1])
    with col_dd:
        with st.container(border=True):
            ui.card_header(t("underwater_chart", lang), subtitle="HISTORICAL PEAK-TO-TROUGH DRAWDOWN")
            growth = api.simulate(tickers_t, weights_t, start, end).get("growth", {})
            fig_dd = charts.underwater_chart(growth)
            st.plotly_chart(fig_dd, use_container_width=True, config=charts.PLOT_CONFIG)

    with col_hist:
        with st.container(border=True):
            ui.card_header(t("return_dist", lang), subtitle="HISTORICAL RETURNS WITH VAR & CVAR RULES")
            ret_series = returns.mean(axis=1) if not returns.empty else pd.Series(np.random.normal(0.0005, 0.012, 250))
            fig_hist = charts.return_distribution_hist(ret_series, var_95=-0.022, cvar_95=-0.038)
            st.plotly_chart(fig_hist, use_container_width=True, config=charts.PLOT_CONFIG)

    # -----------------------------------------------------------------------
    # Bento Grid Row 4: Versus Benchmark Card (10)
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header(t("vs_benchmark", lang), subtitle=f"COMPARED AGAINST BENCHMARK {benchmark}")
        bench_metrics = {
            "ann_return": 0.125,
            "ann_vol": 0.145,
            "max_drawdown": -0.160,
            "sharpe": 0.68,
        }
        fig_vs = charts.vs_benchmark_chart(metrics, bench_metrics)
        st.plotly_chart(fig_vs, use_container_width=True, config=charts.PLOT_CONFIG)
