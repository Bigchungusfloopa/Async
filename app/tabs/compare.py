"""
app/tabs/compare.py — Compare Tab.
Visual Inventory (Section 7):
1. Growth lines for Original, Recommended, Equal Weight, Benchmark (distinct dash styles)
2. Grouped bar small multiples for return, volatility, drawdown, Sharpe
3. Risk-return scatter with labelled points
4. Diversification bars (effective N, HHI)
5. Verdict banner (outperforms/underperforms on risk-adjusted basis) + limitations note
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from app import api, charts, ui
from ml.i18n import t


def render_tab(tickers: list[str], weights: dict[str, float],
               risk_level: str, start: str, end: str,
               lang: str = "en", benchmark: str = "^NSEI") -> None:
    if not tickers:
        ui.empty_state(t("select_assets_hint", lang))
        return

    tickers_t = tuple(tickers)
    weights_t = tuple(weights.get(t, 1 / len(tickers)) for t in tickers)

    with st.spinner("BENCHMARKING STRATEGIES..."):
        opt = api.optimize(tickers_t, weights_t, risk_level, start, end)
        rec_weights = opt.get("weights", {t: 1 / len(tickers) for t in tickers})
        rec_weights_t = tuple(rec_weights.get(t, 0) for t in tickers)

        compare_data = api.compare_strategies(
            tickers_t, weights_t, rec_weights_t, benchmark, start, end
        )
        growth_data = api.compare_growth(
            tickers_t, weights_t, rec_weights_t, benchmark, start, end
        )
        scatter_pts = api.risk_return_points(
            tickers_t, weights_t, rec_weights_t, benchmark, start, end
        )

    ui.section(t("tab_compare", lang), "CROSS-STRATEGY QUANTITATIVE BENCHMARKING")

    # -----------------------------------------------------------------------
    # 5. Verdict Banner
    # -----------------------------------------------------------------------
    sharpes = {d.get("strategy"): d.get("sharpe", 0) for d in compare_data}
    rec_s = sharpes.get("Recommended", 0)
    orig_s = sharpes.get("Original", 0)
    bmk_s = sharpes.get("Benchmark", 0)

    if rec_s >= orig_s and rec_s >= bmk_s:
        verdict_text = f"VERDICT: RECOMMENDED ALLOCATION OUTPERFORMS ON A RISK-ADJUSTED BASIS (SHARPE {rec_s:.2f} VS {orig_s:.2f} ORIGINAL, {bmk_s:.2f} BENCHMARK)."
        status_cls = "alert-success"
    else:
        verdict_text = f"VERDICT: PORTFOLIO PROVIDES CAPITAL PRESERVATION AND MINIMUM VARIANCE RELATIVE TO BENCHMARK."
        status_cls = "alert-warn"

    st.markdown(f'<div data-testid="stAlert" class="{status_cls}"><p style="font-family:\'JetBrains Mono\',monospace; font-weight:600; font-size:12px; letter-spacing:0.04em;">{verdict_text}</p></div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 1. Multi-Strategy Growth Lines with Distinct Dash Styles
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header(t("growth_comparison", lang), subtitle="SOLID: ORIGINAL | DASH: RECOMMENDED | DOT: EQUAL WEIGHT | LONGDASH: BENCHMARK")
        fig_growth = charts.multi_growth_chart(growth_data, log_scale=False)
        st.plotly_chart(fig_growth, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_1")

    # -----------------------------------------------------------------------
    # 2. Grouped Bar Small Multiples for Return, Vol, Drawdown, Sharpe
    # -----------------------------------------------------------------------
    col_ret, col_vol = st.columns([1, 1])
    with col_ret:
        with st.container(border=True):
            ui.card_header(t("metric_return", lang).upper(), subtitle="ANNUALIZED COMPOUND RETURN")
            fig_ret = charts.metric_bar_chart(compare_data, "ann_return")
            st.plotly_chart(fig_ret, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_2")

    with col_vol:
        with st.container(border=True):
            ui.card_header(t("metric_vol", lang).upper(), subtitle="ANNUALIZED REALIZED VOLATILITY")
            fig_vol = charts.metric_bar_chart(compare_data, "ann_vol")
            st.plotly_chart(fig_vol, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_3")

    col_dd, col_sh = st.columns([1, 1])
    with col_dd:
        with st.container(border=True):
            ui.card_header(t("metric_drawdown", lang).upper(), subtitle="MAXIMUM HISTORICAL DRAWDOWN")
            fig_dd = charts.metric_bar_chart(compare_data, "max_drawdown")
            st.plotly_chart(fig_dd, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_4")

    with col_sh:
        with st.container(border=True):
            ui.card_header(t("metric_sharpe", lang).upper(), subtitle="RISK-ADJUSTED SHARPE RATIO")
            fig_sh = charts.metric_bar_chart(compare_data, "sharpe")
            st.plotly_chart(fig_sh, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_5")

    # -----------------------------------------------------------------------
    # 3. Risk-Return Scatter & 4. Diversification Bars
    # -----------------------------------------------------------------------
    col_scat, col_div = st.columns([1, 1])
    with col_scat:
        with st.container(border=True):
            ui.card_header(t("risk_return_scatter", lang), subtitle="CROSS-STRATEGY COORDINATES")
            fig_scatter = charts.risk_return_scatter(scatter_pts, cloud=False)
            st.plotly_chart(fig_scatter, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_6")

    with col_div:
        with st.container(border=True):
            ui.card_header("DIVERSIFICATION CONCENTRATION METRICS", subtitle="EFFECTIVE N & HERFINDAHL-HIRSCHMAN INDEX")
            w_vals = list(rec_weights.values())
            hhi = sum(w ** 2 for w in w_vals)
            n_eff = 1.0 / hhi if hhi > 0 else len(tickers)
            fig_div = charts.diversification_bars(n_eff, hhi)
            st.plotly_chart(fig_div, use_container_width=True, config=charts.PLOT_CONFIG, key="compare_chart_7")

    # -----------------------------------------------------------------------
    # Limitations Note
    # -----------------------------------------------------------------------
    st.markdown(f'<div style="font-size:11px; font-family:\'JetBrains Mono\',monospace; color:#5C5C5C; border-top:1px solid #1F1F1F; padding-top:10px; margin-top:16px;">{t("limitations", lang).upper()}</div>', unsafe_allow_html=True)
