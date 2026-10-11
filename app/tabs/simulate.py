"""
app/tabs/simulate.py — Simulate Tab.
Visual Inventory (Section 7):
1. Growth-of-100 line (log toggle) with flat translucent rectangles marking regimes
2. Underwater drawdown area
3. Monthly returns calendar heatmap (diverging discrete bins)
4. Annual returns bar chart
5. Rolling Sharpe line
6. Metrics table with a column per scenario or asset combination
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import api, charts, ui
from ml.i18n import t


def render_tab(tickers: list[str], weights: dict[str, float],
               start: str, end: str, lang: str = "en") -> None:
    if not tickers:
        ui.empty_state(t("select_assets_hint", lang))
        return

    presets = api.regime_presets()
    tickers_t = tuple(tickers)
    weights_t = tuple(weights.get(t, 1 / len(tickers)) for t in tickers)

    ui.section(t("tab_simulate", lang), "HISTORICAL BACKTESTING & REGIME REPLAY")

    # Controls: Preset selector & Log Scale toggle
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        preset_names = list(presets.keys()) + [t("custom", lang)]
        chosen = st.selectbox(t("regime_presets", lang), preset_names, label_visibility="collapsed")
    with ctrl_col2:
        log_scale = st.checkbox(t("log_scale", lang), value=False)

    if chosen in presets:
        sim_start = presets[chosen]["start"]
        sim_end = presets[chosen]["end"]
    else:
        sim_start = start
        sim_end = end

    with st.spinner("SIMULATING HISTORICAL DYNAMICS..."):
        sim_result = api.simulate(tickers_t, weights_t, sim_start, sim_end)
        growth_data = sim_result.get("growth", {})
        metrics = sim_result.get("metrics", {})
        prices_df = api.get_prices(tickers_t, sim_start, sim_end)

    # -----------------------------------------------------------------------
    # 1. Growth-of-100 line (with regime annotations & log toggle)
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header(t("growth_chart", lang), subtitle=f"PERIOD: {sim_start} TO {sim_end} | BASELINE: INR 100")
        fig_growth = charts.growth_chart(growth_data, log_scale=log_scale, regimes=True)
        st.plotly_chart(fig_growth, use_container_width=True, config=charts.PLOT_CONFIG, key="simulate_chart_1")

    # -----------------------------------------------------------------------
    # 2. Underwater Drawdown Area & 5. Rolling Sharpe Line
    # -----------------------------------------------------------------------
    col_dd, col_sharpe = st.columns([1, 1])
    with col_dd:
        with st.container(border=True):
            ui.card_header(t("underwater_chart", lang), subtitle="PEAK-TO-TROUGH DOWNSIDE VOLATILITY")
            fig_dd = charts.underwater_chart(growth_data)
            st.plotly_chart(fig_dd, use_container_width=True, config=charts.PLOT_CONFIG, key="simulate_chart_2")

    with col_sharpe:
        with st.container(border=True):
            g = pd.Series(growth_data, dtype="float64")
            g.index = pd.to_datetime(g.index)
            g = g.sort_index()
            daily = g.pct_change().dropna()
            win = int(min(126, max(20, len(daily) // 3)))
            try:
                from engine.metrics import default_rf
                rf = default_rf(list(tickers), [weights.get(x, 1 / len(tickers)) for x in tickers])
            except Exception:
                rf = 0.065
            ui.card_header("ROLLING SHARPE RATIO", subtitle=f"{win}-DAY ROLLING, PORTFOLIO RETURNS, RF {rf * 100:.1f}%")
            roll_sharpe = ((daily.rolling(win).mean() * 252 - rf)
                           / (daily.rolling(win).std() * np.sqrt(252)).replace(0, np.nan)).dropna()
            dates = roll_sharpe.index.strftime("%Y-%m-%d").tolist()
            vals = roll_sharpe.tolist()

            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(
                x=dates, y=vals, mode="lines",
                name="Rolling Sharpe",
                line=dict(color="#14B8A6", width=1.75),
                hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>",
            ))
            fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="#2E2E2E", line_width=1)
            fig_sharpe.update_layout(template="ink")
            st.plotly_chart(fig_sharpe, use_container_width=True, config=charts.PLOT_CONFIG, key="simulate_chart_3")

    # -----------------------------------------------------------------------
    # 3. Monthly Returns Calendar Heatmap (Diverging Discrete Bins)
    # 4. Annual Returns Bar Chart
    # -----------------------------------------------------------------------
    col_monthly, col_annual = st.columns([1, 1])
    with col_monthly:
        with st.container(border=True):
            ui.card_header("MONTHLY RETURNS MATRIX", subtitle="7-BIN DISCRETE STEPPED RETURN CALENDAR")
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            m_end = g.resample("ME").last()
            m_ret = m_end.pct_change()
            if len(m_end):
                m_ret.iloc[0] = m_end.iloc[0] / g.iloc[0] - 1.0                 # first (partial) month from the start
            years = sorted({str(d.year) for d in m_ret.index})
            z_m = [[None] * 12 for _ in years]
            for d, v in m_ret.items():
                z_m[years.index(str(d.year))][d.month - 1] = None if pd.isna(v) else float(v * 100)

            fig_m = go.Figure(go.Heatmap(
                z=z_m, x=months, y=years,
                colorscale=charts.DISCRETE_SCALE_DIV,
                zmin=-10, zmax=10,
                text=[[("" if v is None else f"{v:+.1f}%") for v in row] for row in z_m],
                texttemplate="%{text}",
                textfont=dict(family="JetBrains Mono, monospace", size=10, color="#F5F5F5"),
                showscale=False,
                hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
            ))
            fig_m.update_layout(template="ink")
            st.plotly_chart(fig_m, use_container_width=True, config=charts.PLOT_CONFIG, key="simulate_chart_4")

    with col_annual:
        with st.container(border=True):
            ui.card_header("ANNUAL NET RETURNS", subtitle="CALENDAR YEAR PERFORMANCE")
            y_end = g.resample("YE").last()
            y_ret = y_end.pct_change()
            if len(y_end):
                y_ret.iloc[0] = y_end.iloc[0] / g.iloc[0] - 1.0                 # partial first year from the start
            ann_years = [str(d.year) for d in y_ret.index]
            ann_rets = [float(v * 100) for v in y_ret.fillna(0.0)]
            colors = ["#22C55E" if r >= 0 else "#EF4444" for r in ann_rets]
            fig_ann = go.Figure(go.Bar(
                x=ann_years, y=ann_rets,
                marker_color=colors,
                text=[f"{r:+.1f}%" for r in ann_rets],
                textposition="outside",
                textfont=dict(family="JetBrains Mono, monospace", size=10, color="#8A8A8A"),
                hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
            ))
            fig_ann.add_hline(y=0, line_color="#2E2E2E", line_width=1)
            fig_ann.update_layout(template="ink")
            st.plotly_chart(fig_ann, use_container_width=True, config=charts.PLOT_CONFIG, key="simulate_chart_5")

    # -----------------------------------------------------------------------
    # 6. Metrics Table per Regime / Asset Combination
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header("REGIME PERFORMANCE METRICS", subtitle="COMPARATIVE METRIC SUMMARY")
        def _row(name: str, m: dict) -> dict:
            if not m:
                return {"Regime": name, "Ann. Return": "n/a", "Volatility": "n/a", "Max Drawdown": "n/a",
                        "Sharpe": "n/a", "Sortino": "n/a"}
            return {"Regime": name, "Ann. Return": f"{m['ann_return'] * 100:+.1f}%",
                    "Volatility": f"{m['ann_vol'] * 100:.1f}%", "Max Drawdown": f"{m['max_drawdown'] * 100:.1f}%",
                    "Sharpe": f"{m['sharpe']:.2f}", "Sortino": f"{m['sortino']:.2f}"}

        regime_summary = [_row(f"{chosen} (selected)", metrics if growth_data else {})]
        for p_name, p_win in presets.items():
            if p_name == chosen:
                continue
            r = api.simulate(tickers_t, weights_t, p_win["start"], p_win["end"])
            regime_summary.append(_row(p_name, r.get("metrics", {}) if r.get("growth") else {}))
        df_summary = pd.DataFrame(regime_summary)
        st.dataframe(df_summary, hide_index=True, use_container_width=True)
