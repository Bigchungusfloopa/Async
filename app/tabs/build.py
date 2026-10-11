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


def _spark(values, n: int = 30) -> list[float]:
    """Down-sample a real series to <= n points, min-max normalised to [0, 1] (empty if flat / too short)."""
    import numpy as np
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return []
    v = v[np.linspace(0, len(v) - 1, min(n, len(v))).astype(int)]
    lo, hi = float(v.min()), float(v.max())
    return [] if hi - lo < 1e-12 else ((v - lo) / (hi - lo)).tolist()


def render_tab(tickers: list[str], weights: dict[str, float],
               start: str, end: str, lang: str = "en",
               benchmark: str = "^NSEI", risk_level: str = "Medium") -> None:
    if not tickers:
        ui.empty_state(t("select_assets_hint", lang))
        return

    tickers_t = tuple(tickers)
    weights_t = tuple(weights.get(t, 1 / len(tickers)) for t in tickers)

    # Your holdings, and the mix the optimiser recommends for the selected preference
    cur_weights_t = weights_t
    opt = api.optimize(tickers_t, cur_weights_t, risk_level, start, end)
    rec_w = opt.get("weights", {})
    rec_t = tuple(float(rec_w.get(x, 0.0)) for x in tickers)
    cur_label = api.classify_risk(tickers_t, cur_weights_t, start, end).get("label", "Medium")
    cur_metrics = api.portfolio_metrics(tickers_t, cur_weights_t, start, end)

    # -----------------------------------------------------------------------
    # Section Header & Hero Gauge
    # -----------------------------------------------------------------------
    ui.section(t("tab_risk", lang), t("risk_model_desc", lang))

    # The choice is kept in its own session value: the sidebar calls st.rerun() before this tab is drawn, and
    # Streamlit drops the state of widgets that were not drawn in a run, which used to reset the view.
    st.session_state.setdefault("_risk_view", "current")
    view = st.radio(
        "SHOW", ["current", "recommended"], horizontal=True, key="risk_tab_view_widget",
        label_visibility="collapsed", index=0 if st.session_state["_risk_view"] == "current" else 1,
        format_func=lambda v: "YOUR PORTFOLIO" if v == "current" else "RECOMMENDED MIX (FOLLOWS RISK LEVEL)",
        on_change=lambda: st.session_state.update(_risk_view=st.session_state["risk_tab_view_widget"]),
    )
    if view == "recommended" and any(rec_t):
        weights_t = rec_t                                      # every number and chart below uses this mix
        weights = dict(zip(tickers, rec_t))
        st.caption(f"Showing the mix recommended for a {risk_level} preference: "
                   + ", ".join(f"{x} {w:.0%}" for x, w in zip(tickers, rec_t))
                   + ". It changes instantly with the risk level; your holdings are unchanged.")
    else:
        st.caption("Showing the portfolio you hold. These numbers depend on your holdings, not on the risk "
                   "preference; switch to RECOMMENDED to see the mix for your preference.")

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



    col_hero, col_expl = st.columns([1, 1])
    with col_hero:
        with st.container(border=True):
            ui.card_header(t("risk_label", lang), right=f"MODEL CONF: {max(probs.values())*100:.0f}%")
            render_risk_gauge(label, probs, height=240)

    with col_expl:
        with st.container(border=True):
            ui.card_header(t("explanation", lang), right="XAI REPORT")
            st.markdown(f"<div style='font-size:13px; color:#F5F5F5; line-height:1.6;'>{ui.strip_emojis(explanation)}</div>", unsafe_allow_html=True)
            top_factors = risk_data.get("top_factors", ["Concentration in high-beta tech", "Macro inflation sensitivity", "Duration gap"])
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            fig_factors = charts.top_factors_bars(top_factors)
            st.plotly_chart(fig_factors, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_1")

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Preference check: reacts instantly to the Low / Medium / High selector
    # -----------------------------------------------------------------------
    order = {"Low": 0, "Medium": 1, "High": 2}
    rec_label = api.classify_risk(tickers_t, rec_t, start, end).get("label", cur_label) if any(rec_t) else cur_label
    rec_vol = opt.get("metrics", {}).get("ann_vol")
    cur_vol = cur_metrics.get("ann_vol")
    gap = order.get(cur_label, 1) - order.get(risk_level, 1)
    if gap > 0:
        fit, cls = "RISKIER THAN YOUR PREFERENCE", "alert-warn"
    elif gap < 0:
        fit, cls = "SAFER THAN YOUR PREFERENCE", "alert-success"
    else:
        fit, cls = "MATCHES YOUR PREFERENCE", "alert-success"
    reach_txt = ""
    if order.get(rec_label, 1) > order.get(risk_level, 1):     # re-weighting alone cannot get there
        reach_txt = (f" NO MIX OF THESE ASSETS REACHES {risk_level.upper()}: ADD LOWER-RISK HOLDINGS SUCH AS "
                     "LIQUIDBEES.NS (CASH), GOLDBEES.NS (GOLD) OR SHY / AGG (BONDS).")
    vol_txt = (f" Volatility {cur_vol * 100:.1f}% -> {rec_vol * 100:.1f}%."
               if isinstance(cur_vol, (int, float)) and isinstance(rec_vol, (int, float)) else "")
    st.markdown(
        f'<div data-testid="stAlert" class="{cls}"><p style="font-family:\'JetBrains Mono\',monospace; font-size:12px; '
        f'font-weight:600; letter-spacing:0.04em;">YOUR PREFERENCE: {risk_level.upper()} &nbsp;|&nbsp; THIS PORTFOLIO: '
        f'{cur_label.upper()} ({fit}). THE MIX RECOMMENDED FOR A {risk_level.upper()} PREFERENCE IS RATED '
        f'{rec_label.upper()}.{vol_txt} SEE THE ALLOCATION TAB FOR THE WEIGHTS.{reach_txt}</p></div>',
        unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 2. KPI Strip with real deltas vs the benchmark and real sparklines
    # -----------------------------------------------------------------------
    bmk = api.portfolio_metrics((benchmark,), (1.0,), start, end)
    d_ret = metrics.get("ann_return", 0.0) - bmk.get("ann_return", 0.0)
    d_vol = metrics.get("ann_vol", 0.0) - bmk.get("ann_vol", 0.0)
    try:
        from engine.metrics import default_rf
        rf_text = f"Rf = {default_rf(list(tickers), [weights.get(x, 1 / len(tickers)) for x in tickers]) * 100:.1f}%"
    except Exception:
        rf_text = "Risk-free rate"
    growth_s = pd.Series(api.simulate(tickers_t, weights_t, start, end).get("growth", {}), dtype="float64")
    if len(growth_s) > 30:
        growth_s = growth_s.sort_index()
        port_ret = growth_s.pct_change().dropna()
        spark_growth = _spark(growth_s.to_numpy())
        spark_vol = _spark(port_ret.rolling(21).std().dropna().to_numpy())
        spark_dd = _spark((growth_s / growth_s.cummax() - 1.0).to_numpy())
    else:
        spark_growth = spark_vol = spark_dd = []
    kpi_items = [
        {
            "label": t("metric_return", lang),
            "value": f"{metrics.get('ann_return', 0.0)*100:+.1f}%",
            "num_val": abs(metrics.get("ann_return", 0.0)) * 100,
            "prefix": "+" if metrics.get("ann_return", 0.0) >= 0 else "-",
            "suffix": "%",
            "delta": f"{d_ret * 100:+.1f}% vs bmk",
            "delta_pos": d_ret >= 0,
            "sparkline": spark_growth,
        },
        {
            "label": t("metric_vol", lang),
            "value": f"{metrics.get('ann_vol', 0.0)*100:.1f}%",
            "num_val": metrics.get("ann_vol", 0.0) * 100,
            "suffix": "%",
            "delta": f"{d_vol * 100:+.1f}% vs bmk",
            "delta_pos": d_vol <= 0,
            "sparkline": spark_vol,
        },
        {
            "label": t("metric_drawdown", lang),
            "value": f"{metrics.get('max_drawdown', 0.0)*100:.1f}%",
            "num_val": abs(metrics.get("max_drawdown", 0.0)) * 100,
            "prefix": "-",
            "suffix": "%",
            "delta": "Max peak-to-trough",
            "delta_pos": False,
            "sparkline": spark_dd,
        },
        {
            "label": t("metric_sharpe", lang),
            "value": f"{metrics.get('sharpe', 0.0):.2f}",
            "num_val": metrics.get("sharpe", 0.0),
            "delta": rf_text,
            "delta_pos": True,
        },
        {
            "label": t("metric_sortino", lang),
            "value": f"{metrics.get('sortino', 0.0):.2f}",
            "num_val": metrics.get("sortino", 0.0),
            "delta": "Downside only",
            "delta_pos": True,
        },
        {
            "label": t("metric_cvar", lang),
            "value": f"{metrics.get('cvar_95', 0.0)*100:.1f}%",
            "num_val": abs(metrics.get("cvar_95", 0.0)) * 100,
            "prefix": "-",
            "suffix": "%",
            "delta": "Tail Risk 95%",
            "delta_pos": False,
        },
    ]
    render_kpi_strip(kpi_items, height=125)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Bento Grid Row 1: Weight vs Risk Contribution (3) & Correlation Heatmap (4)
    # -----------------------------------------------------------------------
    col_rc, col_heat = st.columns([1, 1])
    with col_rc:
        with st.container(border=True):
            ui.card_header(t("risk_contribution", lang), subtitle="FLAG ASSETS WHERE RISK SHARE EXCEEDS WEIGHT")
            fig_rc = charts.weight_vs_risk_bars(weights, rc)
            st.plotly_chart(fig_rc, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_2")

    with col_heat:
        with st.container(border=True):
            ui.card_header(t("correlation_heatmap", lang), subtitle="5-BIN DISCRETE STEPPED CORRELATION MATRIX")
            returns = prices_df.pct_change().dropna()
            corr = returns.corr() if not returns.empty else pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
            fig_heat = charts.correlation_heatmap(corr)
            st.plotly_chart(fig_heat, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_3")

    # -----------------------------------------------------------------------
    # Bento Grid Row 2: Allocation Treemap (5) & Rolling 60D Volatility (6)
    # -----------------------------------------------------------------------
    col_tree, col_vol = st.columns([1, 1])
    with col_tree:
        with st.container(border=True):
            ui.card_header(t("allocation_treemap", lang), subtitle="CAPITAL DISTRIBUTION HIERARCHY")
            fig_tree = charts.allocation_treemap(weights)
            st.plotly_chart(fig_tree, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_4")

    with col_vol:
        with st.container(border=True):
            ui.card_header(t("rolling_vol", lang), subtitle="60-DAY ANNUALIZED VOLATILITY DRIFT")
            fig_vol = charts.rolling_vol_chart(prices_df)
            st.plotly_chart(fig_vol, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_5")

    # -----------------------------------------------------------------------
    # Bento Grid Row 3: Underwater Drawdown (7) & Daily Return Dist (8)
    # -----------------------------------------------------------------------
    col_dd, col_hist = st.columns([1, 1])
    with col_dd:
        with st.container(border=True):
            ui.card_header(t("underwater_chart", lang), subtitle="HISTORICAL PEAK-TO-TROUGH DRAWDOWN")
            growth = api.simulate(tickers_t, weights_t, start, end).get("growth", {})
            fig_dd = charts.underwater_chart(growth)
            st.plotly_chart(fig_dd, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_6")

    with col_hist:
        with st.container(border=True):
            ui.card_header(t("return_dist", lang), subtitle="HISTORICAL RETURNS WITH VAR & CVAR RULES")
            port_daily = (growth_s.sort_index().pct_change().dropna() if len(growth_s) > 30
                          else pd.Series(dtype="float64"))                 # the weighted portfolio, not an average
            if len(port_daily) > 20:
                var95 = float(np.percentile(port_daily, 5))
                cvar95 = float(port_daily[port_daily <= var95].mean())
                fig_hist = charts.return_distribution_hist(port_daily, var_95=var95, cvar_95=cvar95)
                st.plotly_chart(fig_hist, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_7")
            else:
                st.caption("Not enough history for a return distribution.")

    # -----------------------------------------------------------------------
    # Bento Grid Row 4: Versus Benchmark Card (10)
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header(t("vs_benchmark", lang), subtitle=f"COMPARED AGAINST BENCHMARK {benchmark}")
        bench_metrics = bmk                                       # the benchmark's real metrics, same period
        fig_vs = charts.vs_benchmark_chart(metrics, bench_metrics)
        st.plotly_chart(fig_vs, use_container_width=True, config=charts.PLOT_CONFIG, key="build_chart_8")
