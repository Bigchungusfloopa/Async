"""
app/tabs/build.py — Risk tab (also known as 'build' per the spec).
Shows: risk badge, class probs, metric cards, correlation heatmap,
weight vs risk contribution chart, top factors, explanation text.
"""

from __future__ import annotations

import streamlit as st

import app.api as api
import app.charts as charts
from app.ui import (
    risk_badge, metric_card, prob_bar, fmt_pct, fmt_number, info_banner
)
from ml.i18n import t


def render(lang: str, tickers: list[str], weights: dict[str, float],
           start: str, end: str, **kwargs) -> None:
    """Render the Risk tab."""
    if not tickers:
        st.info(t("no_data", lang))
        return

    tickers_t = tuple(tickers)
    weights_t = tuple(weights.get(tk, 0.0) for tk in tickers)
    total     = sum(weights_t) or 1.0
    weights_t = tuple(w / total for w in weights_t)

    # -----------------------------------------------------------------------
    # Backend calls (all cached)
    # -----------------------------------------------------------------------
    with st.spinner("Analysing portfolio…"):
        try:
            risk_data = api.classify_risk(tickers_t, weights_t, start, end)
        except Exception as e:
            st.error(f"{t('no_data', lang)} ({e})")
            return

        try:
            metrics = api.portfolio_metrics(tickers_t, weights_t, start, end)
        except Exception as e:
            st.warning(t("retry", lang))
            metrics = {}

        try:
            rc = api.risk_contribution(tickers_t, weights_t, start, end)
        except Exception:
            rc = {}

        try:
            explanation = api.explain_portfolio(tickers_t, weights_t,
                                                risk_data.get("label", "Medium"),
                                                lang, start, end)
        except Exception:
            explanation = risk_data.get("explanation", "")

    label = risk_data.get("label", "Medium")
    probs = risk_data.get("probs", {"Low": 0.33, "Medium": 0.34, "High": 0.33})

    # Check for fallback notice
    prices_df = api.get_prices(tickers_t, start, end)
    if prices_df.attrs.get("source") == "fallback":
        info_banner(t("using_fallback", lang))

    # -----------------------------------------------------------------------
    # Layout: badge + probs
    # -----------------------------------------------------------------------
    st.subheader(t("risk_label", lang))
    col_badge, col_probs = st.columns([1, 2])
    with col_badge:
        risk_badge(label, lang)
    with col_probs:
        st.markdown(f"**{t('class_probs', lang)}**")
        prob_bar(probs, lang)

    st.divider()

    # -----------------------------------------------------------------------
    # Metric cards
    # -----------------------------------------------------------------------
    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(t("metric_return", lang),
                    fmt_pct(metrics.get("ann_return", 0)),
                    delta_positive=metrics.get("ann_return", 0) > 0)
    with c2:
        metric_card(t("metric_vol", lang),
                    fmt_pct(metrics.get("ann_vol", 0)))
    with c3:
        metric_card(t("metric_drawdown", lang),
                    fmt_pct(metrics.get("max_drawdown", 0)),
                    delta_positive=False)
    with c4:
        metric_card(t("metric_sharpe", lang),
                    fmt_number(metrics.get("sharpe", 0)))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card(t("metric_sortino", lang),
                    fmt_number(metrics.get("sortino", 0)))
    with c6:
        metric_card(t("metric_cvar", lang),
                    fmt_pct(metrics.get("cvar_95", 0)))
    with c7:
        metric_card("Calmar",
                    fmt_number(metrics.get("calmar", 0)))
    with c8:
        metric_card("Effective N",
                    fmt_number(metrics.get("effective_n", 0), 1))

    st.divider()

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        # Correlation heatmap — we need prices to compute; use a mock matrix
        st.markdown(f"**{t('correlation_heatmap', lang)}**")
        # Build correlation from prices if available
        try:
            prices_df2 = api.get_prices(tickers_t, start, end)
            corr = prices_df2.pct_change().corr()
            fig_corr = charts.correlation_heatmap(corr, "")
        except Exception:
            # Fall back to mock matrix
            import json
            from pathlib import Path
            helpers = json.loads((Path(__file__).parent.parent.parent / "data" / "mocks" / "mock_helpers.json").read_text())
            fig_corr = charts.correlation_heatmap(helpers.get("correlation_matrix", {}), "")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_right:
        st.markdown(f"**{t('risk_contribution', lang)}**")
        weights_dict = dict(zip(tickers, weights_t))
        fig_rc = charts.risk_contribution_chart(weights_dict, rc, "")
        st.plotly_chart(fig_rc, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Top factors
    # -----------------------------------------------------------------------
    top_factors = risk_data.get("top_factors", [])
    if top_factors:
        st.markdown(f"**{t('top_factors', lang)}**")
        for i, factor in enumerate(top_factors, 1):
            st.markdown(f"{i}. {factor}")

    st.divider()

    # -----------------------------------------------------------------------
    # Explanation text
    # -----------------------------------------------------------------------
    st.markdown(f"**{t('explanation', lang)}**")
    st.markdown(
        f'<div style="background:#f8fafb;border-radius:10px;padding:16px;'
        f'border-left:4px solid #4F8EF7;font-size:0.95rem;line-height:1.7;">'
        f'{explanation}</div>',
        unsafe_allow_html=True,
    )
