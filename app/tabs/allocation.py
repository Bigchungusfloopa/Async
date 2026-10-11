"""
app/tabs/allocation.py — Allocation tab.
Shows: original vs recommended weights (side-by-side bars + delta table),
before/after metrics, reason + tradeoffs text, explain_allocation text.
Changing risk level updates it instantly.
"""

from __future__ import annotations

import streamlit as st

import app.api as api
import app.charts as charts
from app.ui import metric_card, fmt_pct, fmt_number
from ml.i18n import t


def render(lang: str, tickers: list[str], weights: dict[str, float],
           risk_level: str, start: str, end: str, total_investment: float = 100000,
           **kwargs) -> None:
    """Render the Allocation tab."""
    if not tickers:
        st.info(t("no_data", lang))
        return

    tickers_t   = tuple(tickers)
    raw_weights = tuple(weights.get(tk, 0.0) for tk in tickers)
    total       = sum(raw_weights) or 1.0
    weights_t   = tuple(w / total for w in raw_weights)

    # Tip about risk level
    st.info(t("change_risk_update", lang))

    with st.spinner("Optimising allocation…"):
        try:
            opt = api.optimize(tickers_t, weights_t, risk_level, start, end)
        except Exception as e:
            st.error(f"{t('no_data', lang)}: {e}")
            return

    rec_weights_raw = opt.get("weights", {})
    # Normalise recommended weights
    rec_total = sum(rec_weights_raw.values()) or 1.0
    rec_weights = {k: v / rec_total for k, v in rec_weights_raw.items()}

    # Ensure all tickers present
    orig_weights_full = {tk: weights.get(tk, 0.0) for tk in tickers}
    orig_total = sum(orig_weights_full.values()) or 1.0
    orig_weights_norm = {k: v / orig_total for k, v in orig_weights_full.items()}

    rec_weights_t = tuple(rec_weights.get(tk, 0.0) for tk in tickers)

    # -----------------------------------------------------------------------
    # Get before / after metrics
    # -----------------------------------------------------------------------
    orig_metrics = api.portfolio_metrics(tickers_t, weights_t, start, end)
    rec_metrics  = opt.get("metrics", {})

    # -----------------------------------------------------------------------
    # Layout: weight comparison chart
    # -----------------------------------------------------------------------
    st.subheader(t("original_weights", lang) + " → " + t("recommended_weights", lang))
    fig = charts.weight_comparison_chart(orig_weights_norm, rec_weights, "")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Delta table
    # -----------------------------------------------------------------------
    import pandas as pd
    rows = []
    for tk in tickers:
        orig_w = orig_weights_norm.get(tk, 0.0)
        rec_w  = rec_weights.get(tk, 0.0)
        delta  = rec_w - orig_w
        orig_amt = orig_w * total_investment
        rec_amt  = rec_w  * total_investment
        rows.append({
            t("ticker_col", lang): tk,
            t("original_weights", lang): f"{orig_w*100:.1f}%",
            t("recommended_weights", lang): f"{rec_w*100:.1f}%",
            t("weight_delta", lang): f"{delta*100:+.1f}%",
            f"Orig {t('amount_col', lang)}": f"₹{orig_amt:,.0f}",
            f"Rec {t('amount_col', lang)}":  f"₹{rec_amt:,.0f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Before / After metrics comparison
    # -----------------------------------------------------------------------
    col_before, col_after = st.columns(2)
    def _metrics_block(metrics_dict: dict, header: str):
        st.markdown(f"**{header}**")
        c1, c2 = st.columns(2)
        with c1:
            metric_card(t("metric_return", lang), fmt_pct(metrics_dict.get("ann_return", 0)))
            metric_card(t("metric_drawdown", lang), fmt_pct(metrics_dict.get("max_drawdown", 0)))
        with c2:
            metric_card(t("metric_vol", lang), fmt_pct(metrics_dict.get("ann_vol", 0)))
            metric_card(t("metric_sharpe", lang), fmt_number(metrics_dict.get("sharpe", 0)))

    with col_before:
        _metrics_block(orig_metrics, t("before_metrics", lang))
    with col_after:
        _metrics_block(rec_metrics, t("after_metrics", lang))

    st.divider()

    # -----------------------------------------------------------------------
    # Reason + Tradeoffs
    # -----------------------------------------------------------------------
    reason     = opt.get("reason", "")
    tradeoffs  = opt.get("tradeoffs", "")

    if reason:
        st.markdown(f"**{t('reason', lang)}**")
        st.markdown(
            f'<div style="background:#f0fdf4;border-radius:10px;padding:14px;'
            f'border-left:4px solid #2ECC71;font-size:0.92rem;line-height:1.65;">'
            f'{reason}</div>',
            unsafe_allow_html=True,
        )

    if tradeoffs:
        st.markdown(f"**{t('tradeoffs', lang)}**")
        st.markdown(
            f'<div style="background:#fffbeb;border-radius:10px;padding:14px;'
            f'border-left:4px solid #F39C12;font-size:0.92rem;line-height:1.65;">'
            f'{tradeoffs}</div>',
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Explainable recommendation text (brownie point)
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown(f"**{t('allocation_explanation', lang)}**")
    with st.spinner("Loading explanation…"):
        try:
            explain_text = api.explain_allocation(
                tickers_t, weights_t, rec_weights_t, risk_level, lang, start, end
            )
        except Exception:
            explain_text = t("no_data", lang)

    st.markdown(
        f'<div style="background:#f8fafb;border-radius:10px;padding:16px;'
        f'border-left:4px solid #4F8EF7;font-size:0.95rem;line-height:1.7;">'
        f'{explain_text}</div>',
        unsafe_allow_html=True,
    )
