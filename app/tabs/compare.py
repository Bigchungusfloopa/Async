"""
app/tabs/compare.py — Compare tab.
Table: Original / Recommended / Equal Weight / Benchmark.
Growth lines, risk-return scatter, metric bar charts, auto verdict.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

import app.api as api
import app.charts as charts
from app.ui import metric_card, fmt_pct, fmt_number, STRATEGY_COLORS
from ml.i18n import t


def render(lang: str, tickers: list[str], weights: dict[str, float],
           risk_level: str, benchmark: str, start: str, end: str, **kwargs) -> None:
    """Render the Compare tab."""
    if not tickers:
        st.info(t("no_data", lang))
        return

    tickers_t = tuple(tickers)
    raw_w     = tuple(weights.get(tk, 0.0) for tk in tickers)
    total     = sum(raw_w) or 1.0
    weights_t = tuple(w / total for w in raw_w)

    with st.spinner("Comparing strategies…"):
        # Get recommended weights
        try:
            opt = api.optimize(tickers_t, weights_t, risk_level, start, end)
            rec_raw = opt.get("weights", {})
            rec_total = sum(rec_raw.values()) or 1.0
            rec_weights_norm = {k: v / rec_total for k, v in rec_raw.items()}
            rec_weights_t = tuple(rec_weights_norm.get(tk, 0.0) for tk in tickers)
        except Exception:
            rec_weights_t = weights_t
            rec_weights_norm = dict(zip(tickers, weights_t))

        try:
            compare_data = api.compare_strategies(
                tickers_t, weights_t, rec_weights_t, benchmark, start, end
            )
        except Exception as e:
            st.error(f"{t('no_data', lang)}: {e}")
            return

        try:
            growth_data = api.compare_growth(
                tickers_t, weights_t, rec_weights_t, benchmark, start, end
            )
        except Exception:
            growth_data = {}

        try:
            scatter_pts = api.risk_return_points(
                tickers_t, weights_t, rec_weights_t, benchmark, start, end
            )
        except Exception:
            scatter_pts = []

    # -----------------------------------------------------------------------
    # Strategy comparison table
    # -----------------------------------------------------------------------
    st.subheader(t("compare_title", lang))

    if compare_data:
        df = pd.DataFrame(compare_data)
        # Format numeric columns
        fmt_map = {
            "ann_return": lambda v: f"{v*100:.1f}%",
            "ann_vol":    lambda v: f"{v*100:.1f}%",
            "max_drawdown": lambda v: f"{v*100:.1f}%",
            "sharpe":       lambda v: f"{v:.2f}",
            "effective_n":  lambda v: f"{v:.1f}" if v else "N/A",
        }
        display_df = df.copy()
        for col, fmt_fn in fmt_map.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: fmt_fn(x) if x is not None and str(x) != "nan" else "N/A"
                )
        col_rename = {
            "strategy": "Strategy",
            "ann_return": t("metric_return", lang),
            "ann_vol": t("metric_vol", lang),
            "max_drawdown": t("metric_drawdown", lang),
            "sharpe": t("metric_sharpe", lang),
            "effective_n": "Effective N",
        }
        display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns},
                          inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning(t("no_data", lang))

    st.divider()

    # -----------------------------------------------------------------------
    # Growth comparison chart
    # -----------------------------------------------------------------------
    st.markdown(f"**{t('growth_comparison', lang)}**")
    if growth_data:
        fig_growth = charts.multi_growth_chart(growth_data, "")
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.info(t("no_data", lang))

    st.divider()

    # -----------------------------------------------------------------------
    # Risk-return scatter
    # -----------------------------------------------------------------------
    st.markdown(f"**{t('risk_return_scatter', lang)}**")
    if scatter_pts:
        fig_scatter = charts.risk_return_scatter(scatter_pts, "")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info(t("no_data", lang))

    st.divider()

    # -----------------------------------------------------------------------
    # Metric bar charts
    # -----------------------------------------------------------------------
    if compare_data:
        col1, col2 = st.columns(2)
        with col1:
            fig_ret = charts.metric_bar_chart(compare_data, "ann_return",
                                              t("metric_return", lang))
            st.plotly_chart(fig_ret, use_container_width=True)
        with col2:
            fig_sharpe = charts.metric_bar_chart(compare_data, "sharpe",
                                                 t("metric_sharpe", lang))
            st.plotly_chart(fig_sharpe, use_container_width=True)

    # -----------------------------------------------------------------------
    # Auto verdict
    # -----------------------------------------------------------------------
    st.markdown(f"**{t('verdict', lang)}**")
    try:
        orig_sharpe = next(d["sharpe"] for d in compare_data if d["strategy"] == "Original")
        bench_sharpe = next(d["sharpe"] for d in compare_data if d["strategy"] in (benchmark, "NIFTY 50", "NASDAQ-100"))
        if orig_sharpe > bench_sharpe:
            verdict = f"✅ Your portfolio **outperforms** {benchmark} on a risk-adjusted basis (Sharpe {orig_sharpe:.2f} vs {bench_sharpe:.2f})."
        else:
            verdict = f"⚠️ Your portfolio **underperforms** {benchmark} on a risk-adjusted basis (Sharpe {orig_sharpe:.2f} vs {bench_sharpe:.2f})."
        st.markdown(verdict)
    except Exception:
        st.markdown("Sharpe comparison not available in mock mode.")

    # -----------------------------------------------------------------------
    # Limitations note
    # -----------------------------------------------------------------------
    st.caption(t("limitations", lang))
