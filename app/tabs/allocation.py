"""
app/tabs/allocation.py — Allocation Tab.
Visual Inventory (Section 7):
1. Original vs recommended donuts side by side (flat slices, thin gaps)
2. Diverging bar of weight changes per asset (red reduce, green increase)
3. Asset comparison table with in-cell bars (ProgressColumn)
4. Risk-return scatter of assets + original, recommended, equal-weight and benchmark,
   with a flat-dot cloud of random portfolios behind them
5. Reason and trade-off text card with numbers highlighted in mono
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

    with st.spinner("SOLVING PORTFOLIO OPTIMIZATION..."):
        opt = api.optimize(tickers_t, weights_t, risk_level, start, end)
        orig_metrics = api.portfolio_metrics(tickers_t, weights_t, start, end)
        rec_weights = opt.get("weights", {t: 1 / len(tickers) for t in tickers})
        rec_weights_t = tuple(rec_weights.get(t, 0) for t in tickers)
        rec_metrics = opt.get("metrics", orig_metrics)
        reason = opt.get("reason", "Adjusted weights to minimize volatility.")
        tradeoffs = opt.get("tradeoffs", "Slight return concession for significant drawdown reduction.")
        explain_text = api.explain_allocation(tickers_t, weights_t, rec_weights_t,
                                              risk_level, lang, start, end)

    ui.section(t("tab_allocation", lang), f"{t('risk_level', lang)}: {risk_level.upper()}")

    # -----------------------------------------------------------------------
    # 1. Original vs Recommended Donuts Side by Side
    # -----------------------------------------------------------------------
    col_donut_orig, col_donut_rec = st.columns([1, 1])
    with col_donut_orig:
        with st.container(border=True):
            ui.card_header(t("original_weights", lang), subtitle="CURRENT HOLDINGS RATIO")
            fig_orig = charts.allocation_pie(weights)
            st.plotly_chart(fig_orig, use_container_width=True, config=charts.PLOT_CONFIG)

    with col_donut_rec:
        with st.container(border=True):
            ui.card_header(t("recommended_weights", lang), subtitle=f"OPTIMIZED FOR {risk_level.upper()} PROFILE")
            fig_rec = charts.allocation_pie(rec_weights)
            st.plotly_chart(fig_rec, use_container_width=True, config=charts.PLOT_CONFIG)

    # -----------------------------------------------------------------------
    # 2. Diverging Bar of Weight Changes & Reason/Tradeoff Card (5)
    # -----------------------------------------------------------------------
    col_diverge, col_reason = st.columns([1, 1])
    with col_diverge:
        with st.container(border=True):
            ui.card_header(t("weight_delta", lang), subtitle="REBALANCING REALLOCATION (RED = TRIM, GREEN = ADD)")
            deltas = {t: rec_weights.get(t, 0) - weights.get(t, 0) for t in tickers}
            fig_deltas = charts.diverging_weight_deltas(deltas)
            st.plotly_chart(fig_deltas, use_container_width=True, config=charts.PLOT_CONFIG)

    with col_reason:
        with st.container(border=True):
            ui.card_header(t("allocation_explanation", lang), right="RATIONALE")
            st.markdown(f"""
            <div style="font-size:12px; color:#8A8A8A; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">OPTIMIZER TARGET</div>
            <div style="font-size:14px; color:#F5F5F5; margin-bottom:12px;">{ui.strip_emojis(reason)}</div>
            <div style="font-size:12px; color:#8A8A8A; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">TRADE-OFF EVALUATION</div>
            <div style="font-size:13px; color:#8A8A8A; line-height:1.5; margin-bottom:12px;">{ui.strip_emojis(tradeoffs)}</div>
            <div style="font-size:12px; color:#8A8A8A; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">EXPLANATION</div>
            <div style="font-size:13px; color:#F5F5F5; line-height:1.5;">{ui.strip_emojis(explain_text)}</div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 3. Asset Comparison Table with Progress Columns
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header("ASSET ALLOCATION COMPARISON", subtitle="DETAILED BEFORE / AFTER WEIGHTS AND DELTAS")
        table_rows = []
        for tkr in tickers:
            w_orig = weights.get(tkr, 0) * 100
            w_rec = rec_weights.get(tkr, 0) * 100
            d = w_rec - w_orig
            table_rows.append({
                "Ticker": tkr,
                "Original (%)": w_orig,
                "Recommended (%)": w_rec,
                "Delta (%)": d,
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(
            df_table,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Original (%)": st.column_config.ProgressColumn("Original (%)", min_value=0, max_value=100, format="%.1f%%"),
                "Recommended (%)": st.column_config.ProgressColumn("Recommended (%)", min_value=0, max_value=100, format="%.1f%%"),
                "Delta (%)": st.column_config.NumberColumn("Delta (%)", format="%+.1f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # 4. Risk-Return Scatter with Flat-Dot Random Portfolio Cloud
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header(t("risk_return_scatter", lang), subtitle="EFFICIENT FRONTIER SIMULATION & ASSET COORDINATES")
        scatter_pts = api.risk_return_points(tickers_t, weights_t, rec_weights_t, benchmark, start, end)
        fig_scatter = charts.risk_return_scatter(scatter_pts, cloud=True)
        st.plotly_chart(fig_scatter, use_container_width=True, config=charts.PLOT_CONFIG)
