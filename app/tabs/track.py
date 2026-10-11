"""
app/tabs/track.py — Track tab.
Holdings editor (ticker, quantity, buy price) with Refresh button.
Table with current price, value, P&L and P&L % (green/red).
Allocation pie; totals per currency; current risk label; timestamp.
Cache TTL ~5 min.
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd

import app.api as api
import app.charts as charts
from app.ui import metric_card, fmt_inr, fmt_pct, fmt_number, pnl_color, info_banner
from ml.i18n import t


_DEFAULT_HOLDINGS = [
    {"ticker": "TCS.NS",      "quantity": 10, "buy_price": 3450.0},
    {"ticker": "INFY.NS",     "quantity": 15, "buy_price": 1540.0},
    {"ticker": "HDFCBANK.NS", "quantity": 20, "buy_price": 1680.0},
]


def render(lang: str, **kwargs) -> None:
    """Render the Track tab."""

    # -----------------------------------------------------------------------
    # Holdings editor
    # -----------------------------------------------------------------------
    st.subheader(t("holdings_editor", lang))
    st.caption(t("data_delayed", lang))

    # Initialize default data
    if "track_holdings_data" not in st.session_state:
        st.session_state["track_holdings_data"] = pd.DataFrame(_DEFAULT_HOLDINGS)

    edited_df = st.data_editor(
        st.session_state["track_holdings_data"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker":    st.column_config.TextColumn(t("ticker_col", lang), required=True),
            "quantity":  st.column_config.NumberColumn(t("qty_col", lang), min_value=0, format="%g"),
            "buy_price": st.column_config.NumberColumn(t("buy_price_col", lang), min_value=0, format="%.2f"),
        },
        key="track_editor",
        hide_index=True,
    )
    st.session_state["track_holdings_data"] = edited_df

    # Refresh button
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        do_refresh = st.button(t("refresh", lang), key="track_refresh")

    st.divider()

    # -----------------------------------------------------------------------
    # Build holdings list from editor
    # -----------------------------------------------------------------------
    holdings = []
    for _, row in edited_df.iterrows():
        tkr = str(row.get("ticker", "")).strip()
        if tkr:
            holdings.append({
                "ticker":    tkr,
                "quantity":  float(row.get("quantity", 0) or 0),
                "buy_price": float(row.get("buy_price", 0) or 0),
            })

    if not holdings:
        st.info(t("no_data", lang))
        return

    # -----------------------------------------------------------------------
    # Fetch holdings data (cached 5 min)
    # -----------------------------------------------------------------------
    holdings_json = json.dumps(holdings, sort_keys=True)
    with st.spinner(f"{t('refresh', lang)}…"):
        try:
            result = api.track_holdings(holdings_json)
        except Exception as e:
            st.error(f"{t('no_data', lang)}: {e}")
            return

    table_data = result.get("table", [])
    total_value = result.get("total_value", 0.0)
    total_pnl   = result.get("total_pnl", 0.0)
    as_of       = result.get("as_of", "")
    currency    = result.get("currency", "INR")

    # -----------------------------------------------------------------------
    # Totals (never mixed currencies — show per currency)
    # -----------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(t("total_value", lang), fmt_inr(total_value))
    with c2:
        pnl_positive = total_pnl >= 0
        metric_card(
            t("total_pnl", lang),
            f"{'+'if pnl_positive else ''}{fmt_inr(total_pnl)}",
            delta=f"{result.get('total_pnl_pct', 0):+.2f}%",
            delta_positive=pnl_positive,
        )
    with c3:
        # Current risk label for the tracked holdings
        if table_data:
            try:
                track_tickers = tuple(row.get("ticker") for row in table_data)
                total_v = sum(row.get("value", 0) for row in table_data)
                track_weights = tuple(
                    row.get("value", 0) / total_v if total_v else 1 / len(track_tickers)
                    for row in table_data
                )
                risk_result = api.classify_risk(
                    track_tickers, track_weights,
                    "2023-01-01", "2026-10-01"
                )
                risk_lbl = risk_result.get("label", "—")
                from app.ui import RISK_COLORS
                color = RISK_COLORS.get(risk_lbl, "#64748b")
                st.markdown(
                    f'<div style="text-align:center;padding:8px;">'
                    f'<div style="font-size:0.78rem;color:#64748b;font-weight:600;">{t("risk_label", lang)}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{color};">{risk_lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.metric(t("risk_label", lang), "—")

    if as_of:
        st.caption(f"{t('as_of', lang)}: {as_of}")

    st.divider()

    # -----------------------------------------------------------------------
    # Holdings table with colored P&L
    # -----------------------------------------------------------------------
    if table_data:
        df = pd.DataFrame(table_data)
        # Build styled display
        display_rows = []
        for row in table_data:
            pnl = row.get("pnl", 0)
            pnl_pct = row.get("pnl_pct", 0)
            display_rows.append({
                t("ticker_col", lang):       row.get("ticker", ""),
                t("qty_col", lang):          row.get("quantity", 0),
                t("buy_price_col", lang):    f"₹{row.get('buy_price', 0):,.2f}",
                t("current_price_col", lang):f"₹{row.get('current_price', 0):,.2f}",
                t("value_col", lang):        f"₹{row.get('value', 0):,.0f}",
                t("pnl_col", lang):          f"{'+'if pnl>=0 else ''}{pnl:,.0f}",
                t("pnl_pct_col", lang):      f"{pnl_pct:+.2f}%",
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Allocation pie
    # -----------------------------------------------------------------------
    if table_data:
        alloc = {row["ticker"]: row.get("value", 0) for row in table_data}
        total_alloc = sum(alloc.values()) or 1
        alloc_pct = {k: v / total_alloc for k, v in alloc.items()}
        fig_pie = charts.allocation_pie(alloc_pct, "Current Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
