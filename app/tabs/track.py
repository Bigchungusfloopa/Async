"""
app/tabs/track.py — Track Tab.
Visual Inventory (Section 7):
1. KPI strip: value, cost, P&L, P&L %, per currency
2. Holdings table with price, value, signed P&L, in-cell bar
3. Allocation donut and P&L bar chart (green/red)
4. Ticker tape and an 'as of' chip with the delay note
"""

from __future__ import annotations

import json
from datetime import datetime
import streamlit as st
import pandas as pd

from app import api, charts, ui
from app.components.kpi_strip import render_kpi_strip
from app.components.ticker_tape import render_ticker_tape
from ml.i18n import t


def render_tab(holdings_input: list[dict] | None = None,
               lang: str = "en") -> None:
    ui.section(t("tab_track", lang), "LIVE PORTFOLIO VALUATION & REAL-TIME PERFORMANCE")

    # 'as of' chip + data source (filled in once the valuation is loaded)
    chip_col1, chip_col2 = st.columns([1, 1])
    asof_ph, feed_ph = chip_col1.empty(), chip_col2.empty()

    # Holdings Data
    if not holdings_input:
        default_holdings = [
            {"ticker": "RELIANCE.NS", "quantity": 15, "buy_price": 2850.0},
            {"ticker": "TCS.NS", "quantity": 12, "buy_price": 4100.0},
            {"ticker": "HDFCBANK.NS", "quantity": 30, "buy_price": 1610.0},
            {"ticker": "INFY.NS", "quantity": 25, "buy_price": 1820.0},
        ]
    else:
        default_holdings = holdings_input

    holdings_df = pd.DataFrame(default_holdings)

    # Editable holdings table
    with st.expander(t("holdings_editor", lang).upper(), expanded=False):
        edited_df = st.data_editor(
            holdings_df,
            num_rows="dynamic",
            use_container_width=True,
            key="track_holdings_editor",
        )
        refresh_btn = st.button("REFRESH MARKET VALUATION", key="refresh_track_btn")

    active_df = edited_df if "edited_df" in locals() else holdings_df
    holdings_list = active_df.to_dict("records")
    holdings_json = json.dumps(holdings_list)

    with st.spinner("FETCHING VALUATIONS..."):
        track_data = api.track_holdings(holdings_json)

    _src = str(track_data.get("source", "cache"))
    _dot = {"live": "live", "cache": "cache", "fallback": "fallback"}.get(_src, "cache")
    asof_ph.markdown(
        f'<div style="font-size:11px; font-family:\'JetBrains Mono\',monospace; color:#8A8A8A; margin-bottom:12px;">'
        f'PRICES AS OF LAST CLOSE: {track_data.get("as_of", "n/a")} | END-OF-DAY DATA, MAY BE DELAYED</div>',
        unsafe_allow_html=True)
    feed_ph.markdown(
        f'<div style="text-align:right;"><span class="status-chip"><span class="status-dot {_dot}"></span>'
        f'<span>PRICE SOURCE: {_src.upper()}</span></span></div>', unsafe_allow_html=True)
    if track_data.get("fx_rate", 1.0) != 1.0:
        st.caption(f"USD holdings are converted to INR at {track_data['fx_rate']:.2f} (latest close) for totals.")
    table_data = track_data.get("table", [])
    total_val = track_data.get("total_value", sum(h["quantity"] * h["buy_price"] for h in holdings_list))
    total_cost = track_data.get("total_cost", sum(h["quantity"] * h["buy_price"] for h in holdings_list))
    pnl = total_val - total_cost
    pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0.0

    # -----------------------------------------------------------------------
    # 1. KPI Strip: Value, Cost, P&L, P&L %
    # -----------------------------------------------------------------------
    kpi_items = [
        {
            "label": "TOTAL VALUE (INR)",
            "value": f"INR {total_val:,.0f}",
            "num_val": total_val,
            "prefix": "INR ",
            "delta": f"{'+' if pnl >= 0 else '-'}INR {abs(pnl):,.0f}",
            "delta_pos": pnl >= 0,
        },
        {
            "label": "INVESTED CAPITAL (INR)",
            "value": f"INR {total_cost:,.0f}",
            "num_val": total_cost,
            "prefix": "INR ",
            "delta": "Cost basis",
            "delta_pos": True,
        },
        {
            "label": "NET UNREALIZED P&L",
            "value": f"{pnl:+,.0f}",
            "num_val": abs(pnl),
            "prefix": "+" if pnl >= 0 else "-",
            "delta": f"{pnl_pct:+.2f}%",
            "delta_pos": pnl >= 0,
        },
        {
            "label": "RETURN ON INVESTMENT",
            "value": f"{pnl_pct:+.2f}%",
            "num_val": abs(pnl_pct),
            "prefix": "+" if pnl_pct >= 0 else "-",
            "suffix": "%",
            "delta": "Holding period return",
            "delta_pos": pnl >= 0,
        },
    ]
    render_kpi_strip(kpi_items, height=115)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 2. Holdings Table with In-Cell Values and Progress Columns
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header("PORTFOLIO ASSET INVENTORY", subtitle="LIVE ASSET PRICING, VALUE AND PERFORMANCE")
        if table_data:
            df_disp = pd.DataFrame(table_data)
        else:
            # Synthetic row build from input
            rows = []
            for h in holdings_list:
                tkr = h["ticker"]
                q = h["quantity"]
                bp = h["buy_price"]
                cp = bp          # no price data: show cost basis (P&L 0), never an invented gain
                v = q * cp
                c_pnl = v - (q * bp)
                c_pnl_pct = (c_pnl / (q * bp)) * 100
                rows.append({
                    "Ticker": tkr,
                    "Quantity": q,
                    "Buy Price": bp,
                    "Current Price": cp,
                    "Total Value": v,
                    "P&L (INR)": c_pnl,
                    "P&L (%)": c_pnl_pct,
                })
            df_disp = pd.DataFrame(rows)

        st.dataframe(
            df_disp,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Quantity": st.column_config.NumberColumn("Quantity", format="%d"),
                "Buy Price": st.column_config.NumberColumn("Buy Price", format="INR %.2f"),
                "Current Price": st.column_config.NumberColumn("Current Price", format="INR %.2f"),
                "Total Value": st.column_config.NumberColumn("Total Value", format="INR %.0f"),
                "P&L (INR)": st.column_config.NumberColumn("P&L (INR)", format="INR %+.0f"),
                "P&L (%)": st.column_config.NumberColumn("P&L (%)", format="%+.2f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # 3. Allocation Donut & P&L Bar Chart
    # -----------------------------------------------------------------------
    col_donut, col_pnl = st.columns([1, 1])
    with col_donut:
        with st.container(border=True):
            ui.card_header("CURRENT CAPITAL ALLOCATION", subtitle="PROPORTIONAL VALUE RATIO")
            alloc_map = {row["Ticker"]: row["Total Value"] for row in df_disp.to_dict("records")}
            tot = sum(alloc_map.values()) or 1.0
            alloc_norm = {k: v / tot for k, v in alloc_map.items()}
            fig_pie = charts.allocation_pie(alloc_norm)
            st.plotly_chart(fig_pie, use_container_width=True, config=charts.PLOT_CONFIG, key="track_chart_1")

    with col_pnl:
        with st.container(border=True):
            ui.card_header("HOLDINGS NET P&L", subtitle="GAIN OR LOSS PER POSITION")
            holdings_recs = [
                {"ticker": row["Ticker"], "pnl": row["P&L (INR)"]}
                for row in df_disp.to_dict("records")
            ]
            fig_pnl_bars = charts.holdings_pnl_bar(holdings_recs)
            st.plotly_chart(fig_pnl_bars, use_container_width=True, config=charts.PLOT_CONFIG, key="track_chart_2")
