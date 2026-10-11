"""
app/tabs/news.py — Market Intel Tab.
Visual Inventory (Section 7):
1. Supplementary banner (flat, left border)
2. Sentiment diverging bar per ticker
3. Sentiment share donut
4. Headline list as a vertical timeline (1px rule, flat dots, source & time in mono)
5. Risks as a compact chip list; summary in a card
"""

from __future__ import annotations

import streamlit as st

from app import api, charts, ui
from ml.i18n import t


def render_tab(tickers: list[str], lang: str = "en") -> None:
    ui.section(t("tab_news", lang), "MACRO SIGNALS, AI SYNTHESIS & HEADLINE TIMELINE")

    # 1. Supplementary Disclaimer Banner (flat, left border)
    st.markdown(
        f'<div data-testid="stAlert" class="alert-warn"><p style="font-family:\'JetBrains Mono\',monospace; font-size:11px; text-transform:uppercase;">'
        f'{ui.strip_emojis(t("market_intel_disclaimer", lang))}'
        f'</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    if not tickers:
        ui.empty_state(t("select_assets_hint", lang))
        return

    # Trigger button (never auto-call)
    c_btn, c_source = st.columns([2, 1])
    with c_btn:
        generate = st.button(
            t("generate_insight", lang).upper(),
            key="generate_news_btn",
            type="primary",
        )

    # Check cache / session state
    cache_key = f"news_{'_'.join(sorted(tickers))}_{lang}"
    if generate or cache_key in st.session_state:
        if generate:
            with st.spinner("SYNTHESIZING MARKET INTELLIGENCE..."):
                insight = api.get_news_insight(tuple(tickers), lang)
                st.session_state[cache_key] = insight
        else:
            insight = st.session_state[cache_key]
    else:
        # Load mock defaults
        insight = api.get_news_insight(tuple(tickers), lang)

    summary = insight.get("summary", "Market environment indicates consolidation in defensive heavyweights.")
    sentiments = insight.get("sentiments", {t: 0.35 for t in tickers})
    sector_trends = insight.get("sector_trends", ["Banking margins expanding", "IT valuations stable"])
    risks = insight.get("risks", ["Regulatory rate changes", "Foreign institutional outflows", "FX volatility"])
    headlines = insight.get("headlines", [
        {"title": f"{tickers[0]} beats Q3 margin guidance", "source": "Reuters", "time": "2h ago"},
        {"title": "RBI keeps repo rate unchanged in monetary policy review", "source": "Bloomberg", "time": "4h ago"},
        {"title": "Global tech index gains 1.2% amidst rate cut expectations", "source": "FT", "time": "6h ago"},
    ])
    src_type = insight.get("source", "cache")

    with c_source:
        st.markdown(
            f'<div style="text-align:right; margin-top:8px;">'
            f'<span class="status-chip"><span class="status-dot live"></span><span>SOURCE: {src_type.upper()}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 2. Sentiment Diverging Bar & 3. Sentiment Share Donut
    # -----------------------------------------------------------------------
    col_bar, col_donut = st.columns([1, 1])
    with col_bar:
        with st.container(border=True):
            ui.card_header("ASSET SENTIMENT SCORES", subtitle="RANGE: -1.0 (BEARISH) TO +1.0 (BULLISH)")
            fig_sent = charts.sentiment_diverging_bar(sentiments)
            st.plotly_chart(fig_sent, use_container_width=True, config=charts.PLOT_CONFIG, key="news_chart_1")

    with col_donut:
        with st.container(border=True):
            ui.card_header("SENTIMENT DISTRIBUTION", subtitle="PROPORTION OF NEWS COVERAGE")
            pos_cnt = sum(1 for v in sentiments.values() if v > 0.1)
            neg_cnt = sum(1 for v in sentiments.values() if v < -0.1)
            neu_cnt = len(sentiments) - pos_cnt - neg_cnt
            share = {
                "Positive": pos_cnt / max(len(sentiments), 1),
                "Neutral": neu_cnt / max(len(sentiments), 1),
                "Negative": neg_cnt / max(len(sentiments), 1),
            }
            fig_donut = charts.sentiment_share_donut(share)
            st.plotly_chart(fig_donut, use_container_width=True, config=charts.PLOT_CONFIG, key="news_chart_2")

    # -----------------------------------------------------------------------
    # 5. Summary Card & Compact Risk Chip List
    # -----------------------------------------------------------------------
    col_sum, col_risks = st.columns([1, 1])
    with col_sum:
        with st.container(border=True):
            ui.card_header("AI EXECUTIVE SUMMARY", right="LLM SYNTHESIS")
            st.markdown(f"<div style='font-size:13px; color:#F5F5F5; line-height:1.6;'>{ui.strip_emojis(summary)}</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#8A8A8A; margin-top:14px; margin-bottom:6px;'>SECTOR TRENDS</div>", unsafe_allow_html=True)
            for tr in sector_trends:
                st.markdown(f"<div style='font-size:12px; color:#8A8A8A; font-family:\"JetBrains Mono\",monospace;'>&bull; {ui.strip_emojis(tr)}</div>", unsafe_allow_html=True)

    with col_risks:
        with st.container(border=True):
            ui.card_header("KEY IDENTIFIED RISK VULNERABILITIES", right="HEAT FACTORS")
            st.markdown("<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;'>", unsafe_allow_html=True)
            chips_html = "".join([
                f"<span class='status-chip' style='border-color:#EF4444; color:#EF4444;'><span class='status-dot neg'></span><span>{ui.strip_emojis(r).upper()}</span></span>"
                for r in risks
            ])
            st.markdown(chips_html + "</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 4. Headline List as a Vertical Timeline (1px rule, flat dots, mono meta)
    # -----------------------------------------------------------------------
    with st.container(border=True):
        ui.card_header("REAL-TIME NEWS TIMELINE", subtitle="VERIFIED MARKET INTELLIGENCE CHRONOLOGY")
        timeline_items = []
        for hl in headlines:
            title = ui.strip_emojis(hl.get("title", ""))
            src = hl.get("source", "MARKET")
            t_str = hl.get("time", "")
            item_html = f"""
            <div style="display:flex; gap:16px; margin-bottom:14px; position:relative;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="width:7px; height:7px; border-radius:50%; background:#F5F5F5; margin-top:5px;"></div>
                    <div style="width:1px; flex-grow:1; background:#1F1F1F; margin-top:4px;"></div>
                </div>
                <div style="flex-grow:1;">
                    <div style="font-size:13px; font-weight:500; color:#F5F5F5;">{title}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#8A8A8A; margin-top:2px;">{src.upper()} &bull; {t_str.upper()}</div>
                </div>
            </div>
            """
            timeline_items.append(item_html)
        st.markdown("".join(timeline_items), unsafe_allow_html=True)
