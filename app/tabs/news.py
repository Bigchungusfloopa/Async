"""
app/tabs/news.py — Market Intel tab.
Disclaimer banner; Generate Insight button (never auto-call);
per-ticker sentiment chips, sector trends, risks, summary, headlines.
Shows source (live/cache/offline).
"""

from __future__ import annotations

import streamlit as st

import app.api as api
from app.ui import sentiment_chip, source_badge, info_banner
from ml.i18n import t


def render(lang: str, tickers: list[str], **kwargs) -> None:
    """Render the Market Intel tab."""

    # Disclaimer — always visible
    st.markdown(
        f'<div style="background:#fffbeb;border:1px solid #F39C12;border-radius:10px;'
        f'padding:12px 16px;font-size:0.9rem;color:#7c3d00;font-weight:500;">'
        f'{t("news_disclaimer", lang)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if not tickers:
        st.info(t("no_data", lang))
        return

    offline = st.session_state.get("offline", False)
    if offline:
        info_banner(t("llm_offline", lang))

    # -----------------------------------------------------------------------
    # Generate Insight button — never auto-call
    # -----------------------------------------------------------------------
    if st.button(t("generate_insight", lang), key="news_generate_btn",
                 type="primary", use_container_width=False):
        st.session_state["_news_generated"] = True

    if not st.session_state.get("_news_generated"):
        st.markdown(
            '<div style="color:#94a3b8;font-size:0.9rem;margin-top:8px;">'
            'Click "Generate Insight" to fetch AI-powered market analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    # -----------------------------------------------------------------------
    # Fetch insight
    # -----------------------------------------------------------------------
    with st.spinner("Fetching market intelligence…"):
        try:
            insight = api.get_news_insight(tuple(tickers), lang)
        except Exception as e:
            st.error(f"{t('no_data', lang)}: {e}")
            return

    source = insight.get("source", "cache")
    source_badge(source, lang)
    st.markdown("")

    # -----------------------------------------------------------------------
    # Overall sentiment
    # -----------------------------------------------------------------------
    sentiment = insight.get("sentiment", "Neutral")
    col_sent, col_empty = st.columns([1, 3])
    with col_sent:
        st.markdown(f"**{t('sentiment', lang)}**")
        sentiment_chip(sentiment)

    st.divider()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary = insight.get("summary", "")
    if summary:
        st.markdown(
            f'<div style="background:#f8fafb;border-radius:10px;padding:16px;'
            f'border-left:4px solid #4F8EF7;font-size:0.95rem;line-height:1.7;">'
            f'{summary}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # -----------------------------------------------------------------------
    # Sector trends
    # -----------------------------------------------------------------------
    sector_trends = insight.get("sector_trends", "")
    if sector_trends:
        st.markdown(f"**{t('sector_trends', lang)}**")
        st.markdown(sector_trends)

    st.divider()

    # -----------------------------------------------------------------------
    # Key risks
    # -----------------------------------------------------------------------
    risks = insight.get("risks", [])
    if risks:
        st.markdown(f"**{t('risks', lang)}**")
        for risk in risks:
            st.markdown(f"- ⚠️ {risk}")

    st.divider()

    # -----------------------------------------------------------------------
    # Headlines
    # -----------------------------------------------------------------------
    headlines = insight.get("headlines", [])
    if headlines:
        st.markdown(f"**{t('headlines', lang)}**")
        for h in headlines:
            url   = h.get("url", "#")
            title = h.get("title", "")
            source_name = h.get("source", "")
            st.markdown(
                f'<div style="margin:6px 0;">'
                f'<a href="{url}" target="_blank" style="color:#4F8EF7;text-decoration:none;font-size:0.92rem;">'
                f'🔗 {title}</a>'
                f'<span style="font-size:0.78rem;color:#94a3b8;margin-left:8px;">{source_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
