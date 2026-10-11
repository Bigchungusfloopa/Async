"""
app/ui.py — UI helper utilities: risk badges, metric cards, formatting, CSS.
"""

from __future__ import annotations

import streamlit as st
from ml.i18n import t


# ---------------------------------------------------------------------------
# Color palette (colorblind-safe)
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Low":    "#2ECC71",   # green
    "Medium": "#F39C12",   # amber
    "High":   "#E74C3C",   # red
}

STRATEGY_COLORS = {
    "Original":    "#4F8EF7",
    "Recommended": "#2ECC71",
    "Equal Weight": "#F39C12",
    "Benchmark":   "#9B59B6",
}

SENTIMENT_COLORS = {
    "Positive": "#2ECC71",
    "Neutral":  "#95A5A6",
    "Negative": "#E74C3C",
}


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inject global CSS for the app."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.10);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a2035;
    line-height: 1.2;
}
.metric-delta {
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 4px;
}
.metric-positive { color: #2ECC71; }
.metric-negative { color: #E74C3C; }
.metric-neutral  { color: #64748b; }

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
}
.risk-low    { background: #d4f5e2; color: #166534; border: 2px solid #2ECC71; }
.risk-medium { background: #fef3cd; color: #7c3d00; border: 2px solid #F39C12; }
.risk-high   { background: #fde8e8; color: #7f1d1d; border: 2px solid #E74C3C; }

/* Tab styling */
div[data-testid="stTabs"] button {
    font-weight: 600;
    font-size: 0.88rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafb 0%, #eef2f7 100%);
}

/* Footer */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.75rem;
    padding: 16px;
    border-top: 1px solid #e2e8f0;
    margin-top: 32px;
}

/* Sentiment chip */
.sentiment-chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Source badge */
.source-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: #e2e8f0;
    color: #475569;
}

/* Info banner */
.info-banner {
    background: #eff6ff;
    border-left: 4px solid #4F8EF7;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.86rem;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Risk badge
# ---------------------------------------------------------------------------
def risk_badge(label: str, lang: str = "en") -> None:
    """Render a large risk level badge."""
    cls = f"risk-{label.lower()}"
    display_label = t(label, lang)
    st.markdown(
        f'<div style="text-align:center; padding:16px 0;">'
        f'<span class="risk-badge {cls}">{display_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric card (HTML-based, prettier than st.metric)
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, delta: str | None = None,
                delta_positive: bool | None = None) -> None:
    """Render a styled metric card."""
    delta_html = ""
    if delta is not None:
        if delta_positive is True:
            cls = "metric-positive"
        elif delta_positive is False:
            cls = "metric-negative"
        else:
            cls = "metric-neutral"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal fraction as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_inr(value: float) -> str:
    """Format value in Indian Rupee notation."""
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.0f}"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a plain number."""
    return f"{value:.{decimals}f}"


def pnl_color(value: float) -> str:
    """Return HTML color string for P&L value."""
    return "#2ECC71" if value >= 0 else "#E74C3C"


def sentiment_chip(sentiment: str) -> None:
    """Render a sentiment chip."""
    color = SENTIMENT_COLORS.get(sentiment, "#95A5A6")
    bg = color + "22"
    st.markdown(
        f'<span class="sentiment-chip" style="background:{bg};color:{color};border:1px solid {color};">'
        f'{sentiment}</span>',
        unsafe_allow_html=True,
    )


def source_badge(source: str, lang: str = "en") -> None:
    """Render a small source badge."""
    label_map = {"live": t("source_live", lang), "cache": t("source_cache", lang),
                 "offline": t("source_offline", lang)}
    label = label_map.get(source.lower(), source)
    st.markdown(f'<span class="source-badge">{label}</span>', unsafe_allow_html=True)


def footer(lang: str = "en") -> None:
    """Render the app footer with disclaimer."""
    st.markdown(
        f'<div class="app-footer">{t("footer", lang)}</div>',
        unsafe_allow_html=True,
    )


def info_banner(text: str) -> None:
    """Render a styled info banner."""
    st.markdown(f'<div class="info-banner">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Probability bar for risk classification
# ---------------------------------------------------------------------------
def prob_bar(probs: dict[str, float], lang: str = "en") -> None:
    """Render colored probability bars for Low/Medium/High."""
    cols = st.columns(3)
    for i, (label, color) in enumerate(RISK_COLORS.items()):
        val = probs.get(label, 0.0)
        with cols[i]:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div style="font-size:0.78rem;font-weight:600;color:#64748b;margin-bottom:4px;">'
                f'{t(label, lang)}</div>'
                f'<div style="background:#f1f5f9;border-radius:8px;overflow:hidden;height:8px;">'
                f'<div style="background:{color};width:{val*100:.0f}%;height:100%;border-radius:8px;"></div>'
                f'</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{color};margin-top:4px;">'
                f'{val*100:.0f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
