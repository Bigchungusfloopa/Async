"""
app/ui.py — Black, Minimal, Animated Trading Terminal UI Components.
HARD RULES:
- No gradients anywhere.
- No shadows, glows, blur, glassmorphism, or 3D.
- No emojis anywhere.
- Tabular monospace numbers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import streamlit as st
from ml.i18n import t

# ---------------------------------------------------------------------------
# Strict Design Tokens & Colors (No gradients, no soft shadows)
# ---------------------------------------------------------------------------
TOKENS = {
    "bg": "#000000",
    "surface": "#0A0A0A",
    "surface_2": "#111111",
    "border": "#1F1F1F",
    "border_strong": "#2E2E2E",
    "text": "#F5F5F5",
    "text_muted": "#8A8A8A",
    "text_dim": "#5C5C5C",
    "pos": "#22C55E",
    "neg": "#EF4444",
    "warn": "#F59E0B",
    "info": "#3B82F6",
}

RISK_COLORS = {
    "Low": "#22C55E",
    "Medium": "#F59E0B",
    "High": "#EF4444",
}

STRATEGY_COLORS = {
    "Original": "#F5F5F5",
    "Recommended": "#3B82F6",
    "Equal Weight": "#F59E0B",
    "Benchmark": "#6B7280",
}

SENTIMENT_COLORS = {
    "Positive": "#22C55E",
    "Neutral": "#8A8A8A",
    "Negative": "#EF4444",
}

# Regex to strip any emoji codepoints
_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F\u2300-\u23FF\u2B50-\u2B55\u200D\u20E3]"
)


def strip_emojis(text: Any) -> str:
    """Strip emojis from any external or internal text."""
    if not isinstance(text, str):
        return str(text)
    return _EMOJI_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# CSS Injection
# ---------------------------------------------------------------------------
@st.cache_data
def _load_style_css() -> str:
    """Read app/style.css from disk."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def inject_css() -> None:
    """Inject the global black minimal design stylesheet."""
    css = _load_style_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Animation State Tracking (Prevent replay on widget interaction reruns)
# ---------------------------------------------------------------------------
def is_first_view(tab_name: str) -> bool:
    """Check if a tab is being viewed for the first time in this session."""
    if "_seen_tabs" not in st.session_state:
        st.session_state["_seen_tabs"] = set()
    first = tab_name not in st.session_state["_seen_tabs"]
    st.session_state["_seen_tabs"].add(tab_name)
    return first


def track_value_change(key: str, new_val: Any) -> str:
    """Return flash-pos or flash-neg CSS class if value changed, else empty string."""
    prev_key = f"_prev_val_{key}"
    prev = st.session_state.get(prev_key)
    st.session_state[prev_key] = new_val
    if prev is not None and prev != new_val:
        try:
            if float(new_val) > float(prev):
                return "flash-pos"
            elif float(new_val) < float(prev):
                return "flash-neg"
        except (ValueError, TypeError):
            return "flash-pos"
    return ""


# ---------------------------------------------------------------------------
# Layout Components
# ---------------------------------------------------------------------------
def top_bar(source: str = "LIVE", llm_mode: str = "ONLINE", lang: str = "en") -> None:
    """Render top bar with wordmark and status chips (no logo image, no emojis)."""
    src_upper = source.upper()
    llm_upper = llm_mode.upper()
    lang_upper = lang.upper()

    src_class = "live" if src_upper == "LIVE" else ("cache" if src_upper == "CACHE" else "fallback")
    llm_class = "online" if llm_upper == "ONLINE" else "offline"

    st.markdown(
        f"""
        <div class="top-bar">
            <div class="top-wordmark">INVESTMENT RISK TERMINAL</div>
            <div class="status-chips-group">
                <span class="status-chip">
                    <span class="status-dot {src_class}"></span>
                    <span>SOURCE: {src_upper}</span>
                </span>
                <span class="status-chip">
                    <span class="status-dot {llm_class}"></span>
                    <span>LLM: {llm_upper}</span>
                </span>
                <span class="status-chip">
                    <span class="status-dot"></span>
                    <span>LANG: {lang_upper}</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    """Page-level heading with a clean 1px hairline rule."""
    sub_html = f'<div class="section-subtitle">{strip_emojis(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-top: 14px; margin-bottom: 12px;">
            <div class="section-title">{strip_emojis(title)}</div>
            {sub_html}
            <div class="section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_header(title: str, subtitle: str | None = None, right: str | None = None) -> None:
    """Header row inside a card container."""
    right_html = f'<div class="status-chip">{strip_emojis(right)}</div>' if right else ""
    sub_html = f'<div class="card-subtitle">{strip_emojis(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="card-title-row">
            <div>
                <div class="card-title">{strip_emojis(title).upper()}</div>
                {sub_html}
            </div>
            {right_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chip(text: str, status: str = "neutral") -> str:
    """Return HTML for a status chip with 1px border and flat status dot."""
    status_map = {
        "live": "live",
        "online": "online",
        "pos": "pos",
        "warn": "warn",
        "neg": "neg",
        "offline": "offline",
        "cache": "cache",
        "fallback": "fallback",
    }
    cls = status_map.get(status.lower(), "")
    return f"""<span class="status-chip"><span class="status-dot {cls}"></span><span>{strip_emojis(text).upper()}</span></span>"""


def delta(value: float | str, is_pct: bool = True) -> str:
    """Return HTML for a signed delta with small CSS triangle glyph (no emojis)."""
    try:
        val_f = float(str(value).replace("%", "").replace("+", "").strip())
        pos = val_f >= 0
        sign = "+" if pos else ""
        glyph = "▲" if pos else "▼"
        cls = "pos" if pos else "neg"
        text = f"{sign}{val_f:.2f}%" if is_pct else f"{sign}{val_f:,.2f}"
    except (ValueError, TypeError):
        pos = True
        glyph = ""
        cls = "pos"
        text = str(value)

    return f"""<span class="delta-val {cls}"><span class="delta-glyph">{glyph}</span> {text}</span>"""


def empty_state(text: str) -> None:
    """Render empty state with a thin-line abstract chart and mono prompt."""
    st.markdown(
        f"""
        <div class="empty-state-box">
            <svg width="120" height="40" viewBox="0 0 120 40" fill="none">
                <line x1="0" y1="35" x2="120" y2="35" stroke="#1F1F1F" stroke-width="1"/>
                <polyline points="5,30 25,25 45,28 65,15 85,20 105,8 115,12" stroke="#2E2E2E" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div class="empty-state-text">{strip_emojis(text).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Backward-compatible Helpers (Refactored to Black Minimal Style)
# ---------------------------------------------------------------------------
def risk_badge(label: str, lang: str = "en") -> None:
    """Render a flat hairline risk badge (no gradient, no shadow, no emoji)."""
    norm = label.capitalize()
    color = RISK_COLORS.get(norm, "#F59E0B")
    display_label = strip_emojis(t(norm, lang)).upper()
    st.markdown(
        f"""
        <div style="display:inline-flex; align-items:center; gap:8px; border:1px solid {color}; border-radius:4px; padding:4px 12px; background:#0A0A0A;">
            <span style="width:6px; height:6px; border-radius:50%; background:{color};"></span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:{color}; letter-spacing:0.06em;">{display_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta_val: str | None = None,
                delta_positive: bool | None = None) -> None:
    """Render a flat card with uppercase label, big tabular mono value, and delta."""
    delta_html = ""
    if delta_val is not None:
        cls = "pos" if delta_positive is not False else "neg"
        glyph = "▲" if delta_positive is not False else "▼"
        delta_html = f"""<div class="delta-val {cls}" style="margin-top:6px;"><span class="delta-glyph">{glyph}</span> {delta_val}</div>"""

    st.markdown(
        f"""
        <div style="background:#0A0A0A; border:1px solid #1F1F1F; border-radius:6px; padding:14px 16px;">
            <div style="font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#8A8A8A;">{strip_emojis(label)}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; font-size:24px; font-weight:600; color:#F5F5F5; line-height:1.2; margin-top:4px;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_chip(sentiment: str) -> None:
    """Render flat sentiment chip."""
    color = SENTIMENT_COLORS.get(sentiment, "#8A8A8A")
    st.markdown(
        f"""
        <span style="display:inline-flex; align-items:center; gap:6px; border:1px solid {color}; border-radius:4px; padding:2px 8px; font-family:'JetBrains Mono',monospace; font-size:11px; text-transform:uppercase; color:{color}; background:#0A0A0A;">
            <span style="width:5px; height:5px; border-radius:50%; background:{color};"></span>
            <span>{sentiment.upper()}</span>
        </span>
        """,
        unsafe_allow_html=True,
    )


def source_badge(source: str, lang: str = "en") -> None:
    """Render small 1px border source badge."""
    label_map = {
        "live": t("source_live", lang),
        "cache": t("source_cache", lang),
        "offline": t("source_offline", lang),
    }
    label = label_map.get(source.lower(), source)
    st.markdown(
        f"""<span class="status-chip"><span>{strip_emojis(label).upper()}</span></span>""",
        unsafe_allow_html=True,
    )


def info_banner(text: str) -> None:
    """Render flat info banner with 2px left border in info blue."""
    st.markdown(
        f"""<div data-testid="stAlert"><p>{strip_emojis(text)}</p></div>""",
        unsafe_allow_html=True,
    )


def footer(lang: str = "en") -> None:
    """Render the app footer disclaimer."""
    st.markdown(
        f"""<div class="footer-disclaimer">{strip_emojis(t("footer", lang))}</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format decimal fraction as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_inr(value: float) -> str:
    """Format value in INR (no emoji, clean currency)."""
    if abs(value) >= 1e7:
        return f"INR {value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"INR {value/1e5:.2f} L"
    return f"INR {value:,.0f}"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format plain number."""
    return f"{value:.{decimals}f}"


def pnl_color(value: float) -> str:
    """Return pos/neg color string."""
    return "#22C55E" if value >= 0 else "#EF4444"
