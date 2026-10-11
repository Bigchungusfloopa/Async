"""
app/main.py — Investment Risk Prediction System
Entry point: streamlit run app/main.py

Layout: sidebar (inputs) + 7 tabs (Risk, Allocation, Simulate, Track, Compare, Scenarios, Market Intel).
Session state, caching, i18n, error handling all wired here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd

# ─── path setup so imports work from project root ──────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─── internal imports ───────────────────────────────────────────────────────
from app.ui import inject_css, footer, risk_badge, info_banner
from ml.i18n import t, set_offline, usage_stats
import app.api as api
from app.report import export_button

# ─── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─── load universe ──────────────────────────────────────────────────────────
@st.cache_data
def _load_universe() -> dict:
    path = _ROOT / "data" / "universe.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


_UNIVERSE = _load_universe()

_ALL_TICKERS: list[str] = (
    [item["ticker"] for item in _UNIVERSE.get("NSE", [])]
    + [item["ticker"] for item in _UNIVERSE.get("US", [])]
)
_TICKER_NAMES: dict[str, str] = {
    item["ticker"]: f"{item['ticker']} – {item['name']}"
    for cat in _UNIVERSE.values()
    for item in cat
    if isinstance(item, dict) and "ticker" in item
}


# ─── session state defaults ─────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "lang": "en",
        "offline": False,
        "tickers": ["TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "amounts": {"TCS.NS": 33000.0, "INFY.NS": 33000.0, "HDFCBANK.NS": 34000.0},
        "total_investment": 100000.0,
        "risk_level": "Medium",
        "start": (date.today() - timedelta(days=3*365)).isoformat(),
        "end":   date.today().isoformat(),
        "benchmark": "^NSEI",
        "rf": 6.5,
        # internal
        "_llm_usage": {
            "gemini":  {"calls": 0, "cache_hits": 0, "failures": 0},
            "groq":    {"calls": 0, "cache_hits": 0, "failures": 0},
            "mistral": {"calls": 0, "cache_hits": 0, "failures": 0},
        },
        "_news_generated": False,
        "_sim_preset": None,
        "_sim_custom": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ─── derived helpers ─────────────────────────────────────────────────────────
def _weights_from_amounts() -> dict[str, float]:
    """Normalised weights from current amounts."""
    amounts = st.session_state["amounts"]
    total = sum(amounts.values()) or 1.0
    return {k: v / total for k, v in amounts.items()}


def _on_lang_change():
    pass  # just triggers rerun


def _on_offline_change():
    set_offline(st.session_state["_offline_toggle"])
    st.session_state["offline"] = st.session_state["_offline_toggle"]


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Language + Offline ──────────────────────────────────────────────────
    st.markdown(f"## 📊 {t('sidebar_title', st.session_state['lang'])}")

    col_lang, col_offline = st.columns([2, 1])
    with col_lang:
        lang_map = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
        lang_display = {v: k for k, v in lang_map.items()}
        chosen_lang_label = st.selectbox(
            t("language", st.session_state["lang"]),
            options=list(lang_map.keys()),
            index=list(lang_map.values()).index(st.session_state["lang"]),
            key="_lang_select",
            on_change=lambda: st.session_state.update(
                {"lang": lang_map[st.session_state["_lang_select"]]}
            ),
        )
        st.session_state["lang"] = lang_map[chosen_lang_label]
    with col_offline:
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        st.toggle(
            t("offline_mode", st.session_state["lang"]),
            value=st.session_state["offline"],
            key="_offline_toggle",
            on_change=_on_offline_change,
        )

    lang = st.session_state["lang"]

    # ── LLM Usage Panel ─────────────────────────────────────────────────────
    with st.expander(t("llm_usage", lang), expanded=False):
        stats = usage_stats()
        rows = []
        for provider, data in stats.items():
            rows.append({
                t("provider", lang): provider.capitalize(),
                t("calls", lang):     data["calls"],
                t("cache_hits", lang):data["cache_hits"],
                t("failures", lang):  data["failures"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Voice Input ─────────────────────────────────────────────────────────
    with st.expander(t("voice_input", lang), expanded=False):
        lang_hint_map = {"en": "en", "hi": "hi", "mr": "mr"}
        audio = st.audio_input(t("record_audio", lang), key="voice_audio_input")
        if audio is not None:
            with st.spinner("Transcribing…"):
                try:
                    transcript_text = api.transcribe(
                        audio.read(), lang_hint=lang_hint_map.get(lang, "hi")
                    )
                except Exception:
                    transcript_text = ""
                    st.warning("Please type instead.")
            if "voice_transcript" not in st.session_state or audio:
                st.session_state["voice_transcript"] = transcript_text

        transcript = st.text_area(
            t("transcript", lang),
            value=st.session_state.get("voice_transcript", ""),
            key="voice_transcript_area",
            height=80,
        )

        if st.button(t("fill_portfolio", lang), key="voice_fill_btn"):
            with st.spinner("Extracting portfolio…"):
                try:
                    extracted = api.extract_portfolio(transcript)
                    if extracted:
                        new_tickers = extracted.get("tickers", [])
                        new_amounts = extracted.get("amounts", {})
                        new_risk    = extracted.get("risk_level")
                        unresolved  = extracted.get("unresolved", [])

                        if new_tickers:
                            st.session_state["tickers"] = new_tickers
                        if new_amounts:
                            st.session_state["amounts"] = {k: float(v) for k, v in new_amounts.items()}
                            st.session_state["total_investment"] = sum(st.session_state["amounts"].values())
                        if new_risk:
                            st.session_state["risk_level"] = new_risk.capitalize()

                        if unresolved:
                            st.warning(f"{t('unresolved', lang)}: {', '.join(unresolved)}")
                        else:
                            st.success("✅ Portfolio filled from transcript!")
                        st.rerun()
                except Exception as e:
                    st.warning(f"Extraction failed: {e}. {t('retry', lang)}")

    st.divider()

    # ── Asset Selection ─────────────────────────────────────────────────────
    st.markdown(f"**{t('assets', lang)}**")
    selected_tickers = st.multiselect(
        t("assets", lang),
        options=_ALL_TICKERS,
        default=st.session_state["tickers"],
        format_func=lambda tkr: _TICKER_NAMES.get(tkr, tkr),
        key="ticker_multiselect",
        label_visibility="collapsed",
    )

    # Free-text ticker entry + validation
    extra_ticker = st.text_input(t("free_text_ticker", lang), key="extra_ticker_input",
                                 placeholder="e.g. BAJAJFINSV.NS").strip().upper()
    if extra_ticker and st.button("+ Add", key="add_ticker_btn"):
        with st.spinner("Validating…"):
            try:
                valid = api.validate_tickers((extra_ticker,))
                if valid.get(extra_ticker, False):
                    if extra_ticker not in selected_tickers:
                        selected_tickers = selected_tickers + [extra_ticker]
                    st.success(f"✅ {extra_ticker} added.")
                else:
                    st.error(t("invalid_ticker", lang, extra_ticker))
            except Exception:
                selected_tickers = selected_tickers + [extra_ticker]
                st.warning(f"Could not validate {extra_ticker} (offline).")

    # Update session tickers
    if selected_tickers != st.session_state["tickers"]:
        # Keep amounts for existing tickers, default 0 for new ones
        new_amounts = {}
        for tkr in selected_tickers:
            new_amounts[tkr] = st.session_state["amounts"].get(tkr, 0.0)
        st.session_state["tickers"]  = selected_tickers
        st.session_state["amounts"]  = new_amounts

    st.divider()

    # ── Amounts / Weights Table ─────────────────────────────────────────────
    st.markdown(f"**{t('amounts_weights', lang)}**")
    tickers_now = st.session_state["tickers"]

    if tickers_now:
        total_inv = st.session_state["total_investment"]
        amounts   = st.session_state["amounts"]

        # Build DataFrame for editor
        editor_data = pd.DataFrame([
            {
                t("ticker_col", lang): tkr,
                t("amount_col", lang): amounts.get(tkr, 0.0),
                t("weight_col", lang): round(amounts.get(tkr, 0.0) / max(sum(amounts.values()), 1) * 100, 1),
            }
            for tkr in tickers_now
        ])

        edited = st.data_editor(
            editor_data,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                t("ticker_col", lang):  st.column_config.TextColumn(disabled=True),
                t("amount_col", lang):  st.column_config.NumberColumn(min_value=0, format="%.0f"),
                t("weight_col", lang):  st.column_config.NumberColumn(disabled=True, format="%.1f"),
            },
            key="amounts_editor",
        )

        # Update amounts from editor
        new_amounts = {}
        for _, row in edited.iterrows():
            tkr = row[t("ticker_col", lang)]
            amt = float(row[t("amount_col", lang)] or 0)
            new_amounts[tkr] = amt
        st.session_state["amounts"] = new_amounts

        new_total = sum(new_amounts.values())
        total_inv_input = st.number_input(
            t("total_investment", lang),
            min_value=0.0,
            value=float(new_total) if new_total > 0 else st.session_state["total_investment"],
            step=1000.0,
            format="%.0f",
            key="total_inv_input",
        )
        st.session_state["total_investment"] = total_inv_input

    st.divider()

    # ── Risk Level ──────────────────────────────────────────────────────────
    risk_options = ["Low", "Medium", "High"]
    risk_labels  = [t(r, lang) for r in risk_options]
    risk_idx     = risk_options.index(st.session_state["risk_level"]) if st.session_state["risk_level"] in risk_options else 1

    chosen_risk_label = st.radio(
        t("risk_level", lang),
        options=risk_labels,
        index=risk_idx,
        horizontal=True,
        key="risk_radio",
    )
    st.session_state["risk_level"] = risk_options[risk_labels.index(chosen_risk_label)]

    st.divider()

    # ── Date Range + Benchmark ──────────────────────────────────────────────
    st.markdown(f"**{t('date_range', lang)}**")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            t("start_date", lang),
            value=pd.to_datetime(st.session_state["start"]),
            key="start_date_input",
        )
    with col_d2:
        end_date = st.date_input(
            t("end_date", lang),
            value=pd.to_datetime(st.session_state["end"]),
            key="end_date_input",
        )
    st.session_state["start"] = start_date.strftime("%Y-%m-%d")
    st.session_state["end"]   = end_date.strftime("%Y-%m-%d")

    benchmark_opts = {"NIFTY 50": "^NSEI", "NASDAQ-100": "^NDX"}
    bench_display  = {v: k for k, v in benchmark_opts.items()}
    chosen_bench   = st.selectbox(
        t("benchmark", lang),
        options=list(benchmark_opts.keys()),
        index=list(benchmark_opts.values()).index(st.session_state["benchmark"])
            if st.session_state["benchmark"] in benchmark_opts.values() else 0,
        key="benchmark_select",
    )
    st.session_state["benchmark"] = benchmark_opts[chosen_bench]

    with st.expander(t("advanced", lang), expanded=False):
        rf = st.slider(t("risk_free_rate", lang), min_value=0.0, max_value=15.0,
                       value=st.session_state["rf"], step=0.25,
                       key="rf_slider")
        st.session_state["rf"] = rf

    st.divider()

    # ── Save / Load / Export ────────────────────────────────────────────────
    session_name = st.text_input(t("session_name", lang),
                                 value="My Portfolio",
                                 key="session_name_input")

    col_save, col_load = st.columns(2)
    with col_save:
        if st.button(t("save_portfolio", lang), key="save_btn", use_container_width=True):
            with st.spinner("Saving…"):
                try:
                    ok = api.save_portfolio(
                        session_name,
                        st.session_state["tickers"],
                        st.session_state["amounts"],
                        st.session_state["risk_level"],
                        lang,
                    )
                    if ok:
                        st.success("Saved ✅")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with col_load:
        if st.button(t("load_portfolio", lang), key="load_btn", use_container_width=True):
            with st.spinner("Loading…"):
                try:
                    saved = api.load_portfolio(session_name)
                    if saved:
                        st.session_state["tickers"]          = saved.get("tickers", [])
                        st.session_state["amounts"]          = saved.get("amounts", {})
                        st.session_state["risk_level"]       = saved.get("risk_level", "Medium")
                        st.session_state["total_investment"] = sum(saved.get("amounts", {}).values())
                        st.success("Loaded ✅")
                        st.rerun()
                    else:
                        st.warning("Portfolio not found.")
                except Exception as e:
                    st.error(f"Load failed: {e}")

    # PDF Export
    st.divider()
    _exp_tickers = st.session_state["tickers"]
    _exp_weights = _weights_from_amounts()
    if _exp_tickers:
        export_button(
            lang=lang,
            tickers=_exp_tickers,
            weights=st.session_state["amounts"],
            risk_label=st.session_state["risk_level"],
            metrics={},   # Will be populated from Risk tab calls
            rec_weights={},
            explanation="See the Risk tab for full analysis.",
            session_name=session_name,
        )


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ════════════════════════════════════════════════════════════════════════════
lang            = st.session_state["lang"]
tickers         = st.session_state["tickers"]
amounts         = st.session_state["amounts"]
total_investment= st.session_state["total_investment"]
risk_level      = st.session_state["risk_level"]
start           = st.session_state["start"]
end             = st.session_state["end"]
benchmark       = st.session_state["benchmark"]
weights         = _weights_from_amounts()

# Header
st.markdown(
    '<h1 style="font-size:1.8rem;font-weight:800;color:#1a2035;margin-bottom:4px;">'
    '📊 Investment Risk Prediction System</h1>'
    '<p style="color:#64748b;font-size:0.95rem;margin-bottom:0;">AI-powered portfolio analysis · ML risk classification · Multilingual</p>',
    unsafe_allow_html=True,
)
st.markdown("")

# Warnings
if st.session_state["offline"]:
    st.warning(f"🔌 {t('llm_offline', lang)}")

if not tickers:
    st.info("👈 Add assets in the sidebar to get started.")
    footer(lang)
    st.stop()

# Single asset notice
if len(tickers) == 1:
    info_banner(t("single_asset", lang))

# Mixed currency detection
nse_count = sum(1 for tk in tickers if tk.endswith(".NS") or tk.endswith(".BO"))
us_count  = sum(1 for tk in tickers if not tk.endswith(".NS") and not tk.endswith(".BO") and not tk.startswith("^"))
if nse_count > 0 and us_count > 0:
    st.warning(t("mixed_currencies", lang))

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_labels = [
    t("tab_risk", lang),
    t("tab_allocation", lang),
    t("tab_simulate", lang),
    t("tab_track", lang),
    t("tab_compare", lang),
    t("tab_scenarios", lang),
    t("tab_news", lang),
]
tabs = st.tabs(tab_labels)

common = dict(
    lang=lang,
    tickers=tickers,
    weights=weights,
    risk_level=risk_level,
    start=start,
    end=end,
    benchmark=benchmark,
    total_investment=total_investment,
)

# Tab 0: Risk
with tabs[0]:
    try:
        from app.tabs.build import render as _render_risk
        _render_risk(**common)
    except Exception as e:
        st.error(f"Risk tab error: {e}")

# Tab 1: Allocation
with tabs[1]:
    try:
        from app.tabs.allocation import render as _render_alloc
        _render_alloc(**common)
    except Exception as e:
        st.error(f"Allocation tab error: {e}")

# Tab 2: Simulate
with tabs[2]:
    try:
        from app.tabs.simulate import render as _render_sim
        _render_sim(**common)
    except Exception as e:
        st.error(f"Simulate tab error: {e}")

# Tab 3: Track
with tabs[3]:
    try:
        from app.tabs.track import render as _render_track
        _render_track(**common)
    except Exception as e:
        st.error(f"Track tab error: {e}")

# Tab 4: Compare
with tabs[4]:
    try:
        from app.tabs.compare import render as _render_compare
        _render_compare(**common)
    except Exception as e:
        st.error(f"Compare tab error: {e}")

# Tab 5: Scenarios
with tabs[5]:
    try:
        from app.tabs.scenarios import render as _render_scenarios
        _render_scenarios(**common)
    except Exception as e:
        st.error(f"Scenarios tab error: {e}")

# Tab 6: Market Intel
with tabs[6]:
    try:
        from app.tabs.news import render as _render_news
        _render_news(**common)
    except Exception as e:
        st.error(f"Market Intel tab error: {e}")

# ── Footer ──────────────────────────────────────────────────────────────────
footer(lang)
