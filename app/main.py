"""
app/main.py — Investment Risk Prediction System.
Black, Minimal, Animated Trading Terminal Design.
HARD RULES:
- Black canvas (#000000), hairline 1px borders, uppercase labels, tabular figures.
- NO gradients anywhere.
- NO shadows, glows, blur, or 3D.
- NO emojis anywhere in labels, tabs, toasts, or page icon.
- Respect prefers-reduced-motion.
"""

from __future__ import annotations

import hashlib
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
from app.ui import inject_css, footer, top_bar, info_banner, strip_emojis
from app.components.ticker_tape import render_ticker_tape
from ml.i18n import t, set_offline, usage_stats
import app.api as api
from app.report import export_button

# ─── page config (Strictly dark, expanded, no emoji page icon) ──────────────
st.set_page_config(
    page_title="Investment Risk Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─── load universe ──────────────────────────────────────────────────────────
# data/universe.json is the single source of truth (flat list shared with engine/ and ml/); group it by
# market for the pickers: NSE stocks, US stocks, ETFs (gold / bonds / cash / index funds), benchmarks.
@st.cache_data
def _load_universe() -> dict:
    path = _ROOT / "data" / "universe.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):                                  # legacy grouped schema
        return raw
    groups: dict[str, list] = {"NSE": [], "US": [], "ETF": [], "Indices": []}
    for e in raw:
        kind = e.get("type", "stock")
        if kind == "index":
            groups["Indices"].append(e)
        elif kind == "etf":
            groups["ETF"].append(e)
        elif e.get("currency") == "INR":
            groups["NSE"].append(e)
        else:
            groups["US"].append(e)
    return groups


_UNIVERSE = _load_universe()

_ALL_TICKERS: list[str] = (
    [item["ticker"] for item in _UNIVERSE.get("NSE", [])]
    + [item["ticker"] for item in _UNIVERSE.get("US", [])]
    + [item["ticker"] for item in _UNIVERSE.get("ETF", [])]
)
_TICKER_NAMES: dict[str, str] = {
    item["ticker"]: f"{item['ticker']} - {item['name']}"
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
        "end": date.today().isoformat(),
        "benchmark": "^NSEI",
        "rf": 6.5,
        "_seen_tabs": set(),
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


def _weights_from_amounts() -> dict[str, float]:
    """Normalised weights from current amounts."""
    amounts = st.session_state["amounts"]
    total = sum(amounts.values()) or 1.0
    return {k: v / total for k, v in amounts.items()}


def _on_offline_change():
    set_offline(st.session_state["_offline_toggle"])
    st.session_state["offline"] = st.session_state["_offline_toggle"]


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR (320px, strictly grouped under 11px uppercase headings, no emojis)
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-header">LANGUAGE AND MODE</div>', unsafe_allow_html=True)
    col_lang, col_offline = st.columns([2, 1])
    with col_lang:
        lang_map = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
        chosen_lang_label = st.selectbox(
            "LANGUAGE",
            options=list(lang_map.keys()),
            index=list(lang_map.values()).index(st.session_state["lang"]),
            key="_lang_select",
            label_visibility="collapsed",
            on_change=lambda: st.session_state.update(
                {"lang": lang_map[st.session_state["_lang_select"]]}
            ),
        )
        st.session_state["lang"] = lang_map[chosen_lang_label]
    with col_offline:
        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        st.toggle(
            "OFFLINE",
            value=st.session_state["offline"],
            key="_offline_toggle",
            on_change=_on_offline_change,
        )

    lang = st.session_state["lang"]

    # ── LLM Usage Panel ─────────────────────────────────────────────────────
    with st.expander("LLM USAGE PANEL", expanded=False):
        stats = usage_stats()
        rows = []
        for provider, data in stats.items():
            rows.append({
                "Provider": provider.upper(),
                "Calls": data["calls"],
                "Hits": data["cache_hits"],
                "Failures": data["failures"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Voice Input ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">VOICE INPUT</div>', unsafe_allow_html=True)
    with st.expander("RECORD / AUDIO TRANSCRIPT", expanded=False):
        lang_hint_map = {"en": "en", "hi": "hi", "mr": "mr"}
        audio = st.audio_input("AUDIO CAPTURE", key="voice_audio_input")
        if audio is not None:
            # CSS waveform-style static bars while recording / processing
            st.markdown("""
            <div style="display:flex; gap:3px; align-items:center; height:18px; margin:8px 0;">
                <div style="width:2px; height:8px; background:#F5F5F5;"></div>
                <div style="width:2px; height:16px; background:#F5F5F5;"></div>
                <div style="width:2px; height:10px; background:#F5F5F5;"></div>
                <div style="width:2px; height:14px; background:#F5F5F5;"></div>
                <div style="width:2px; height:6px; background:#F5F5F5;"></div>
                <div style="width:2px; height:12px; background:#F5F5F5;"></div>
                <div style="width:2px; height:8px; background:#F5F5F5;"></div>
            </div>
            """, unsafe_allow_html=True)
            raw_audio = audio.getvalue()
            audio_id = hashlib.md5(raw_audio).hexdigest()
            if st.session_state.get("voice_audio_hash") != audio_id:      # once per recording, not per rerun
                with st.spinner("TRANSCRIBING..."):
                    try:
                        transcript_text = api.transcribe(raw_audio, lang_hint=lang_hint_map.get(lang, "hi"))
                    except Exception:
                        transcript_text = ""
                        st.warning("Please type transcript manually.")
                st.session_state["voice_audio_hash"] = audio_id
                st.session_state["voice_transcript"] = transcript_text
                st.session_state["voice_transcript_area"] = transcript_text

        transcript = st.text_area(
            "TRANSCRIPT (EDITABLE)",
            value=st.session_state.get("voice_transcript", ""),
            key="voice_transcript_area",
            height=75,
        )

        if st.button("FILL PORTFOLIO FROM TRANSCRIPT", key="voice_fill_btn"):
            with st.spinner("EXTRACTING PORTFOLIO ENTITIES..."):
                try:
                    extracted = api.extract_portfolio(transcript)
                    if extracted:
                        new_tickers = extracted.get("tickers", [])
                        new_amounts = extracted.get("amounts", {})
                        new_risk = extracted.get("risk_level")
                        unresolved = extracted.get("unresolved", [])

                        if new_tickers:
                            st.session_state["tickers"] = new_tickers
                        if new_amounts:
                            st.session_state["amounts"] = {k: float(v) for k, v in new_amounts.items()}
                            st.session_state["total_investment"] = sum(st.session_state["amounts"].values())
                        if new_risk:
                            st.session_state["risk_level"] = new_risk.capitalize()

                        if unresolved:
                            st.warning(f"Unresolved tickers: {', '.join(unresolved)}")
                        else:
                            st.success("Portfolio populated from transcript.")
                        st.rerun()
                except Exception as e:
                    st.warning(f"Extraction failed: {e}")

    # ── Asset Selection ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">ASSETS</div>', unsafe_allow_html=True)
    selected_tickers = st.multiselect(
        "ASSETS",
        options=_ALL_TICKERS,
        default=st.session_state["tickers"],
        format_func=lambda tkr: _TICKER_NAMES.get(tkr, tkr),
        key="ticker_multiselect",
        label_visibility="collapsed",
    )

    extra_ticker = st.text_input("ADD TICKER MANUALLY", key="extra_ticker_input",
                                 placeholder="e.g. BAJAJFINSV.NS").strip().upper()
    if extra_ticker and st.button("+ ADD TICKER", key="add_ticker_btn"):
        with st.spinner("VALIDATING TICKER SYMBOL..."):
            valid = api.validate_tickers((extra_ticker,))
        if valid.get(extra_ticker, True):
            if extra_ticker not in selected_tickers:
                selected_tickers = selected_tickers + [extra_ticker]
                st.session_state["tickers"] = selected_tickers
                st.session_state["amounts"][extra_ticker] = 10000.0
                st.session_state["total_investment"] = sum(st.session_state["amounts"].values())
                st.success(f"Added {extra_ticker}")
                st.rerun()
        else:
            st.error(f"Ticker {extra_ticker} not recognized.")

    if selected_tickers != st.session_state["tickers"]:
        st.session_state["tickers"] = selected_tickers
        current_amounts = st.session_state["amounts"]
        new_amounts = {}
        for tkr in selected_tickers:
            new_amounts[tkr] = current_amounts.get(
                tkr, st.session_state["total_investment"] / max(len(selected_tickers), 1)
            )
        st.session_state["amounts"] = new_amounts
        st.session_state["total_investment"] = sum(new_amounts.values())
        st.rerun()

    # ── Amounts / Weights ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">AMOUNTS / WEIGHTS</div>', unsafe_allow_html=True)
    total_inv = st.number_input(
        "TOTAL INVESTMENT (INR)",
        min_value=1000.0,
        value=float(st.session_state["total_investment"]),
        step=5000.0,
        key="total_inv_input",
    )
    if total_inv != st.session_state["total_investment"]:
        old_total = st.session_state["total_investment"] or 1.0
        scale = total_inv / old_total
        st.session_state["amounts"] = {
            k: v * scale for k, v in st.session_state["amounts"].items()
        }
        st.session_state["total_investment"] = total_inv
        st.rerun()

    if selected_tickers:
        with st.expander("POSITION ALLOCATION EDITOR", expanded=False):
            norm_weights = _weights_from_amounts()
            editor_rows = [
                {"Ticker": tkr, "Amount": st.session_state["amounts"].get(tkr, 0.0),
                 "Weight %": norm_weights.get(tkr, 0.0) * 100}
                for tkr in selected_tickers
            ]
            df_edit = pd.DataFrame(editor_rows)
            edited = st.data_editor(
                df_edit,
                column_config={
                    "Ticker": st.column_config.TextColumn(disabled=True),
                    "Amount": st.column_config.NumberColumn(format="INR %d", min_value=0),
                    "Weight %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                },
                use_container_width=True,
                hide_index=True,
                key="amounts_data_editor",
            )
            for row in edited.to_dict("records"):
                tkr = row["Ticker"]
                st.session_state["amounts"][tkr] = float(row["Amount"])
            st.session_state["total_investment"] = sum(st.session_state["amounts"].values())

    # ── Risk Level ──────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">RISK LEVEL</div>', unsafe_allow_html=True)
    risk_opts = ["Low", "Medium", "High"]
    cur_risk = st.session_state["risk_level"]
    idx_risk = risk_opts.index(cur_risk) if cur_risk in risk_opts else 1
    new_risk = st.radio(
        "RISK PREFERENCE",
        options=risk_opts,
        index=idx_risk,
        horizontal=True,
        label_visibility="collapsed",
    )
    if new_risk != st.session_state["risk_level"]:
        st.session_state["risk_level"] = new_risk
        st.rerun()

    # ── Period ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">PERIOD</div>', unsafe_allow_html=True)
    c_start, c_end = st.columns(2)
    with c_start:
        s_date = st.date_input("START", value=date.fromisoformat(st.session_state["start"]),
                               key="start_date_input")
        st.session_state["start"] = s_date.isoformat()
    with c_end:
        e_date = st.date_input("END", value=date.fromisoformat(st.session_state["end"]),
                               key="end_date_input")
        st.session_state["end"] = e_date.isoformat()

    # ── Benchmark ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">BENCHMARK</div>', unsafe_allow_html=True)
    bench_opts = ["^NSEI", "^NDX", "^GSPC"]
    bench_labels = {"^NSEI": "NIFTY 50", "^NDX": "NASDAQ 100", "^GSPC": "S&P 500"}
    st.session_state["benchmark"] = st.selectbox(
        "BENCHMARK",
        options=bench_opts,
        index=bench_opts.index(st.session_state["benchmark"]),
        format_func=lambda b: bench_labels.get(b, b),
        label_visibility="collapsed",
    )

    # ── Save and Export ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">SAVE AND EXPORT</div>', unsafe_allow_html=True)
    session_name = st.text_input("PORTFOLIO NAME", value="Primary Portfolio",
                                key="session_name_input")
    c_save, c_load = st.columns(2)
    with c_save:
        if st.button("SAVE", key="save_portfolio_btn", use_container_width=True):
            with st.spinner("SAVING..."):
                ok = api.save_portfolio(
                    session_name=session_name,
                    tickers=st.session_state["tickers"],
                    amounts=st.session_state["amounts"],
                    risk_level=st.session_state["risk_level"],
                    lang=lang,
                )
            if ok:
                st.success("Portfolio saved.")
    with c_load:
        if st.button("LOAD", key="load_portfolio_btn", use_container_width=True):
            with st.spinner("LOADING..."):
                saved = api.load_portfolio(session_name)
            if saved:
                st.session_state["tickers"] = saved.get("tickers", st.session_state["tickers"])
                st.session_state["amounts"] = saved.get("amounts", st.session_state["amounts"])
                st.session_state["risk_level"] = saved.get("risk_level", "Medium")
                st.session_state["total_investment"] = sum(saved.get("amounts", {}).values())
                st.success("Portfolio loaded.")
                st.rerun()

    # PDF Export Button
    _exp_tickers = st.session_state["tickers"]
    if _exp_tickers:
        # the report carries the real analysis (cached calls, same arguments as the tabs use)
        _tt = tuple(_exp_tickers)
        _w = _weights_from_amounts()
        _wt = tuple(_w.get(x, 1 / len(_tt)) for x in _tt)
        _s0, _e0 = st.session_state["start"], st.session_state["end"]
        _label, _metrics, _rec, _expl = st.session_state["risk_level"], {}, {}, ""
        try:
            _label = api.classify_risk(_tt, _wt, _s0, _e0).get("label", _label)
            _metrics = api.portfolio_metrics(_tt, _wt, _s0, _e0)
            _opt = api.optimize(_tt, _wt, st.session_state["risk_level"], _s0, _e0)
            _rec = _opt.get("weights", {})
            _expl = api.explain_portfolio(_tt, _wt, _label, lang, _s0, _e0)
            if _opt.get("reason"):
                _expl = f"{_expl}\n\n{_opt['reason']}"
        except Exception:
            _expl = "Analysis unavailable for the current selection."
        export_button(
            lang=lang,
            tickers=_exp_tickers,
            weights=st.session_state["amounts"],
            risk_label=_label,
            metrics=_metrics,
            rec_weights=_rec,
            explanation=_expl,
            session_name=session_name,
        )


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ════════════════════════════════════════════════════════════════════════════
lang = st.session_state["lang"]
tickers = st.session_state["tickers"]
amounts = st.session_state["amounts"]
total_investment = st.session_state["total_investment"]
risk_level = st.session_state["risk_level"]
start = st.session_state["start"]
end = st.session_state["end"]
benchmark = st.session_state["benchmark"]
weights = _weights_from_amounts()

# Top bar: wordmark, status chips
_px_src = "FALLBACK"
if tickers and not api.USE_MOCKS:
    _px_src = {"live": "LIVE", "cache": "CACHE", "fallback": "FALLBACK"}.get(
        api.get_prices(tuple(tickers), start, end).attrs.get("source", "fallback"), "FALLBACK")
top_bar(
    source=_px_src,
    llm_mode="OFFLINE" if st.session_state["offline"] else "ONLINE",
    lang=lang,
)

# Ticker tape marquee
render_ticker_tape([], height=34)

if not tickers:
    ui.empty_state("NO ASSETS SELECTED. CONFIGURE HOLDINGS IN THE SIDEBAR.")
    footer(lang)
    st.stop()

# ── 7 Tabs Layout (Strictly uppercase, no emojis) ───────────────────────────
tab_labels = [
    strip_emojis(t("tab_risk", lang)).upper(),
    strip_emojis(t("tab_allocation", lang)).upper(),
    strip_emojis(t("tab_simulate", lang)).upper(),
    strip_emojis(t("tab_track", lang)).upper(),
    strip_emojis(t("tab_compare", lang)).upper(),
    strip_emojis(t("tab_scenarios", lang)).upper(),
    strip_emojis(t("tab_news", lang)).upper(),
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
        from app.tabs.build import render_tab as _render_risk
        _render_risk(tickers=tickers, weights=weights, start=start, end=end, lang=lang, benchmark=benchmark)
    except Exception as e:
        st.error(f"Risk module error: {e}")

# Tab 1: Allocation
with tabs[1]:
    try:
        from app.tabs.allocation import render_tab as _render_alloc
        _render_alloc(tickers=tickers, weights=weights, risk_level=risk_level, start=start, end=end, lang=lang, benchmark=benchmark)
    except Exception as e:
        st.error(f"Allocation module error: {e}")

# Tab 2: Simulate
with tabs[2]:
    try:
        from app.tabs.simulate import render_tab as _render_sim
        _render_sim(tickers=tickers, weights=weights, start=start, end=end, lang=lang)
    except Exception as e:
        st.error(f"Simulate module error: {e}")

# Tab 3: Track
with tabs[3]:
    try:
        from app.tabs.track import render_tab as _render_track
        _render_track(holdings_input=None, lang=lang)
    except Exception as e:
        st.error(f"Track module error: {e}")

# Tab 4: Compare
with tabs[4]:
    try:
        from app.tabs.compare import render_tab as _render_compare
        _render_compare(tickers=tickers, weights=weights, risk_level=risk_level, start=start, end=end, lang=lang, benchmark=benchmark)
    except Exception as e:
        st.error(f"Compare module error: {e}")

# Tab 5: Scenarios
with tabs[5]:
    try:
        from app.tabs.scenarios import render_tab as _render_scenarios
        _render_scenarios(tickers=tickers, weights=weights, total_investment=total_investment, start=start, end=end, lang=lang)
    except Exception as e:
        st.error(f"Scenarios module error: {e}")

# Tab 6: Market Intel
with tabs[6]:
    try:
        from app.tabs.news import render_tab as _render_news
        _render_news(tickers=tickers, lang=lang)
    except Exception as e:
        st.error(f"Market Intel module error: {e}")

# ── Footer Line ─────────────────────────────────────────────────────────────
footer(lang)
