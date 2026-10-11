"""
app/api.py — Backend adapter with mock / real switch.

Every function tries the real backend first. If USE_MOCKS=True in secrets
(or the import fails), it returns data from data/mocks/*.json.
Flip one USE_MOCKS flag per function in Phase 3 as real backends come online.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_MOCKS_DIR = _ROOT / "data" / "mocks"

# Read from secrets; default to True so the app always works standalone.
USE_MOCKS: bool = st.secrets.get("USE_MOCKS", True)


def _load_mock(filename: str) -> Any:
    """Load a JSON mock file."""
    path = _MOCKS_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# engine/data.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Return adjusted-close price DataFrame. Cols = tickers, index = date."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _real
            return _real(list(tickers), start, end)
        except Exception as e:
            st.warning(f"Live price fetch failed ({e}). Using fallback data.")

    # Mock: generate a simple random-walk DataFrame
    import numpy as np
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(42)
    data = {}
    for tkr in tickers:
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates))))
        data[tkr] = prices
    df = pd.DataFrame(data, index=dates)
    df.attrs["source"] = "fallback"
    return df


# ---------------------------------------------------------------------------
# engine/metrics.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def portfolio_metrics(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                      start: str, end: str) -> dict:
    """Return portfolio metric dict."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.metrics import portfolio_metrics as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            return _real(prices, weights)
        except Exception as e:
            st.warning(f"Metrics computation failed ({e}). Using mock data.")
    return _load_mock("mock_portfolio_metrics.json")


@st.cache_data(ttl=3600)
def risk_contribution(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                      start: str, end: str) -> dict:
    """Return per-asset risk contribution as a dict {ticker: fraction}."""
    helpers = _load_mock("mock_helpers.json")
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.metrics import risk_contribution as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            rc = _real(prices, weights)
            return rc.to_dict() if hasattr(rc, "to_dict") else rc
        except Exception as e:
            st.warning(f"Risk contribution failed ({e}). Using mock.")
    return helpers.get("risk_contribution", {t: 1 / len(tickers) for t in tickers})


# ---------------------------------------------------------------------------
# engine/optimizer.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def optimize(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
             risk_level: str, start: str, end: str) -> dict:
    """Return optimized weights, metrics, reason, tradeoffs."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.optimizer import optimize as _real
            prices = _gp(list(tickers), start, end)
            return _real(prices, risk_level)
        except Exception as e:
            st.warning(f"Optimizer failed ({e}). Using mock.")
    return _load_mock("mock_optimize.json")


# ---------------------------------------------------------------------------
# engine/simulate.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def simulate(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
             start: str, end: str) -> dict:
    """Return {'growth': dict of date->value, 'metrics': dict}."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.simulate import simulate as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            result = _real(prices, weights, start, end)
            # Serialize Series to dict for caching
            if hasattr(result.get("growth"), "to_dict"):
                result["growth"] = {str(k): v for k, v in result["growth"].to_dict().items()}
            return result
        except Exception as e:
            st.warning(f"Simulation failed ({e}). Using mock.")
    return _load_mock("mock_simulate.json")


@st.cache_data(ttl=3600)
def regime_presets() -> dict:
    """Return named historical regime windows."""
    if not USE_MOCKS:
        try:
            from engine.simulate import regime_presets as _real
            return _real()
        except Exception:
            pass
    return {
        "COVID Crash (2020)":  {"start": "2020-01-01", "end": "2020-12-31"},
        "2021 Bull Run":       {"start": "2021-01-01", "end": "2021-12-31"},
        "2022 Bear / Volatile":{"start": "2022-01-01", "end": "2022-12-31"},
        "Last 1 Year":         {"start": "2025-10-01", "end": "2026-10-01"},
    }


@st.cache_data(ttl=3600)
def compare_strategies(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                       recommended_weights: tuple[float, ...],
                       benchmark: str, start: str, end: str) -> list[dict]:
    """Return list of strategy comparison dicts."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.simulate import compare_strategies as _real
            prices = _gp(list(tickers) + [benchmark], start, end)
            orig_w = dict(zip(tickers, original_weights))
            rec_w  = dict(zip(tickers, recommended_weights))
            result = _real(prices, orig_w, rec_w, benchmark)
            if hasattr(result, "to_dict"):
                return result.to_dict("records")
            return result
        except Exception as e:
            st.warning(f"Strategy comparison failed ({e}). Using mock.")
    return _load_mock("mock_compare_strategies.json")


@st.cache_data(ttl=3600)
def apply_scenario(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                   scenario: str, scenario_params: tuple,
                   start: str, end: str) -> dict:
    """Apply a market scenario and return impact dict."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.simulate import apply_scenario as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            scen_dict = dict(zip(["type", "value"], scenario_params))
            return _real(prices, weights, scen_dict)
        except Exception as e:
            st.warning(f"Scenario analysis failed ({e}). Using mock.")
    return _load_mock("mock_scenario.json")


@st.cache_data(ttl=3600)
def monte_carlo(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                start: str, end: str, n: int = 1000, horizon: int = 252) -> dict:
    """Run Monte Carlo simulation."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.simulate import monte_carlo as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            return _real(prices, weights, n=n, horizon=horizon)
        except Exception as e:
            st.warning(f"Monte Carlo failed ({e}). Using mock.")
    mock = _load_mock("mock_scenario.json")
    return mock.get("monte_carlo", {})


# ---------------------------------------------------------------------------
# engine/track.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def track_holdings(holdings_json: str) -> dict:
    """Return current value, P&L table for holdings."""
    import json as _json
    if not USE_MOCKS:
        try:
            from engine.track import track_holdings as _real
            holdings = _json.loads(holdings_json)
            result = _real(holdings)
            if hasattr(result.get("table"), "to_dict"):
                result["table"] = result["table"].to_dict("records")
            return result
        except Exception as e:
            st.warning(f"Holdings tracking failed ({e}). Using mock.")
    return _load_mock("mock_track_holdings.json")


# ---------------------------------------------------------------------------
# engine/db.py
# ---------------------------------------------------------------------------
def save_portfolio(session_name: str, tickers: list, amounts: dict,
                   risk_level: str, lang: str) -> bool:
    """Save portfolio to Supabase. Returns True on success."""
    if not USE_MOCKS:
        try:
            from engine.db import save_portfolio as _real
            return _real(session_name, tickers, amounts, risk_level, lang)
        except Exception as e:
            st.warning(f"Save failed ({e}).")
    st.info("Mock mode: portfolio not persisted.")
    return True


def load_portfolio(session_name: str) -> dict | None:
    """Load portfolio from Supabase. Returns dict or None."""
    if not USE_MOCKS:
        try:
            from engine.db import load_portfolio as _real
            return _real(session_name)
        except Exception as e:
            st.warning(f"Load failed ({e}).")
    # Return a demo portfolio in mock mode
    return {
        "tickers": ["TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "amounts": {"TCS.NS": 33000, "INFY.NS": 33000, "HDFCBANK.NS": 34000},
        "risk_level": "Medium",
        "lang": "en",
    }


# ---------------------------------------------------------------------------
# ml/classify.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def classify_risk(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                  start: str, end: str) -> dict:
    """Return {'label', 'probs', 'top_factors'}."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from ml.classify import classify_risk as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            return _real(prices, weights)
        except Exception as e:
            st.warning(f"Risk classification failed ({e}). Using mock.")
    return _load_mock("mock_classify_risk.json")


# ---------------------------------------------------------------------------
# ml/explain.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def explain_portfolio(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                      label: str, lang: str, start: str, end: str) -> str:
    """Return plain-text explanation of risk classification."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from ml.explain import explain_portfolio as _real
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            return _real(prices, weights, label, lang)
        except Exception as e:
            st.warning(f"Explanation failed ({e}). Using template.")
    mock = _load_mock("mock_classify_risk.json")
    return mock.get("explanation", f"Portfolio classified as {label} risk.")


@st.cache_data(ttl=3600)
def explain_allocation(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                       recommended_weights: tuple[float, ...],
                       risk_level: str, lang: str, start: str, end: str) -> str:
    """Return text explaining why weights changed."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from ml.explain import explain_allocation as _real
            prices = _gp(list(tickers), start, end)
            orig_w = dict(zip(tickers, original_weights))
            rec_w  = dict(zip(tickers, recommended_weights))
            return _real(prices, orig_w, rec_w, risk_level, lang)
        except Exception as e:
            st.warning(f"Allocation explanation failed ({e}). Using mock.")
    helpers = _load_mock("mock_helpers.json")
    return helpers.get("explain_allocation", "Weights adjusted to improve risk-adjusted returns.")


# ---------------------------------------------------------------------------
# ml/news.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_news_insight(tickers: tuple[str, ...], lang: str = "en") -> dict:
    """Return news sentiment and summary dict."""
    if not USE_MOCKS:
        try:
            from ml.news import get_news_insight as _real
            return _real(list(tickers), lang)
        except Exception as e:
            st.warning(f"News insight failed ({e}). Using mock.")
    return _load_mock("mock_news_insight.json")


# ---------------------------------------------------------------------------
# ml/voice.py
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes, lang_hint: str = "hi") -> str:
    """Transcribe audio to text."""
    if not USE_MOCKS:
        try:
            from ml.voice import transcribe as _real
            return _real(audio_bytes, lang_hint)
        except Exception as e:
            st.warning(f"Transcription failed ({e}). Please type instead.")
    return "Mere paas 1 lakh rupaye hain, TCS, Infosys aur HDFC Bank mein lagana hai, risk kam chahiye."


def extract_portfolio(text: str) -> dict:
    """Extract portfolio JSON from transcript text."""
    if not USE_MOCKS:
        try:
            from ml.voice import extract_portfolio as _real
            return _real(text)
        except Exception as e:
            st.warning(f"Portfolio extraction failed ({e}). Using mock.")
    return _load_mock("mock_extract_portfolio.json")


# ---------------------------------------------------------------------------
# Additional helpers (compare tab, scenarios tab)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def compare_growth(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                   recommended_weights: tuple[float, ...],
                   benchmark: str, start: str, end: str) -> dict:
    """Return growth-of-100 series for each strategy."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.simulate import compare_strategies as _cs
            prices = _gp(list(tickers) + [benchmark], start, end)
            orig_w = dict(zip(tickers, original_weights))
            rec_w  = dict(zip(tickers, recommended_weights))
            result = _cs(prices, orig_w, rec_w, benchmark)
            return result
        except Exception:
            pass
    helpers = _load_mock("mock_helpers.json")
    return helpers.get("compare_growth", {})


@st.cache_data(ttl=3600)
def risk_return_points(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                       recommended_weights: tuple[float, ...],
                       benchmark: str, start: str, end: str) -> list[dict]:
    """Return risk-return coordinates for scatter plot."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.metrics import portfolio_metrics as _pm
            prices = _gp(list(tickers) + [benchmark], start, end)
            points = []
            for tkr in tickers:
                m = _pm(prices[[tkr]], {tkr: 1.0})
                points.append({"name": tkr, "ann_return": m["ann_return"],
                                "ann_vol": m["ann_vol"], "type": "asset"})
            orig_w = dict(zip(tickers, original_weights))
            rec_w  = dict(zip(tickers, recommended_weights))
            mo = _pm(prices[list(tickers)], orig_w)
            mr = _pm(prices[list(tickers)], rec_w)
            points.append({"name": "Original",    "ann_return": mo["ann_return"], "ann_vol": mo["ann_vol"], "type": "portfolio"})
            points.append({"name": "Recommended", "ann_return": mr["ann_return"], "ann_vol": mr["ann_vol"], "type": "portfolio"})
            mb = _pm(prices[[benchmark]], {benchmark: 1.0})
            points.append({"name": benchmark, "ann_return": mb["ann_return"], "ann_vol": mb["ann_vol"], "type": "benchmark"})
            return points
        except Exception:
            pass
    helpers = _load_mock("mock_helpers.json")
    return helpers.get("risk_return_points", [])


@st.cache_data(ttl=3600)
def validate_tickers(tickers: tuple[str, ...]) -> dict[str, bool]:
    """Return {ticker: is_valid} dict."""
    if not USE_MOCKS:
        try:
            import yfinance as yf
            result = {}
            for tkr in tickers:
                try:
                    info = yf.Ticker(tkr).fast_info
                    result[tkr] = info.get("lastPrice", None) is not None
                except Exception:
                    result[tkr] = False
            return result
        except Exception:
            pass
    helpers = _load_mock("mock_helpers.json")
    return helpers.get("validate_tickers", {t: True for t in tickers})


@st.cache_data(ttl=3600)
def classify_stressed(tickers: tuple[str, ...], weights_tuple: tuple[float, ...],
                      scenario_params: tuple, start: str, end: str) -> dict:
    """Classify risk after applying a stress scenario."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _gp
            from engine.simulate import apply_scenario as _as
            from ml.classify import classify_risk as _cr
            prices = _gp(list(tickers), start, end)
            weights = dict(zip(tickers, weights_tuple))
            scen_dict = {"type": scenario_params[0], "value": scenario_params[1]}
            stressed = _as(prices, weights, scen_dict)
            if "stressed_prices" in stressed:
                return _cr(stressed["stressed_prices"], weights)
        except Exception:
            pass
    helpers = _load_mock("mock_helpers.json")
    return helpers.get("classify_stressed", {"label": "High", "probs": {"Low": 0.1, "Medium": 0.3, "High": 0.6}})
