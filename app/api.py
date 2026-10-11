"""
app/api.py - backend adapter between the Streamlit tabs and the real ``engine/`` + ``ml/`` packages.

* Each function calls the real backend and **normalises its output to the shape the tabs consume**
  (percent vs fraction, records vs DataFrame, UI scenario names -> engine scenario dicts, ...).
* ``USE_MOCKS = true`` in ``.streamlit/secrets.toml`` switches every function to ``data/mocks/*.json``
  (UI development without any backend). The default is the real backend, which itself degrades gracefully
  (offline price files, template explanations, local caches).
* In real mode a backend failure never fabricates data: prices come back empty with a warning; other results
  fall back to the clearly-warned mock payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent
_MOCKS_DIR = _ROOT / "data" / "mocks"


def _use_mocks() -> bool:
    try:
        return bool(st.secrets.get("USE_MOCKS", False))
    except Exception:  # no secrets file at all
        return False


USE_MOCKS: bool = _use_mocks()


def _load_mock(filename: str) -> Any:
    path = _MOCKS_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# scenario vocabulary: the UI's names -> engine scenario dicts
# ---------------------------------------------------------------------------
def _engine_scenario(params: tuple) -> dict:
    """``(ui_kind, value)`` -> ``engine.simulate.apply_scenario`` dict.

    Rate shocks, stagflation and custom shocks are modelled as a broad market move (beta-scaled per asset);
    the tech downturn is a direct shock to the IT sector with spillover to the rest.
    """
    kind, val = params[0], float(params[1])
    if kind in ("market_crash", "rate_shock", "stagflation", "custom", "market_shock"):
        return {"type": "market_shock", "pct": val}
    if kind in ("bull_run", "recovery"):
        return {"type": "recovery", "pct": val}
    if kind in ("sector_tech", "sector_shock"):
        return {"type": "sector_shock", "sector": "IT", "pct": val}
    if kind == "vol_multiplier":
        return {"type": "vol_multiplier", "k": val}
    return {"type": "market_shock", "pct": val}


# ---------------------------------------------------------------------------
# engine/data.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Adjusted-close prices (cols = tickers). ``df.attrs["source"]`` is live / cache / fallback."""
    if not USE_MOCKS:
        try:
            from engine.data import get_prices as _real
            df = _real(list(tickers), start, end)
            if df.empty:
                reasons = df.attrs.get("dropped_reasons", {})
                raise ValueError("no price data (" + "; ".join(f"{k}: {v}" for k, v in reasons.items()) + ")")
            return df
        except Exception as e:
            st.warning(f"Price data unavailable ({e}).")
            out = pd.DataFrame()
            out.attrs["source"] = "fallback"
            return out

    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates)))) for t in tickers},
                      index=dates)
    df.attrs["source"] = "fallback"
    return df


# ---------------------------------------------------------------------------
# engine/metrics.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def portfolio_metrics(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], start: str, end: str) -> dict:
    if not USE_MOCKS:
        try:
            from engine.metrics import portfolio_metrics as _real
            return _real(get_prices(tickers, start, end), dict(zip(tickers, weights_tuple)))
        except Exception as e:
            st.warning(f"Metrics computation failed ({e}). Using mock data.")
    return _load_mock("mock_portfolio_metrics.json")


@st.cache_data(ttl=3600, show_spinner=False)
def risk_contribution(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], start: str, end: str) -> dict:
    """``{ticker: share of portfolio risk}`` (sums to 1)."""
    helpers = _load_mock("mock_helpers.json")
    if not USE_MOCKS:
        try:
            from engine.metrics import risk_contribution as _real
            rc = _real(get_prices(tickers, start, end), dict(zip(tickers, weights_tuple)))
            return {str(k): float(v) for k, v in rc.items()}
        except Exception as e:
            st.warning(f"Risk contribution failed ({e}). Using mock.")
    return helpers.get("risk_contribution", {t: 1 / len(tickers) for t in tickers})


# ---------------------------------------------------------------------------
# engine/optimizer.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def optimize(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], risk_level: str,
             start: str, end: str) -> dict:
    """``{"weights","metrics","reason","tradeoffs","mode", ...}``; the explanation compares with the user's mix."""
    if not USE_MOCKS:
        try:
            from engine.optimizer import optimize as _real
            return _real(get_prices(tickers, start, end), risk_level,
                         current_weights=dict(zip(tickers, weights_tuple)))
        except Exception as e:
            st.warning(f"Optimizer failed ({e}). Using mock.")
    return _load_mock("mock_optimize_ui.json")


# ---------------------------------------------------------------------------
# engine/simulate.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def simulate(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], start: str, end: str) -> dict:
    """``{"growth": {iso_date: value}, "metrics": {...}, "dropped_tickers": [...]}``."""
    if not USE_MOCKS:
        try:
            from engine.simulate import simulate as _real
            prices = get_prices(tickers, start, end)
            result = _real(prices, dict(zip(tickers, weights_tuple)), start, end)
            result["growth"] = {_iso(k): float(v) for k, v in result["growth"].items()}
            return result
        except Exception as e:
            st.warning(f"Simulation failed ({e}). Using mock.")
    return _load_mock("mock_simulate_ui.json")


@st.cache_data(ttl=3600, show_spinner=False)
def regime_presets() -> dict:
    """``{name: {"start", "end"}}`` named historical windows."""
    if not USE_MOCKS:
        try:
            from engine.simulate import regime_presets as _real
            return {name: {"start": a, "end": b} for name, (a, b) in _real().items()}
        except Exception:
            pass
    return {
        "COVID crash": {"start": "2020-01-20", "end": "2020-04-30"},
        "2021 bull run": {"start": "2021-01-01", "end": "2021-12-31"},
        "2022 bear/volatile": {"start": "2022-01-01", "end": "2022-12-31"},
        "Last 1 year": {"start": "2025-10-01", "end": "2026-10-01"},
    }


@st.cache_data(ttl=3600, show_spinner=False)
def compare_strategies(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                       recommended_weights: tuple[float, ...], benchmark: str, start: str, end: str) -> list[dict]:
    """Records ``{"strategy", "ann_return", "ann_vol", "max_drawdown", "sharpe", ..., "total_return"}``."""
    if not USE_MOCKS:
        try:
            from engine.simulate import compare_strategies as _real
            df = _real(get_prices(tickers, start, end), dict(zip(tickers, original_weights)),
                       dict(zip(tickers, recommended_weights)), benchmark)
            return df.reset_index(names="strategy").to_dict("records")
        except Exception as e:
            st.warning(f"Strategy comparison failed ({e}). Using mock.")
    return _load_mock("mock_compare_strategies.json")


@st.cache_data(ttl=3600, show_spinner=False)
def compare_growth(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                   recommended_weights: tuple[float, ...], benchmark: str, start: str, end: str) -> dict:
    """``{strategy: {iso_date: growth_of_100}}`` for Original / Recommended / Equal Weight / Benchmark."""
    if not USE_MOCKS:
        try:
            from engine.simulate import compare_growth as _real
            df = _real(get_prices(tickers, start, end), dict(zip(tickers, original_weights)),
                       dict(zip(tickers, recommended_weights)), benchmark)
            return {c: {_iso(i): float(v) for i, v in df[c].dropna().items()} for c in df.columns}
        except Exception as e:
            st.warning(f"Growth comparison failed ({e}). Using mock.")
    return _load_mock("mock_helpers.json").get("compare_growth", {})


@st.cache_data(ttl=3600, show_spinner=False)
def risk_return_points(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                       recommended_weights: tuple[float, ...], benchmark: str, start: str, end: str) -> list[dict]:
    """``[{"name", "ann_return", "ann_vol", "type": asset|portfolio|benchmark}]`` for the scatter chart."""
    if not USE_MOCKS:
        try:
            from engine.simulate import risk_return_points as _real
            pts = _real(get_prices(tickers, start, end),
                        {"Original": dict(zip(tickers, original_weights)),
                         "Recommended": dict(zip(tickers, recommended_weights))}, benchmark)
            return [{"name": n, "ann_return": p["ret"], "ann_vol": p["vol"], "type": p["kind"]}
                    for n, p in pts.items()]
        except Exception as e:
            st.warning(f"Risk-return points failed ({e}). Using mock.")
    return _load_mock("mock_helpers.json").get("risk_return_points", [])


@st.cache_data(ttl=3600, show_spinner=False)
def apply_scenario(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], scenario: str,
                   scenario_params: tuple, start: str, end: str) -> dict:
    """Stress result. ``portfolio_change_pct`` is in percent (the UI's unit); ``asset_shocks`` are fractions."""
    if not USE_MOCKS:
        try:
            from engine.simulate import apply_scenario as _real
            r = _real(get_prices(tickers, start, end), dict(zip(tickers, weights_tuple)),
                      _engine_scenario(scenario_params))
            return {
                "portfolio_change_pct": float(r["value_change_pct"]) * 100.0,
                "value_change_pct": float(r["value_change_pct"]),
                "value_change_basis": r["value_change_basis"],
                "asset_shocks": r["asset_shocks"], "betas": r["betas"],
                "before": r["before"], "after": r["after"],
                "new_volatility": r["after"]["ann_vol"], "warnings": r["warnings"],
                "scenario": r["scenario"],
            }
        except Exception as e:
            st.warning(f"Scenario analysis failed ({e}). Using mock.")
    m = _load_mock("mock_scenario_ui.json")
    return {**m, "portfolio_change_pct": m.get("pct_change", 0.0)}


@st.cache_data(ttl=3600, show_spinner=False)
def monte_carlo(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], start: str, end: str,
                n: int = 1000, horizon: int = 252) -> dict:
    """``{"percentiles", "time_labels", "prob_of_loss", "expected_shortfall", "terminal_values"}``."""
    if not USE_MOCKS:
        try:
            from engine.simulate import monte_carlo as _real
            r = _real(get_prices(tickers, start, end), dict(zip(tickers, weights_tuple)), n=n, horizon=horizon)
            return {**r, "time_labels": list(range(1, r["horizon"] + 1)), "prob_of_loss": r["prob_loss"]}
        except Exception as e:
            st.warning(f"Monte Carlo failed ({e}). Using mock.")
    return _load_mock("mock_scenario_ui.json").get("monte_carlo", {})


@st.cache_data(ttl=3600, show_spinner=False)
def classify_stressed(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], scenario_params: tuple,
                      start: str, end: str) -> dict:
    """Risk label of the portfolio after the scenario's shock."""
    if not USE_MOCKS:
        try:
            from engine.simulate import apply_scenario as _as
            from ml.classify import classify_stressed as _cs
            weights = dict(zip(tickers, weights_tuple))
            stressed = _as(get_prices(tickers, start, end), weights, _engine_scenario(scenario_params))
            return _ui_classification(_cs(stressed["stressed_prices"], weights))
        except Exception as e:
            st.warning(f"Stressed classification failed ({e}). Using mock.")
    return _load_mock("mock_helpers.json").get(
        "classify_stressed", {"label": "High", "probs": {"Low": 0.1, "Medium": 0.3, "High": 0.6}})


# ---------------------------------------------------------------------------
# engine/track.py
# ---------------------------------------------------------------------------
def _fx_to_inr(currencies: set[str]) -> float:
    """INR per USD (latest close) when any USD holding exists, else 1.0."""
    if "USD" not in currencies:
        return 1.0
    from engine.data import get_prices as _gp
    today = pd.Timestamp.today().normalize()
    fx = _gp(["INR=X"], (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"),
             align="outer", min_rows=1)
    return float(fx["INR=X"].dropna().iloc[-1]) if "INR=X" in fx.columns and fx["INR=X"].notna().any() else 1.0


@st.cache_data(ttl=300, show_spinner=False)
def track_holdings(holdings_json: str) -> dict:
    """Holdings valuation in the tab's display columns. USD holdings are converted to INR at today's rate
    (labelled by the ``Currency`` column and ``fx_rate``); per-currency totals stay in ``totals_by_currency``."""
    if not USE_MOCKS:
        try:
            from engine.track import track_holdings as _real
            r = _real(json.loads(holdings_json))
            tab = r["table"]
            fx = _fx_to_inr(set(tab["currency"])) if not tab.empty else 1.0
            rows = []
            for rec in tab.to_dict("records"):
                k = fx if rec["currency"] == "USD" else 1.0
                rows.append({
                    "Ticker": rec["ticker"], "Quantity": rec["quantity"],
                    "Buy Price": rec["buy_price"] * k, "Current Price": rec["last_price"] * k,
                    "Total Value": rec["value"] * k, "P&L (INR)": rec["pnl"] * k,
                    "P&L (%)": rec["pnl_pct"] * 100.0, "Currency": rec["currency"],
                })
            tv = sum(x["Total Value"] for x in rows)
            tc = sum(rec["cost"] * (fx if rec["currency"] == "USD" else 1.0) for rec in tab.to_dict("records"))
            return {"table": rows, "total_value": tv, "total_cost": tc, "total_pnl": tv - tc,
                    "total_pnl_pct": ((tv / tc - 1) * 100.0) if tc else 0.0, "currency": "INR",
                    "fx_rate": fx, "as_of": r["as_of"], "source": r["source"], "warnings": r["warnings"],
                    "totals_by_currency": r["total_value"], "risk_metrics": r["risk_metrics"]}
        except Exception as e:
            st.warning(f"Holdings tracking failed ({e}). Using mock.")
    m = _load_mock("mock_track_holdings.json")
    rows = [{"Ticker": h["ticker"], "Quantity": h["quantity"], "Buy Price": h["buy_price"],
             "Current Price": h["current_price"], "Total Value": h["value"], "P&L (INR)": h["pnl"],
             "P&L (%)": h["pnl_pct"], "Currency": h.get("currency", "INR")} for h in m.get("table", [])]
    cost = sum(h["quantity"] * h["buy_price"] for h in m.get("table", []))
    return {**m, "table": rows, "total_cost": cost}


# ---------------------------------------------------------------------------
# engine/db.py
# ---------------------------------------------------------------------------
def save_portfolio(session_name: str, tickers: list, amounts: dict, risk_level: str, lang: str) -> bool:
    """Persist the current portfolio (Supabase when reachable, otherwise a local file). True on success."""
    if not USE_MOCKS:
        try:
            from engine.db import save_portfolio as _real
            total = sum(float(amounts.get(t, 0.0)) for t in tickers) or 1.0
            holdings = [{"ticker": t, "amount": float(amounts.get(t, 0.0)),
                         "weight": float(amounts.get(t, 0.0)) / total} for t in tickers]
            return bool(_real(session_name, session_name, risk_level, lang, holdings))
        except Exception as e:
            st.warning(f"Save failed ({e}).")
            return False
    st.info("Mock mode: portfolio not persisted.")
    return True


def load_portfolio(session_name: str) -> dict | None:
    """Most recently saved portfolio for this session name, as ``{tickers, amounts, risk_level, lang}``."""
    if not USE_MOCKS:
        try:
            from engine.db import list_portfolios, load_portfolio as _load
            saved = list_portfolios(session_name)
            if not saved:
                st.info(f"No saved portfolio named '{session_name}'.")
                return None
            p = _load(saved[0]["id"])
            if not p:
                return None
            hs = p.get("holdings", [])
            return {"tickers": [h["ticker"] for h in hs], "amounts": {h["ticker"]: h["amount"] for h in hs},
                    "risk_level": p.get("risk_level") or "Medium", "lang": p.get("language", "en")}
        except Exception as e:
            st.warning(f"Load failed ({e}).")
            return None
    return {"tickers": ["TCS.NS", "INFY.NS", "HDFCBANK.NS"],
            "amounts": {"TCS.NS": 33000, "INFY.NS": 33000, "HDFCBANK.NS": 34000},
            "risk_level": "Medium", "lang": "en"}


# ---------------------------------------------------------------------------
# ml/classify.py, ml/explain.py
# ---------------------------------------------------------------------------
def _ui_classification(r: dict) -> dict:
    """Real classifier output -> what the tabs draw.

    The factor chart wants signed numbers (negative = pushes risk up, drawn red), so each factor becomes
    ``{"name": "Volatility, last 6 months (28%)", "impact": -0.31}``; the raw dicts stay under ``factors``.
    """
    tf = []
    for f in r.get("top_factors", []):
        if isinstance(f, dict):
            mag = abs(float(f.get("effect", 0.0))) or 0.05
            sign = -1.0 if f.get("impact", "raises risk") == "raises risk" else 1.0
            tf.append({"name": f"{f.get('name', '')} ({f.get('value', '')})", "impact": sign * mag})
        else:
            tf.append(f)
    return {**r, "top_factors": tf, "factors": r.get("top_factors", [])}


@st.cache_data(ttl=3600, show_spinner=False)
def classify_risk(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], start: str, end: str) -> dict:
    """``{"label", "probs", "top_factors", "factors", ...}`` from the trained model."""
    if not USE_MOCKS:
        try:
            from ml.classify import classify_risk as _real
            return _ui_classification(_real(get_prices(tickers, start, end), dict(zip(tickers, weights_tuple))))
        except Exception as e:
            st.warning(f"Risk classification failed ({e}). Using mock.")
    return _load_mock("mock_classify_risk.json")


@st.cache_data(ttl=3600, show_spinner=False)
def explain_portfolio(tickers: tuple[str, ...], weights_tuple: tuple[float, ...], label: str, lang: str,
                      start: str, end: str) -> str:
    if not USE_MOCKS:
        try:
            from ml.explain import explain_portfolio as _real
            return _real(get_prices(tickers, start, end), dict(zip(tickers, weights_tuple)), label, lang)
        except Exception as e:
            st.warning(f"Explanation failed ({e}). Using template.")
    return _load_mock("mock_classify_risk.json").get("explanation", f"Portfolio classified as {label} risk.")


@st.cache_data(ttl=3600, show_spinner=False)
def explain_allocation(tickers: tuple[str, ...], original_weights: tuple[float, ...],
                       recommended_weights: tuple[float, ...], risk_level: str, lang: str,
                       start: str, end: str) -> str:
    if not USE_MOCKS:
        try:
            from ml.explain import explain_allocation as _real
            return _real(get_prices(tickers, start, end), dict(zip(tickers, original_weights)),
                         dict(zip(tickers, recommended_weights)), risk_level, lang)
        except Exception as e:
            st.warning(f"Allocation explanation failed ({e}). Using mock.")
    return _load_mock("mock_helpers.json").get("explain_allocation",
                                               "Weights adjusted to improve risk-adjusted returns.")


# ---------------------------------------------------------------------------
# ml/news.py, ml/voice.py
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_news_insight(tickers: tuple[str, ...], lang: str = "en") -> dict:
    if not USE_MOCKS:
        try:
            from ml.news import get_news_insight as _real
            return _real(list(tickers), lang)
        except Exception as e:
            st.warning(f"News insight failed ({e}). Using mock.")
    return _load_mock("mock_news_insight.json")


def transcribe(audio_bytes: bytes, lang_hint: str = "hi") -> str:
    """Speech -> text. Empty string (with a warning) when every STT path failed, so the user can type."""
    if not USE_MOCKS:
        try:
            from ml.voice import transcribe_ex as _real
            r = _real(audio_bytes, lang_hint)
            if r["error"]:
                msg = {"offline": "Offline mode is on: speech recognition is disabled.",
                       "empty_audio": "No audio was captured."}.get(r["error"],
                                                                    "Speech recognition is unavailable right now.")
                st.warning(f"{msg} Please type the transcript instead.")
            return r["text"]
        except Exception as e:
            st.warning(f"Transcription failed ({e}). Please type instead.")
            return ""
    return "Mere paas 1 lakh rupaye hain, TCS, Infosys aur HDFC Bank mein lagana hai, risk kam chahiye."


def extract_portfolio(text: str) -> dict:
    """Transcript -> ``{"tickers", "amounts", "risk_level", "unresolved", "confidence"}``."""
    if not USE_MOCKS:
        try:
            from ml.voice import extract_portfolio as _real
            return _real(text)
        except Exception as e:
            st.warning(f"Portfolio extraction failed ({e}).")
            return {}
    return _load_mock("mock_extract_portfolio.json")


# ---------------------------------------------------------------------------
# ticker validation
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def validate_tickers(tickers: tuple[str, ...]) -> dict[str, bool]:
    """``{ticker: is_valid}``."""
    if not USE_MOCKS:
        try:
            from engine.data import validate_tickers as _real
            r = _real(list(tickers))
            valid = set(r["valid"])
            return {t.strip().upper(): t.strip().upper() in valid for t in tickers}
        except Exception:
            pass
    return _load_mock("mock_helpers.json").get("validate_tickers", {t: True for t in tickers})
