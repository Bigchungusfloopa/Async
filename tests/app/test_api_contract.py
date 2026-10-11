"""Contract tests for app/api.py: the real engine/ml outputs must be normalised into the shapes the Streamlit tabs
(and app/charts.py) consume. Offline: prices come from the committed fallback parquet files, no LLM, no Supabase.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import api, charts  # noqa: E402
from engine import data, db  # noqa: E402
from ml import llm  # noqa: E402

TICKERS = ("TCS.NS", "INFY.NS", "HDFCBANK.NS")
WEIGHTS = (0.5, 0.3, 0.2)
REC = (0.3, 0.3, 0.4)
START, END, BENCH = "2023-01-02", "2025-12-30", "^NSEI"


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "USE_MOCKS", False)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "prices")
    monkeypatch.setattr(data, "SYNC_TO_SUPABASE", False)
    monkeypatch.setattr(data, "_sleep", lambda s: None)
    monkeypatch.setattr(data, "_download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    monkeypatch.setattr(db, "LOCAL_DIR", tmp_path / "local")
    monkeypatch.setattr(db, "_read_secrets", lambda: (None, None))
    monkeypatch.setattr(db, "_client", None)
    data._tail_tried.clear()
    data._head_tried.clear()
    llm.set_offline(True)
    st.cache_data.clear()
    yield
    llm.set_offline(False)
    st.cache_data.clear()


def test_prices_are_real_or_empty_never_fabricated():
    px = api.get_prices(TICKERS, START, END)
    assert list(px.columns) == list(TICKERS) and len(px) > 600
    assert px.attrs["source"] in ("fallback", "cache", "live")
    bad = api.get_prices(("NOTAREALTICKER123",), START, END)
    assert bad.empty                                       # no random-walk substitute in real mode


def test_classification_matches_what_the_factor_chart_needs():
    r = api.classify_risk(TICKERS, WEIGHTS, START, END)
    assert r["label"] in ("Low", "Medium", "High") and abs(sum(r["probs"].values()) - 1) < 1e-9
    for f in r["top_factors"]:
        assert isinstance(f["name"], str) and isinstance(f["impact"], float)      # numeric: the chart does `< 0`
    assert r["factors"] and r["factors"][0]["impact"] in ("raises risk", "lowers risk")
    charts.top_factors_bars(r["top_factors"])                                      # must not raise


def test_regime_presets_are_start_end_dicts():
    p = api.regime_presets()
    assert "COVID crash" in p and set(p["COVID crash"]) == {"start", "end"}
    assert p["COVID crash"]["start"] == "2020-01-20"


def test_comparison_shapes_feed_the_charts():
    cmp_ = api.compare_strategies(TICKERS, WEIGHTS, REC, BENCH, START, END)
    assert [d["strategy"] for d in cmp_] == ["Original", "Recommended", "Equal Weight", "Benchmark"]
    assert {"ann_return", "ann_vol", "max_drawdown", "sharpe"} <= set(cmp_[0])
    charts.metric_bar_chart(cmp_, "sharpe")
    growth = api.compare_growth(TICKERS, WEIGHTS, REC, BENCH, START, END)
    assert set(growth) == {"Original", "Recommended", "Equal Weight", "Benchmark"}
    assert abs(next(iter(growth["Original"].values())) - 100.0) < 1e-6
    charts.multi_growth_chart(growth, log_scale=False)
    pts = api.risk_return_points(TICKERS, WEIGHTS, REC, BENCH, START, END)
    assert {p["type"] for p in pts} == {"asset", "portfolio", "benchmark"}
    assert all({"name", "ann_return", "ann_vol"} <= set(p) for p in pts)
    charts.risk_return_scatter(pts, cloud=False)


def test_simulate_and_optimize_shapes():
    sim = api.simulate(TICKERS, WEIGHTS, START, END)
    assert all(isinstance(k, str) and len(k) == 10 for k in sim["growth"])          # ISO date keys
    assert abs(next(iter(sim["growth"].values())) - 100.0) < 1e-6
    charts.growth_chart(sim["growth"], log_scale=False, regimes=True)
    opt = api.optimize(TICKERS, WEIGHTS, "Low", START, END)
    assert abs(sum(opt["weights"].values()) - 1) < 1e-9 and opt["reason"] and opt["tradeoffs"]


@pytest.mark.parametrize("ui_kind,value,expected", [
    ("market_crash", -0.38, {"type": "market_shock", "pct": -0.38}),
    ("rate_shock", -0.08, {"type": "market_shock", "pct": -0.08}),
    ("stagflation", -0.12, {"type": "market_shock", "pct": -0.12}),
    ("sector_tech", -0.25, {"type": "sector_shock", "sector": "IT", "pct": -0.25}),
    ("bull_run", 0.30, {"type": "recovery", "pct": 0.30}),
    ("custom", -0.20, {"type": "market_shock", "pct": -0.20}),
])
def test_ui_scenario_names_map_to_engine_scenarios(ui_kind, value, expected):
    assert api._engine_scenario((ui_kind, value)) == expected


def test_scenario_result_uses_percent_and_real_attribution():
    crash = api.apply_scenario(TICKERS, WEIGHTS, "x", ("market_crash", -0.28), START, END)
    assert -100 < crash["portfolio_change_pct"] < 0                                 # percent, not fraction
    assert abs(sum(crash["asset_shocks"][t] * w for t, w in zip(TICKERS, WEIGHTS)) * 100
               - crash["portfolio_change_pct"]) < 1e-9                              # waterfall sums to the headline
    rally = api.apply_scenario(TICKERS, WEIGHTS, "x", ("bull_run", 0.30), START, END)
    assert rally["portfolio_change_pct"] > 0
    stressed = api.classify_stressed(TICKERS, WEIGHTS, ("market_crash", -0.28), START, END)
    assert stressed["label"] in ("Low", "Medium", "High")


def test_monte_carlo_feeds_fan_and_terminal_charts():
    mc = api.monte_carlo(TICKERS, WEIGHTS, START, END)
    assert len(mc["time_labels"]) == len(mc["percentiles"]["p50"]) == 252
    assert 0 <= mc["prob_of_loss"] <= 1 and len(mc["terminal_values"]) == 1000
    charts.monte_carlo_chart(mc)
    charts.terminal_values_hist(mc["terminal_values"], threshold=100.0)


def test_track_holdings_display_columns_and_fx():
    import json
    hs = [{"ticker": "TCS.NS", "quantity": 10, "buy_price": 3200.0},
          {"ticker": "AAPL", "quantity": 5, "buy_price": 150.0}]
    r = api.track_holdings(json.dumps(hs))
    cols = {"Ticker", "Quantity", "Buy Price", "Current Price", "Total Value", "P&L (INR)", "P&L (%)"}
    assert all(cols <= set(row) for row in r["table"]) and {row["Ticker"] for row in r["table"]} == {"TCS.NS", "AAPL"}
    assert isinstance(r["total_value"], float) and r["currency"] == "INR" and r["fx_rate"] > 1.0
    aapl = next(row for row in r["table"] if row["Ticker"] == "AAPL")
    assert aapl["Currency"] == "USD" and aapl["Total Value"] > 5 * 150.0            # converted to INR
    assert abs(r["total_pnl"] - (r["total_value"] - r["total_cost"])) < 1e-6


def test_validate_save_load_and_voice_helpers():
    v = api.validate_tickers(("TCS.NS", "NOTAREALTICKER123"))
    assert v == {"TCS.NS": True, "NOTAREALTICKER123": False}
    assert api.load_portfolio("never-saved") is None
    ok = api.save_portfolio("my session", ["TCS.NS", "INFY.NS"], {"TCS.NS": 60000.0, "INFY.NS": 40000.0}, "Low", "hi")
    assert ok is True
    back = api.load_portfolio("my session")
    assert back["tickers"] == ["TCS.NS", "INFY.NS"] and back["risk_level"] == "Low" and back["lang"] == "hi"
    assert back["amounts"]["TCS.NS"] == pytest.approx(60000.0)
    ex = api.extract_portfolio("50000 rupees in TCS and Infosys low risk")
    assert ex["tickers"] == ["TCS.NS", "INFY.NS"] and ex["risk_level"] == "Low"
    assert api.transcribe(b"RIFF", "hi") == ""                                       # offline: typed fallback


def test_text_helpers_work_offline():
    label = api.classify_risk(TICKERS, WEIGHTS, START, END)["label"]
    txt = api.explain_portfolio(TICKERS, WEIGHTS, label, "hi", START, END)
    assert isinstance(txt, str) and len(txt) > 40                                    # English template fallback
    assert "Increase" in api.explain_allocation(TICKERS, WEIGHTS, REC, "Medium", "en", START, END) or \
           "Reduce" in api.explain_allocation(TICKERS, WEIGHTS, REC, "Medium", "en", START, END)
    news = api.get_news_insight(TICKERS, "en")
    assert {"sentiment", "summary", "headlines", "risks"} <= set(news)
