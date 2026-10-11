"""Acceptance + regression tests for the finance engine. Fully offline: synthetic prices, faked downloader,
temporary cache dirs, Supabase disabled. Run:  python -m pytest tests -q
"""
from __future__ import annotations

import ast
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import data, db, metrics, optimizer, simulate, track  # noqa: E402
from engine.serialize import load_mock  # noqa: E402


# --------------------------------------------------------------------------- helpers / fixtures
def make_prices(tickers, n=600, seed=0, drifts=None, vols=None, start="2021-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(tickers)
    drifts = drifts if drifts is not None else rng.uniform(0.02, 0.2, k)
    vols = vols if vols is not None else rng.uniform(0.15, 0.35, k)
    market = rng.normal(0, 0.008, n)
    cols = {}
    for t, d, v in zip(tickers, drifts, vols):
        r = d / 252 + 0.6 * market + rng.normal(0, 0.8 * v / np.sqrt(252), n)
        cols[t] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=pd.bdate_range(start, periods=n))


def fake_series(ticker: str, start, end) -> pd.Series:
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    full = pd.bdate_range("2019-01-01", "2024-12-31")
    path = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(full))))
    return pd.Series(path, index=full).loc[start:end]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No network, no Supabase, private cache/fallback dirs, no sleeping."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "prices")
    monkeypatch.setattr(data, "FALLBACK_DIR", tmp_path / "fallback")
    monkeypatch.setattr(data, "SYNC_TO_SUPABASE", False)
    monkeypatch.setattr(data, "_sleep", lambda s: None)
    monkeypatch.setattr(db, "LOCAL_DIR", tmp_path / "local")
    monkeypatch.setattr(db, "_read_secrets", lambda: (None, None))
    monkeypatch.setattr(db, "_client", None)
    monkeypatch.setattr(db, "_disabled_until", 0.0)
    data._tail_tried.clear()
    data._head_tried.clear()
    return tmp_path


@pytest.fixture
def downloader(monkeypatch):
    """Replace the network call; records every request."""
    calls = []

    def fake(tickers, start, end_inclusive):
        calls.append((list(tickers), pd.Timestamp(start), pd.Timestamp(end_inclusive)))
        return pd.DataFrame({t: fake_series(t, start, end_inclusive) for t in tickers}).dropna(how="all")

    monkeypatch.setattr(data, "_download", fake)
    return calls


# --------------------------------------------------------------------------- metrics
def test_two_asset_equal_weight_matches_hand_calculation():
    px = make_prices(["TCS.NS", "INFY.NS"], n=300, seed=1)
    m = metrics.portfolio_metrics(px, {"TCS.NS": 0.5, "INFY.NS": 0.5})

    r = np.log(px.to_numpy()[1:] / px.to_numpy()[:-1])
    rp = r.mean(axis=1)
    w = np.array([0.5, 0.5])
    cov = LedoitWolf().fit(r).covariance_ * 252
    ann_return = np.exp(rp.mean() * 252) - 1
    vol = np.sqrt(w @ cov @ w)
    value = np.concatenate([[1.0], np.exp(np.cumsum(rp))])
    mdd = (value / np.maximum.accumulate(value) - 1).min()
    rf = 0.065
    rf_d = np.log1p(rf) / 252
    downside = np.sqrt(np.mean(np.minimum(rp - rf_d, 0) ** 2)) * np.sqrt(252)
    k = int(np.ceil(0.05 * len(rp)))
    expected = {
        "ann_return": ann_return, "ann_vol": vol, "max_drawdown": mdd,
        "sharpe": (ann_return - rf) / vol, "sortino": (ann_return - rf) / downside,
        "calmar": ann_return / abs(mdd), "cvar_95": np.sort(rp)[:k].mean(),
        "avg_corr": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), "hhi": 0.5, "effective_n": 2.0,
    }
    assert set(m) == set(metrics.METRIC_KEYS) == set(expected)
    for key, val in expected.items():
        assert m[key] == pytest.approx(val, abs=1e-6), key


def test_risk_and_return_contribution_sum_to_one():
    px = make_prices(["A.NS", "B.NS", "C.NS", "D"], seed=3)
    w = {"A.NS": 50, "B.NS": 30, "C.NS": 15, "D": 5}          # percentages, not summing to 1
    assert metrics.risk_contribution(px, w).sum() == pytest.approx(1.0, abs=1e-9)
    assert metrics.return_contribution(px, w).sum() == pytest.approx(1.0, abs=1e-9)


def test_risk_contribution_flags_the_dominant_risk_asset():
    px = make_prices(["CALM.NS", "WILD.NS"], seed=4, vols=[0.10, 0.60], drifts=[0.1, 0.1])
    rc = metrics.risk_contribution(px, {"CALM.NS": 0.5, "WILD.NS": 0.5})
    assert rc["WILD.NS"] > 0.8


def test_weights_are_normalised_and_negatives_rejected():
    px = make_prices(["A.NS", "B.NS"], seed=5)
    a = metrics.portfolio_metrics(px, {"A.NS": 60, "B.NS": 40})
    b = metrics.portfolio_metrics(px, {"A.NS": 0.6, "B.NS": 0.4})
    assert a == pytest.approx(b)
    with pytest.raises(ValueError):
        metrics.portfolio_metrics(px, {"A.NS": 1.2, "B.NS": -0.2})
    with pytest.raises(ValueError):
        metrics.portfolio_metrics(px, {"A.NS": 0.0, "B.NS": 0.0})


def test_single_asset_and_zero_variance_are_finite():
    px = make_prices(["A.NS", "B.NS"], seed=6)
    single = metrics.portfolio_metrics(px[["A.NS"]], {"A.NS": 1.0})
    assert single["avg_corr"] == 0.0 and single["hhi"] == 1.0 and single["effective_n"] == 1.0
    flat = px.copy()
    flat["B.NS"] = 50.0
    m = metrics.portfolio_metrics(flat, {"A.NS": 0.5, "B.NS": 0.5})
    assert all(np.isfinite(v) for v in m.values())
    rc = metrics.risk_contribution(flat, {"A.NS": 0.5, "B.NS": 0.5})
    assert rc.sum() == pytest.approx(1.0)


def test_rf_defaults_follow_currency():
    assert metrics.default_rf(["TCS.NS"]) == pytest.approx(0.065)
    assert metrics.default_rf(["AAPL"]) == pytest.approx(0.04)
    assert metrics.default_rf(["TCS.NS", "AAPL"], [0.5, 0.5]) == pytest.approx(0.0525)


def test_feature_inputs_keys():
    px = make_prices(["A.NS", "B.NS", "C.NS"], seed=7)
    f = metrics.feature_inputs(px, {"A.NS": 0.5, "B.NS": 0.3, "C.NS": 0.2})
    assert set(f) == {"ann_return", "ann_vol", "max_drawdown", "avg_corr", "hhi", "effective_n",
                      "top_weight", "skew", "kurtosis"}
    assert f["top_weight"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- optimizer
@pytest.mark.parametrize("seed", range(5))
def test_optimizer_ordering_sum_and_bounds(seed):
    tickers = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]
    px = make_prices(tickers, n=800, seed=seed, drifts=[0.22, 0.04, 0.14, 0.02, 0.10],
                     vols=[0.30, 0.15, 0.25, 0.12, 0.22])
    ew = {t: 0.2 for t in tickers}
    m_ew = metrics.portfolio_metrics(px, ew)
    lo, hi = 0.05, 0.40
    out = {lvl: optimizer.optimize(px, lvl, bounds=(lo, hi)) for lvl in ("Low", "Medium", "High")}
    for lvl, r in out.items():
        w = np.array(list(r["weights"].values()))
        assert w.sum() == pytest.approx(1.0, abs=1e-9), lvl
        assert w.min() >= lo - 1e-9 and w.max() <= hi + 1e-9, lvl
        assert set(r) >= {"weights", "metrics", "reason", "tradeoffs", "mode"}
    assert out["Low"]["metrics"]["ann_vol"] <= m_ew["ann_vol"] + 1e-9
    assert out["Low"]["metrics"]["ann_vol"] <= out["Medium"]["metrics"]["ann_vol"] + 1e-9
    assert out["Medium"]["metrics"]["sharpe"] >= m_ew["sharpe"] - 1e-9
    assert out["High"]["metrics"]["ann_vol"] <= 1.3 * m_ew["ann_vol"] * (1 + 1e-6)
    assert out["High"]["metrics"]["ann_return"] >= out["Low"]["metrics"]["ann_return"] - 1e-9
    assert [out[k]["mode"] for k in ("Low", "Medium", "High")] == ["min_variance", "max_sharpe", "max_return_vol_cap"]


def test_optimizer_is_deterministic():
    px = make_prices(["A.NS", "B.NS", "C.NS", "D.NS"], seed=11)
    a = optimizer.optimize(px, "Medium", current_weights={"A.NS": 0.7, "B.NS": 0.3})
    b = optimizer.optimize(px, "Medium", current_weights={"A.NS": 0.7, "B.NS": 0.3})
    assert a == b


def test_optimizer_relaxes_infeasible_bounds_and_says_so():
    many = make_prices([f"T{i}.NS" for i in range(25)], n=400, seed=2)      # 25 * 5% > 100%
    r = optimizer.optimize(many, "Low")
    w = np.array(list(r["weights"].values()))
    assert w.sum() == pytest.approx(1.0) and r["mode"] == "min_variance"
    assert "relaxed" in r["reason"]
    two = make_prices(["A.NS", "B.NS"], seed=2)                              # 2 * 40% < 100%
    r2 = optimizer.optimize(two, "Medium")
    assert sum(r2["weights"].values()) == pytest.approx(1.0) and "relaxed" in r2["reason"]


def test_optimizer_single_asset_and_solver_failure_fall_back_to_equal(monkeypatch):
    px = make_prices(["A.NS", "B.NS", "C.NS"], seed=8)
    one = optimizer.optimize(px[["A.NS"]], "High")
    assert one["weights"] == {"A.NS": 1.0} and one["mode"] == "fallback_equal"
    monkeypatch.setattr(optimizer, "_solve", lambda *a, **k: None)
    r = optimizer.optimize(px, "Medium")
    assert r["mode"] == "fallback_equal"
    assert list(r["weights"].values()) == pytest.approx([1 / 3] * 3)


def test_optimizer_rejects_unknown_risk_level():
    with pytest.raises(ValueError):
        optimizer.optimize(make_prices(["A.NS", "B.NS"]), "Extreme")


def test_optimizer_reason_names_the_biggest_changes():
    px = make_prices(["A.NS", "B.NS", "C.NS", "D.NS"], seed=9)
    r = optimizer.optimize(px, "Low", current_weights={"A.NS": 0.7, "B.NS": 0.1, "C.NS": 0.1, "D.NS": 0.1})
    assert "Objective: minimum variance" in r["reason"] and "Biggest changes" in r["reason"]
    assert r["tradeoffs"]


# --------------------------------------------------------------------------- simulate / compare
def test_simulate_growth_and_dropped_assets():
    px = make_prices(["A.NS", "B.NS"], seed=12)
    px["GHOST.NS"] = np.nan                                  # no data at all in the window
    out = simulate.simulate(px, {"A.NS": 0.4, "B.NS": 0.4, "GHOST.NS": 0.2, "MISSING.NS": 0.0}, "2021-03-01", "2022-03-01")
    assert out["growth"].iloc[0] == pytest.approx(100.0)
    assert set(out["dropped_tickers"]) == {"GHOST.NS", "MISSING.NS"}
    assert set(out["metrics"]) == set(metrics.METRIC_KEYS)
    empty = simulate.simulate(px, {"A.NS": 1.0}, "1990-01-01", "1990-02-01")
    assert empty["growth"].empty and "note" in empty


def test_regime_presets():
    p = simulate.regime_presets()
    assert list(p) == ["COVID crash", "2021 bull run", "2022 bear/volatile", "Last 1 year"]
    assert p["COVID crash"] == ("2020-01-20", "2020-04-30")
    start, end = map(pd.Timestamp, p["Last 1 year"])
    assert (end - start).days == 365


def test_compare_strategies_growth_and_risk_return():
    tickers = ["A.NS", "B.NS", "C.NS"]
    px = make_prices(tickers, seed=13)
    bench = make_prices(["^NSEI"], seed=14)["^NSEI"]
    orig, rec = {"A.NS": 0.8, "B.NS": 0.1, "C.NS": 0.1}, {"A.NS": 0.34, "B.NS": 0.33, "C.NS": 0.33}
    table = simulate.compare_strategies(px, orig, rec, bench)
    assert list(table.index) == ["Original", "Recommended", "Equal Weight", "Benchmark"]
    assert set(metrics.METRIC_KEYS) <= set(table.columns)
    assert table.loc["Equal Weight", "hhi"] == pytest.approx(1 / 3)
    growth = simulate.compare_growth(px, orig, rec, bench)
    assert list(growth.columns) == list(table.index)
    assert np.allclose(growth.iloc[0], 100.0)
    # growth endpoint agrees with the table's total return
    assert growth["Original"].iloc[-1] / 100 - 1 == pytest.approx(table.loc["Original", "total_return"], abs=1e-9)
    pts = simulate.risk_return_points(px, {"Original": orig, "Recommended": rec}, bench)
    assert {pts[k]["kind"] for k in pts} == {"asset", "portfolio", "benchmark"}
    assert pts["Original"]["vol"] == pytest.approx(table.loc["Original", "ann_vol"], rel=0.05)


# --------------------------------------------------------------------------- scenarios / Monte Carlo
@pytest.fixture
def scenario_world(monkeypatch):
    # same seed -> same market factor, so the indices are genuinely correlated with the assets (beta > 0)
    px = make_prices(["TCS.NS", "HDFCBANK.NS", "AAPL"], n=400, seed=21, vols=[0.25, 0.25, 0.25])
    bench = make_prices(["^NSEI", "^NDX"], n=400, seed=21, vols=[0.05, 0.05])
    monkeypatch.setattr(simulate, "get_prices", lambda syms, *a, **k: bench[[s for s in syms if s in bench.columns]])
    return px, {"TCS.NS": 0.4, "HDFCBANK.NS": 0.4, "AAPL": 0.2}


def test_market_shock_uses_beta_and_appends_one_day(scenario_world):
    px, w = scenario_world
    r = simulate.apply_scenario(px, w, {"type": "market_shock", "pct": -0.20})
    assert set(r) >= {"before", "after", "value_change_pct", "stressed_prices"}
    assert len(r["stressed_prices"]) == len(px) + 1
    for t, shock in r["asset_shocks"].items():
        assert shock == pytest.approx(max(r["betas"][t] * -0.20, -0.99))
    assert r["value_change_pct"] == pytest.approx(sum(w[t] * s for t, s in r["asset_shocks"].items()))
    assert all(b > 0.1 for b in r["betas"].values())
    assert r["value_change_pct"] < 0 and r["after"]["max_drawdown"] <= r["before"]["max_drawdown"]
    last = r["stressed_prices"].iloc[-1] / px.iloc[-1] - 1
    assert last.to_dict() == pytest.approx(r["asset_shocks"])


def test_recovery_vol_multiplier_and_sector_shock(scenario_world):
    px, w = scenario_world
    up = simulate.apply_scenario(px, w, {"type": "recovery", "pct": 0.15})
    assert up["value_change_pct"] > 0
    vm = simulate.apply_scenario(px, w, {"type": "vol_multiplier", "k": 2})
    assert vm["after"]["ann_vol"] > vm["before"]["ann_vol"]
    assert len(vm["stressed_prices"]) == len(px)
    assert vm["stressed_prices"].iloc[-1].to_numpy() == pytest.approx(px.iloc[-1].to_numpy())  # mean-preserving
    sec = simulate.apply_scenario(px, w, {"type": "sector_shock", "sector": "it", "pct": -0.30})
    assert sec["asset_shocks"]["TCS.NS"] == pytest.approx(-0.30)                  # IT: direct
    assert sec["asset_shocks"]["AAPL"] == pytest.approx(-0.30)
    assert sec["asset_shocks"]["HDFCBANK.NS"] == pytest.approx(0.3 * sec["betas"]["HDFCBANK.NS"] * -0.30)  # spillover
    with pytest.raises(ValueError):
        simulate.apply_scenario(px, w, {"type": "meteor"})


def test_scenario_works_without_benchmark_data(monkeypatch):
    px = make_prices(["TCS.NS", "INFY.NS"], n=300, seed=23)
    monkeypatch.setattr(simulate, "get_prices", lambda *a, **k: pd.DataFrame())
    r = simulate.apply_scenario(px, {"TCS.NS": 0.5, "INFY.NS": 0.5}, {"type": "market_shock", "pct": -0.3})
    assert r["warnings"] and r["value_change_pct"] < 0


def test_monte_carlo_shape_order_and_determinism():
    px = make_prices(["A.NS", "B.NS"], n=500, seed=31)
    w = {"A.NS": 0.5, "B.NS": 0.5}
    a = simulate.monte_carlo(px, w, n=1000, horizon=252)
    b = simulate.monte_carlo(px, w, n=1000, horizon=252)
    assert a == b
    p = a["percentiles"]
    assert set(p) == {"p5", "p25", "p50", "p75", "p95"} and all(len(v) == 252 for v in p.values())
    stacked = np.array([p[k] for k in ("p5", "p25", "p50", "p75", "p95")])
    assert (np.diff(stacked, axis=0) >= -1e-9).all()
    assert 0.0 <= a["prob_loss"] <= 1.0 and a["horizon"] == 252
    assert a["expected_shortfall"] <= stacked[2, -1] / 100 - 1 + 1e-9
    assert simulate.monte_carlo(px, w, n=50, horizon=30, seed=1) != simulate.monte_carlo(px, w, n=50, horizon=30, seed=2)


# --------------------------------------------------------------------------- data layer
def test_get_prices_one_batched_call_then_cache(downloader):
    t = ["tcs.ns", "TCS.NS", " AAPL ", "INFY.NS"]
    a = data.get_prices(t, "2021-01-01", "2023-12-29")
    assert len(downloader) == 1 and downloader[0][0] == ["TCS.NS", "AAPL", "INFY.NS"]      # normalised, de-duped
    assert a.attrs["source"] == "live" and a.attrs["mixed_currency"]
    assert a.attrs["currencies"]["AAPL"] == "USD" and not a.isna().any().any()
    b = data.get_prices(t, "2021-01-01", "2023-12-29")
    assert len(downloader) == 1 and b.attrs["source"] == "cache"
    pd.testing.assert_frame_equal(a, b)


def test_get_prices_only_fetches_missing_tail_and_head(downloader):
    data.get_prices(["TCS.NS"], "2021-01-01", "2022-06-30")
    downloader.clear()
    data.get_prices(["TCS.NS"], "2021-01-01", "2023-06-30")           # newer end -> tail only
    assert len(downloader) == 1
    assert downloader[0][1] >= pd.Timestamp("2022-06-01")             # overlap week, not the full history
    downloader.clear()
    out = data.get_prices(["TCS.NS"], "2020-01-01", "2023-06-30")     # older start -> head
    assert len(downloader) == 1 and downloader[0][1] == pd.Timestamp("2020-01-01")
    assert out.index[0] <= pd.Timestamp("2020-01-03")


def test_retries_then_fallback_when_offline(monkeypatch, isolated):
    attempts = []

    def boom(*a, **k):
        attempts.append(1)
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "_download", boom)
    fb = data.FALLBACK_DIR
    fb.mkdir(parents=True)
    s = fake_series("TCS.NS", "2020-01-01", "2024-06-28")
    s.rename("adj_close").rename_axis("date").to_frame().to_parquet(fb / "TCS.NS.parquet")
    out = data.get_prices(["TCS.NS", "NOPE.NS"], "2021-01-01", "2023-12-29")
    assert len(attempts) == 3                                         # 3 attempts, one batch
    assert out.attrs["source"] == "fallback"
    assert list(out.columns) == ["TCS.NS"] and out.attrs["dropped"] == ["NOPE.NS"]
    assert "NOPE.NS" in out.attrs["dropped_reasons"]


def test_short_history_ticker_is_dropped(monkeypatch, downloader):
    real = data._download

    def short(tickers, start, end):
        df = real(tickers, start, end)
        if "NEW.NS" in df.columns:
            df.loc[df.index < "2023-06-01", "NEW.NS"] = np.nan
        return df

    monkeypatch.setattr(data, "_download", short)
    out = data.get_prices(["TCS.NS", "NEW.NS"], "2021-01-01", "2023-12-29")
    assert "NEW.NS" in out.attrs["dropped"] and list(out.columns) == ["TCS.NS"]


def test_adjustment_event_triggers_full_refresh(monkeypatch, downloader):
    first = data.get_prices(["TCS.NS"], "2021-01-04", "2023-06-30")
    real = data._download

    def split_adjusted(tickers, start, end):                          # whole history re-scaled by 2
        return real(tickers, start, end) / 2

    monkeypatch.setattr(data, "_download", split_adjusted)
    data._tail_tried.clear()
    second = data.get_prices(["TCS.NS"], "2021-01-04", "2023-12-29")
    ratio = (second["TCS.NS"].loc[first.index] / first["TCS.NS"]).to_numpy()
    assert ratio == pytest.approx(0.5)                                # old rows were refreshed, not mixed


def test_outer_alignment_keeps_nan_and_inner_ffills_gaps(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=400)
    a = pd.Series(np.linspace(100, 150, 400), index=idx)
    b = a.copy()
    b.iloc[200:202] = np.nan                                          # 2-day hole -> bridged by ffill(limit=3)
    c = a.copy()
    c.iloc[:100] = np.nan                                             # late starter
    monkeypatch.setattr(data, "_download", lambda t, s, e: pd.DataFrame({"A": a, "B": b, "C": c})[t])
    inner = data.get_prices(["A", "B", "C"], "2021-01-04", "2022-07-01")
    assert inner.index[0] == idx[100] and not inner.isna().any().any()
    assert any("truncated" in w for w in inner.attrs["warnings"]) or (inner.index[0] - pd.Timestamp("2021-01-04")).days <= 30
    outer = data.get_prices(["A", "B", "C"], "2021-01-04", "2022-07-01", align="outer")
    assert outer["C"].iloc[:100].isna().all() and len(outer) > len(inner)


def test_validate_tickers_offline_uses_known_sources(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    out = data.validate_tickers(["TCS.NS", "aapl", "ZZZZ.NS"])
    assert out == {"valid": ["TCS.NS", "AAPL"], "invalid": ["ZZZZ.NS"]}


def test_get_benchmark_aliases(downloader):
    s = data.get_benchmark("NIFTY 50", "2021-01-01", "2022-12-30")
    assert s.name == "^NSEI" and len(s) > 400
    assert downloader[0][0] == ["^NSEI"]
    assert data.get_benchmark("NASDAQ-100", "2021-01-01", "2022-12-30").name == "^NDX"


# --------------------------------------------------------------------------- tracking
def test_track_holdings_totals_never_mix_currencies(monkeypatch):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    frame = pd.DataFrame({"TCS.NS": np.linspace(3000, 3600, 300), "AAPL": np.linspace(150, 200, 300),
                          "INR=X": np.full(300, 80.0)}, index=idx)
    monkeypatch.setattr(track, "get_prices", lambda *a, **k: frame)
    out = track.track_holdings([
        {"ticker": "TCS.NS", "quantity": 10, "buy_price": 3000},
        {"ticker": "aapl", "quantity": 5, "buy_price": 100},
        {"ticker": "AAPL", "quantity": 5, "buy_price": 200},          # merged: 10 @ 150
        {"ticker": "JUNK", "quantity": 0, "buy_price": 5},            # skipped
    ])
    assert out["total_value"] == pytest.approx({"INR": 36000.0, "USD": 2000.0})
    assert out["total_pnl"] == pytest.approx({"INR": 6000.0, "USD": 500.0})
    assert set(out["table"]["ticker"]) == {"TCS.NS", "AAPL"}
    assert sum(out["allocation"].values()) == pytest.approx(1.0)
    assert out["allocation"]["TCS.NS"] == pytest.approx(36000 / (36000 + 2000 * 80))
    assert set(out["risk_metrics"]) == set(metrics.METRIC_KEYS)
    assert any("JUNK" in w for w in out["warnings"])


def test_track_holdings_empty_input():
    out = track.track_holdings([])
    assert out["table"].empty and out["total_value"] == {} and out["risk_metrics"] == {}


# --------------------------------------------------------------------------- db
def test_db_degrades_gracefully_without_supabase():
    assert db.get_client() is None
    assert db.read_price_cache(["TCS.NS"], "2021-01-01", "2021-02-01") is None
    assert db.upsert_prices(make_prices(["A"], n=5)) == 0
    assert db.upload_model(__file__) is False and db.download_model("risk_model.joblib") is None
    assert db.health() == {"configured": False, "reachable": False, "schema_ready": False}
    assert db.llm_cache_get("k") is None
    db.llm_cache_put("k", "news", "hi", "सार", "gemini")
    assert db.llm_cache_get("k")["response"] == "सार"
    pid = db.save_portfolio("My mix", "sess-1", "Low", "hi",
                            [{"ticker": "tcs.ns", "quantity": 10, "buy_price": 3000},
                             {"ticker": "AAPL", "quantity": 2, "buy_price": 150}])
    listed = db.list_portfolios("sess-1")
    assert [p["id"] for p in listed] == [pid] and db.list_portfolios("other") == []
    loaded = db.load_portfolio(pid)
    assert loaded["risk_level"] == "Low" and loaded["language"] == "hi"
    by_t = {h["ticker"]: h for h in loaded["holdings"]}
    assert by_t["TCS.NS"]["buy_price"] == pytest.approx(3000) and by_t["TCS.NS"]["quantity"] == 10
    assert sum(h["weight"] for h in loaded["holdings"]) == pytest.approx(1.0)
    assert db.save_run(pid, "Low", {"sharpe": 1.0}, {"TCS.NS": 1.0}) is False   # stored locally
    assert db.load_portfolio("does-not-exist") is None


# --------------------------------------------------------------------------- robustness regressions
def test_max_sharpe_never_scores_below_equal_weight_on_random_markets():
    """The acceptance property holds by construction (shrinkage relaxes, equal weight is a candidate)."""
    worse = []
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        k = int(rng.integers(3, 9))
        tickers = [f"S{i}.NS" for i in range(k)]
        px = make_prices(tickers, n=int(rng.integers(300, 900)), seed=seed,
                         drifts=rng.uniform(-0.1, 0.3, k), vols=rng.uniform(0.1, 0.5, k))
        r = optimizer.optimize(px, "Medium", bounds=(0.02, 0.5))
        if r["mode"] != "max_sharpe":
            continue
        ew = metrics.portfolio_metrics(px, {t: 1 / k for t in tickers})
        if r["metrics"]["sharpe"] < ew["sharpe"] - 1e-9:
            worse.append(seed)
    assert worse == []


def test_optimizer_validates_bounds_and_handles_dead_columns():
    px = make_prices(["A.NS", "B.NS", "C.NS"], seed=15)
    for bad in [(0.5, 0.2), (-0.1, 0.4), (0.0, 1.5)]:
        with pytest.raises(ValueError):
            optimizer.optimize(px, "Low", bounds=bad)
    px["DEAD.NS"] = np.nan
    r = optimizer.optimize(px, "Low")
    assert "DEAD.NS" not in r["weights"] and sum(r["weights"].values()) == pytest.approx(1.0)


def test_ticker_case_insensitive_weights_everywhere():
    px = make_prices(["AAPL", "TCS.NS"], seed=16)
    lower, upper = {"aapl": 0.6, "tcs.ns": 0.4}, {"AAPL": 0.6, "TCS.NS": 0.4}
    assert metrics.portfolio_metrics(px, lower) == pytest.approx(metrics.portfolio_metrics(px, upper))
    sim = simulate.simulate(px, lower, "2021-02-01", "2022-01-01")
    assert sim["dropped_tickers"] == [] and not sim["growth"].empty
    r = optimizer.optimize(px, "Low", current_weights=lower)
    assert r["before_weights"]["AAPL"] == pytest.approx(0.6)


def test_network_failure_is_retried_sooner_than_missing_ticker(monkeypatch):
    monkeypatch.setattr(data, "_download", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    data.get_prices(["TCS.NS"], "2021-01-01", "2023-12-29")
    assert data._tail_tried["TCS.NS"][2] == data.FAILURE_TTL_S
    monkeypatch.setattr(data, "_download",
                        lambda t, s, e: pd.DataFrame({"TCS.NS": fake_series("TCS.NS", s, e)}).reindex(columns=list(t)))
    data._tail_tried.clear()
    data.get_prices(["TCS.NS", "GHOST.NS"], "2021-01-01", "2023-12-29")
    assert data._tail_tried["GHOST.NS"][2] == data.RECHECK_TTL_S      # data came back, ticker simply has none


class _ApiError(Exception):
    code = "42501"


class _FakeTable:
    def __init__(self, store, name):
        self.store, self.name, self.op, self.filters = store, name, None, []

    def insert(self, rows):
        self.op = ("insert", rows)
        return self

    def delete(self):
        self.op = ("delete", None)
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def execute(self):
        kind, rows = self.op
        if self.name in self.store.get("fail", set()) and kind == "insert":
            raise _ApiError("permission denied")
        table = self.store.setdefault(self.name, [])
        if kind == "insert":
            rows = [rows] if isinstance(rows, dict) else rows
            out = [{"id": f"id-{len(table) + i}", **r} for i, r in enumerate(rows)]
            table += out
            return type("R", (), {"data": out})()
        table[:] = [r for r in table if not all(r.get(c) == v for c, v in self.filters)]
        return type("R", (), {"data": []})()


class _FakeClient:
    def __init__(self, fail=()):
        self.store = {"fail": set(fail)}

    def table(self, name):
        return _FakeTable(self.store, name)


def test_db_rolls_back_half_written_portfolio_and_falls_back_locally(monkeypatch):
    fake = _FakeClient(fail={"holdings"})
    monkeypatch.setattr(db, "get_client", lambda: fake)
    pid = db.save_portfolio("p", "s", "Low", "en", [{"ticker": "TCS.NS", "quantity": 1, "buy_price": 10}])
    assert fake.store.get("portfolios") == []                         # orphan portfolio row was deleted
    assert db.load_portfolio(pid)["holdings"][0]["ticker"] == "TCS.NS"   # still saved (locally)
    assert db._disabled_until == 0.0                                  # an API error is not a network outage


def test_db_circuit_breaker_trips_only_on_transport_errors(monkeypatch):
    import httpx

    monkeypatch.setattr(db, "get_client", lambda: object())
    ok, err = db._remote(lambda c: (_ for _ in ()).throw(_ApiError("rls")), "t")
    assert not ok and db._disabled_until == 0.0
    ok, err = db._remote(lambda c: (_ for _ in ()).throw(httpx.ConnectError("down")), "t")
    assert not ok and db._disabled_until > 0.0
    db.reset_circuit()
    assert db._disabled_until == 0.0


# --------------------------------------------------------------------------- repo-level contracts
def test_engine_never_imports_streamlit():
    for path in (ROOT / "engine").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] == "streamlit" for n in names), f"{path.name} imports streamlit"


def track_with_frame(px):
    """Run the real track_holdings on a synthetic price frame (no network)."""
    import engine.track as tr

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    frame = pd.DataFrame({"TCS.NS": np.linspace(3000, 3600, 300), "AAPL": np.linspace(150, 200, 300),
                          "INR=X": np.full(300, 80.0)}, index=idx)
    orig = tr.get_prices
    tr.get_prices = lambda *a, **k: frame
    try:
        return tr.track_holdings([{"ticker": "TCS.NS", "quantity": 1, "buy_price": 3000},
                                  {"ticker": "AAPL", "quantity": 1, "buy_price": 100}])
    finally:
        tr.get_prices = orig


def test_mock_files_match_real_return_shapes(scenario_world):
    px, w = scenario_world
    assert set(load_mock("metrics")) == set(metrics.portfolio_metrics(px, w))
    real_opt = optimizer.optimize(px, "Medium")
    assert set(load_mock("optimize")) == set(real_opt)
    assert set(load_mock("optimize")["metrics"]) == set(real_opt["metrics"])
    sim = simulate.simulate(px, w, "2021-06-01", "2022-06-01")
    assert set(load_mock("simulate")) == set(sim)
    assert isinstance(load_mock("simulate")["growth"], pd.Series)
    cmp_real = simulate.compare_strategies(px, w, w, make_prices(["^NSEI"], n=400, seed=22)["^NSEI"])
    assert list(load_mock("compare").index) == list(cmp_real.index)
    assert list(load_mock("compare").columns) == list(cmp_real.columns)
    sc = simulate.apply_scenario(px, w, {"type": "market_shock", "pct": -0.2})
    assert set(load_mock("scenario")) == set(sc)
    mc = simulate.monte_carlo(px, w)
    assert set(load_mock("monte_carlo")) == set(mc) and len(load_mock("monte_carlo")["percentiles"]["p5"]) == 252
    trk = load_mock("track")
    real_trk = track_with_frame(px)
    assert set(trk) == set(real_trk)
    assert list(trk["table"].columns) == track.TABLE_COLUMNS
