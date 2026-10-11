"""Historical simulation, regime presets, strategy comparison, stress scenarios and Monte Carlo."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .data import BENCHMARKS, get_benchmark, get_prices
from .metrics import (METRIC_KEYS, default_rf, match_columns, normalize_weights, portfolio_metrics,
                      portfolio_returns, portfolio_value, prepare)
from .universe import currency_of, sector_of

log = logging.getLogger("engine.simulate")

STRATEGIES = ("Original", "Recommended", "Equal Weight", "Benchmark")
MIN_COVERAGE = 0.5   # an asset needs data on at least half of the window rows to stay in a simulation


def _empty_metrics() -> dict:
    return {k: 0.0 for k in METRIC_KEYS}


def _slice_clean(prices: pd.DataFrame, weights: dict, start=None, end=None) -> tuple[pd.DataFrame, dict, list[str]]:
    """Slice to the window, drop weighted assets that are missing/too sparse, return (prices, weights, dropped)."""
    w_all = match_columns(normalize_weights(weights), prices.columns)
    window = prices.loc[start:end] if (start is not None or end is not None) else prices
    dropped, keep = [], []
    for t in w_all:
        if t not in window.columns:
            dropped.append(t)
        elif window[t].notna().sum() < max(2, MIN_COVERAGE * len(window)):
            dropped.append(t)
        else:
            keep.append(t)
    if dropped:
        log.warning("dropped %s: no usable data in the window", dropped)
    sub = window[keep].ffill(limit=3).dropna(how="any") if keep else window.iloc[0:0][[]]
    return sub, {t: w_all[t] for t in keep}, dropped


# --------------------------------------------------------------------------- simulate / regimes
def simulate(prices: pd.DataFrame, weights: dict, start: str, end: str) -> dict:
    """Growth of 100 and metrics for ``weights`` over ``[start, end]``.

    Returns ``{"growth": Series (base 100), "metrics": dict, "dropped_tickers": [...]}``. Assets without
    data in the window are dropped (and weights renormalised) rather than raising; if nothing usable is
    left, ``growth`` is empty and ``metrics`` are zeros, with a ``"note"`` explaining why.
    """
    sub, w, dropped = _slice_clean(prices, weights, start, end)
    if not w or len(sub) < 3:
        return {"growth": pd.Series(dtype="float64", name="portfolio"), "metrics": _empty_metrics(),
                "dropped_tickers": dropped, "note": "not enough price history in this window"}
    return {"growth": portfolio_value(sub, w, 100.0), "metrics": portfolio_metrics(sub, w),
            "dropped_tickers": dropped}


def regime_presets() -> dict:
    """Named historical windows (inclusive ISO dates); "Last 1 year" is computed from today."""
    today = pd.Timestamp.today().normalize()
    return {
        "COVID crash": ("2020-01-20", "2020-04-30"),
        "2021 bull run": ("2021-01-01", "2021-12-31"),
        "2022 bear/volatile": ("2022-01-01", "2022-12-31"),
        "Last 1 year": ((today - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")),
    }


# --------------------------------------------------------------------------- strategy comparison
def _benchmark_series(prices: pd.DataFrame, benchmark) -> pd.Series:
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]
    if isinstance(benchmark, pd.Series):
        return benchmark.dropna().astype(float)
    return get_benchmark(str(benchmark), prices.index[0].strftime("%Y-%m-%d"), prices.index[-1].strftime("%Y-%m-%d"))


def _compare_inputs(prices, original_w, recommended_w, benchmark):
    """Common price window + one weights dict per strategy + benchmark series on the same dates."""
    orig = match_columns(normalize_weights(original_w), prices.columns)
    rec = match_columns(normalize_weights(recommended_w), prices.columns)
    tickers = [c for c in prices.columns if c in orig or c in rec]
    if not tickers:
        raise ValueError("none of the portfolio tickers are in the price data")
    px = prices[tickers].ffill(limit=3).dropna(how="any")
    ew = {t: 1.0 / len(tickers) for t in tickers}
    bench = _benchmark_series(prices, benchmark)
    common = px.index.intersection(bench.index) if len(bench) else px.index[:0]
    if len(common) >= 20:
        px, bench = px.loc[common], bench.loc[common]
    else:
        log.warning("benchmark shares only %d dates with the portfolio; comparing on each series' own dates", len(common))
    weights = {"Original": orig, "Recommended": rec, "Equal Weight": ew}
    return px, weights, bench


def compare_strategies(prices: pd.DataFrame, original_w: dict, recommended_w: dict, benchmark) -> pd.DataFrame:
    """Rows Original / Recommended / Equal Weight / Benchmark over the same period; columns are the ten
    metric names plus ``total_return``. ``benchmark`` is a symbol/name (fetched) or a price Series.
    All rows use the original portfolio's risk-free rate so Sharpe/Sortino are comparable."""
    px, weights, bench = _compare_inputs(prices, original_w, recommended_w, benchmark)
    rf = default_rf(px.columns, [weights["Original"].get(t, 0.0) for t in px.columns])
    rows = {}
    for name, w in weights.items():
        sub = px[[t for t in px.columns if w.get(t, 0) > 0]]
        m = portfolio_metrics(sub, w, rf=rf)
        m["total_return"] = float(portfolio_value(sub, w, 1.0).iloc[-1] - 1.0)
        rows[name] = m
    if len(bench) >= 3:
        bdf = bench.to_frame("benchmark")
        m = portfolio_metrics(bdf, {"benchmark": 1.0}, rf=rf)
        m["total_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
    else:
        m = {**_empty_metrics(), "total_return": 0.0}
    rows["Benchmark"] = m
    return pd.DataFrame.from_dict(rows, orient="index").loc[list(STRATEGIES)]


def compare_growth(prices: pd.DataFrame, original_w: dict, recommended_w: dict, benchmark) -> pd.DataFrame:
    """Growth of 100 for Original / Recommended / Equal Weight / Benchmark (date index, one column each)."""
    px, weights, bench = _compare_inputs(prices, original_w, recommended_w, benchmark)
    out = {name: portfolio_value(px[[t for t in px.columns if w.get(t, 0) > 0]], w, 100.0)
           for name, w in weights.items()}
    if len(bench):
        out["Benchmark"] = 100.0 * bench / bench.iloc[0]
    return pd.DataFrame(out).reindex(columns=[s for s in STRATEGIES if s in out])


def risk_return_points(prices: pd.DataFrame, portfolios: dict, benchmark=None) -> dict:
    """``{name: {"ret": annual return, "vol": annual vol, "kind": "asset"|"portfolio"|"benchmark"}}`` for
    every asset, every named portfolio (``portfolios = {name: weights}``) and the benchmark."""
    out: dict[str, dict] = {}
    px = prices.ffill(limit=3).dropna(how="any")
    if len(px) >= 3:
        lr = np.diff(np.log(px.to_numpy()), axis=0)
        ret = np.expm1(lr.mean(axis=0) * 252)
        vol = lr.std(axis=0, ddof=1) * np.sqrt(252)
        for t, r, v in zip(px.columns, ret, vol):
            out[str(t)] = {"ret": float(r), "vol": float(v), "kind": "asset"}
    for name, w in (portfolios or {}).items():
        m = portfolio_metrics(prices, w)
        out[str(name)] = {"ret": m["ann_return"], "vol": m["ann_vol"], "kind": "portfolio"}
    if benchmark is not None:
        b = _benchmark_series(prices, benchmark)
        b = b.loc[b.index.intersection(px.index)] if len(px) and len(b.index.intersection(px.index)) >= 20 else b
        if len(b) >= 3:
            m = portfolio_metrics(b.to_frame("benchmark"), {"benchmark": 1.0})
            out["Benchmark"] = {"ret": m["ann_return"], "vol": m["ann_vol"], "kind": "benchmark"}
    return out


# --------------------------------------------------------------------------- scenarios
BETA_CLIP = (-1.0, 3.0)
SECTOR_SPILLOVER = 0.3         # share of the market-beta effect that non-targeted assets feel in a sector shock
VOL_WINDOW = 60
DEFAULT_BENCHMARK = {"INR": "^NSEI", "USD": "^NDX"}
MIN_BETA_OBS = 30
SCENARIO_TYPES = ("market_shock", "recovery", "vol_multiplier", "sector_shock")


def _asset_betas(px: pd.DataFrame, benchmark: str | None, warnings: list[str]) -> dict[str, float]:
    """Beta of each asset vs its market index (INR -> NIFTY 50, USD -> NASDAQ-100, or an explicit
    ``benchmark``). Falls back to beta vs the equal-weight basket of the assets when an index has no data."""
    ccy = {t: currency_of(t) for t in px.columns}
    bench_for = {t: (BENCHMARKS.get(str(benchmark).strip().upper(), str(benchmark).strip().upper()) if benchmark else DEFAULT_BENCHMARK.get(ccy[t], "^NDX"))
                 for t in px.columns}
    syms = sorted(set(bench_for.values()))
    start, end = px.index[0].strftime("%Y-%m-%d"), px.index[-1].strftime("%Y-%m-%d")
    try:
        bpx = get_prices(syms, start, end, align="outer", min_rows=1)
    except Exception as exc:  # noqa: BLE001 - scenarios must still work offline
        log.warning("benchmark fetch failed: %s", type(exc).__name__)
        bpx = pd.DataFrame()
    lr = np.log(px).diff()
    proxy = lr.mean(axis=1)
    betas: dict[str, float] = {}
    fell_back: list[str] = []
    for t in px.columns:
        b = bench_for[t]
        r_b = np.log(bpx[b].dropna()).diff() if b in bpx.columns else pd.Series(dtype="float64")
        joined = pd.concat([lr[t], r_b.rename("b")], axis=1, join="inner").dropna()
        if len(joined) < MIN_BETA_OBS or joined["b"].var() <= 0:
            joined = pd.concat([lr[t], proxy.rename("b")], axis=1, join="inner").dropna()
            fell_back.append(t)
            if len(joined) < MIN_BETA_OBS or joined["b"].var() <= 0:
                betas[t] = 1.0
                continue
        betas[t] = float(np.clip(joined[t].cov(joined["b"]) / joined["b"].var(), *BETA_CLIP))
    if fell_back:
        warnings.append("no benchmark data for " + ", ".join(fell_back) + "; beta measured against the portfolio's own average")
    return betas



def apply_scenario(prices: pd.DataFrame, weights: dict, scenario: dict) -> dict:
    """Stress the portfolio and return ``{"before", "after", "value_change_pct", "stressed_prices", ...}``.

    Scenario dicts:
    * ``{"type": "market_shock", "pct": -0.20}`` -- each asset moves ``beta_i * pct`` (beta vs its market index)
    * ``{"type": "recovery", "pct": 0.15}``      -- same mechanism, positive
    * ``{"type": "vol_multiplier", "k": 2}``     -- last 60 days' deviations around the mean scaled by ``k``
    * ``{"type": "sector_shock", "sector": "IT", "pct": -0.30}`` -- that sector moves ``pct``; others feel
      ``0.3 * beta_i * pct``
    Optional ``"benchmark"`` overrides the per-currency default index.

    ``stressed_prices`` is the price history with the shock appended as one synthetic final-day move
    (so P2's ``classify_risk`` can re-label it). For ``vol_multiplier`` the last 60 rows are rebuilt instead
    (the terminal price is unchanged by construction, so ``value_change_pct`` is the worst point of the
    stressed window vs its start; ``value_change_basis`` says which definition applies).
    """
    stype = scenario.get("type")
    if stype not in SCENARIO_TYPES:
        raise ValueError(f"unknown scenario type {stype!r}; expected one of {SCENARIO_TYPES}")
    p = prepare(prices, weights)
    px, w, tickers = p.prices, dict(zip(p.tickers, p.w.tolist())), p.tickers
    warnings: list[str] = []
    before = portfolio_metrics(px, w)

    if stype == "vol_multiplier":
        k = float(scenario.get("k", 2.0))
        if k <= 0:
            raise ValueError("vol_multiplier k must be > 0")
        n = min(VOL_WINDOW, len(px) - 1)
        lr = np.diff(np.log(px.to_numpy()), axis=0)
        seg = lr[-n:]
        mean = seg.mean(axis=0)
        seg_k = mean + k * (seg - mean)
        rebuilt = np.exp(np.log(px.iloc[-n - 1].to_numpy()) + np.cumsum(seg_k, axis=0))
        stressed = px.copy()
        stressed.iloc[-n:] = rebuilt
        rp = seg_k @ p.w
        trough = float(np.exp(np.cumsum(rp)).min() - 1.0)
        change, basis, shocks = min(trough, 0.0), "trough_of_stress_window", {t: 0.0 for t in tickers}
        betas: dict[str, float] = {}
    else:
        betas = _asset_betas(px, scenario.get("benchmark"), warnings)
        pct = scenario.get("pct")
        if pct is None:
            raise ValueError(f"scenario {stype!r} needs a 'pct' value")
        pct = float(pct)
        if stype == "sector_shock":
            sector = str(scenario.get("sector", "")).strip().lower()
            if not sector:
                raise ValueError("sector_shock needs a 'sector'")
            hit = [t for t in tickers if (sector_of(t) or "").lower() == sector]
            if not hit:
                warnings.append(f"no holdings in sector {scenario['sector']!r}; only spillover is applied")
            shocks = {t: (pct if t in hit else SECTOR_SPILLOVER * betas[t] * pct) for t in tickers}
        else:
            shocks = {t: betas[t] * pct for t in tickers}
        shocks = {t: max(v, -0.99) for t, v in shocks.items()}
        last = px.iloc[-1]
        new_row = pd.DataFrame([[last[t] * (1.0 + shocks[t]) for t in tickers]], columns=tickers,
                               index=[px.index[-1] + pd.offsets.BDay(1)])
        stressed = pd.concat([px, new_row])
        change, basis = float(sum(w[t] * shocks[t] for t in tickers)), "instant_shock"

    return {
        "before": before, "after": portfolio_metrics(stressed, w), "value_change_pct": change,
        "stressed_prices": stressed, "value_change_basis": basis, "asset_shocks": shocks,
        "betas": betas, "scenario": dict(scenario), "warnings": warnings,
    }


# --------------------------------------------------------------------------- Monte Carlo
def monte_carlo(prices: pd.DataFrame, weights: dict, n: int = 1000, horizon: int = 252,
                block: int = 10, seed: int = 42) -> dict:
    """Block-bootstrap Monte Carlo of the portfolio's daily log returns.

    Resampling 10-day blocks (instead of single days) keeps fat tails and short-term autocorrelation.
    Seeded for reproducibility. Values are growth of 100 over ``horizon`` trading days.
    Returns ``{"percentiles": {"p5","p25","p50","p75","p95": [horizon values]}, "prob_loss",
    "expected_shortfall" (mean terminal return of the worst 5% paths), "horizon", ...}``.
    """
    r = portfolio_returns(prices, weights).to_numpy()
    if len(r) < 20:
        raise ValueError("need at least 20 daily returns for a Monte Carlo simulation")
    n, horizon = max(1, int(n)), max(1, int(horizon))
    block = max(1, min(int(block), len(r)))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(horizon / block)
    starts = rng.integers(0, len(r) - block + 1, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :horizon]
    cum = np.cumsum(r[idx], axis=1)                       # (n, horizon) cumulative log return
    paths = 100.0 * np.exp(cum)
    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    terminal = np.expm1(cum[:, -1])
    k = max(1, math.ceil(0.05 * n))
    return {
        "percentiles": {f"p{q}": pcts[i].tolist() for i, q in enumerate((5, 25, 50, 75, 95))},
        "prob_loss": float(np.mean(terminal < 0)),
        "expected_shortfall": float(np.sort(terminal)[:k].mean()),
        "horizon": horizon, "n_paths": n, "block": block, "seed": seed, "start_value": 100.0,
    }
