"""Portfolio math: vectorised numpy, log returns, Ledoit-Wolf covariance, 252 trading days.

Conventions (shared with the optimizer, simulator and P2's feature pipeline):

* Daily returns are **log returns** of adjusted close.
* Portfolio daily return = ``returns @ w`` (constant weights, i.e. rebalanced daily). It is treated as a
  log return, so the value series is ``exp(cumsum(r_p))`` and ``ann_return = exp(mean(r_p)*252) - 1``
  is exactly the CAGR of that series.
* Volatility uses the Ledoit-Wolf shrunk covariance (annualised): ``sqrt(w' Σ w)``.
* Undefined ratios (zero variance / zero drawdown) are reported as ``0.0`` so the UI never sees NaN.
* Everything is pure (no Streamlit, no I/O).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .universe import currency_of

TRADING_DAYS = 252
RF_DEFAULTS = {"INR": 0.065, "USD": 0.04}

METRIC_KEYS = ("ann_return", "ann_vol", "max_drawdown", "sharpe", "sortino", "calmar",
               "cvar_95", "avg_corr", "hhi", "effective_n")


# --------------------------------------------------------------------------- input preparation
@dataclass(frozen=True)
class Prepared:
    tickers: list[str]
    w: np.ndarray            # normalised weights, aligned with tickers
    prices: pd.DataFrame     # clean (no NaN, positive) prices for tickers
    logret: np.ndarray       # (T-1, n) daily log returns


def default_rf(tickers, weights=None) -> float:
    """Risk-free rate: 6.5% for INR assets, 4% for USD, weight-averaged for mixed portfolios."""
    tickers = list(tickers)
    if not tickers:
        return RF_DEFAULTS["USD"]
    w = np.ones(len(tickers)) / len(tickers) if weights is None else np.abs(np.asarray(weights, float))
    if w.sum() <= 0:
        w = np.ones(len(tickers))
    w = w / w.sum()
    rf = np.array([RF_DEFAULTS.get(currency_of(t), RF_DEFAULTS["USD"]) for t in tickers])
    return float(rf @ w)


def normalize_weights(weights: dict, allow_negative: bool = False) -> dict:
    """Clean a weights dict and scale it to sum to 1. Raises ValueError on negatives / all-zero."""
    clean = {str(k): float(v) for k, v in (weights or {}).items() if v is not None and np.isfinite(v)}
    if not clean:
        raise ValueError("weights are empty")
    if not allow_negative and any(v < 0 for v in clean.values()):
        raise ValueError("negative weights are not allowed (long-only)")
    total = sum(clean.values())
    if abs(total) < 1e-12:
        raise ValueError("weights sum to zero")
    return {k: v / total for k, v in clean.items()}


def match_columns(weights: dict, columns) -> dict:
    """Map weight keys onto actual column labels case-insensitively (``"aapl"`` -> ``"AAPL"``); keys
    that collapse onto the same column are summed. Unmatched keys are kept as-is (and later ignored)."""
    by_upper = {str(c).strip().upper(): c for c in columns}
    out: dict = {}
    for k, v in weights.items():
        col = k if k in columns else by_upper.get(str(k).strip().upper(), k)
        out[col] = out.get(col, 0.0) + v
    return out


def prepare(prices, weights: dict, allow_negative: bool = False) -> Prepared:
    """Align weights with price columns, drop NaN/non-positive rows and compute log returns."""
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(prices.name or "asset")
    w_all = match_columns(normalize_weights(weights, allow_negative), prices.columns)
    cols = [c for c in prices.columns if c in w_all and w_all[c] != 0.0]
    if not cols:
        raise ValueError("none of the weighted tickers are present in the price data")
    px = prices[cols].astype(float)
    px = px[(px > 0).all(axis=1)].dropna(how="any")
    if len(px) < 3:
        raise ValueError("not enough overlapping price history (need at least 3 rows)")
    w = np.array([w_all[c] for c in cols])
    w = w / w.sum()  # renormalise over the assets that actually have data
    logret = np.diff(np.log(px.to_numpy()), axis=0)
    return Prepared(cols, w, px, logret)


# --------------------------------------------------------------------------- core calculations
def lw_covariance(logret: np.ndarray, annualize: bool = True) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of daily log returns (annualised by default)."""
    logret = np.asarray(logret, float)
    if logret.ndim == 1:
        logret = logret[:, None]
    n_obs, n_assets = logret.shape
    if n_obs < 2:
        cov = np.zeros((n_assets, n_assets))
    elif n_assets == 1:
        cov = np.array([[np.var(logret[:, 0], ddof=1)]])  # nothing to shrink towards
    else:
        cov = LedoitWolf().fit(logret).covariance_
    return cov * TRADING_DAYS if annualize else cov


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _avg_corr(cov: np.ndarray) -> float:
    n = cov.shape[0]
    if n < 2:
        return 0.0
    corr = _corr_from_cov(cov)
    return float((corr.sum() - n) / (n * (n - 1)))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) and abs(b) > 1e-12 and np.isfinite(a) else 0.0


def _max_drawdown(value: np.ndarray) -> float:
    return float((value / np.maximum.accumulate(value) - 1.0).min())


def _compute(rp: np.ndarray, w: np.ndarray, cov: np.ndarray, rf: float) -> dict:
    """All ten headline metrics from portfolio daily log returns ``rp`` and annualised ``cov``."""
    ann_return = float(np.expm1(rp.mean() * TRADING_DAYS))
    ann_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    value = np.exp(np.concatenate(([0.0], np.cumsum(rp))))
    mdd = _max_drawdown(value)

    rf_daily = np.log1p(rf) / TRADING_DAYS
    downside = np.minimum(rp - rf_daily, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS))

    k = max(1, int(np.ceil(0.05 * len(rp))))
    cvar = float(np.sort(rp)[:k].mean())

    hhi = float(np.sum(w ** 2))
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "sharpe": _safe_div(ann_return - rf, ann_vol),
        "sortino": _safe_div(ann_return - rf, downside_dev),
        "calmar": _safe_div(ann_return, abs(mdd)),
        "cvar_95": cvar,
        "avg_corr": _avg_corr(cov),
        "hhi": hhi,
        "effective_n": _safe_div(1.0, hhi),
    }


# --------------------------------------------------------------------------- public API
def portfolio_metrics(prices: pd.DataFrame, weights: dict, rf: float | None = None,
                      allow_negative: bool = False) -> dict:
    """Headline metrics: ann_return, ann_vol, max_drawdown, sharpe, sortino, calmar, cvar_95,
    avg_corr, hhi, effective_n. ``rf`` defaults to 6.5% (INR) / 4% (USD) by the tickers' currency."""
    p = prepare(prices, weights, allow_negative)
    rf = default_rf(p.tickers, p.w) if rf is None else float(rf)
    return _compute(p.logret @ p.w, p.w, lw_covariance(p.logret), rf)


def portfolio_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """Daily portfolio log-return series (``returns @ w``) indexed by date."""
    p = prepare(prices, weights)
    return pd.Series(p.logret @ p.w, index=p.prices.index[1:], name="portfolio")


def portfolio_value(prices: pd.DataFrame, weights: dict, base: float = 100.0) -> pd.Series:
    """Growth of ``base`` for the constant-weight portfolio; first date equals ``base``."""
    p = prepare(prices, weights)
    rp = p.logret @ p.w
    value = base * np.exp(np.concatenate(([0.0], np.cumsum(rp))))
    return pd.Series(value, index=p.prices.index, name="portfolio")


def risk_contribution(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """Fraction of portfolio variance each asset is responsible for: ``w_i (Σw)_i / σ_p²`` (sums to 1)."""
    p = prepare(prices, weights)
    cov = lw_covariance(p.logret)
    marginal = cov @ p.w
    var = float(p.w @ marginal)
    rc = p.w * marginal / var if var > 1e-18 else p.w.copy()
    return pd.Series(rc, index=p.tickers, name="risk_contribution")


def return_contribution(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """Share of portfolio return from each asset: ``w_i * mean_i / Σ_j w_j * mean_j`` (sums to 1)."""
    p = prepare(prices, weights)
    parts = p.w * p.logret.mean(axis=0)
    total = parts.sum()
    rc = parts / total if abs(total) > 1e-14 else np.zeros_like(parts)
    return pd.Series(rc, index=p.tickers, name="return_contribution")


def feature_inputs(prices: pd.DataFrame, weights: dict, rf: float | None = None) -> dict:
    """Raw ingredients for P2's ``ml/features.py`` so training and inference use identical code.

    Keys: ann_return, ann_vol, max_drawdown, avg_corr, hhi, effective_n, top_weight, skew, kurtosis
    (``skew``/``kurtosis`` = unbiased skewness and *excess* kurtosis of daily portfolio log returns).
    """
    p = prepare(prices, weights)
    rp = p.logret @ p.w
    m = _compute(rp, p.w, lw_covariance(p.logret), default_rf(p.tickers, p.w) if rf is None else rf)
    s = pd.Series(rp)
    skew = s.skew() if len(s) >= 3 else 0.0
    kurt = s.kurt() if len(s) >= 4 else 0.0
    return {
        "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "max_drawdown": m["max_drawdown"],
        "avg_corr": m["avg_corr"], "hhi": m["hhi"], "effective_n": m["effective_n"],
        "top_weight": float(p.w.max()),
        "skew": float(skew) if np.isfinite(skew) else 0.0,
        "kurtosis": float(kurt) if np.isfinite(kurt) else 0.0,
    }
