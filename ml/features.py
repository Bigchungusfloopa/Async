"""Single source of truth for model features: dataset generation and live inference both call
:func:`build_features`, so there is no train/serve skew. The 12 base features are P1's
``engine.metrics.feature_inputs``; the groups below add the short-term / structure / market-regime signals
(volatility clusters, so *recent* volatility predicts future volatility far better than the 1-year number).

``build_features`` returns the whole superset (``ALL_FEATURES``); the deployed model uses ``FEATURE_NAMES``
(the subset recorded in ``feature_meta.json``) so the two can evolve independently.
"""
from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from engine.metrics import feature_inputs, prepare
from engine.universe import sector_of

log = logging.getLogger("ml.features")

LOOKBACK_RETURNS = 252                      # one trading year of daily returns ...
LOOKBACK_ROWS = LOOKBACK_RETURNS + 1        # ... needs one extra price row
EWMA_LAMBDA = 0.94                          # RiskMetrics decay
MARKET_PROXIES = ("^NSEI", "^NDX")          # market regime proxy = mean of the two index log returns

BASE_FEATURES = ["ann_return", "ann_vol", "max_drawdown", "sharpe", "avg_corr", "hhi", "effective_n",
                 "top_weight", "skew", "kurtosis", "n_assets", "downside_dev"]
SHORT_TERM_FEATURES = ["vol_21", "vol_63", "vol_126", "vol_ratio_21", "vol_ratio_63", "ewma_vol", "vol_of_vol",
                       "ret_21", "ret_63", "cur_drawdown"]
STRUCTURE_FEATURES = ["max_asset_vol", "undiversified_vol", "diversification_ratio", "avg_corr_63",
                      "corr_shift", "sector_hhi"]
MARKET_FEATURES = ["market_vol_21", "market_vol_ratio", "market_drawdown", "beta_to_market"]
ALL_FEATURES = BASE_FEATURES + SHORT_TERM_FEATURES + STRUCTURE_FEATURES + MARKET_FEATURES

# The deployed model's columns. Replaced by the chosen subset after the experiment in scripts/.
FEATURE_NAMES = list(ALL_FEATURES)


def lookback_slice(prices: pd.DataFrame) -> pd.DataFrame:
    """The trailing window the model was trained on (the last 253 price rows)."""
    return prices.iloc[-LOOKBACK_ROWS:]


# --------------------------------------------------------------------------- market proxy
def market_log_returns(prices_by_symbol: pd.DataFrame) -> pd.Series:
    """Mean daily log return of the available index columns (ffilled across holidays)."""
    cols = [c for c in MARKET_PROXIES if c in prices_by_symbol.columns]
    if not cols:
        return pd.Series(dtype="float64")
    px = prices_by_symbol[cols].ffill()
    return np.log(px).diff().mean(axis=1).dropna().rename("market")


@functools.lru_cache(maxsize=64)
def _fetch_market(start: str, end: str) -> pd.Series:
    from engine import data

    px = data.get_prices(list(MARKET_PROXIES), start, end, align="outer", min_rows=1)
    return market_log_returns(px)


def _auto_market(index: pd.DatetimeIndex) -> pd.Series | None:
    try:
        m = _fetch_market(index[0].strftime("%Y-%m-%d"), index[-1].strftime("%Y-%m-%d"))
        return m if len(m) else None
    except Exception as exc:  # noqa: BLE001 - features must degrade, never raise, when the market is unavailable
        log.info("market proxy unavailable: %s", type(exc).__name__)
        return None


# --------------------------------------------------------------------------- feature maths
def _avg_offdiag_corr(x: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 5:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x, rowvar=False)
    n = c.shape[0]
    c = np.nan_to_num(c, nan=0.0)
    return float((c.sum() - np.trace(c)) / (n * (n - 1)))


def _drawdown_now(log_ret: np.ndarray) -> float:
    v = np.exp(np.cumsum(log_ret))
    return float(v[-1] / v.max() - 1.0) if len(v) else 0.0


def _extras(win: pd.DataFrame, weights: dict, market: pd.Series | None) -> dict:
    p = prepare(win, weights)
    R, w = p.logret, p.w                                       # (T, n) asset log returns, normalised weights
    rp = R @ w
    s = np.sqrt(252.0)
    vol252 = float(rp.std(ddof=1) * s) if len(rp) > 1 else 0.0

    def vol(k: int) -> float:
        x = rp[-k:]
        return float(x.std(ddof=1) * s) if len(x) > 1 else 0.0

    out: dict[str, float] = {"vol_21": vol(21), "vol_63": vol(63), "vol_126": vol(126)}
    out["vol_ratio_21"] = out["vol_21"] / vol252 if vol252 > 1e-9 else 1.0
    out["vol_ratio_63"] = out["vol_63"] / vol252 if vol252 > 1e-9 else 1.0

    decay = EWMA_LAMBDA ** np.arange(len(rp))[::-1]            # most recent observation weighs most
    out["ewma_vol"] = float(np.sqrt(np.sum(decay * rp ** 2) / decay.sum()) * s) if len(rp) else 0.0
    if len(rp) >= 42:
        roll = np.lib.stride_tricks.sliding_window_view(rp, 21).std(axis=1, ddof=1)
        out["vol_of_vol"] = float(roll.std(ddof=1) / roll.mean()) if roll.mean() > 1e-12 else 0.0
    else:
        out["vol_of_vol"] = 0.0
    out["ret_21"], out["ret_63"] = float(rp[-21:].sum()), float(rp[-63:].sum())
    out["cur_drawdown"] = _drawdown_now(rp)

    asset_vol = R.std(axis=0, ddof=1) * s if len(R) > 1 else np.zeros(R.shape[1])
    out["max_asset_vol"] = float(asset_vol.max())
    out["undiversified_vol"] = float(w @ asset_vol)
    out["diversification_ratio"] = (out["undiversified_vol"] / vol252) if vol252 > 1e-9 else 1.0
    c63, c252 = _avg_offdiag_corr(R[-63:]), _avg_offdiag_corr(R)
    out["avg_corr_63"], out["corr_shift"] = c63, c63 - c252
    sectors: dict[str, float] = {}
    for t, wt in zip(p.tickers, w):
        sectors[sector_of(t) or "Other"] = sectors.get(sector_of(t) or "Other", 0.0) + float(wt)
    out["sector_hhi"] = float(sum(v * v for v in sectors.values()))

    nan = float("nan")
    out.update(market_vol_21=nan, market_vol_ratio=nan, market_drawdown=nan, beta_to_market=nan)
    if market is not None and len(market):
        m = market.reindex(p.prices.index[1:]).fillna(0.0).to_numpy()
        if len(m) == len(rp) and m.std(ddof=1) > 1e-12:
            out["market_vol_21"] = float(m[-21:].std(ddof=1) * s)
            out["market_vol_ratio"] = float(m[-21:].std(ddof=1) / m.std(ddof=1))
            out["market_drawdown"] = _drawdown_now(m)
            out["beta_to_market"] = float(np.cov(rp, m, ddof=1)[0, 1] / m.var(ddof=1))
    return out


def build_features(prices: pd.DataFrame, weights: dict, market: pd.Series | None = None,
                   fetch_market: bool = True) -> dict:
    """Feature dict (keys = ``ALL_FEATURES``) for a portfolio, using only the trailing lookback window.

    ``market`` = daily log returns of the market proxy (see :func:`market_log_returns`); when ``None`` and
    ``fetch_market`` it is loaded (cached) for the window dates. Market features are ``NaN`` if unavailable
    (the classifier imputes training medians). ``prices`` may be longer than the window: only its tail is used.
    """
    win = lookback_slice(prices)
    raw = feature_inputs(win, weights)
    out = {k: float(raw[k]) for k in BASE_FEATURES}
    if market is None and fetch_market:
        market = _auto_market(win.index)
    out.update(_extras(win, weights, market))
    for k, v in out.items():
        if not np.isfinite(v) and k not in MARKET_FEATURES:
            out[k] = 0.0
    return {k: out[k] for k in ALL_FEATURES}


def to_vector(features: dict, names: list[str] | None = None, fill: dict | None = None) -> np.ndarray:
    """Feature dict -> 1-D array in the model's column order; NaNs are replaced by ``fill[name]`` if given."""
    names = names or FEATURE_NAMES
    v = np.array([features[n] for n in names], dtype="float64")
    if fill is not None:
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = [fill[n] for n, b in zip(names, bad) if b]
    return v
