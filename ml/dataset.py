"""Synthetic portfolio dataset with leakage-safe labels.

* Rolling windows over the universe's history: **lookback 252 returns, forward 126 trading days**, window
  start stepped every 21 days. Forward stats are stored for 63 and 126 days.
* Each window: random portfolios of 3-10 assets (stocks + equity / bond / gold / cash ETFs so that all three
  risk classes exist), Dirichlet weights with alpha in {0.3, 1, 3} (concentrated -> diversified).
* **Features** use only the lookback window (``ml.features.build_features``, shared with inference).
* **Labels** use only the forward window of the *same* portfolio. Two definitions are supported:
  - ``composite``: ``0.6*z(fwd_vol) + 0.4*z(|fwd_mdd|)`` cut at terciles fitted on the training period only;
  - ``abs_vol``: forward realised volatility cut at fixed, human-meaningful thresholds (default 15% / 25%).
  Either way the labeler is frozen into ``feature_meta.json``.
* Per row we keep ``window_start``, ``window_end``, ``label_start`` and ``label_end`` so leakage can be
  audited (see ``tests/test_ml.py``).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from engine import data
from engine.universe import info, load_universe
from .features import ALL_FEATURES, LOOKBACK_ROWS, MARKET_PROXIES, build_features, market_log_returns

log = logging.getLogger("ml.dataset")

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "synthetic_portfolios.parquet"

HORIZON = 126            # forward trading days used for the main label
SHORT_HORIZON = 63       # also recorded (closer to the features, usually more predictable)
STEP = 21                # window start step
MIN_ASSETS, MAX_ASSETS = 3, 10
ALPHAS = (0.3, 1.0, 3.0)
HISTORY_START = "2015-01-01"
LABELS = ("Low", "Medium", "High")
SCORE_WEIGHTS = (0.6, 0.4)
# Annualised forward volatility: Low < 12% <= Medium < 17% <= High. Rounded terciles of the earliest 40% of
# history (so no test data influenced them) and then FIXED, which makes the labels interpretable.
# 15%/25% was tried and rejected: only 9% of portfolios (4% in the latest test period) were "High".
ABS_THRESHOLDS = (0.12, 0.17)


def universe_tickers(types=("stock", "etf")) -> list[str]:
    return [e["ticker"] for e in load_universe() if e.get("type") in types]


BREAK_THRESHOLD = 0.4    # |log return| of one day above this is treated as a corporate action


def repair_breaks(prices: pd.DataFrame, threshold: float = BREAK_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    """Back-adjust unadjusted corporate actions so they are not learned as crashes.

    A single-day move beyond ``threshold`` in a large-cap / ETF universe is a split or demerger the data
    provider did not adjust (TMPV.NS, 2025-10-14). History before the break is rescaled so that day's return
    becomes zero. Only used for ML training data; the engine itself only reports such breaks.
    """
    out, fixed = prices.copy(), {}
    for c in out.columns:
        lr = np.log(out[c]).diff()
        for d in lr.index[lr.abs() > threshold]:
            pos = out.index.get_loc(d)
            out.iloc[:pos, out.columns.get_loc(c)] *= out[c].iloc[pos] / out[c].iloc[pos - 1]
            fixed.setdefault(c, []).append(str(d.date()))
    return out, fixed


def _forward_stats(fwd_prices: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Realised annualised vol and max drawdown (positive number) of a constant-weight portfolio."""
    rp = np.diff(np.log(fwd_prices), axis=0) @ w
    vol = float(rp.std(ddof=1) * np.sqrt(252))
    value = np.exp(np.concatenate(([0.0], np.cumsum(rp))))
    mdd = float(1.0 - (value / np.maximum.accumulate(value)).min())
    return vol, mdd


def build_dataset(target_rows: int = 8000, seed: int = 42, end: str | None = None, save: bool = True,
                  prices: pd.DataFrame | None = None, market: pd.Series | None = None,
                  path: Path | None = None) -> pd.DataFrame:
    """Generate the (unlabelled) dataset: all features + forward stats + window dates.

    Uses ``engine.data.get_prices`` for the whole universe (falls back to the committed parquet files when
    offline). ``prices`` / ``market`` can be injected for tests (market features are NaN without ``market``).
    """
    if prices is None:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        prices = data.get_prices(universe_tickers(), HISTORY_START, end)
        if prices.attrs.get("dropped"):
            log.warning("dropped from universe: %s", prices.attrs["dropped"])
        idx_px = data.get_prices(list(MARKET_PROXIES), HISTORY_START, end, align="outer", min_rows=1)
        market = market_log_returns(idx_px)
    prices, repaired = repair_breaks(prices.dropna(how="any"))
    if repaired:
        log.warning("treated as corporate actions (history rescaled): %s", repaired)
    tickers = list(prices.columns)
    if len(tickers) < MIN_ASSETS:
        raise ValueError("universe has too few assets with complete history")
    stock = np.array([(info(t) or {}).get("type") == "stock" for t in tickers])
    T, L, H = len(prices), LOOKBACK_ROWS, HORIZON
    starts = list(range(0, T - L - H + 1, STEP))
    if not starts:
        raise ValueError("not enough history for one lookback + forward window")
    per_window = max(1, math.ceil(target_rows / len(starts)))
    rng = np.random.default_rng(seed)
    values = prices.to_numpy()
    index = prices.index
    rows: list[dict] = []
    for wid, s in enumerate(starts):
        t = s + L - 1                                   # last lookback row (the "as of" date)
        window = prices.iloc[s:s + L]
        for _ in range(per_window):
            k = int(rng.integers(MIN_ASSETS, min(MAX_ASSETS, len(tickers)) + 1))
            cols = rng.choice(len(tickers), size=k, replace=False)
            w = rng.dirichlet(np.full(k, rng.choice(ALPHAS)))
            names = [tickers[c] for c in cols]
            feats = build_features(window[names], dict(zip(names, w)), market=market, fetch_market=False)
            fvol, fmdd = _forward_stats(values[t:t + H + 1][:, cols], w)
            fvol63, _ = _forward_stats(values[t:t + SHORT_HORIZON + 1][:, cols], w)
            rows.append({
                **feats, "fwd_vol": fvol, "fwd_vol_63": fvol63, "fwd_mdd": fmdd, "window_id": wid,
                "window_start": index[s], "window_end": index[t],
                "label_start": index[t + 1], "label_end": index[t + H], "label_end_63": index[t + SHORT_HORIZON],
                "equity_share": float(w[stock[cols]].sum()),
                "tickers": "|".join(names), "weights": "|".join(repr(float(x)) for x in w),
            })
    df = pd.DataFrame(rows)
    df.attrs["data_end"] = str(index[-1].date())
    if save:
        out = path or DATASET_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    log.info("dataset: %d rows, %d windows, %d per window", len(df), len(starts), per_window)
    return df


def load_dataset(path: Path | None = None) -> pd.DataFrame | None:
    p = path or DATASET_PATH
    return pd.read_parquet(p) if p.is_file() else None


# --------------------------------------------------------------------------- split + labels
def time_split(df: pd.DataFrame, test_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier windows -> train, latest ``test_frac`` of windows -> test, with a purge gap.

    Training rows whose forward window (``label_end``) reaches into the test period's first lookback
    (``window_start``) are dropped, so ``max(train.label_end) < min(test.window_start)`` always holds.
    """
    ids = np.sort(df["window_id"].unique())
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_ids = set(ids[-n_test:])
    test = df[df["window_id"].isin(test_ids)]
    cutoff = test["window_start"].min()
    train = df[~df["window_id"].isin(test_ids) & (df["label_end"] < cutoff)]
    if train.empty:
        raise ValueError("purged training set is empty; need more history")
    assert train["label_end"].max() < test["window_start"].min()
    return train.copy(), test.copy()


def fit_labeler(train: pd.DataFrame, mode: str = "composite", horizon: int = HORIZON,
                thresholds: tuple[float, float] = ABS_THRESHOLDS) -> dict:
    """Frozen label definition. ``composite`` fits z-scores + terciles on the TRAIN period only;
    ``abs_vol`` is fixed thresholds on forward volatility (nothing is fitted)."""
    col = "fwd_vol" if horizon == HORIZON else "fwd_vol_63"
    if mode == "abs_vol":
        return {"mode": "abs_vol", "horizon": int(horizon), "column": col, "thresholds": [float(x) for x in thresholds],
                "labels": list(LABELS)}
    if mode == "vol_tercile":                    # forward vol only (no noisy drawdown), terciles fitted on train
        lo, hi = np.quantile(train[col], [1 / 3, 2 / 3])
        return {"mode": "abs_vol", "horizon": int(horizon), "column": col, "thresholds": [float(lo), float(hi)],
                "labels": list(LABELS), "fitted_terciles": True}
    mv, sv = float(train["fwd_vol"].mean()), float(train["fwd_vol"].std(ddof=0)) or 1.0
    mm, sm = float(train["fwd_mdd"].mean()), float(train["fwd_mdd"].std(ddof=0)) or 1.0
    score = SCORE_WEIGHTS[0] * (train["fwd_vol"] - mv) / sv + SCORE_WEIGHTS[1] * (train["fwd_mdd"] - mm) / sm
    lo, hi = np.quantile(score, [1 / 3, 2 / 3])
    return {"mode": "composite", "horizon": HORIZON, "vol_mean": mv, "vol_std": sv, "mdd_mean": mm, "mdd_std": sm,
            "score_weights": list(SCORE_WEIGHTS), "t_low": float(lo), "t_high": float(hi), "labels": list(LABELS)}


def risk_score(df: pd.DataFrame, p: dict) -> pd.Series:
    """Continuous realised-risk score the labels are cut from (log forward vol for ``abs_vol``)."""
    if p.get("mode") == "abs_vol":
        return np.log(df[p["column"]].clip(lower=1e-4))
    return (p["score_weights"][0] * (df["fwd_vol"] - p["vol_mean"]) / p["vol_std"]
            + p["score_weights"][1] * (df["fwd_mdd"] - p["mdd_mean"]) / p["mdd_std"])


def apply_labels(df: pd.DataFrame, p: dict) -> np.ndarray:
    """Integer class (0 Low, 1 Medium, 2 High) using the frozen definition."""
    if p.get("mode") == "abs_vol":
        return np.digitize(df[p["column"]].to_numpy(), p["thresholds"])
    return np.digitize(risk_score(df, p).to_numpy(), [p["t_low"], p["t_high"]])
