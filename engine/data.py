"""Market data: one batched yfinance download, tiered caching, parquet fallback, date alignment.

Tiers, fastest first (per ticker):  local parquet cache  ->  Supabase ``price_cache``  ->  yfinance
(retried 3x, 1s/2s/4s)  ->  ``data/fallback/<TICKER>.parquet``.

API notes
* ``end`` is **inclusive** here (yfinance's own ``end`` is exclusive; we add a day internally).
* ``df.attrs`` carries metadata for the UI: ``source`` ("live" | "cache" | "fallback"), ``dropped``,
  ``dropped_reasons``, ``currencies``, ``mixed_currency``, ``warnings``, ``as_of``, ``bad_ticks`` (glitch
  prices removed) and ``suspect_breaks`` (huge one-day moves left in place, likely unadjusted splits).
  pandas keeps attrs through most slicing, but read them right after the call.
* Pure python: no Streamlit. Caching decorators belong to the UI layer.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .universe import UNIVERSE_PATH, currency_of, info, load_universe  # noqa: F401  (re-exported)

log = logging.getLogger("engine.data")

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_DIR = ROOT / "data" / "fallback"
CACHE_DIR = ROOT / "data" / "cache" / "prices"

MIN_ROWS = 252            # a ticker needs >= 1 year of rows in the window (scaled down for short windows)
FFILL_LIMIT = 3
RETRY_BACKOFF = (1, 2, 4)
RECHECK_TTL_S = 30 * 60   # do not re-ask the network for the same end date within this window
FAILURE_TTL_S = 60        # ...but after a network failure retry sooner, so recovery is quick
OVERLAP_DAYS = 7          # re-fetch a week of overlap to detect split/dividend re-adjustments
ADJ_TOLERANCE = 1e-3
SYNC_TO_SUPABASE = True   # tests switch this off

BENCHMARKS = {"NIFTY 50": "^NSEI", "NIFTY": "^NSEI", "^NSEI": "^NSEI", "NSEI": "^NSEI",
              "NASDAQ-100": "^NDX", "NASDAQ 100": "^NDX", "NASDAQ100": "^NDX", "^NDX": "^NDX", "NDX": "^NDX"}

_sleep = time.sleep                      # patched in tests
_tail_tried: dict[str, tuple[float, pd.Timestamp, float]] = {}   # ticker -> (when, end asked for, ttl)
_head_tried: dict[str, pd.Timestamp] = {}                 # ticker -> earliest start we asked for
_state_lock = threading.Lock()


# --------------------------------------------------------------------------- helpers
def normalize_tickers(tickers) -> list[str]:
    """Strip, upper-case and de-duplicate while preserving order (``.NS`` / ``.BO`` suffixes kept)."""
    if isinstance(tickers, str):
        tickers = [tickers]
    seen, out = set(), []
    for t in tickers or []:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def _read_parquet(path: Path) -> pd.Series | None:
    try:
        if not path.is_file():
            return None
        s = pd.read_parquet(path)["adj_close"].dropna().astype("float64")
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as exc:  # noqa: BLE001 - a corrupt cache file must not break the app
        log.warning("unreadable cache file %s: %s", path.name, type(exc).__name__)
        return None


def _write_cache(ticker: str, s: pd.Series) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _cache_path(ticker).with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")   # unique: sessions may race
        s.rename("adj_close").rename_axis("date").to_frame().to_parquet(tmp)
        os.replace(tmp, _cache_path(ticker))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not write cache for %s: %s", ticker, type(exc).__name__)


def _tail_fresh(ticker: str, end: pd.Timestamp) -> bool:
    """True if we already asked the network for data up to ``end`` within the last RECHECK_TTL_S."""
    with _state_lock:
        rec = _tail_tried.get(ticker)
    return bool(rec and rec[1] >= end and time.time() - rec[0] < rec[2])


def _head_tried_for(ticker: str, start: pd.Timestamp) -> bool:
    """True if a fetch reaching back to ``start`` was already attempted (IPO later than start is final)."""
    with _state_lock:
        t = _head_tried.get(ticker)
    return t is not None and t <= start


def _record_attempt(ticker: str, fetch_from: pd.Timestamp, fetch_to: pd.Timestamp, head: bool,
                    ttl: float = RECHECK_TTL_S) -> None:
    with _state_lock:
        _tail_tried[ticker] = (time.time(), fetch_to, ttl)
        if head:
            prev = _head_tried.get(ticker)
            _head_tried[ticker] = fetch_from if prev is None else min(prev, fetch_from)


def clean_bad_ticks(s: pd.Series, window: int = 11, ratio: float = 2.0) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Blank out transient bad ticks: points more than ``ratio``x away from the centred rolling median.

    Yahoo occasionally serves prices off by 10x/100x for a day or two (e.g. GOLDBEES.NS, NIFTYBEES.NS on
    2019-12-19). A genuine crash persists, so it moves the rolling median with it and is *not* flagged.
    """
    if len(s) < window:
        return s, []
    med = s.rolling(window, center=True, min_periods=window // 2).median()
    r = s / med
    bad = (r > ratio) | (r < 1.0 / ratio)
    if not bad.any():
        return s, []
    return s.mask(bad), list(s.index[bad.to_numpy()])


def suspect_breaks(s: pd.Series, threshold: float = 0.4) -> list[tuple[pd.Timestamp, float]]:
    """Single-day |log return| above ``threshold`` that did NOT revert: usually a split / demerger that the
    data provider failed to back-adjust (e.g. TMPV.NS on 2025-10-14). Reported, never silently altered."""
    lr = np.log(s.dropna()).diff().dropna()
    return [(d, float(v)) for d, v in lr[lr.abs() > threshold].items()]


def _read_fallback(ticker: str) -> pd.Series | None:
    return _read_parquet(FALLBACK_DIR / f"{ticker}.parquet")


# --------------------------------------------------------------------------- network
def _download(tickers: list[str], start: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.DataFrame:
    """ONE batched ``yf.download`` -> wide adjusted-close frame (empty on no data). May raise."""
    import yfinance as yf

    raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                      end=(end_inclusive + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) or "Close" in raw.columns else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.copy()
    close.columns = [str(c).upper() for c in close.columns]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.astype("float64").round(4).dropna(how="all")   # round away float32 noise


def _download_with_retries(tickers, start, end_inclusive) -> pd.DataFrame:
    """3 attempts, backing off 1s/2s between them; both exceptions and empty results are retried."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate(RETRY_BACKOFF):
        try:
            df = _download(tickers, start, end_inclusive)
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted errors
            last_exc = exc
            log.warning("yfinance attempt %d failed: %s", attempt + 1, type(exc).__name__)
        if attempt < len(RETRY_BACKOFF) - 1:
            _sleep(delay)
    if last_exc is not None:
        log.warning("yfinance gave up: %s", type(last_exc).__name__)
    return pd.DataFrame()


# --------------------------------------------------------------------------- public API
def get_prices(tickers: list[str], start: str, end: str, align: str = "inner",
               min_rows: int | None = None) -> pd.DataFrame:
    """Adjusted-close prices for ``[start, end]`` (both inclusive): columns=tickers, index=date.

    ``align="inner"`` (default) forward-fills gaps of up to 3 days then keeps only common dates;
    ``align="outer"`` keeps every date and leaves NaN where a ticker has no data (used for regime
    windows where some assets did not exist yet).

    Tickers with fewer than ``min_rows`` (default 252, scaled down for windows shorter than a year)
    non-NaN rows in the window are dropped and listed in ``df.attrs["dropped"]``.
    """
    tickers = normalize_tickers(tickers)
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    today = pd.Timestamp.today().normalize()
    end_eff = min(end_ts, today)
    warnings: list[str] = []
    series: dict[str, pd.Series] = {}
    live_used = fallback_used = False

    if not tickers or end_ts < start_ts:
        out = pd.DataFrame()
        out.attrs.update(source="cache", dropped=[], dropped_reasons={}, currencies={}, mixed_currency=False,
                         warnings=["no tickers requested" if not tickers else "end date is before start date"],
                         as_of=None)
        return out

    # 1) local parquet cache, then Supabase for tickers we have never seen on this machine
    for t in tickers:
        s = _read_parquet(_cache_path(t))
        if s is not None:
            series[t] = s
    cold = [t for t in tickers if t not in series]
    if cold:
        remote = db.read_price_cache(cold, start_ts, end_eff)
        if remote is not None and not remote.empty:
            for t in remote.columns:
                s = remote[t].dropna()
                if len(s):
                    series[t] = s
                    _write_cache(t, s)

    # 2) work out what is missing: a tail (newer than the cache) and/or a head (older than the cache)
    need: dict[str, tuple[pd.Timestamp, pd.Timestamp, bool]] = {}   # ticker -> (from, to, includes_head)
    for t in tickers:
        s = series.get(t)
        if s is None or s.empty:
            if not _tail_fresh(t, end_eff):
                need[t] = (start_ts, end_eff, True)
            continue
        lo, hi = s.index[0], s.index[-1]
        fetch_from, fetch_to, head = None, end_eff, False
        # head: cache starts well after the requested start (a week of slack for holidays)
        if lo - start_ts > pd.Timedelta(days=7) and not _head_tried_for(t, start_ts):
            fetch_from, head = start_ts, True
        # tail: any business day after the last cached date up to the requested end
        if hi < end_eff and np.busday_count((hi + pd.Timedelta(days=1)).date(),
                                            (end_eff + pd.Timedelta(days=1)).date()) > 0 \
                and not _tail_fresh(t, end_eff):
            fetch_from = fetch_from if head else hi - pd.Timedelta(days=OVERLAP_DAYS)
        if fetch_from is not None:
            need[t] = (fetch_from, fetch_to, head)

    # 3) one batched download for everything that needs it
    failed: list[str] = []
    new_rows: dict[str, pd.Series] = {}
    if need:
        win_start = min(a for a, _, _ in need.values())
        win_end = max(b for _, b, _ in need.values())
        fetched = _download_with_retries(list(need), win_start, win_end)
        batch_ok = not fetched.empty        # empty batch = network trouble or all-invalid: retry soon
        for t, (f_from, f_to, head) in need.items():
            _record_attempt(t, f_from, f_to, head, RECHECK_TTL_S if batch_ok else FAILURE_TTL_S)
            s_new = fetched[t].dropna() if t in fetched.columns else pd.Series(dtype="float64")
            if s_new.empty:
                failed.append(t)
                continue
            live_used = True
            old = series.get(t)
            if old is not None and not old.empty:
                overlap = old.index.intersection(s_new.index)
                if len(overlap) >= 3:
                    drift = np.abs(s_new[overlap].to_numpy() / old[overlap].to_numpy() - 1).max()
                    if drift > ADJ_TOLERANCE:       # split / dividend re-adjusted history -> refetch it all
                        full = _download_with_retries([t], min(start_ts, old.index[0]), end_eff)
                        if t in full.columns and not full[t].dropna().empty:
                            s_new, old = full[t].dropna(), None
                if old is not None:
                    s_new = pd.concat([old, s_new])          # fresh rows win on the overlap
                    s_new = s_new[~s_new.index.duplicated(keep="last")].sort_index()
            series[t] = s_new
            new_rows[t] = s_new
            _write_cache(t, s_new)
        if new_rows and SYNC_TO_SUPABASE and db.is_configured():
            frame = pd.DataFrame(new_rows)
            threading.Thread(target=db.upsert_prices, args=(frame,), daemon=True).start()

    # 4) failures: keep stale cache, and top it up from the parquet fallback
    for t in failed:
        fb = _read_fallback(t)
        cur = series.get(t)
        if fb is not None and not fb.empty:
            merged = fb if cur is None else cur.combine_first(fb)
            if cur is None or len(merged) > len(cur):
                fallback_used = True
            series[t] = merged
        elif cur is not None:
            warnings.append(f"{t}: live refresh failed, using cached data up to {cur.index[-1].date()}")

    # 5) assemble the window
    dropped: list[str] = []
    reasons: dict[str, str] = {}
    cols: dict[str, pd.Series] = {}
    bad_ticks: dict[str, list[str]] = {}
    breaks: dict[str, list] = {}
    win_days = max(1, int(np.busday_count(start_ts.date(), (end_eff + pd.Timedelta(days=1)).date())))
    required = int(min_rows if min_rows is not None else max(20, min(MIN_ROWS, 0.8 * win_days)))
    for t in tickers:
        s = series.get(t)
        if s is None or s.empty:
            dropped.append(t)
            reasons[t] = "no data found (invalid ticker or offline with no cache)"
            continue
        s, bad_dates = clean_bad_ticks(s)
        if bad_dates:
            bad_ticks[t] = [d.date().isoformat() for d in bad_dates]
            warnings.append(f"{t}: ignored {len(bad_dates)} bad price tick(s) around {bad_dates[0].date()} "
                            f"(provider glitch)")
        s = s.loc[start_ts:end_ts]
        brk = suspect_breaks(s)
        if brk:
            breaks[t] = [(d.date().isoformat(), round(v, 4)) for d, v in brk]
            warnings.append(f"{t}: price moved {np.expm1(brk[0][1]):+.0%} in one day on {brk[0][0].date()} - "
                            f"possibly an unadjusted split/demerger, not a real loss")
        if len(s) < required:
            dropped.append(t)
            reasons[t] = f"only {len(s)} rows in the window (< {required} required)"
            continue
        cols[t] = s

    df = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    df.index.name = "date"                      # same index name whether served live or from cache
    if align == "inner" and not df.empty:
        df = df.ffill(limit=FFILL_LIMIT).dropna(how="any")
        if not df.empty and (df.index[0] - start_ts).days > 30:
            warnings.append(f"history truncated to start at {df.index[0].date()}: the shortest-history "
                            f"asset limits the common window")
    # align="outer": leave NaN where a ticker has no data so last-valid dates stay truthful
    if df.empty and cols:
        warnings.append("no common dates across the selected tickers")

    currencies = {t: currency_of(t) for t in df.columns}
    mixed = len(set(currencies.values())) > 1
    if mixed:
        warnings.append("mixed currencies (" + ", ".join(sorted(set(currencies.values()))) +
                        "): compare in % terms, do not add raw prices")
    for t, why in reasons.items():
        warnings.append(f"{t} dropped: {why}")

    df.attrs.update(
        source="fallback" if fallback_used else ("live" if live_used else "cache"),
        dropped=dropped, dropped_reasons=reasons, currencies=currencies, mixed_currency=mixed,
        warnings=warnings, as_of=(df.index[-1].date().isoformat() if not df.empty else None),
        bad_ticks=bad_ticks, suspect_breaks=breaks,
    )
    return df


def validate_tickers(tickers: list[str]) -> dict:
    """Fast validity check with a single 5-day batched download -> ``{"valid": [...], "invalid": [...]}``.

    Tickers we already know (universe / local cache / fallback file) are valid even when offline."""
    tickers = normalize_tickers(tickers)
    known = {t for t in tickers if info(t) or _cache_path(t).is_file() or (FALLBACK_DIR / f"{t}.parquet").is_file()}
    live_ok: set[str] = set()
    if tickers:
        try:
            import yfinance as yf

            raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False, group_by="column")
            if raw is not None and not raw.empty:
                close = raw["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame(tickers[0])
                live_ok = {str(c).upper() for c in close.columns if close[c].notna().any()}
        except Exception as exc:  # noqa: BLE001
            log.warning("validate_tickers network check failed: %s", type(exc).__name__)
    valid = [t for t in tickers if t in live_ok or t in known]
    return {"valid": valid, "invalid": [t for t in tickers if t not in valid]}


def get_benchmark(name: str, start: str, end: str) -> pd.Series:
    """Adjusted-close Series for a benchmark (``^NSEI`` / ``^NDX`` or "NIFTY 50" / "NASDAQ-100")."""
    sym = BENCHMARKS.get(str(name).strip().upper(), str(name).strip().upper())
    df = get_prices([sym], start, end, align="outer", min_rows=1)
    if sym not in df.columns:
        s = pd.Series(dtype="float64", name=sym)
        s.attrs.update(df.attrs)
        return s
    s = df[sym].dropna().rename(sym)
    s.attrs.update(df.attrs)
    return s
