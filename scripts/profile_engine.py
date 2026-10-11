"""Real-data hardening + latency check (needs network once to warm the cache; fallback test is offline).

    python scripts/profile_engine.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import data, metrics, optimizer, simulate  # noqa: E402

START, END = "2021-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
IN = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "SUNPHARMA.NS"]
US = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
TEN = IN + US
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def equal(cols):
    return {c: 1 / len(cols) for c in cols}


# ---- portfolios of every flavour -------------------------------------------------------------
for label, tickers in [("Indian-only", IN), ("US-only", US), ("Mixed currencies", ["TCS.NS", "INFY.NS", "AAPL", "MSFT"]),
                       ("10 assets", TEN)]:
    px = data.get_prices(tickers, START, END)
    w = equal(list(px.columns))
    m = metrics.portfolio_metrics(px, w)
    r = optimizer.optimize(px, "Medium", current_weights=w)
    ok = (len(px.columns) == len(tickers) and all(np.isfinite(v) for v in m.values())
          and abs(sum(r["weights"].values()) - 1) < 1e-9)
    check(label, ok, f"{px.shape} source={px.attrs['source']} mixed={px.attrs['mixed_currency']} sharpe={m['sharpe']:.2f}")

px1 = data.get_prices(["TCS.NS"], START, END)
m1 = metrics.portfolio_metrics(px1, {"TCS.NS": 1.0})
r1 = optimizer.optimize(px1, "Low")
check("Single asset", m1["avg_corr"] == 0 and r1["weights"] == {"TCS.NS": 1.0}, f"mode={r1['mode']}")

v = data.validate_tickers(["TCS.NS", "NOTAREALTICKER123"])
bad = data.get_prices(["TCS.NS", "NOTAREALTICKER123"], START, END)
check("Bad ticker", v["invalid"] == ["NOTAREALTICKER123"] and bad.attrs["dropped"] == ["NOTAREALTICKER123"],
      f"validate={v} dropped_reason={bad.attrs['dropped_reasons']}")

short = data.get_prices(["TCS.NS", "TMCV.NS"], START, END)       # TMCV.NS listed Nov 2025
check("Short history dropped", "TMCV.NS" in short.attrs["dropped"] and list(short.columns) == ["TCS.NS"],
      f"{short.attrs['dropped_reasons']}")

px = data.get_prices(IN, START, END)
a = metrics.portfolio_metrics(px, {t: 20 for t in IN})           # percentages summing to 100
b = metrics.portfolio_metrics(px, {t: 7 for t in IN})            # arbitrary scale
check("Weights not summing to 100%", a == b, "normalised identically")

# ---- regime windows (assets that did not exist yet are dropped, not raised) --------------------
wide = data.get_prices(IN + ["TMCV.NS"], "2019-06-01", END, align="outer", min_rows=1)
covid = simulate.simulate(wide, equal(list(wide.columns)), *simulate.regime_presets()["COVID crash"])
check("Regime window drops missing asset", covid["dropped_tickers"] == ["TMCV.NS"] and len(covid["growth"]) > 40,
      f"growth end={covid['growth'].iloc[-1]:.1f} dropped={covid['dropped_tickers']}")

# ---- latency budgets ---------------------------------------------------------------------------
data.get_prices(TEN, START, END)                                  # warm
_, t = timed(data.get_prices, TEN, START, END)
check("get_prices cached (10 tickers) < 1 s", t < 1.0, f"{t * 1e3:.0f} ms")
pxt = data.get_prices(TEN, START, END)
w10 = equal(TEN)
_, t = timed(metrics.portfolio_metrics, pxt, w10)
check("portfolio_metrics < 50 ms", t < 0.05, f"{t * 1e3:.1f} ms")
worst = 0.0
for lvl in ("Low", "Medium", "High"):
    _, t = timed(optimizer.optimize, pxt, lvl)
    worst = max(worst, t)
check("optimize < 2 s (worst of Low/Medium/High)", worst < 2.0, f"{worst * 1e3:.0f} ms")
_, t = timed(simulate.monte_carlo, pxt, w10, 1000, 252)
check("monte_carlo 1000x252 < 2 s", t < 2.0, f"{t * 1e3:.0f} ms")

# ---- offline: no network and an empty cache must fall back to the parquet files -----------------
cache_backup = data.CACHE_DIR.with_name("prices_backup")
shutil.rmtree(cache_backup, ignore_errors=True)
if data.CACHE_DIR.exists():
    data.CACHE_DIR.rename(cache_backup)
orig_dl, orig_sleep = data._download, data._sleep
data._tail_tried.clear(); data._head_tried.clear()
data._sleep = lambda s: None
def _offline(*a, **k):
    raise ConnectionError("network disabled")
data._download = _offline
try:
    off, t = timed(data.get_prices, ["TCS.NS", "AAPL"], START, "2025-12-31")
    check("Offline -> parquet fallback", off.attrs["source"] == "fallback" and off.shape[1] == 2 and len(off) > 900,
          f"{off.shape} in {t:.2f}s warnings={off.attrs['warnings']}")
finally:
    data._download, data._sleep = orig_dl, orig_sleep
    shutil.rmtree(data.CACHE_DIR, ignore_errors=True)
    if cache_backup.exists():
        cache_backup.rename(data.CACHE_DIR)

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
raise SystemExit(1 if failed else 0)
