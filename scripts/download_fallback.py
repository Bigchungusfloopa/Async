"""Pre-download the demo universe + benchmarks to ``data/fallback/<TICKER>.parquet`` (batched).

Run from the repo root (re-run before the demo so the files are fresh):
    python scripts/download_fallback.py [--start 2015-01-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import data  # noqa: E402
from engine.universe import load_universe  # noqa: E402

EXTRA = ["INR=X", "^GSPC"]   # USD->INR for mixed portfolios; S&P 500 is a benchmark option in the UI
CHUNK = 20          # tickers per batched request, gentle on Yahoo's rate limits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    tickers = data.normalize_tickers([e["ticker"] for e in load_universe()] + EXTRA)
    end = pd.Timestamp.today().normalize()
    data.FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    ok, empty = [], []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        df = data._download_with_retries(chunk, pd.Timestamp(args.start), end)
        for t in chunk:
            s = df[t].dropna() if t in df.columns else pd.Series(dtype="float64")
            if s.empty:
                empty.append(t)
                continue
            s.rename("adj_close").rename_axis("date").to_frame().to_parquet(data.FALLBACK_DIR / f"{t}.parquet")
            ok.append((t, len(s), s.index[0].date(), s.index[-1].date()))
        print(f"chunk {i // CHUNK + 1}: {len(chunk)} requested")
    for t, n, a, b in ok:
        print(f"  {t:<16}{n:>6} rows  {a} -> {b}")
    if empty:
        print("NO DATA for:", ", ".join(empty))
    print(f"saved {len(ok)} files to {data.FALLBACK_DIR}")
    return 0 if not empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
