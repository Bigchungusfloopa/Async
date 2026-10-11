"""Write the Phase-0 mock JSON files (fake but realistic numbers, exact engine return shapes).

Run from the repo root:  python scripts/make_mocks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.serialize import MOCK_DIR, dump_json  # noqa: E402

rng = np.random.default_rng(7)


def walk(n: int, drift: float, vol: float, start: float = 100.0) -> np.ndarray:
    r = rng.normal(drift / 252, vol / np.sqrt(252), n)
    return start * np.exp(np.cumsum(r))


def metrics(ann_return, ann_vol, mdd, sharpe, sortino, calmar, cvar, corr, hhi) -> dict:
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "max_drawdown": mdd, "sharpe": sharpe,
        "sortino": sortino, "calmar": calmar, "cvar_95": cvar, "avg_corr": corr,
        "hhi": hhi, "effective_n": round(1 / hhi, 2),
    }


M_ORIG = metrics(0.14, 0.19, -0.23, 0.55, 0.80, 0.60, -0.031, 0.42, 0.31)
M_REC = metrics(0.125, 0.145, -0.17, 0.62, 0.91, 0.74, -0.024, 0.38, 0.26)
M_EW = metrics(0.13, 0.16, -0.19, 0.58, 0.85, 0.68, -0.026, 0.40, 0.25)
M_BM = metrics(0.115, 0.15, -0.21, 0.45, 0.66, 0.55, -0.025, 0.0, 1.0)

dates = pd.bdate_range("2021-01-01", "2021-12-31")
n = len(dates)

# ---- portfolio_metrics
dump_json(M_ORIG, MOCK_DIR / "mock_metrics.json")

# ---- optimize
dump_json({
    "weights": {"TCS.NS": 0.25, "INFY.NS": 0.15, "HDFCBANK.NS": 0.30, "AAPL": 0.30},
    "metrics": M_REC,
    "reason": ("Objective: maximum Sharpe ratio (Medium risk). Volatility goes from 19.0% to 14.5% and "
               "expected return from 14.0% to 12.5%. Biggest changes: TCS.NS 40% -> 25% (it drives 52% of "
               "your risk); HDFCBANK.NS 20% -> 30% (low correlation with the rest); AAPL 15% -> 30% "
               "(better return per unit of risk)."),
    "tradeoffs": ("You give up 1.5 points of historical annual return to cut volatility by 4.5 points "
                  "and shrink the worst drawdown from -23% to -17%."),
    "mode": "max_sharpe",
    "before_weights": {"TCS.NS": 0.40, "INFY.NS": 0.30, "HDFCBANK.NS": 0.20, "AAPL": 0.10},
    "before_metrics": M_ORIG,
    "bounds_used": [0.05, 0.40],
    "risk_level": "Medium",
    "return_shrinkage": 0.5,
}, MOCK_DIR / "mock_optimize.json")

# ---- simulate
growth = pd.Series(np.round(walk(n, 0.18, 0.17), 2), index=dates)
growth.iloc[0] = 100.0
dump_json({"growth": growth, "metrics": M_ORIG, "dropped_tickers": []}, MOCK_DIR / "mock_simulate.json")

# ---- track_holdings
table = pd.DataFrame(
    [
        ["TCS.NS", "Tata Consultancy Services", "INR", 10, 3200.0, 3650.0, 32000.0, 36500.0, 4500.0, 0.1406, 0.58, "2025-06-13"],
        ["INFY.NS", "Infosys", "INR", 20, 1400.0, 1580.0, 28000.0, 31600.0, 3600.0, 0.1286, 0.42, "2025-06-13"],
        ["AAPL", "Apple", "USD", 5, 150.0, 195.0, 750.0, 975.0, 225.0, 0.30, 1.0, "2025-06-12"],
    ],
    columns=["ticker", "name", "currency", "quantity", "buy_price", "last_price", "cost", "value",
             "pnl", "pnl_pct", "alloc_pct", "as_of"],
)
dump_json({
    "table": table,
    "total_value": {"INR": 68100.0, "USD": 975.0},
    "total_cost": {"INR": 60000.0, "USD": 750.0},
    "total_pnl": {"INR": 8100.0, "USD": 225.0},
    "allocation": {"TCS.NS": 0.44, "INFY.NS": 0.38, "AAPL": 0.18},
    "risk_metrics": M_ORIG,
    "as_of": "2025-06-13",
    "source": "live",
    "warnings": [],
    "data_note": "Prices are the latest available daily close and may be delayed.",
}, MOCK_DIR / "mock_track.json")

# ---- compare_strategies / compare_growth / risk_return_points
cmp_df = pd.DataFrame(
    [M_ORIG, M_REC, M_EW, M_BM],
    index=["Original", "Recommended", "Equal Weight", "Benchmark"],
)
cmp_df["total_return"] = [0.31, 0.27, 0.29, 0.24]
dump_json(cmp_df, MOCK_DIR / "mock_compare.json")

growth_df = pd.DataFrame(
    {
        "Original": np.round(walk(n, 0.18, 0.19), 2),
        "Recommended": np.round(walk(n, 0.16, 0.145), 2),
        "Equal Weight": np.round(walk(n, 0.17, 0.16), 2),
        "Benchmark": np.round(walk(n, 0.12, 0.15), 2),
    },
    index=dates,
)
growth_df.iloc[0] = 100.0
dump_json(growth_df, MOCK_DIR / "mock_compare_growth.json")

dump_json({
    "TCS.NS": {"ret": 0.16, "vol": 0.24, "kind": "asset"},
    "INFY.NS": {"ret": 0.13, "vol": 0.26, "kind": "asset"},
    "HDFCBANK.NS": {"ret": 0.12, "vol": 0.22, "kind": "asset"},
    "AAPL": {"ret": 0.21, "vol": 0.28, "kind": "asset"},
    "Original": {"ret": 0.14, "vol": 0.19, "kind": "portfolio"},
    "Recommended": {"ret": 0.125, "vol": 0.145, "kind": "portfolio"},
    "Benchmark": {"ret": 0.115, "vol": 0.15, "kind": "benchmark"},
}, MOCK_DIR / "mock_risk_return.json")

# ---- apply_scenario
stressed = pd.DataFrame(
    {
        "TCS.NS": np.round(walk(120, 0.1, 0.22, 3500), 2),
        "INFY.NS": np.round(walk(120, 0.1, 0.25, 1500), 2),
        "AAPL": np.round(walk(120, 0.2, 0.27, 190), 2),
    },
    index=pd.bdate_range(end="2025-06-16", periods=120),
)
dump_json({
    "before": M_ORIG,
    "after": metrics(-0.05, 0.27, -0.34, -0.40, -0.55, -0.15, -0.045, 0.60, 0.31),
    "value_change_pct": -0.18,
    "stressed_prices": stressed,
    "value_change_basis": "instant_shock",
    "asset_shocks": {"TCS.NS": -0.17, "INFY.NS": -0.19, "AAPL": -0.22},
    "betas": {"TCS.NS": 0.85, "INFY.NS": 0.95, "AAPL": 1.10},
    "scenario": {"type": "market_shock", "pct": -0.20},
    "warnings": [],
}, MOCK_DIR / "mock_scenario.json")

# ---- monte_carlo
h = 252
t = np.arange(1, h + 1)


def pct_path(z: float) -> list:
    return [round(100 * float(np.exp((0.12 / 252) * k + z * 0.19 * np.sqrt(k / 252))), 2) for k in t]


dump_json({
    "percentiles": {"p5": pct_path(-1.645), "p25": pct_path(-0.674), "p50": pct_path(0.0),
                    "p75": pct_path(0.674), "p95": pct_path(1.645)},
    "prob_loss": 0.27,
    "expected_shortfall": -0.21,
    "horizon": h, "n_paths": 1000, "block": 10, "seed": 42, "start_value": 100.0,
}, MOCK_DIR / "mock_monte_carlo.json")

print("wrote", len(list(MOCK_DIR.glob("mock_*.json"))), "mock files to", MOCK_DIR)
