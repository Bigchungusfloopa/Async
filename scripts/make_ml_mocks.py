"""Write P2's mock JSON files (exact shapes of the real ml/ functions; contract-tested in tests/test_ml.py).

    python scripts/make_ml_mocks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml import i18n  # noqa: E402

MOCKS = ROOT / "data" / "mocks"


def dump(name: str, obj) -> None:
    (MOCKS / f"mock_{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


dump("classify", {
    "label": "High",
    "probs": {"Low": 0.05, "Medium": 0.25, "High": 0.70},
    "top_factors": [
        {"name": "Volatility, last 6 months", "value": "28%", "impact": "raises risk", "feature": "vol_126", "effect": 0.62},
        {"name": "Recent-weighted volatility", "value": "30%", "impact": "raises risk", "feature": "ewma_vol", "effect": 0.31},
        {"name": "Sector concentration", "value": "0.52", "impact": "raises risk", "feature": "sector_hhi", "effect": 0.18},
    ],
    "features": {
        "ann_return": 0.14, "ann_vol": 0.27, "max_drawdown": -0.31, "sharpe": 0.42, "avg_corr": 0.71, "hhi": 0.45,
        "effective_n": 2.2, "top_weight": 0.55, "skew": -0.2, "kurtosis": 2.8, "n_assets": 4.0, "downside_dev": 0.19,
        "vol_21": 0.31, "vol_63": 0.29, "vol_126": 0.28, "vol_ratio_21": 1.15, "vol_ratio_63": 1.07,
        "ewma_vol": 0.30, "vol_of_vol": 0.35, "ret_21": -0.04, "ret_63": 0.02, "cur_drawdown": -0.12,
        "max_asset_vol": 0.38, "undiversified_vol": 0.31, "diversification_ratio": 1.15, "avg_corr_63": 0.74,
        "corr_shift": 0.03, "sector_hhi": 0.52, "market_vol_21": 0.14, "market_vol_ratio": 1.1,
        "market_drawdown": -0.05, "beta_to_market": 1.2},
    "window_rows": 252,
    "note": "",
})

dump("explain", {
    "explain_portfolio": ("This portfolio is rated High risk: its annualised volatility is 27% and the worst "
                          "peak-to-trough fall over the past year was 31%. TCS.NS is 55% of the money but drives "
                          "71% of the risk, so the portfolio's swings depend heavily on it. Average correlation "
                          "between holdings is 0.71 (low diversification); the portfolio behaves like 2.2 equally "
                          "sized positions out of 4. The model's main drivers for this rating: Volatility 27% "
                          "(raises risk); Concentration (HHI) 0.45 (raises risk); Average correlation 0.71 "
                          "(raises risk)."),
    "explain_allocation": ("Reduce TCS.NS from 55% to 30% (-25 points): it was 55% of the portfolio, a very "
                           "concentrated position.\nIncrease HDFCBANK.NS from 15% to 30% (+15 points): it "
                           "diversifies the mix (average correlation 0.28).\nBetter on both counts: volatility "
                           "falls 6.1 points and historical annual return rises 1.2 points."),
    "explain_scenario": ("Under a market shock of -20%, the portfolio value would change by about -17.0%. The "
                         "hardest hit holdings would be HDFCBANK.NS (-22%) and INFY.NS (-18%). Annual volatility "
                         "would move from 19% to 27% and the worst drawdown from 23% to 34%. Its risk rating "
                         "would change from Medium to High."),
})

dump("news", {
    "sentiment": "Neutral",
    "per_ticker": [
        {"ticker": "TCS.NS", "sentiment": "Positive", "drivers": "Quarterly results beat expectations."},
        {"ticker": "INFY.NS", "sentiment": "Neutral", "drivers": "Mixed commentary on deal pipeline."},
    ],
    "sector_trends": "Indian IT stocks rallied after results, while US visa rules remain a watch item.",
    "risks": ["US visa policy changes", "Weak discretionary tech spending"],
    "summary": "Sentiment is broadly neutral to positive after results; policy risk remains.",
    "headlines": [
        {"title": "TCS shares jump after Q2 results", "source": "Moneycontrol", "url": "https://example.com/1",
         "ticker": "TCS.NS"},
        {"title": "Infosys keeps FY guidance unchanged", "source": "Business Standard",
         "url": "https://example.com/2", "ticker": "INFY.NS"},
    ],
    "source": "cache",
    "ai_available": True,
})

dump("voice", {
    "transcribe": "Mere paas 50,000 rupaye hain, TCS aur Infosys mein lagana hai, risk kam chahiye.",
    "extract_portfolio": {"tickers": ["TCS.NS", "INFY.NS"], "amounts": {"TCS.NS": 25000, "INFY.NS": 25000},
                          "risk_level": "Low", "unresolved": [], "confidence": 0.95, "source": "llm"},
})

keys = ["app_title", "tab_risk", "tab_allocation", "analyze", "risk_low", "risk_medium", "risk_high",
        "ann_vol", "max_drawdown", "disclaimer"]
dump("i18n_sample", {lang: {k: i18n.t(k, lang) for k in keys} for lang in ("en", "hi", "mr")})
print("wrote", len(list(MOCKS.glob("mock_*.json"))), "mock files")
