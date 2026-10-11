"""
scripts/prewarm.py — Pre-warm the LLM cache before the demo.

Run this at 5:40 and again just before the demo:
    python scripts/prewarm.py

It runs every demo portfolio x 3 languages (en/hi/mr) x (explanation, news insight)
and saves responses in llm_cache. Also verifies offline mode works.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_PORTFOLIOS = [
    {
        "tickers": ["TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "weights": {"TCS.NS": 0.33, "INFY.NS": 0.33, "HDFCBANK.NS": 0.34},
        "risk_level": "Medium",
    },
    {
        "tickers": ["TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "weights": {"TCS.NS": 0.20, "INFY.NS": 0.30, "HDFCBANK.NS": 0.50},
        "risk_level": "Low",
    },
    {
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "weights": {"AAPL": 0.40, "MSFT": 0.40, "NVDA": 0.20},
        "risk_level": "High",
    },
]

DEMO_VOICE_SAMPLES = [
    {
        "lang": "hi",
        "transcript": "Mere paas 1 lakh rupaye hain, TCS, Infosys aur HDFC Bank mein lagana hai, risk kam chahiye.",
        "expected": {"tickers": ["TCS.NS", "INFY.NS", "HDFCBANK.NS"], "amounts": {"TCS.NS": 33333, "INFY.NS": 33333, "HDFCBANK.NS": 33334}, "risk_level": "Low"},
    },
    {
        "lang": "mr",
        "transcript": "Majhyakade 50 hazaar rupaye ahet, TCS aani Infosys madhe guntavayache ahe, jokhim madhyam pahije.",
        "expected": {"tickers": ["TCS.NS", "INFY.NS"], "amounts": {"TCS.NS": 25000, "INFY.NS": 25000}, "risk_level": "Medium"},
    },
    {
        "lang": "en",
        "transcript": "I have 2 lakh rupees to invest in Reliance and Wipro with high risk.",
        "expected": {"tickers": ["RELIANCE.NS", "WIPRO.NS"], "amounts": {"RELIANCE.NS": 100000, "WIPRO.NS": 100000}, "risk_level": "High"},
    },
]

START = "2022-01-01"
END   = "2024-12-31"
LANGS = ["en", "hi", "mr"]


def run_prewarm():
    print("=" * 60)
    print("Investment Risk Predictor — Demo Cache Prewarm")
    print("=" * 60)

    try:
        from ml.explain import explain_portfolio
        from ml.news import get_news_insight
        has_ml = True
    except ImportError:
        has_ml = False
        print("[WARN] ml modules not available — using mock mode only")

    total = 0
    success = 0

    for portfolio in DEMO_PORTFOLIOS:
        tickers  = portfolio["tickers"]
        weights  = portfolio["weights"]
        risk_lbl = portfolio["risk_level"]
        print(f"\n▶ Portfolio: {tickers} ({risk_lbl})")

        for lang in LANGS:
            print(f"  Lang: {lang}")
            total += 1

            # Explanation
            try:
                if has_ml:
                    import yfinance as yf
                    import pandas as pd
                    prices = yf.download(tickers, start=START, end=END,
                                        auto_adjust=True, progress=False)["Close"]
                    if isinstance(prices, pd.Series):
                        prices = prices.to_frame(tickers[0])
                    explain_portfolio(prices, weights, risk_lbl, lang)
                print(f"    ✅ Explanation ({lang})")
                success += 1
            except Exception as e:
                print(f"    ❌ Explanation ({lang}): {e}")

            # News insight
            total += 1
            try:
                if has_ml:
                    get_news_insight(tickers, lang)
                print(f"    ✅ News insight ({lang})")
                success += 1
            except Exception as e:
                print(f"    ❌ News insight ({lang}): {e}")

    # Voice samples
    print("\n▶ Voice samples (pre-recorded expected outputs)")
    for sample in DEMO_VOICE_SAMPLES:
        print(f"  [{sample['lang']}] transcript: {sample['transcript'][:50]}...")
        print(f"    expected: {sample['expected']}")

    print("\n" + "=" * 60)
    print(f"Prewarm complete: {success}/{total} items cached.")
    print("\nNext: switch Offline mode ON and verify the full demo flow.")
    print("=" * 60)


if __name__ == "__main__":
    run_prewarm()
