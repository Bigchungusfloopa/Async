"""Fill the LLM cache for the demo so every screen works with Offline mode ON.

For each demo portfolio x language (en/hi/mr) it runs the real functions once:
  * portfolio explanation, allocation explanation, scenario text   (translation calls for hi/mr)
  * news insight (one call per portfolio x language)
and it runs the saved voice samples through ``extract_portfolio``.

    python scripts/prewarm.py            # run at 5:40 and again just before the demo
    python scripts/prewarm.py --offline-check   # verify everything renders from cache, zero network LLM calls

Cache keys depend on the numbers in the text, so re-run it on the day (prices move daily).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import data, optimizer, simulate  # noqa: E402
from ml import classify, explain, llm, news, voice  # noqa: E402

LANGS = ("en", "hi", "mr")
DEMO_PORTFOLIOS = {            # chosen to span High / Medium / Low (and mix INR + USD) on the current model
    "IT heavy (India)": {"TCS.NS": 0.35, "INFY.NS": 0.35, "WIPRO.NS": 0.15, "HCLTECH.NS": 0.15},
    "Balanced: stocks + gold + bonds": {"HDFCBANK.NS": 0.25, "ITC.NS": 0.25, "TCS.NS": 0.15, "GOLDBEES.NS": 0.20,
                                        "SHY": 0.15},
    "Index core + gold + bonds": {"NIFTYBEES.NS": 0.35, "SPY": 0.30, "GOLDBEES.NS": 0.20, "IEF": 0.15},
}
SCENARIO = {"type": "market_shock", "pct": -0.20}


def run(offline_check: bool = False, pause: float | None = None) -> int:
    llm.set_offline(offline_check)
    llm.reload_config()
    start = (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"providers configured: {llm.providers_configured() or 'NONE (templates + rules only)'}"
          f" | offline={offline_check}")
    problems: list[str] = []
    t0 = time.time()
    # free tiers allow only a handful of requests per minute: pace calls (4 s) unless told otherwise
    if pause is None:
        pause = 0.0 if (offline_check or not llm.providers_configured()) else 4.0

    def pace() -> None:
        if pause:
            time.sleep(pause)

    for name, w in DEMO_PORTFOLIOS.items():
        px = data.get_prices(list(w), start, end)
        if px.empty or len(px) < 252:
            problems.append(f"{name}: not enough price data")
            continue
        label = classify.classify_risk(px, w)["label"]
        opt = optimizer.optimize(px, "Medium", current_weights=w)
        sc = simulate.apply_scenario(px, w, SCENARIO)
        after = classify.classify_stressed(sc["stressed_prices"], w)["label"]
        for lang in LANGS:
            checks = {
                "portfolio": explain.explain_portfolio_ex(px, w, label, lang),
                "allocation": explain.explain_allocation_ex(px, w, opt["weights"], "Medium", lang, opt["tradeoffs"]),
                "scenario": explain.explain_scenario_ex(sc, label, after, lang),
            }
            pace()
            for task, r in checks.items():
                if lang != "en" and r["fallback"]:
                    problems.append(f"{name} / {lang} / {task}: translation unavailable (English shown)")
            n = news.get_news_insight(list(w), lang)
            pace()
            if not n.get("ai_available"):
                problems.append(f"{name} / {lang} / news: no AI summary ({n.get('source')})")
        print(f"  done: {name} (label {label}, stressed {after})")

    for s in voice.demo_samples():
        r = voice.extract_portfolio(s["text"])
        pace()
        if r["tickers"] != s["expected"]["tickers"]:
            problems.append(f"voice sample [{s['lang']}] mismatch: {r['tickers']} != {s['expected']['tickers']}")
    print(f"voice samples checked: {len(voice.demo_samples())}")

    print(f"usage this run: {llm.usage_stats()}  ({time.time() - t0:.0f}s)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("\nall demo content is cached / renders")
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-check", action="store_true",
                    help="Offline mode ON: no network LLM calls; fails if anything needs one")
    ap.add_argument("--pause", type=float, default=None, help="seconds between LLM-capable steps (default 4)")
    a = ap.parse_args()
    raise SystemExit(run(a.offline_check, a.pause))
