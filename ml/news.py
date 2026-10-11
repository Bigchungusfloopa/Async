"""AI market intelligence (supplementary - never feeds the ML risk label).

Headlines come from ``yfinance`` ``Ticker.news`` (old flat and new nested schema) and Google News RSS via
``feedparser``; they are de-duplicated and sent to the LLM in ONE call for all tickers. Results carry
``source``: "live" (fresh headlines + fresh answer), "cache" (answer served from the LLM cache) or "offline"
(no network/LLM: last saved snapshot, or an empty result). Every failure path returns a well-formed dict.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote_plus

import httpx

from engine import db
from engine.universe import name_of
from . import i18n, llm

log = logging.getLogger("ml.news")

MAX_PER_TICKER = 5
MAX_TOTAL = 20
MAX_TICKERS = 10
RSS_TIMEOUT_S = 8.0
SENTIMENTS = ("Positive", "Neutral", "Negative")
NEWS_TASK = "news_insight"
RSS_URL = "https://news.google.com/rss/search?q={q}+stock&hl=en-IN&gl=IN&ceid=IN:en"


# --------------------------------------------------------------------------- headline collection
def _norm_title(title: str) -> str:
    t = re.sub(r"\s+-\s+[^-]{2,40}$", "", title.strip())          # Google News appends " - Publisher"
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _parse_yf_item(item: dict) -> dict | None:
    """Handle both yfinance news schemas: flat {title, publisher, link} and nested {content: {...}}."""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (c.get("title") or "").strip()
    if not title:
        return None
    provider = c.get("provider") if isinstance(c.get("provider"), dict) else {}
    source = provider.get("displayName") or c.get("publisher") or item.get("publisher") or "Yahoo Finance"
    url = ""
    for key in ("canonicalUrl", "clickThroughUrl"):
        if isinstance(c.get(key), dict) and c[key].get("url"):
            url = c[key]["url"]
            break
    url = url or c.get("link") or item.get("link") or ""
    return {"title": title, "source": str(source), "url": str(url)}


def _yf_news(ticker: str) -> list[dict]:
    try:
        import yfinance as yf

        items = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001 - yfinance raises assorted errors; news is best-effort
        log.info("yfinance news failed for %s: %s", ticker, type(exc).__name__)
        return []
    return [h for h in (_parse_yf_item(i) for i in items if isinstance(i, dict)) if h]


def _rss_news(ticker: str) -> list[dict]:
    try:
        import feedparser

        company = re.sub(r"\.(NS|BO)$", "", name_of(ticker))
        r = httpx.get(RSS_URL.format(q=quote_plus(company)), timeout=RSS_TIMEOUT_S, follow_redirects=True)
        feed = feedparser.parse(r.content)
    except Exception as exc:  # noqa: BLE001
        log.info("RSS failed for %s: %s", ticker, type(exc).__name__)
        return []
    out = []
    for e in feed.entries[:10]:
        title = (getattr(e, "title", "") or "").strip()
        src = getattr(getattr(e, "source", None), "title", None) or "Google News"
        if title.endswith(" - " + str(src)):
            title = title[: -len(str(src)) - 3].rstrip()
        if title:
            out.append({"title": title, "source": str(src), "url": getattr(e, "link", "")})
    return out


def fetch_headlines(tickers: list[str]) -> list[dict]:
    """De-duplicated headlines, <= 5 per ticker and <= 20 in total, tagged with their ticker."""
    def one(t: str) -> list[dict]:
        return _yf_news(t) + _rss_news(t)

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(tickers)))) as pool:
        per = list(pool.map(one, tickers))
    seen: set[str] = set()
    kept: list[list[dict]] = []
    for t, items in zip(tickers, per):
        mine = []
        for h in items:
            k = _norm_title(h["title"])
            if k and k not in seen:
                seen.add(k)
                mine.append({**h, "ticker": t})
            if len(mine) >= MAX_PER_TICKER:
                break
        kept.append(mine)
    out: list[dict] = []
    for rank in range(MAX_PER_TICKER):                              # round-robin so every ticker is represented
        for mine in kept:
            if rank < len(mine) and len(out) < MAX_TOTAL:
                out.append(mine[rank])
    return out


# --------------------------------------------------------------------------- LLM call
def _prompt(tickers: list[str], headlines: list[dict], lang: str) -> tuple[str, str]:
    system = ("You are a cautious financial news analyst. Use ONLY the headlines provided. Never invent facts, "
              "numbers, prices or events. Output ONLY valid JSON.")
    lines = "\n".join(f"{i + 1}. [{h['ticker']}] {h['title']} ({h['source']})" for i, h in enumerate(headlines))
    tick = ", ".join(f"{t} = {name_of(t)}" for t in tickers)
    glossary = i18n.glossary_prompt(lang) if lang != "en" else ""
    prompt = (
        f"Tickers: {tick}\nHeadlines:\n{lines}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"sentiment": "Positive|Neutral|Negative" (overall), '
        '"per_ticker": [{"ticker": str, "sentiment": "Positive|Neutral|Negative", "drivers": str (<= 25 words)}], '
        '"sector_trends": str (<= 40 words), "risks": [str] (max 4, each <= 20 words), "summary": str (<= 60 words)}\n'
        f"Write the text fields in {i18n.language_name(lang)}; keep the sentiment values in English exactly as "
        "listed and keep ticker symbols unchanged.\n"
        + (f"Use this fixed glossary:\n{glossary}\n" if glossary else "")
        + "If the headlines say little about a ticker, give it Neutral and say so."
    )
    return system, prompt


def _clean_sentiment(v: Any) -> str:
    s = str(v or "").strip().lower()
    return next((x for x in SENTIMENTS if x.lower() == s), "Neutral")


def _shape(obj: Any, tickers: list[str], headlines: list[dict]) -> dict:
    obj = obj if isinstance(obj, dict) else {}
    per = []
    for r in obj.get("per_ticker") or []:
        if isinstance(r, dict) and str(r.get("ticker", "")).upper() in tickers:
            per.append({"ticker": str(r["ticker"]).upper(), "sentiment": _clean_sentiment(r.get("sentiment")),
                        "drivers": str(r.get("drivers", "")).strip()})
    return {
        "sentiment": _clean_sentiment(obj.get("sentiment")),
        "per_ticker": per,
        "sector_trends": str(obj.get("sector_trends", "")).strip(),
        "risks": [str(x).strip() for x in (obj.get("risks") or []) if str(x).strip()][:5],
        "summary": str(obj.get("summary", "")).strip(),
        "headlines": [{"title": h["title"], "source": h["source"], "url": h["url"], "ticker": h.get("ticker", "")}
                      for h in headlines],
    }


def _empty(headlines: list[dict], note: str, source: str) -> dict:
    return {"sentiment": "Neutral", "per_ticker": [], "sector_trends": "", "risks": [], "summary": note,
            "headlines": [{"title": h["title"], "source": h["source"], "url": h["url"], "ticker": h.get("ticker", "")}
                          for h in headlines], "source": source, "ai_available": False}


def _snapshot_key(tickers: list[str], lang: str) -> str:
    return hashlib.sha256(("news_snapshot|" + ",".join(sorted(tickers)) + "|" + lang).encode()).hexdigest()


def _load_snapshot(tickers: list[str], lang: str, remote: bool) -> dict | None:
    row = db.llm_cache_get(_snapshot_key(tickers, lang), remote=remote)
    if not row:
        return None
    try:
        return json.loads(row["response"])
    except (ValueError, KeyError, TypeError):
        return None


# --------------------------------------------------------------------------- public API
def get_news_insight(tickers: list[str], lang: str = "en") -> dict:
    """Sentiment, per-ticker drivers, sector trends, risks, summary and headlines for ``tickers``.

    Shape: ``{"sentiment", "per_ticker": [{"ticker","sentiment","drivers"}], "sector_trends", "risks": [...],
    "summary", "headlines": [{"title","source","url","ticker"}], "source": "live|cache|offline",
    "ai_available": bool}``. Call only from an explicit user action (one LLM call for all tickers).
    """
    tickers = [t for t in dict.fromkeys(str(x).strip().upper() for x in (tickers or []) if str(x).strip())][:MAX_TICKERS]
    if not tickers:
        return _empty([], "No tickers selected.", "offline")
    unavailable = "AI market summary is unavailable right now (offline mode or provider limit)."

    if llm.is_offline():
        snap = _load_snapshot(tickers, lang, remote=False)
        return {**snap, "source": "offline"} if snap else _empty([], unavailable, "offline")

    headlines = fetch_headlines(tickers)
    if not headlines:
        snap = _load_snapshot(tickers, lang, remote=True)
        if snap:
            return {**snap, "source": "offline"}
        return _empty([], "No headlines could be fetched (no network?).", "offline")

    system, prompt = _prompt(tickers, headlines, lang)
    try:
        out = llm.ask_llm(NEWS_TASK, prompt, system=system, json_mode=True, lang=lang, max_tokens=1200)
        result = _shape(llm.parse_json(out["text"]), tickers, headlines)
        result.update(source="cache" if out["cached"] else "live", ai_available=True)
    except (llm.LLMUnavailable, ValueError) as exc:
        log.info("news summary unavailable: %s", exc)
        snap = _load_snapshot(tickers, lang, remote=True)
        if snap:                                              # older AI summary + the fresh headlines
            snap["headlines"] = _empty(headlines, "", "offline")["headlines"]
            return {**snap, "source": "cache"}
        return _empty(headlines, unavailable, "offline")
    db.llm_cache_put(_snapshot_key(tickers, lang), "news_snapshot", lang, json.dumps(result, ensure_ascii=False),
                     "snapshot", remote=True)
    return result
