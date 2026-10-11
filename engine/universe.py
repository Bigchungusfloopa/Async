"""Read-only helpers over ``data/universe.json`` (sector / currency / name lookups). No network, no Streamlit."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"

_SUFFIX_CCY = {".NS": "INR", ".BO": "INR", ".L": "GBP", ".TO": "CAD", ".HK": "HKD", ".T": "JPY",
               ".DE": "EUR", ".PA": "EUR", ".AX": "AUD"}
_INR_INDICES = {"^NSEI", "^BSESN", "^NSEBANK", "^CNXIT", "^CNXMIDCAP", "INR=X"}


@lru_cache(maxsize=1)
def _load() -> tuple[dict, ...]:
    try:
        return tuple(json.loads(UNIVERSE_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return tuple()


def load_universe() -> list[dict]:
    """All universe entries: ``{"ticker","name","sector","currency","market","type"}``."""
    return [dict(e) for e in _load()]


@lru_cache(maxsize=1)
def _by_ticker() -> dict[str, dict]:
    return {e["ticker"].upper(): e for e in _load()}


def info(ticker: str) -> dict | None:
    return _by_ticker().get(str(ticker).strip().upper())


def currency_of(ticker: str) -> str:
    """Universe currency if known, otherwise inferred from the exchange suffix (default USD)."""
    t = str(ticker).strip().upper()
    e = _by_ticker().get(t)
    if e:
        return e["currency"]
    if t in _INR_INDICES:
        return "INR"
    for suf, ccy in _SUFFIX_CCY.items():
        if t.endswith(suf):
            return ccy
    return "USD"


def sector_of(ticker: str) -> str | None:
    e = info(ticker)
    return e["sector"] if e else None


def name_of(ticker: str) -> str:
    e = info(ticker)
    return e["name"] if e else str(ticker)
