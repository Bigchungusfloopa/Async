"""Live holdings tracking: latest close, value, P&L, allocation and trailing-1y risk.

Rules: totals are always **per currency** (INR and USD are never added together). Allocation and the
risk metrics need one common basis, so for mixed portfolios USD values are converted with the latest
``INR=X`` rate *for weighting only*; if the rate is unavailable, each currency group is weighted equally
and value-share within the group (and a warning is returned).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .data import get_prices, normalize_tickers
from .metrics import portfolio_metrics
from .universe import currency_of, name_of

log = logging.getLogger("engine.track")

TABLE_COLUMNS = ["ticker", "name", "currency", "quantity", "buy_price", "last_price", "cost", "value",
                 "pnl", "pnl_pct", "alloc_pct", "as_of"]
FX_TICKER = "INR=X"          # INR per 1 USD
LOOKBACK_DAYS = 400          # enough for a trailing year plus ffill slack
RISK_WINDOW_DAYS = 365
MIN_RISK_ROWS = 60
STALE_DAYS = 7
DATA_NOTE = "Prices are the latest available daily close and may be delayed."


def _clean_holdings(holdings: list[dict], warnings: list[str]) -> dict[str, dict]:
    """Validate and merge duplicate tickers (quantities add, buy price is the cost-weighted average)."""
    agg: dict[str, dict] = {}
    for h in holdings or []:
        try:
            t = normalize_tickers([h["ticker"]])[0]
            q, bp = float(h["quantity"]), float(h["buy_price"])
        except (KeyError, IndexError, TypeError, ValueError):
            warnings.append(f"skipped a holding with missing or invalid fields: {h!r}")
            continue
        if not (np.isfinite(q) and np.isfinite(bp)) or q <= 0 or bp < 0:
            warnings.append(f"{t}: skipped (quantity must be > 0 and buy price >= 0)")
            continue
        a = agg.setdefault(t, {"quantity": 0.0, "cost": 0.0})
        a["quantity"] += q
        a["cost"] += q * bp
    return agg


def track_holdings(holdings: list[dict]) -> dict:
    """``holdings``: ``[{"ticker","quantity","buy_price"}, ...]``.

    Returns ``{"table": DataFrame, "total_value": {ccy: x}, "total_cost": {ccy: x}, "total_pnl": {ccy: x},
    "allocation": {ticker: fraction}, "risk_metrics": {...trailing 1y...}, "as_of": "YYYY-MM-DD",
    "source": ..., "warnings": [...], "data_note": str}``.
    """
    warnings: list[str] = []
    agg = _clean_holdings(holdings, warnings)
    result = {"table": pd.DataFrame(columns=TABLE_COLUMNS), "total_value": {}, "total_cost": {}, "total_pnl": {},
              "allocation": {}, "risk_metrics": {}, "as_of": None, "source": "cache",
              "warnings": warnings, "data_note": DATA_NOTE}
    if not agg:
        return result

    tickers = list(agg)
    today = pd.Timestamp.today().normalize()
    need_fx = len({currency_of(t) for t in tickers}) > 1
    px = get_prices(tickers + ([FX_TICKER] if need_fx else []), (today - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
                    today.strftime("%Y-%m-%d"), align="outer", min_rows=1)
    result["source"] = px.attrs.get("source", "cache")
    warnings += [w for w in px.attrs.get("warnings", []) if "mixed currencies" not in w]

    rows, no_price = [], []
    for t, a in agg.items():
        s = px[t].dropna() if t in px.columns else pd.Series(dtype="float64")
        if s.empty:
            no_price.append(t)
            continue
        last, as_of = float(s.iloc[-1]), s.index[-1]
        if (today - as_of).days > STALE_DAYS:
            warnings.append(f"{t}: latest price is from {as_of.date()} (older than {STALE_DAYS} days)")
        qty, cost = a["quantity"], a["cost"]
        value = qty * last
        rows.append({"ticker": t, "name": name_of(t), "currency": currency_of(t), "quantity": qty,
                     "buy_price": cost / qty, "last_price": last, "cost": cost, "value": value,
                     "pnl": value - cost, "pnl_pct": (value / cost - 1.0) if cost > 0 else 0.0,
                     "alloc_pct": 0.0, "as_of": as_of.date().isoformat()})
    if no_price:
        warnings.append("no price data for: " + ", ".join(no_price) + " (excluded from totals)")
    if not rows:
        return result

    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)

    # one common weighting basis for allocation + risk
    fx = None
    if need_fx and FX_TICKER in px.columns and px[FX_TICKER].notna().any():
        fx = float(px[FX_TICKER].dropna().iloc[-1])
    to_inr = {"INR": 1.0, "USD": fx} if fx else None
    ccys = set(table["currency"])
    if len(ccys) == 1:
        basis = table["value"].to_numpy()
        share = basis / basis.sum() if basis.sum() > 0 else np.zeros(len(table))
    elif to_inr and ccys <= set(to_inr):
        basis = table["value"].to_numpy() * table["currency"].map(to_inr).to_numpy()
        share = basis / basis.sum() if basis.sum() > 0 else np.zeros(len(table))
        warnings.append(f"allocation uses USD->INR {fx:.2f} for weighting only; totals stay per currency")
    else:
        grp = table.groupby("currency")["value"].transform("sum")
        share = (table["value"] / grp.where(grp > 0, 1.0)).to_numpy() / len(ccys)
        warnings.append("mixed currencies and no FX rate available: each currency group is weighted equally")
    table["alloc_pct"] = share

    result["table"] = table
    for key, col in (("total_value", "value"), ("total_cost", "cost"), ("total_pnl", "pnl")):
        result[key] = {c: float(v) for c, v in table.groupby("currency")[col].sum().items()}
    result["allocation"] = dict(zip(table["ticker"], (float(x) for x in share)))
    result["as_of"] = max(table["as_of"])

    # trailing-1y risk on the portfolio's current weights
    window = px[table["ticker"]].loc[today - pd.Timedelta(days=RISK_WINDOW_DAYS):]
    usable = [t for t in window.columns if window[t].notna().sum() >= MIN_RISK_ROWS]
    short = [t for t in window.columns if t not in usable]
    if short:
        warnings.append("too little 1-year history for risk metrics: " + ", ".join(short))
    if usable:
        sub = window[usable].ffill(limit=3).dropna(how="any")
        w = {t: result["allocation"][t] for t in usable}
        if len(sub) >= MIN_RISK_ROWS and sum(w.values()) > 0:
            result["risk_metrics"] = portfolio_metrics(sub, w)
        else:
            warnings.append("holdings share too few common trading days for risk metrics")
    return result
