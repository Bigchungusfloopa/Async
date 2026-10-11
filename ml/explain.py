"""Template-based explanations built from real numbers. English never calls an LLM; for Hindi/Marathi ONE
``ask_llm`` translation call is made per explanation (with the fixed glossary), and any failure returns the
English template instead of raising.

* ``explain_portfolio(prices, weights, label, lang)`` -> why the portfolio got its label
* ``explain_allocation(prices, original_w, optimized_w, risk_level, lang)`` -> why each weight moved
* ``*_ex`` variants return ``{"text", "lang", "translated", "fallback"}`` so the UI can show a notice.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine import metrics as M
from engine.optimizer import tradeoffs_text
from . import i18n, llm
from .features import lookback_slice

log = logging.getLogger("ml.explain")

MIN_CHANGE = 0.03          # explain weight changes larger than 3 percentage points
TRANSLATE_TASK = "translate_explanation"


# --------------------------------------------------------------------------- translation
def _translate(text: str, lang: str) -> tuple[str, bool]:
    """(text, translated?) - falls back to the English text when the LLM is unavailable."""
    if lang == "en" or not text:
        return text, False
    prompt = (
        f"Translate the following investment explanation into {i18n.language_name(lang)}.\n"
        "Rules: keep every number, percentage, ticker symbol (such as TCS.NS) and company name exactly as "
        "written; do not add or remove information; use a plain, friendly tone; output ONLY the translation.\n"
        f"Fixed glossary (use these terms consistently):\n{i18n.glossary_prompt(lang)}\n\n"
        f"TEXT:\n{text}"
    )
    try:
        out = llm.ask_llm(TRANSLATE_TASK, prompt, system="You are a careful financial translator.", lang=lang)
        return out["text"].strip(), True
    except llm.LLMUnavailable as exc:
        log.info("translation unavailable (%s); returning English", exc)
        return text, False


# --------------------------------------------------------------------------- portfolio explanation
def _diversification_phrase(avg_corr: float) -> str:
    if avg_corr >= 0.6:
        return "low diversification"
    if avg_corr >= 0.35:
        return "moderate diversification"
    return "good diversification"


def _portfolio_text(prices: pd.DataFrame, weights: dict, label: str, top_factors: list[dict] | None) -> str:
    window = lookback_slice(prices)
    m = M.portfolio_metrics(window, weights)
    rc = M.risk_contribution(window, weights)
    w = pd.Series(M.normalize_weights(weights)).reindex(rc.index).fillna(0.0)
    w = w / w.sum()
    ratio = (rc / w.replace(0, np.nan)).fillna(0.0)
    top_asset = rc.idxmax()
    top_rc, top_w = float(rc[top_asset]), float(w[top_asset])

    parts = [f"This portfolio is rated {label} risk: its annualised volatility is {m['ann_vol']:.0%} and the "
             f"worst peak-to-trough fall over the past year was {abs(m['max_drawdown']):.0%}."]
    if len(rc) > 1 and top_rc >= 1.3 * top_w and top_rc >= 0.4:
        parts.append(f"{top_asset} is {top_w:.0%} of the money but drives {top_rc:.0%} of the risk, "
                     "so the portfolio's swings depend heavily on it.")
    elif len(rc) > 1:
        parts.append(f"Risk is spread roughly in line with the weights; the biggest risk share is "
                     f"{top_asset} at {top_rc:.0%} (weight {top_w:.0%}).")
    else:
        parts.append("With a single holding all of the risk comes from that one asset.")
    if len(rc) > 1:
        parts.append(f"Average correlation between holdings is {m['avg_corr']:.2f} "
                     f"({_diversification_phrase(m['avg_corr'])}); the portfolio behaves like "
                     f"{m['effective_n']:.1f} equally sized positions out of {len(rc)}.")
    if top_factors:
        drivers = "; ".join(f"{f['name']} {f['value']} ({f['impact']})" for f in top_factors[:3])
        parts.append(f"The model's main drivers for this rating: {drivers}.")
    return " ".join(parts)


def explain_portfolio_ex(prices: pd.DataFrame, weights: dict, label: str, lang: str = "en",
                         top_factors: list[dict] | None = None) -> dict:
    if top_factors is None:
        try:
            from .classify import classify_risk
            top_factors = classify_risk(prices, weights)["top_factors"]
        except Exception as exc:  # noqa: BLE001 - explanation must still work without the model
            log.info("factors unavailable: %s", type(exc).__name__)
            top_factors = []
    en = _portfolio_text(prices, weights, label, top_factors)
    text, translated = _translate(en, lang)
    return {"text": text, "lang": lang if translated else "en", "translated": translated,
            "fallback": lang != "en" and not translated}


def explain_portfolio(prices: pd.DataFrame, weights: dict, label: str, lang: str = "en") -> str:
    """Plain-text explanation (3-5 sentences) of the risk label. Never raises because of the LLM."""
    return explain_portfolio_ex(prices, weights, label, lang)["text"]


# --------------------------------------------------------------------------- allocation explanation
def _change_reason(i: int, up: bool, level: str, base: np.ndarray, rc: np.ndarray, corr_i: np.ndarray,
                   vol_i: np.ndarray, mu_i: np.ndarray, sharpe_i: np.ndarray) -> str:
    avg_corr, med_vol = float(corr_i.mean()), float(np.median(vol_i))
    med_sharpe, med_mu = float(np.median(sharpe_i)), float(np.median(mu_i))
    n = len(base)
    if not up:
        if base[i] > max(0.35, 1.5 / n):
            return f"it was {base[i]:.0%} of the portfolio, a very concentrated position"
        if base[i] > 0 and rc[i] / base[i] >= 1.25 and rc[i] - base[i] > 0.03:
            return f"it drives {rc[i]:.0%} of the portfolio's risk on only {base[i]:.0%} of the money"
        if corr_i[i] > avg_corr + 0.08:
            return f"it moves closely with the rest of the portfolio (average correlation {corr_i[i]:.2f})"
        if sharpe_i[i] < med_sharpe - 0.05:
            return f"it has earned less return per unit of risk (Sharpe {sharpe_i[i]:.2f} vs median {med_sharpe:.2f})"
        if vol_i[i] > 1.1 * med_vol:
            return f"it is more volatile than most holdings ({vol_i[i]:.0%} vs median {med_vol:.0%})"
        return "this rebalances the mix toward the chosen objective"
    order = {"low": ("corr", "vol", "sharpe"), "medium": ("sharpe", "corr", "vol"),
             "high": ("ret", "sharpe", "corr")}.get(level.lower(), ("sharpe", "corr", "vol"))
    for key in order:
        if key == "corr" and corr_i[i] < avg_corr - 0.08:
            return f"it diversifies the mix (average correlation {corr_i[i]:.2f})"
        if key == "vol" and vol_i[i] < 0.9 * med_vol:
            return f"it is calmer than most holdings ({vol_i[i]:.0%} vs median {med_vol:.0%})"
        if key == "sharpe" and sharpe_i[i] > med_sharpe + 0.05:
            if sharpe_i[i] <= 0:
                return (f"it has held up better than most holdings per unit of risk "
                        f"(Sharpe {sharpe_i[i]:.2f} vs median {med_sharpe:.2f}; returns were weak overall)")
            return f"it has earned more return per unit of risk (Sharpe {sharpe_i[i]:.2f} vs median {med_sharpe:.2f})"
        if key == "ret" and mu_i[i] > med_mu + 0.01:
            return f"it has the strongest historical return ({mu_i[i]:.0%} vs median {med_mu:.0%})"
    return "this rebalances the mix toward the chosen objective"


def _allocation_text(prices: pd.DataFrame, original_w: dict, optimized_w: dict, risk_level: str,
                     tradeoffs: str | None) -> str:
    orig = M.normalize_weights(original_w)
    opt = M.normalize_weights(optimized_w)
    names = [c for c in prices.columns if c in orig or c in opt]
    p = M.prepare(prices[names], {t: 1.0 for t in names})
    tickers = p.tickers
    cov = M.lw_covariance(p.logret)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(sd, sd)
    n = len(tickers)
    corr_i = (corr.sum(axis=1) - 1.0) / max(n - 1, 1)
    rf = M.default_rf(tickers)
    mu_i = np.expm1(p.logret.mean(axis=0) * M.TRADING_DAYS)
    sharpe_i = (mu_i - rf) / sd
    base = np.array([orig.get(t, 0.0) for t in tickers])
    base = base / base.sum() if base.sum() > 0 else np.full(n, 1.0 / n)
    new = np.array([opt.get(t, 0.0) for t in tickers])
    new = new / new.sum() if new.sum() > 0 else base
    var = float(base @ cov @ base)
    rc = base * (cov @ base) / var if var > 0 else base

    lines = []
    for i in np.argsort(-np.abs(new - base)):
        d = new[i] - base[i]
        if abs(d) < MIN_CHANGE:
            break
        up = d > 0
        verb = "Increase" if up else "Reduce"
        why = _change_reason(i, up, risk_level, base, rc, corr_i, sd, mu_i, sharpe_i)
        lines.append(f"{verb} {tickers[i]} from {base[i]:.0%} to {new[i]:.0%} ({d * 100:+.0f} points): {why}.")
    if not lines:
        lines.append("The recommended allocation is very close to the current one: no holding changes by more "
                     f"than {MIN_CHANGE * 100:.0f} percentage points.")
    if tradeoffs is None:
        tradeoffs = tradeoffs_text(M.portfolio_metrics(p.prices, dict(zip(tickers, base)), rf=rf),
                                   M.portfolio_metrics(p.prices, dict(zip(tickers, new)), rf=rf),
                                   "your current portfolio")
    return "\n".join(lines) + "\n" + tradeoffs


def explain_allocation_ex(prices: pd.DataFrame, original_w: dict, optimized_w: dict, risk_level: str,
                          lang: str = "en", tradeoffs: str | None = None) -> dict:
    en = _allocation_text(prices, original_w, optimized_w, risk_level, tradeoffs)
    text, translated = _translate(en, lang)
    return {"text": text, "lang": lang if translated else "en", "translated": translated,
            "fallback": lang != "en" and not translated}


def explain_allocation(prices: pd.DataFrame, original_w: dict, optimized_w: dict, risk_level: str,
                       lang: str = "en", tradeoffs: str | None = None) -> str:
    """Per-asset reasons for every weight change over 3 points, plus the trade-off sentence.
    Pass ``tradeoffs=optimize(...)["tradeoffs"]`` to reuse P1's wording; otherwise it is recomputed."""
    return explain_allocation_ex(prices, original_w, optimized_w, risk_level, lang, tradeoffs)["text"]


# --------------------------------------------------------------------------- scenario explanation
def _scenario_text(result: dict, label_before: str | None, label_after: str | None) -> str:
    sc = result.get("scenario", {})
    kind = sc.get("type")
    pct = sc.get("pct")
    desc = {
        "market_shock": f"a market shock of {pct:+.0%}" if pct is not None else "a market shock",
        "recovery": f"a market recovery of {pct:+.0%}" if pct is not None else "a market recovery",
        "vol_multiplier": f"volatility rising to {sc.get('k', 2):g}x its recent level",
        "sector_shock": f"a {sc.get('sector', 'sector')} sector shock of {pct:+.0%}" if pct is not None
        else f"a {sc.get('sector', 'sector')} sector shock",
    }.get(kind, "this scenario")
    chg = float(result.get("value_change_pct", 0.0))
    b, a = result.get("before", {}), result.get("after", {})
    if result.get("value_change_basis") == "trough_of_stress_window":
        first = (f"Under {desc}, the portfolio's value would dip about {abs(chg):.1%} below its starting point "
                 "at the worst moment of the stressed period.")
    else:
        first = f"Under {desc}, the portfolio value would change by about {chg:+.1%}."
    parts = [first]
    shocks = result.get("asset_shocks") or {}
    hit = sorted(((v, k) for k, v in shocks.items() if v < -0.005))[:2]
    if hit:
        parts.append("The hardest hit holdings would be " + " and ".join(f"{k} ({v:+.0%})" for v, k in hit) + ".")
    if b and a:
        parts.append(f"Annual volatility would move from {b['ann_vol']:.0%} to {a['ann_vol']:.0%} and the worst "
                     f"drawdown from {abs(b['max_drawdown']):.0%} to {abs(a['max_drawdown']):.0%}.")
    if label_before and label_after:
        parts.append(f"Its risk rating would change from {label_before} to {label_after}."
                     if label_before != label_after else f"Its risk rating would stay {label_after}.")
    return " ".join(parts)


def explain_scenario_ex(result: dict, label_before: str | None = None, label_after: str | None = None,
                        lang: str = "en") -> dict:
    """``result`` is ``engine.simulate.apply_scenario(...)``; labels come from ``classify_risk`` /
    ``classify_stressed`` (optional). One translation call for non-English."""
    en = _scenario_text(result, label_before, label_after)
    text, translated = _translate(en, lang)
    return {"text": text, "lang": lang if translated else "en", "translated": translated,
            "fallback": lang != "en" and not translated}


def explain_scenario(result: dict, label_before: str | None = None, label_after: str | None = None,
                     lang: str = "en") -> str:
    return explain_scenario_ex(result, label_before, label_after, lang)["text"]
