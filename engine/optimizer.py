"""Allocation optimizer: SciPy SLSQP, long-only, fully invested, per-asset bounds.

Risk levels
* ``Low``    -> minimum variance                                  (mode ``min_variance``)
* ``Medium`` -> maximum Sharpe ratio                              (mode ``max_sharpe``)
              or, when no allocation beats the risk-free rate, equal risk contribution (``risk_parity``)
* ``High``   -> maximise return s.t. vol <= 1.3 x equal-weight vol (mode ``max_return_vol_cap``)
* solver failure / single asset                                   (mode ``fallback_equal``)

Two safeguards against over-fitting noisy history:
* covariance is Ledoit-Wolf shrunk (shared with ``metrics``);
* expected returns are **shrunk 50% toward the cross-sectional mean**. Raw historical means are the
  noisiest input of mean-variance optimisation and push solutions into corners (one asset at the cap,
  the rest at the floor). Shrinking keeps the optimiser driven mostly by risk, which is estimated far
  better than return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import TRADING_DAYS, default_rf, lw_covariance, match_columns, portfolio_metrics, prepare

RETURN_SHRINKAGE = 0.5
FALLBACK_SHRINKAGES = (0.25, 0.0)   # tried in order if the 50% solution scores below equal weight
VOL_CAP_MULTIPLE = 1.3
N_RANDOM_STARTS = 4
SEED = 42
FEASIBILITY_TOL = 1e-6

_LEVELS = {"low": "Low", "medium": "Medium", "med": "Medium", "mid": "Medium", "high": "High"}
_MODES = {"Low": "min_variance", "Medium": "max_sharpe", "High": "max_return_vol_cap"}
_OBJECTIVE_TEXT = {
    "min_variance": "minimum variance (Low risk)",
    "max_sharpe": "maximum Sharpe ratio (Medium risk)",
    "risk_parity": "equal risk contribution (Medium risk)",
    "max_return_vol_cap": "maximum return within a volatility cap of 1.3x equal-weight volatility (High risk)",
    "fallback_equal": "equal weight",
}


# --------------------------------------------------------------------------- helpers
def _project(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Nearest-ish point with sum(w) == 1 and lo <= w <= hi (water-filling on the slack)."""
    w = np.clip(np.asarray(w, float), lo, hi)
    for _ in range(50):
        diff = 1.0 - w.sum()
        if abs(diff) < 1e-13:
            break
        room = (hi - w) if diff > 0 else (w - lo)
        total = room.sum()
        if total <= 1e-15:
            break
        w = np.clip(w + diff * room / total, lo, hi)
    return w


def _max_linear(mu: np.ndarray, lo: float, hi: float) -> float:
    """max mu@w over the bounded simplex (greedy: fill the highest-mu assets first)."""
    w = np.full(len(mu), lo)
    remaining = 1.0 - w.sum()
    for i in np.argsort(-mu):
        add = min(hi - lo, remaining)
        w[i] += add
        remaining -= add
        if remaining <= 1e-15:
            break
    return float(mu @ w)


def _relax_bounds(n: int, lo: float, hi: float) -> tuple[float, float, str]:
    note = ""
    if n * lo > 1 + 1e-12:
        new_lo = 0.5 / n
        note += f"Minimum weight relaxed from {lo:.0%} to {new_lo:.1%} because {n} assets cannot each hold {lo:.0%}. "
        lo = new_lo
    if n * hi < 1 - 1e-12:
        new_hi = min(1.0, 1.5 / n)
        note += f"Maximum weight relaxed from {hi:.0%} to {new_hi:.0%} because {n} assets cannot be fully invested under a {hi:.0%} cap. "
        hi = new_hi
    return lo, hi, note.strip()


def _solve(mode: str, mu: np.ndarray, cov: np.ndarray, rf: float, lo: float, hi: float,
           cap: float, seed: int) -> np.ndarray | None:
    n = len(mu)
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n)] + [_project(rng.dirichlet(np.ones(n)), lo, hi) for _ in range(N_RANDOM_STARTS)]

    if mode == "min_variance":
        fun = lambda w: 100.0 * (w @ cov @ w)
        jac = lambda w: 200.0 * (cov @ w)
        extra: list[dict] = []
        score = lambda w: float(w @ cov @ w)
    elif mode == "max_sharpe":
        # exactly the reported Sharpe: (exp(mu'w) - 1 - rf) / sqrt(w' C w)
        def fun(w):
            s = np.sqrt(max(w @ cov @ w, 1e-16))
            return -(np.expm1(mu @ w) - rf) / s

        def jac(w):
            cw = cov @ w
            s = np.sqrt(max(w @ cw, 1e-16))
            m = mu @ w
            return -(np.exp(m) * mu / s - (np.expm1(m) - rf) * cw / s ** 3)
        extra = []
        score = fun
    elif mode == "risk_parity":
        # every asset contributes the same share of portfolio variance: a balanced middle ground between
        # minimum variance (Low) and maximum return (High) that needs no return forecast at all
        def fun(w):
            cw = cov @ w
            var = max(float(w @ cw), 1e-16)
            return 1e4 * float(np.sum((w * cw / var - 1.0 / n) ** 2))
        jac = None
        extra = []
        score = fun
    else:  # max_return_vol_cap
        fun = lambda w: -float(mu @ w)
        jac = lambda w: -mu
        extra = [{"type": "ineq", "fun": lambda w: cap ** 2 - w @ cov @ w, "jac": lambda w: -2.0 * (cov @ w)}]
        score = fun

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}] + extra
    best, best_score = None, np.inf
    for x0 in starts:
        try:
            res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=[(lo, hi)] * n, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-12})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        w = _project(res.x, lo, hi)
        if mode == "max_return_vol_cap" and np.sqrt(w @ cov @ w) > cap * (1 + FEASIBILITY_TOL):
            continue
        s = score(w)
        if s < best_score - 1e-14:
            best, best_score = w, s
    if best is None:
        return None                       # no valid solver run -> caller falls back to equal weight
    ew = _project(np.full(n, 1.0 / n), lo, hi)   # feasible by construction (bounds are relaxed first)
    return ew if score(ew) <= best_score else best     # never return something worse than equal weight


# --------------------------------------------------------------------------- explanations
def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _asset_reason(i: int, direction: str, mode: str, base: np.ndarray, rc: np.ndarray, corr_i: np.ndarray,
                  avg_corr: float, vol_i: np.ndarray, mu: np.ndarray, sharpe_i: np.ndarray) -> str:
    med_sharpe, med_vol, med_mu = np.median(sharpe_i), np.median(vol_i), np.median(mu)
    if direction == "down":
        if base[i] > 0 and rc[i] / base[i] >= 1.25 and rc[i] - base[i] > 0.03:
            return f"it drives {_pct(rc[i])} of portfolio risk on {_pct(base[i])} of the money"
        if corr_i[i] > avg_corr + 0.08:
            return f"it moves closely with the rest of the portfolio (average correlation {corr_i[i]:.2f})"
        if sharpe_i[i] < med_sharpe - 0.05:
            return f"it earns less return per unit of risk (Sharpe {sharpe_i[i]:.2f} vs median {med_sharpe:.2f})"
        if vol_i[i] > 1.1 * med_vol:
            return f"it is more volatile than most holdings ({vol_i[i]:.0%} vs median {med_vol:.0%})"
        return "this rebalances toward the objective"
    order = ("ret", "sharpe", "corr", "vol") if mode == "max_return_vol_cap" else \
            ("sharpe", "corr", "vol") if mode == "max_sharpe" else ("corr", "vol", "sharpe")
    for key in order:
        if key == "ret" and mu[i] > med_mu + 0.01:
            return f"it has the strongest expected return ({mu[i]:.0%} vs median {med_mu:.0%})"
        if key == "sharpe" and sharpe_i[i] > med_sharpe + 0.05:
            return f"it earns more return per unit of risk (Sharpe {sharpe_i[i]:.2f} vs median {med_sharpe:.2f})"
        if key == "corr" and corr_i[i] < avg_corr - 0.08:
            return f"it diversifies the mix (average correlation {corr_i[i]:.2f})"
        if key == "vol" and vol_i[i] < 0.9 * med_vol:
            return f"it is calmer than most holdings ({vol_i[i]:.0%} vs median {med_vol:.0%})"
    return "this rebalances toward the objective"


def tradeoffs_text(m0: dict, m1: dict, baseline: str) -> str:
    """Plain-English trade-off sentence(s) for before/after metrics (also used by ml/explain.py)."""
    dv, dr = m1["ann_vol"] - m0["ann_vol"], m1["ann_return"] - m0["ann_return"]
    dd = m1["max_drawdown"] - m0["max_drawdown"]
    pts = lambda x: f"{abs(x) * 100:.1f} points"
    if abs(dv) < 0.0025 and abs(dr) < 0.0025:
        return f"The recommended mix is already close to {baseline}: volatility and return barely change."
    dd_txt = f" The worst past drawdown moves from {m0['max_drawdown']:.0%} to {m1['max_drawdown']:.0%}." if abs(dd) >= 0.005 else ""
    if dv < 0 and dr < 0:
        s = f"Lower risk costs return: volatility falls {pts(dv)} ({m0['ann_vol']:.1%} to {m1['ann_vol']:.1%}) while historical annual return falls {pts(dr)} ({m0['ann_return']:.1%} to {m1['ann_return']:.1%})."
    elif dv <= 0 <= dr:
        s = f"Better on both counts: volatility falls {pts(dv)} and historical annual return rises {pts(dr)}."
    elif dv > 0 and dr > 0:
        s = f"More return needs more risk: volatility rises {pts(dv)} ({m0['ann_vol']:.1%} to {m1['ann_vol']:.1%}) for {pts(dr)} more historical annual return."
    else:
        s = f"Volatility rises {pts(dv)} while historical annual return falls {pts(dr)}; the mix is rebalanced toward the stated objective."
    return s + dd_txt


# --------------------------------------------------------------------------- public API
def optimize(prices: pd.DataFrame, risk_level: str, bounds: tuple[float, float] = (0.05, 0.40),
             current_weights: dict | None = None, rf: float | None = None, seed: int = SEED) -> dict:
    """Recommend weights for ``risk_level`` in {"Low","Medium","High"} (case-insensitive).

    ``current_weights`` (optional) is the user's portfolio, used as the "before" in the explanation;
    equal weight is used when omitted. Returns ``{"weights","metrics","reason","tradeoffs","mode"}`` plus
    ``before_weights``, ``before_metrics``, ``bounds_used``, ``risk_level``.
    """
    level = _LEVELS.get(str(risk_level).strip().lower())
    if level is None:
        raise ValueError(f"risk_level must be Low, Medium or High (got {risk_level!r})")
    lo_b, hi_b = float(bounds[0]), float(bounds[1])
    if not (0.0 <= lo_b <= hi_b <= 1.0 and hi_b > 0.0):
        raise ValueError(f"bounds must satisfy 0 <= min <= max <= 1 (got {bounds!r})")
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("no price data to optimise")
    p = prepare(prices, {c: 1.0 for c in prices.columns})
    tickers, n = p.tickers, len(p.tickers)
    cov = lw_covariance(p.logret)
    mu_hist = p.logret.mean(axis=0) * TRADING_DAYS
    rf = default_rf(tickers) if rf is None else float(rf)
    lo, hi, relax_note = _relax_bounds(n, lo_b, hi_b)

    # "before" portfolio for the explanation
    ew = np.full(n, 1.0 / n)
    if current_weights:
        cw = match_columns({str(k): v for k, v in current_weights.items()}, tickers)
        raw = np.array([max(float(cw.get(t, 0.0) or 0.0), 0.0) for t in tickers])
        base = raw / raw.sum() if raw.sum() > 0 else ew
        baseline_label = "your current portfolio"
    else:
        base, baseline_label = ew, "an equal-weight portfolio"
    base_w = dict(zip(tickers, base.tolist()))

    mode = _MODES[level]
    shrinkage_used = RETURN_SHRINKAGE
    mu = (1 - RETURN_SHRINKAGE) * mu_hist + RETURN_SHRINKAGE * mu_hist.mean()   # replaced if the loop below relaxes it
    notes: list[str] = [relax_note] if relax_note else []
    w: np.ndarray | None = None
    if n == 1:
        w, mode = np.array([1.0]), "fallback_equal"
        notes.append("Only one asset was provided, so there is nothing to allocate.")
    else:
        vol_ew = float(np.sqrt(ew @ cov @ ew))
        cap = VOL_CAP_MULTIPLE * vol_ew
        if mode == "max_sharpe" and np.expm1(_max_linear(mu_hist, lo, hi)) <= rf:
            mode = "risk_parity"
            notes.append("No allocation beats the risk-free rate on historical return, so maximum Sharpe is not "
                         "meaningful; an equal-risk-contribution mix is used for Medium instead.")

        def hist_sharpe(x: np.ndarray) -> float:      # identical to the reported Sharpe
            v = float(np.sqrt(x @ cov @ x))
            return (float(np.expm1(mu_hist @ x)) - rf) / v if v > 1e-12 else 0.0

        # Return shrinkage: start at 50%. For max-Sharpe, if that cautious solution still scores below
        # equal weight on the historical Sharpe it is reported against, relax the shrinkage (25%, then 0%)
        # so the recommendation is never worse than equal weight on the numbers the user sees.
        levels = (RETURN_SHRINKAGE,) + (FALLBACK_SHRINKAGES if mode == "max_sharpe" else ())
        for shrink in levels:
            mu = (1 - shrink) * mu_hist + shrink * mu_hist.mean()
            w = _solve(mode, mu, cov, rf, lo, hi, cap, seed)
            if w is None or mode != "max_sharpe" or hist_sharpe(w) >= hist_sharpe(ew) - 1e-9:
                break
        shrinkage_used = shrink
        if w is None:
            w, mode = _project(ew, lo, hi), "fallback_equal"
            notes.append("The optimiser could not find a valid solution, so equal weights are shown.")
        elif shrinkage_used != RETURN_SHRINKAGE:
            notes.append(f"Expected returns were shrunk {shrinkage_used:.0%} instead of {RETURN_SHRINKAGE:.0%} "
                         "because the more cautious mix scored below equal weight on historical Sharpe.")
    w = _project(w, lo, hi) if n > 1 else w
    weights = dict(zip(tickers, [float(x) for x in w]))

    px = p.prices
    m1 = portfolio_metrics(px, weights, rf=rf)
    m0 = portfolio_metrics(px, base_w, rf=rf)

    # per-asset diagnostics (computed on the "before" portfolio)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(sd, sd)
    corr_i = (corr.sum(axis=1) - 1.0) / max(n - 1, 1)
    base_var = float(base @ cov @ base)
    rc = base * (cov @ base) / base_var if base_var > 0 else base
    sharpe_i = (mu - rf) / sd
    avg_corr_all = float(corr_i.mean())

    delta = w - base
    top = [i for i in np.argsort(-np.abs(delta))[:3] if abs(delta[i]) >= 0.01]
    changes = []
    for i in top:
        direction = "up" if delta[i] > 0 else "down"
        why = _asset_reason(i, "up" if direction == "up" else "down", mode, base, rc, corr_i, avg_corr_all, sd, mu, sharpe_i)
        verb = "raised" if direction == "up" else "reduced"
        changes.append(f"{tickers[i]} {verb} from {_pct(base[i])} to {_pct(w[i])} because {why}")

    reason = (f"Objective: {_OBJECTIVE_TEXT[mode]}. Compared with {baseline_label}, volatility goes from "
              f"{m0['ann_vol']:.1%} to {m1['ann_vol']:.1%} and historical annual return from "
              f"{m0['ann_return']:.1%} to {m1['ann_return']:.1%}.")
    reason += (" Biggest changes: " + "; ".join(changes) + ".") if changes else " No holding changes by more than 1 point."
    if notes:
        reason += " " + " ".join(notes)

    return {
        "weights": weights, "metrics": m1, "reason": reason,
        "tradeoffs": tradeoffs_text(m0, m1, baseline_label), "mode": mode,
        "before_weights": base_w, "before_metrics": m0,
        "bounds_used": (float(lo), float(hi)), "risk_level": level, "return_shrinkage": shrinkage_used,
    }
