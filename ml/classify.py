"""Risk classification: ``classify_risk(prices, weights)`` -> Low / Medium / High with probabilities and the
factors that drove the call.

Model loading order: process memory -> local ``ml/models`` -> Supabase Storage -> quick retrain from the
dataset. Load happens once per process (P3 can additionally wrap :func:`load_model` in
``@st.cache_resource``).

Factor attribution is *occlusion*: replace one feature at a time by its training median and see how the
model's expected risk class moves. It is model-based, cheap (one batch predict) and needs no SHAP.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, LOOKBACK_ROWS, build_features, lookback_slice, to_vector
from . import train as _train

log = logging.getLogger("ml.classify")

MODEL_DIR = _train.MODEL_DIR
MODEL_FILE, META_FILE = _train.MODEL_PATH.name, _train.META_PATH.name
LABELS = ["Low", "Medium", "High"]
MIN_ROWS = 60                 # below this a one-year-style feature set is meaningless
SHORT_HISTORY_ROWS = 126

_lock = threading.Lock()
_loaded: dict = {}

# feature -> (display name, formatter)
_pct = lambda v: f"{v:.0%}"
_DISPLAY = {
    "vol_21": ("Volatility, last month", _pct), "vol_63": ("Volatility, last 3 months", _pct),
    "vol_126": ("Volatility, last 6 months", _pct), "ewma_vol": ("Recent-weighted volatility", _pct),
    "vol_ratio_21": ("Recent vs 1-year volatility", lambda v: f"{v:.2f}x"),
    "vol_ratio_63": ("3-month vs 1-year volatility", lambda v: f"{v:.2f}x"),
    "vol_of_vol": ("Volatility instability", lambda v: f"{v:.2f}"),
    "ret_21": ("1-month return", lambda v: f"{v:+.0%}"), "ret_63": ("3-month return", lambda v: f"{v:+.0%}"),
    "cur_drawdown": ("Current drop from peak", lambda v: f"{abs(v):.0%}"),
    "max_asset_vol": ("Most volatile holding", _pct), "undiversified_vol": ("Average holding volatility", _pct),
    "diversification_ratio": ("Diversification ratio", lambda v: f"{v:.2f}"),
    "avg_corr_63": ("Recent average correlation", lambda v: f"{v:.2f}"),
    "corr_shift": ("Correlation change (3m vs 1y)", lambda v: f"{v:+.2f}"),
    "sector_hhi": ("Sector concentration", lambda v: f"{v:.2f}"),
    "market_vol_21": ("Market volatility, last month", _pct),
    "market_vol_ratio": ("Market stress vs normal", lambda v: f"{v:.2f}x"),
    "market_drawdown": ("Market drop from peak", lambda v: f"{abs(v):.0%}"),
    "beta_to_market": ("Sensitivity to the market (beta)", lambda v: f"{v:.2f}"),
    "ann_vol": ("Volatility", lambda v: f"{v:.0%}"),
    "max_drawdown": ("Max drawdown", lambda v: f"{abs(v):.0%}"),
    "downside_dev": ("Downside volatility", lambda v: f"{v:.0%}"),
    "ann_return": ("Annual return", lambda v: f"{v:+.0%}"),
    "sharpe": ("Sharpe ratio", lambda v: f"{v:.2f}"),
    "avg_corr": ("Average correlation", lambda v: f"{v:.2f}"),
    "hhi": ("Concentration (HHI)", lambda v: f"{v:.2f}"),
    "effective_n": ("Effective no. of assets", lambda v: f"{v:.1f}"),
    "top_weight": ("Largest holding", lambda v: f"{v:.0%}"),
    "skew": ("Return skew", lambda v: f"{v:+.2f}"),
    "kurtosis": ("Fat tails (kurtosis)", lambda v: f"{v:.1f}"),
    "n_assets": ("Number of assets", lambda v: f"{int(round(v))}"),
}


# --------------------------------------------------------------------------- loading
class _BoosterModel:
    """``predict_proba`` over a raw LightGBM booster (loaded from its text dump)."""

    def __init__(self, booster_str: str):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_str=booster_str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(np.asarray(X, dtype="float64")))


def _read_local(model_dir: Path) -> dict | None:
    try:
        bundle = joblib.load(model_dir / MODEL_FILE)
        meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        if list(meta.get("feature_names", [])) != list(bundle["feature_names"]):
            return None
        model = _BoosterModel(bundle["booster_str"]) if "booster_str" in bundle else bundle["model"]
        return {"model": model, "meta": meta, "names": bundle["feature_names"]}
    except Exception as exc:  # noqa: BLE001 - corrupt/missing file -> try the next source
        log.info("local model not usable: %s", type(exc).__name__)
        return None


def load_model(force: bool = False, model_dir: Path | None = None, allow_retrain: bool = True) -> dict:
    """``{"model", "meta", "names", "source"}``; source is "local" | "storage" | "retrained"."""
    global _loaded
    with _lock:
        if _loaded and not force:
            return _loaded
        model_dir = Path(model_dir or MODEL_DIR)
        got = _read_local(model_dir)
        source = "local"
        if got is None:
            from engine import db
            if db.download_model(MODEL_FILE, dest_dir=model_dir) and db.download_model(META_FILE, dest_dir=model_dir):
                got, source = _read_local(model_dir), "storage"
        if got is None and allow_retrain:
            log.warning("no model found locally or in Storage: retraining a quick model")
            _train.train(quick=True, upload=False, model_dir=model_dir)
            got, source = _read_local(model_dir), "retrained"
        if got is None:
            raise FileNotFoundError("risk model unavailable (no local file, no Storage copy, retrain disabled)")
        got["source"] = source
        _loaded = got
        return got


def model_info() -> dict:
    """Metadata of the loaded model (loads it if needed): training dates, metrics, thresholds, source."""
    m = load_model()
    return {**m["meta"], "source": m["source"]}


# --------------------------------------------------------------------------- inference
def _top_factors(bundle: dict, x: np.ndarray, proba: np.ndarray, k: int = 3) -> list[dict]:
    meta, model, names = bundle["meta"], bundle["model"], bundle["names"]
    med = np.array([meta["train_medians"][n] for n in names])
    batch = np.tile(x, (len(names), 1))
    for j in range(len(names)):
        batch[j, j] = med[j]
    base_risk = float(proba @ np.array([0.0, 1.0, 2.0]))
    occluded = model.predict_proba(batch) @ np.array([0.0, 1.0, 2.0])
    effect = base_risk - occluded                      # >0: this feature pushes the portfolio toward higher risk
    out = []
    for j in np.argsort(-np.abs(effect))[:k]:
        n = names[j]
        label, fmt = _DISPLAY.get(n, (n, lambda v: f"{v:.2f}"))
        out.append({"name": label, "value": fmt(float(x[j])),
                    "impact": "raises risk" if effect[j] > 0 else "lowers risk",
                    "feature": n, "effect": float(effect[j])})
    return out


def classify_risk(prices: pd.DataFrame, weights: dict) -> dict:
    """``{"label", "probs", "top_factors", "features", "window_rows", "note"}`` for a portfolio.

    Uses the trailing 252 trading days of ``prices`` (exactly the training window). Needs >= 60 rows.
    ``top_factors``: three dicts ``{"name", "value", "impact"}`` (+ ``feature``, ``effect``).
    """
    n_rows = len(prices.dropna(how="any"))
    if n_rows < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} trading days of data to classify (got {n_rows})")
    bundle = load_model()
    feats = build_features(prices, weights)                            # market features may be NaN (offline)
    x = to_vector(feats, bundle["names"], fill=bundle["meta"]["train_medians"])
    proba = bundle["model"].predict_proba(x[None, :])[0]
    probs = {LABELS[i]: float(p) for i, p in enumerate(proba)}
    note = ""
    if n_rows < SHORT_HISTORY_ROWS:
        note = "Less than 6 months of history: treat this rating as indicative only."
    elif n_rows < LOOKBACK_ROWS:
        note = "Less than a full year of history was available for the features."
    safe = {k: (float(v) if np.isfinite(v) else None) for k, v in feats.items()}     # NaN is not valid JSON
    return {"label": max(probs, key=probs.get), "probs": probs, "top_factors": _top_factors(bundle, x, proba),
            "features": safe, "window_rows": min(n_rows, LOOKBACK_ROWS), "note": note}


def classify_stressed(stressed_prices: pd.DataFrame, weights: dict) -> dict:
    """Re-label a stressed portfolio (``engine.simulate.apply_scenario(...)["stressed_prices"]``)."""
    out = classify_risk(stressed_prices, weights)
    out["stressed"] = True
    return out
