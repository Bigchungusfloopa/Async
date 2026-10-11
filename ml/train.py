"""Train, evaluate honestly, and store the risk classifier.

Protocol (no leakage anywhere):
* split by *time*: earlier windows train, the latest 25% of windows test, with a purge so no training label's
  forward window reaches into the test period;
* validation slice for LightGBM early stopping = the END of the training period (again purged);
* the label is the portfolio's realised forward volatility cut at fixed thresholds (``dataset.ABS_THRESHOLDS``,
  chosen from the earliest part of history and frozen into ``feature_meta.json``);
* the test period is touched exactly once, at the end;
* compared with honest baselines: majority class, and one-line rules on trailing volatility (1-year and EWMA);
* ordered-class metrics: macro-F1, balanced accuracy, within-one-class accuracy, quadratic weighted kappa;
  plus an equity-heavy subset, because bond/cash ETFs make the Low class easy.

Run:  ``python -m ml.train``  (``--quick`` for a fast, smaller model)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)

from . import dataset
from .features import FEATURE_NAMES

log = logging.getLogger("ml.train")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "ml" / "models"
REPORT_DIR = ROOT / "ml" / "reports"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
META_PATH = MODEL_DIR / "feature_meta.json"
LABELS = list(dataset.LABELS)
SEED = 42
EQUITY_HEAVY = 0.8
# regularised: overlapping windows make rows highly correlated, so keep trees small and leaves heavy
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=-1)


def _xy(df: pd.DataFrame, labeler: dict) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURE_NAMES].to_numpy(dtype="float64"), dataset.apply_labels(df, labeler)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "within_one_class": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)),
        "opposite_extreme": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) == 2)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2])),
        "recall": {LABELS[i]: float(r) for i, r in
                   enumerate(recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0))},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def _bootstrap_delta(test: pd.DataFrame, y: np.ndarray, p_model: np.ndarray, p_rule: np.ndarray,
                     n: int = 300, seed: int = SEED) -> dict:
    """Window-level bootstrap of macro-F1(model) - macro-F1(rule). Windows, not rows, are resampled because
    portfolios inside one window share the same market period."""
    rng = np.random.default_rng(seed)
    wid = test["window_id"].to_numpy()
    groups = {w: np.flatnonzero(wid == w) for w in np.unique(wid)}
    keys = list(groups)
    deltas = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        a = f1_score(y[idx], p_model[idx], average="macro", labels=[0, 1, 2], zero_division=0)
        b = f1_score(y[idx], p_rule[idx], average="macro", labels=[0, 1, 2], zero_division=0)
        deltas.append(a - b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"mean": float(np.mean(deltas)), "ci95": [float(lo), float(hi)], "n_boot": n}


def _plot_confusion(cm: list, title: str, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual (future realised risk)")
    ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _bin(values: pd.Series | np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype="float64"), thresholds)


def train(df: pd.DataFrame | None = None, quick: bool = False, upload: bool = True, seed: int = SEED,
          model_dir: Path | None = None, report_dir: Path | None = None,
          labeled_path: Path | None = None, save_labeled: bool = True) -> dict:
    """Train + evaluate + save. Returns the report dict (also written to ``ml/reports/metrics.json``)."""
    import lightgbm as lgb

    t0 = time.time()
    model_dir, report_dir = Path(model_dir or MODEL_DIR), Path(report_dir or REPORT_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = dataset.build_dataset(target_rows=3000 if quick else 8000, seed=seed)

    train_all, test = dataset.time_split(df, test_frac=0.25)
    try:                                                               # validation = end of the training period
        fit_df, val_df = dataset.time_split(train_all, test_frac=0.2)
    except ValueError:                                                 # too little history for a purged slice
        fit_df, val_df = train_all, None
    labeler = dataset.fit_labeler(train_all, mode="abs_vol", horizon=dataset.HORIZON,
                                  thresholds=dataset.ABS_THRESHOLDS)   # fixed by definition, frozen
    thr = labeler["thresholds"]
    X_all, y_all = _xy(train_all, labeler)
    X_fit, y_fit = _xy(fit_df, labeler)
    X_val, y_val = _xy(val_df, labeler) if val_df is not None else (None, None)
    X_te, y_te = _xy(test, labeler)

    # ---- baselines: nothing learned, same thresholds as the label
    majority = int(np.bincount(y_all, minlength=3).argmax())
    p_major = np.full(len(y_te), majority)
    p_rule = _bin(test["ann_vol"], thr)                                # trailing 1-year volatility
    p_ewma = _bin(test["ewma_vol"], thr)                               # EWMA (RiskMetrics) volatility

    # ---- RandomForest baseline
    rf = RandomForestClassifier(n_estimators=100 if quick else 300, class_weight="balanced", min_samples_leaf=5,
                                n_jobs=-1, random_state=seed).fit(np.nan_to_num(X_all), y_all)
    p_rf = rf.predict(np.nan_to_num(X_te))

    # ---- LightGBM main model: early stopping on the validation slice, then refit on all training data
    n_trees = 150 if quick else 400
    params = {**LGB_PARAMS, "random_state": seed}
    if val_df is not None:
        es = lgb.LGBMClassifier(n_estimators=n_trees, **params)
        es.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        best_iter = max(20, int(es.best_iteration_ or n_trees))
    else:
        best_iter = max(20, n_trees // 4)                              # no purged validation slice available
    model = lgb.LGBMClassifier(n_estimators=best_iter, **params).fit(X_all, y_all)
    proba_te = model.predict_proba(X_te)
    p_lgb = proba_te.argmax(axis=1)

    lgb_s, rf_s = _scores(y_te, p_lgb), _scores(y_te, p_rf)
    major_s, rule_s, ewma_s = _scores(y_te, p_major), _scores(y_te, p_rule), _scores(y_te, p_ewma)
    best_rule_name, p_best_rule = ("ewma_volatility_rule", p_ewma) \
        if ewma_s["macro_f1"] >= rule_s["macro_f1"] else ("volatility_rule_1y", p_rule)
    delta = _bootstrap_delta(test.reset_index(drop=True), y_te, p_lgb, p_best_rule)
    gain = model.booster_.feature_importance("gain")
    importance = dict(zip(FEATURE_NAMES, (gain / max(gain.sum(), 1e-12)).tolist()))

    eq = (test["equity_share"].to_numpy() >= EQUITY_HEAVY) if "equity_share" in test else np.zeros(len(test), bool)
    equity_heavy = None
    if eq.sum() >= 50:
        equity_heavy = {"rows": int(eq.sum()), "lightgbm": _scores(y_te[eq], p_lgb[eq]),
                        "volatility_rule_1y": _scores(y_te[eq], p_rule[eq]),
                        "ewma_volatility_rule": _scores(y_te[eq], p_ewma[eq]),
                        "class_mix": np.round(np.bincount(y_te[eq], minlength=3) / eq.sum(), 3).tolist()}

    split_info = {
        "train_window_start": str(train_all["window_start"].min().date()),
        "train_label_end_max": str(train_all["label_end"].max().date()),
        "test_window_start": str(test["window_start"].min().date()),
        "test_window_end_max": str(test["window_end"].max().date()),
        "test_label_end_max": str(test["label_end"].max().date()),
        "rows": {"train": len(train_all), "fit": len(fit_df), "validation": 0 if val_df is None else len(val_df),
                 "test": len(test), "dropped_by_purge": int(len(df) - len(train_all) - len(test))},
        "windows": {"train": int(train_all["window_id"].nunique()), "test": int(test["window_id"].nunique())},
        "horizon_days": dataset.HORIZON, "lookback_returns": 252,
    }
    class_dist = {"train": np.round(np.bincount(y_all, minlength=3) / len(y_all), 3).tolist(),
                  "test": np.round(np.bincount(y_te, minlength=3) / len(y_te), 3).tolist(), "labels": LABELS}
    verdict = (f"LightGBM beats the {best_rule_name.replace('_', ' ')} on macro-F1 (bootstrap 95% CI excludes 0)"
               if delta["ci95"][0] > 0 else
               f"LightGBM is NOT clearly better than the {best_rule_name.replace('_', ' ')} (bootstrap 95% CI "
               "includes 0): most of the signal is simply 'recent volatility persists'")
    report = {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
        "quick": quick, "best_iteration": best_iter, "early_stopping": val_df is not None,
        "data_end": df.attrs.get("data_end"), "label": {k: labeler[k] for k in ("mode", "horizon", "thresholds")},
        "split": split_info, "class_distribution": class_dist,
        "models": {"lightgbm": lgb_s, "random_forest": rf_s},
        "baselines": {"majority_class": {"predicts": LABELS[majority], **major_s},
                      "volatility_rule_1y": rule_s, "ewma_volatility_rule": ewma_s},
        "lightgbm_minus_best_rule_macro_f1": {"rule": best_rule_name, **delta}, "verdict": verdict,
        "equity_heavy_subset": equity_heavy,
        "feature_importance_gain": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _plot_confusion(lgb_s["confusion_matrix"], f"LightGBM, test period (macro-F1 {lgb_s['macro_f1']:.2f})",
                    report_dir / "confusion_matrix.png")

    med = np.nanmedian(X_all, axis=0)
    std = np.nanstd(X_all, axis=0)
    meta = {
        "feature_names": FEATURE_NAMES, "labels": LABELS, "labeler": labeler, "model": "lightgbm", "format": 3,
        "lookback_returns": 252, "horizon_days": dataset.HORIZON,
        # NaN-safe: market features may be missing at inference time and are imputed with these medians
        "train_medians": {n: (float(v) if np.isfinite(v) else 0.0) for n, v in zip(FEATURE_NAMES, med)},
        "train_std": {n: (float(v) if np.isfinite(v) and v > 0 else 1.0) for n, v in zip(FEATURE_NAMES, std)},
        "feature_importance_gain": importance, "split": split_info, "class_distribution": class_dist,
        "metrics": {"macro_f1": lgb_s["macro_f1"], "accuracy": lgb_s["accuracy"],
                    "within_one_class": lgb_s["within_one_class"], "quadratic_kappa": lgb_s["quadratic_kappa"],
                    "rule_1y_macro_f1": rule_s["macro_f1"], "ewma_rule_macro_f1": ewma_s["macro_f1"],
                    "majority_macro_f1": major_s["macro_f1"]},
        "data_end": df.attrs.get("data_end"), "trained_at": report["trained_at"], "quick": quick,
    }
    # Version-stable on purpose: the booster's text dump plus plain lists. No sklearn / lightgbm class pickles,
    # so a different library version on the deploy target cannot break loading.
    joblib.dump({"booster_str": model.booster_.model_to_string(), "feature_names": FEATURE_NAMES,
                 "labels": LABELS, "format": 3}, model_dir / MODEL_PATH.name, compress=3)
    (model_dir / META_PATH.name).write_text(json.dumps(meta, indent=1), encoding="utf-8")

    if save_labeled:
        labeled = df.copy()
        labeled["split"] = "purged"
        labeled.loc[train_all.index, "split"] = "train"
        labeled.loc[test.index, "split"] = "test"
        labeled["label"] = np.nan
        labeled.loc[train_all.index, "label"] = y_all
        labeled.loc[test.index, "label"] = y_te
        labeled.to_parquet(labeled_path or dataset.DATASET_PATH)

    if upload:
        from engine import db
        for f in (model_dir / MODEL_PATH.name, model_dir / META_PATH.name):
            report.setdefault("uploaded", {})[f.name] = bool(db.upload_model(f))
    log.info("trained in %.1fs: macro-F1 lgbm %.3f | ewma rule %.3f | 1y rule %.3f | majority %.3f",
             time.time() - t0, lgb_s["macro_f1"], ewma_s["macro_f1"], rule_s["macro_f1"], major_s["macro_f1"])
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller dataset and fewer trees")
    ap.add_argument("--no-upload", action="store_true", help="skip the Supabase Storage upload")
    args = ap.parse_args()
    r = train(quick=args.quick, upload=not args.no_upload)
    print("label:", r["label"], "| class mix train/test:", r["class_distribution"]["train"], r["class_distribution"]["test"])
    print(f"\n{'':<24}{'macro-F1':>9}{'bal.acc':>9}{'accuracy':>9}{'within-1':>9}{'QW-kappa':>9}")
    for name, s in {**r["models"], **r["baselines"]}.items():
        print(f"{name:<24}{s['macro_f1']:>9.3f}{s['balanced_accuracy']:>9.3f}{s['accuracy']:>9.3f}"
              f"{s['within_one_class']:>9.3f}{s['quadratic_kappa']:>9.3f}")
    if r["equity_heavy_subset"]:
        e = r["equity_heavy_subset"]
        print(f"\nequity-heavy subset (>= {EQUITY_HEAVY:.0%} stocks, n={e['rows']}, mix {e['class_mix']}):")
        for name in ("lightgbm", "volatility_rule_1y", "ewma_volatility_rule"):
            s = e[name]
            print(f"  {name:<22}{s['macro_f1']:>9.3f}{s['balanced_accuracy']:>9.3f}{s['accuracy']:>9.3f}"
                  f"{s['within_one_class']:>9.3f}{s['quadratic_kappa']:>9.3f}")
    print("\nLightGBM minus best rule (macro-F1):", r["lightgbm_minus_best_rule_macro_f1"])
    print("verdict:", r["verdict"])
    print("top features:", {k: round(v, 3) for k, v in list(r["feature_importance_gain"].items())[:8]})
    print("uploaded:", r.get("uploaded"))


if __name__ == "__main__":
    main()
