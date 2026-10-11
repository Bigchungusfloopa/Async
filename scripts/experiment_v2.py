"""Accuracy experiment v2: labels x features x model form x horizon, with honest evaluation.

* Walk-forward, 4 consecutive test blocks (expanding train, purged). Folds 1-3 are the DESIGN folds (all
  decisions come from their mean); fold 4 (the most recent ~2.4 years) is the untouched HOLDOUT.
* Same fixed, regularised LightGBM settings for every learned variant; label thresholds fitted per fold on
  training data only (or fixed by definition).
* Metrics for ordered classes: macro-F1, balanced accuracy, accuracy, within-one-class accuracy, quadratic
  weighted kappa, and Spearman rank correlation of the predicted vs realised risk.

    python scripts/experiment_v2.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402

from ml import dataset  # noqa: E402
from ml.features import BASE_FEATURES, SHORT_TERM_FEATURES, ALL_FEATURES  # noqa: E402

N_FOLDS, DESIGN = 4, 3
LGB = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=-1)
FEATSETS = {"base12": BASE_FEATURES, "base+short": BASE_FEATURES + SHORT_TERM_FEATURES, "all": ALL_FEATURES}
RULE_VARS = {"vol252": "ann_vol", "vol63": "vol_63", "vol21": "vol_21", "ewma": "ewma_vol"}


def folds(df):
    ids = np.sort(df["window_id"].unique())
    later = ids[int(len(ids) * 0.4):]
    for block in np.array_split(later, N_FOLDS):
        test = df[df["window_id"].isin(block)]
        train = df[(df["window_id"] < block.min()) & (df["label_end"] < test["window_start"].min())]
        yield train, test


def metrics(y, pred, rank_pred, rank_true):
    return {"macroF1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
            "balAcc": balanced_accuracy_score(y, pred), "acc": accuracy_score(y, pred),
            "within1": float(np.mean(np.abs(y - pred) <= 1)),
            "QWK": cohen_kappa_score(y, pred, weights="quadratic", labels=[0, 1, 2]),
            "rank": float(spearmanr(rank_pred, rank_true).statistic)}


def main() -> None:
    df = dataset.load_dataset()
    if df is None or "vol_21" not in df:
        raise SystemExit("regenerate the dataset first: python -m ml.build_dataset (or ml.train)")
    ids = np.sort(df.window_id.unique())
    early = df[df.window_id <= ids[int(len(ids) * 0.4)]]                 # only the first 40% of history
    t12, t18 = np.round(np.quantile(early["fwd_vol"], [1 / 3, 2 / 3]), 2)
    print(f"terciles of forward vol in the earliest 40% of history, rounded: {t12:.2f} / {t18:.2f}")
    LABELS = {                          # name -> (fit_labeler kwargs, forward-vol column)
        "A composite(126d)": dict(mode="composite"),
        "B vol-tercile 126d": dict(mode="vol_tercile", horizon=126),
        "C abs 15/25 126d": dict(mode="abs_vol", horizon=126, thresholds=(0.15, 0.25)),
        f"D abs {t12:.0%}/{t18:.0%} 126d": dict(mode="abs_vol", horizon=126, thresholds=(float(t12), float(t18))),
        "E vol-tercile 63d": dict(mode="vol_tercile", horizon=63),
    }
    rows: dict[tuple, list] = {}
    mix: dict[str, list] = {}
    for k, (train, test) in enumerate(folds(df)):
        for lname, kw in LABELS.items():
            lab = dataset.fit_labeler(train, **kw)
            y_tr, y_te = dataset.apply_labels(train, lab), dataset.apply_labels(test, lab)
            s_tr, s_te = dataset.risk_score(train, lab).to_numpy(), dataset.risk_score(test, lab).to_numpy()
            mix.setdefault(lname, []).append(np.bincount(y_te, minlength=3) / len(y_te))
            # rules: bin a trailing-volatility variable with the label's own thresholds
            cuts = ([lab["t_low"], lab["t_high"]] if lab["mode"] == "composite" else lab["thresholds"])
            for rname, col in RULE_VARS.items():
                if lab["mode"] == "composite" or lab.get("fitted_terciles"):
                    c = np.quantile(train[col], [1 / 3, 2 / 3])           # terciles of the rule variable
                else:
                    c = cuts                                              # same absolute thresholds
                rows.setdefault((lname, f"rule {rname}"), [None] * N_FOLDS)[k] = metrics(
                    y_te, np.digitize(test[col], c), test[col].to_numpy(), s_te)
            for fname, feats in FEATSETS.items():
                clf = lgb.LGBMClassifier(**LGB).fit(train[feats], y_tr)
                pr = clf.predict_proba(test[feats])
                rows.setdefault((lname, f"clf {fname}"), [None] * N_FOLDS)[k] = metrics(
                    y_te, pr.argmax(1), pr @ np.array([0, 1, 2.0]), s_te)
                if fname != "base12":
                    reg = lgb.LGBMRegressor(**LGB).fit(train[feats], s_tr)
                    pred = reg.predict(test[feats])
                    # bin the predicted score exactly like the realised score was binned
                    if lab["mode"] == "composite":
                        b = [lab["t_low"], lab["t_high"]]
                    else:
                        b = np.log(np.clip(lab["thresholds"], 1e-4, None))
                    rows.setdefault((lname, f"reg {fname}"), [None] * N_FOLDS)[k] = metrics(
                        y_te, np.digitize(pred, b), pred, s_te)
    print("\nclass mix of the test blocks (Low/Med/High, mean over folds):")
    for lname, v in mix.items():
        print(f"  {lname:<24}", np.round(np.mean(v, axis=0), 2))

    def agg(flist, sl):
        return {m: float(np.mean([f[m] for f in flist[sl]])) for m in flist[0]}

    table = []
    for (lname, mname), flist in rows.items():
        table.append((lname, mname, agg(flist, slice(0, DESIGN)), flist[DESIGN]))
    hdr = f"{'label':<24}{'model':<18}{'macroF1':>8}{'balAcc':>8}{'within1':>8}{'QWK':>7}{'rank':>7} | holdout: {'macroF1':>8}{'within1':>8}{'QWK':>7}"
    print("\nDESIGN folds 1-3 (mean) | HOLDOUT fold 4   (sorted by design QWK within each label)\n" + hdr)
    for lname in LABELS:
        for lab_, m, d, h in sorted([t for t in table if t[0] == lname], key=lambda t: -t[2]["QWK"]):
            print(f"{lab_:<24}{m:<18}{d['macroF1']:>8.3f}{d['balAcc']:>8.3f}{d['within1']:>8.3f}{d['QWK']:>7.3f}{d['rank']:>7.3f} | "
                  f"{'':>8}{h['macroF1']:>8.3f}{h['within1']:>8.3f}{h['QWK']:>7.3f}")
        print()


if __name__ == "__main__":
    main()
