"""Metrics with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _valid(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(np.asarray(a, dtype=float))
    return m


def auroc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = _valid(y, s)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], s[m])


def auprc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = _valid(y, s)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return np.nan
    return average_precision_score(y[m], s[m])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = _valid(a, b)
    if m.sum() < 4:
        return np.nan
    return spearmanr(a[m], b[m]).correlation


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = _valid(a, b)
    if m.sum() < 4:
        return np.nan
    return pearsonr(a[m], b[m])[0]


def cindex(time, event, risk):
    """Harrell's C where higher risk = shorter survival."""
    t, e, r = (np.asarray(v, float) for v in (time, event, risk))
    m = _valid(t, e, r) & (t > 0)
    if m.sum() < 10 or e[m].sum() < 2:
        return np.nan
    return concordance_index(t[m], -r[m], e[m])


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _valid(y, p)
    if m.sum() < 4 or len(np.unique(y[m])) < 2 or p[m].min() < 0 or p[m].max() > 1:
        return np.nan  # only defined for probabilities
    return brier_score_loss(y[m], p[m])


def bootstrap_ci(fn, *arrays, n_boot: int = 2000, seed: int = 0, stratify=None):
    """Point estimate and percentile 95% CI for fn(*arrays)."""
    arrays = [np.asarray(a) for a in arrays]
    est = fn(*arrays)
    if not np.isfinite(est):
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    if stratify is not None:
        strat = np.asarray(stratify)
        groups = [np.where(strat == g)[0] for g in np.unique(strat[np.isfinite(strat.astype(float))])]
    for _ in range(n_boot):
        if stratify is not None:
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        else:
            idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return est, np.nan, np.nan
    return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_predictions(pred: pd.DataFrame, labels: pd.DataFrame, n_boot: int = 2000,
                         seed: int = 0) -> pd.DataFrame:
    """Standard TCGA-style evaluation of all heads present in `pred`."""
    rows = []
    df = pred.join(labels, how="inner", rsuffix="_lab")

    def add(task, metric, fn, cols, strat=None):
        if not all(c in df.columns for c in cols):
            return
        arrays = [df[c].values.astype(float) for c in cols]
        n = int(_valid(*arrays).sum())
        est, lo, hi = bootstrap_ci(fn, *arrays, n_boot=n_boot, seed=seed,
                                   stratify=df[strat].values.astype(float) if strat else None)
        rows.append({"task": task, "metric": metric, "estimate": est, "ci_low": lo, "ci_high": hi, "n": n})

    add("hr_status", "AUROC", auroc, ["hr_status", "p_hr"])
    add("hr_status", "AUPRC", auprc, ["hr_status", "p_hr"])
    add("hr_status", "Brier", brier, ["hr_status", "p_hr"])
    add("hrd_score", "Spearman", spearman, ["hrd_score", "pred_hrd"])
    add("hrd_score", "Pearson", pearson, ["hrd_score", "pred_hrd"])
    add("brca_alt", "AUROC", auroc, ["brca_alt", "p_brca"])
    add("brca_alt", "AUPRC", auprc, ["brca_alt", "p_brca"])
    add("platinum_resistant", "AUROC", auroc, ["platinum_resistant", "p_plat"])
    add("purity", "Spearman", spearman, ["purity", "pred_purity"])
    add("os", "C-index", cindex, ["os_time", "os", "risk"])
    add("pfi", "C-index", cindex, ["pfi_time", "pfi", "risk"])
    return pd.DataFrame(rows)


def composite_score(m: dict) -> float:
    keys = ["auroc_hr", "spearman_hrd", "auroc_brca", "auroc_plat", "cindex"]
    vals = [m[k] for k in keys if k in m and np.isfinite(m[k])]
    return float(np.mean(vals)) if vals else -np.inf


def paired_bootstrap_diff(fn, y, s1, s2, n_boot: int = 2000, seed: int = 0, extra=None):
    """Difference fn(y, s1) - fn(y, s2) with 95% CI and two-sided bootstrap p-value.

    `extra` (optional) is a tuple of additional aligned arrays passed before the scores
    (e.g. (time, event) for C-index: fn(time, event, score)).
    """
    y = np.asarray(y, float)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    extra = tuple(np.asarray(e, float) for e in (extra or ()))
    args1 = (y,) + extra if extra else (y,)

    def f(idx, s):
        return fn(*[a[idx] for a in args1], s[idx])

    base = np.arange(len(y))
    d0 = f(base, s1) - f(base, s2)
    if not np.isfinite(d0):
        return d0, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        d = f(idx, s1) - f(idx, s2)
        if np.isfinite(d):
            ds.append(d)
    ds = np.asarray(ds)
    if len(ds) < 50:
        return d0, np.nan, np.nan, np.nan
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(min(1.0, max(p, 1.0 / len(ds))))
