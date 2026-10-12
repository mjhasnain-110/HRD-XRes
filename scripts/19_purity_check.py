#!/usr/bin/env python
"""19 - Is the spatial HRD-response association explained by tumour purity?

For each Visium cohort, section-level HRD metrics are related to response after adjusting, one at a time, for
  (a) predicted purity of the section pseudobulk          (pseudobulk_pred_purity, from step 11)
  (b) mean predicted purity of tumour spots              (spot-level purity head)
  (c) mean deconvolved malignant fraction of all spots    (as in the manuscript)
  (d) fraction of spots that are tumour spots
Partial Spearman correlations are reported with exact permutation P values (response labels permuted
across sections; feasible because there are only 8 and 12 sections).

Run:  python scripts/19_purity_check.py > Logs/19.out 2>&1
Output: Results/spatial/purity_adjusted_response.csv
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from hrdx.utils import get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("19_purity_check", cfg)
R = cfg["paths"]["results"] / "spatial"


def _resid_rows(Y, rz):
    """Residuals of each row of Y (already ranked) regressed on rz (ranks of covariate)."""
    zc = rz - rz.mean()
    beta = (Y - Y.mean(axis=1, keepdims=True)) @ zc / (zc @ zc)
    return Y - Y.mean(axis=1, keepdims=True) - np.outer(beta, zc)


def _rank_rows(M):
    return np.argsort(np.argsort(M, axis=1), axis=1).astype(float)


def _corr_rows(a, B):
    a = (a - a.mean()) / a.std()
    Bc = (B - B.mean(axis=1, keepdims=True)) / B.std(axis=1, keepdims=True)
    return Bc @ a / len(a)


def partial_rho(x, y, z):
    rz = rankdata(z)
    ex = _resid_rows(rankdata(x)[None, :], rz)[0]
    ey = _resid_rows(rankdata(y)[None, :], rz)[0]
    return spearmanr(ex, ey).correlation


def exact_perm_p(x, y, z, max_perm=20000, seed=0):
    """Two-sided permutation P for the partial Spearman correlation (response labels permuted across sections)."""
    n = len(y)
    rz, ry = rankdata(z), rankdata(y)
    ex = rankdata(_resid_rows(rankdata(x)[None, :], rz)[0])
    if n <= 9:
        P = np.array(list(itertools.permutations(range(n))))
    else:
        rng = np.random.default_rng(seed)
        P = np.array([rng.permutation(n) for _ in range(max_perm)])
    EY = _resid_rows(ry[P], rz)
    vals = _corr_rows(ex, _rank_rows(EY))
    obs = partial_rho(x, y, z)
    return obs, float(np.mean(np.abs(vals) >= abs(obs) - 1e-9))


def section_table(name, response_map):
    sec = pd.read_csv(R / f"{name}_section_metrics.csv")
    if "section" not in sec.columns:
        sec = sec.rename(columns={sec.columns[0]: "section"})
    spots = pd.read_csv(R / f"{name}_spot_predictions.csv", index_col=0)
    spots["tumor_spot"] = spots["tumor_spot"].astype(bool)
    agg = spots.groupby("section").agg(mean_malignant_fraction=("malignant_fraction", "mean"),
                                       tumour_spot_fraction=("tumor_spot", "mean"))
    if "pred_purity" in spots:
        agg["tumour_spot_pred_purity"] = spots[spots.tumor_spot].groupby("section")["pred_purity"].mean()
    sec = sec.merge(agg, left_on="section", right_index=True, how="left")
    sec["y"] = sec["response"].map(response_map)
    return sec.dropna(subset=["y"])


rows = []
for name, cohort, rmap, note in (
        ("denisenko_spots", "Denisenko (post-NACT)", {"good": 0, "partial": 1, "poor": 2}, "rho vs CRS (good<partial<poor); expected negative"),
        ("stur_spots", "Stur (pre-treatment)", {"excellent": 0, "poor": 1}, "rho vs poor response; expected negative")):
    sec = section_table(name, rmap)
    covs = [c for c in ("pseudobulk_pred_purity", "tumour_spot_pred_purity", "mean_malignant_fraction",
                        "tumour_spot_fraction") if c in sec]
    log.info(f"{cohort}: {len(sec)} sections; covariates {covs}")
    for c in covs:
        r = spearmanr(sec[c], sec["y"])
        rows.append({"cohort": cohort, "hrd_metric": "(covariate alone)", "adjusted_for": c, "test": note,
                     "rho": r.correlation, "p_value": r.pvalue, "n_sections": len(sec)})
    for m in ("frac_hrd_high", "mean_p_hr", "pseudobulk_p_hr"):
        if m not in sec:
            continue
        d = sec.dropna(subset=[m])
        r = spearmanr(d[m], d["y"])
        rows.append({"cohort": cohort, "hrd_metric": m, "adjusted_for": "none", "test": note,
                     "rho": r.correlation, "p_value": r.pvalue, "n_sections": len(d)})
        for c in covs:
            dd = d.dropna(subset=[c])
            rc = spearmanr(dd[m], dd[c]).correlation
            rho, p = exact_perm_p(dd[m].values, dd["y"].values.astype(float), dd[c].values)
            rows.append({"cohort": cohort, "hrd_metric": m, "adjusted_for": c, "test": note,
                         "rho": rho, "p_value": p, "p_type": "exact permutation" if len(dd) <= 9 else "20,000 permutations",
                         "rho_hrd_vs_covariate": rc, "n_sections": len(dd)})

out = pd.DataFrame(rows)
out.to_csv(R / "purity_adjusted_response.csv", index=False)
pd.set_option("display.width", 250)
log.info("Response association before and after purity adjustment:\n" + out.round(4).to_string(index=False))
