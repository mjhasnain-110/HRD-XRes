#!/usr/bin/env python
"""15 - Head-to-head with IdentifiHR (Weir et al., Commun Med 2026) + adjusted spatial neighbourhood test.

IdentifiHR was trained on TCGA HGSC, so it is compared only on data that neither model saw with labels:
  SPECTRUM  (a) site pseudobulk, all cells        -> patient mean   (same input as HRD-XRes "pseudobulk")
            (b) patient pseudobulk, malignant cells only (sum of malignant metacells; IdentifiHR's own design)
  Visium    section pseudobulk of tumour spots (Denisenko, Stur)
HRD-XRes and the elastic net are re-scored on exactly the same pseudobulk inputs.

Part N (no R needed): spot-level association of P(HRD) with neighbouring cell types, adjusted for malignant
fraction (partial Spearman per section), sign test across sections, BH-FDR.

Run (one command):
  nohup python scripts/15_identifihr.py > Logs/15.out 2>&1 &
Options: --skip_r  (use existing *_identifihr.csv)   --parts I,N
Outputs: Results/identifihr/
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest, kruskal, mannwhitneyu, rankdata, spearmanr

from hrdx.data import load_features, load_inputs, model_input
from hrdx.metrics import auroc, bootstrap_ci, paired_bootstrap_diff
from hrdx.trainer import ensemble_predict, get_device, labels_to_source, load_ensemble
from hrdx.utils import ensure_dir, get_logger, load_config

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--parts", default="I,N")
ap.add_argument("--skip_r", action="store_true")
ap.add_argument("--rscript", default="Rscript")
args = ap.parse_args()
parts = set(args.parts.upper().split(","))

cfg = load_config(args.config)
log = get_logger("15_identifihr", cfg)
P, RES = cfg["paths"]["processed"], cfg["paths"]["results"]
R = ensure_dir(RES / "identifihr")
nb = cfg["evaluation"]["n_bootstrap"]
ROOT = Path(__file__).resolve().parents[1]


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.clip(q, 0, 1)
    return out


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)


# ====================================================================== I. IdentifiHR head-to-head
if "I" in parts:
    device = get_device(cfg["training"]["device"])
    feat = load_features(P)
    genes = feat["genes"]
    ens = load_ensemble(cfg["paths"]["models"], "full", device)
    Xtr, labtr = load_inputs(P, "tcga")
    src = labels_to_source(Xtr, labtr)
    plabels = pd.read_csv(P / "spectrum" / "spectrum_patient_labels.csv", index_col=0)
    plabels.index = plabels.index.astype(str)

    # elastic net exactly as the baseline (fitted on all TCGA)
    from hrdx.baselines import classical
    _enet = {}

    def enet_p(X, key):
        if key not in _enet:
            _enet[key] = classical("enet", Xtr, src.y, X, seed=0)["p_hr"].values
        return _enet[key]

    def write_counts(adata, name):
        M = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        df = pd.DataFrame(M.T, index=adata.var_names.astype(str), columns=adata.obs_names.astype(str))
        df = df.groupby(level=0).sum()  # collapse duplicated symbols
        df.round(0).astype(np.int64).to_csv(R / f"{name}_counts.csv.gz", compression="gzip")
        log.info(f"  exported {name}: {df.shape[0]} genes x {df.shape[1]} samples")

    def pseudobulk(adata, groups):
        codes, levels = pd.factorize(pd.Series(groups).astype(str))
        G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(levels), len(codes)))
        X = sp.csr_matrix(adata.X)
        pb = ad.AnnData(X=sp.csr_matrix(G @ X), obs=pd.DataFrame(index=levels), var=pd.DataFrame(index=adata.var_names))
        pb.uns["value_type"] = "counts"
        return pb

    sets = {}  # name -> AnnData of raw counts
    # (a) SPECTRUM site pseudobulk (all cells)
    spb = ad.read_h5ad(P / "spectrum" / "spectrum_pseudobulk.h5ad")
    spb.obs_names = spb.obs_names.astype(str)
    sets["spectrum_site_allcells"] = spb
    # (b) SPECTRUM patient pseudobulk of malignant cells (sum of malignant metacells)
    mc = ad.read_h5ad(P / "spectrum" / "spectrum_metacells.h5ad")
    sets["spectrum_patient_malignant"] = pseudobulk(mc, mc.obs["patient"].values)
    # (c) Visium section pseudobulk of tumour spots
    for name in ("denisenko_spots", "stur_spots"):
        a = ad.read_h5ad(P / "spatial" / f"{name}.h5ad")
        spots = pd.read_csv(RES / "spatial" / f"{name}_spot_predictions.csv", index_col=0)
        spots.index = spots.index.astype(str)
        a = a[spots.index].copy()
        keep = np.zeros(len(spots), bool)
        for sec, s in spots.groupby("section"):
            idx = spots.index.get_indexer(s.index)
            t = s["tumor_spot"].astype(bool).values
            keep[idx] = t if t.sum() >= 10 else True
        sub = a[keep].copy()
        sets[f"{name.split('_')[0]}_section_tumour"] = pseudobulk(sub, spots.loc[keep, "section"].values)

    if not args.skip_r:
        for name, adata in sets.items():
            write_counts(adata, name)
        cmd = [args.rscript, str(ROOT / "scripts" / "R" / "run_identifihr.R"), str(R)]
        log.info("Running IdentifiHR: " + " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.info(r.stderr[-3000:])
        if r.returncode != 0:
            raise RuntimeError("IdentifiHR (R) failed - see message above; install instructions in scripts/R/run_identifihr.R")

    # score HRD-XRes and elastic net on the identical inputs
    st_pb = np.load(P / "features" / "cohort_stats_spectrum_pseudobulk.npz")
    preds = {}
    for name, adata in sets.items():
        idh = pd.read_csv(R / f"{name}_identifihr.csv")
        idh["Sample"] = idh["Sample"].astype(str)
        idh = idh.set_index("Sample").reindex(adata.obs_names.astype(str))
        if name.startswith(("denisenko", "stur")):
            X, cov = model_input(adata, genes, stats=(st_pb["mu"], st_pb["sd"]))
        else:
            X, cov = model_input(adata, genes)
        d = pd.DataFrame(index=adata.obs_names.astype(str))
        d["IdentifiHR"] = idh["predictionProbability"].values
        d["HRD-XRes"] = ensemble_predict(ens, X, device)["p_hr"].values
        d["Elastic net"] = enet_p(X, name)
        d["identifihr_genes_present"] = idh.get("genes_present", pd.Series(np.nan, index=idh.index)).values
        preds[name] = d
        d.to_csv(R / f"{name}_predictions.csv")
        log.info(f"  {name}: HRD-XRes gene coverage {cov:.2f}; IdentifiHR genes present "
                 f"{d['identifihr_genes_present'].iloc[0]} / 2604")

    METHODS = ["HRD-XRes", "Elastic net", "IdentifiHR"]
    rows, comps = [], []

    # ---------- SPECTRUM patient-level
    site = preds["spectrum_site_allcells"].join(spb.obs[["patient"]].set_axis(spb.obs_names.astype(str)))
    site["patient"] = site["patient"].astype(str)
    pat_a = site.groupby("patient")[METHODS].mean()
    pat_b = preds["spectrum_patient_malignant"]
    for level, pat in (("site pseudobulk, all cells", pat_a), ("patient pseudobulk, malignant cells", pat_b)):
        d = pat.join(plabels, how="inner")
        d = d[d["hr_status_wgs"].notna()]
        y = d["hr_status_wgs"].values.astype(float)
        for m in METHODS:
            est, lo, hi = bootstrap_ci(auroc, y, d[m].values, n_boot=nb)
            rows.append({"dataset": "SPECTRUM", "input": level, "method": m, "metric": "AUROC vs WGS HR status",
                         "estimate": est, "ci_low": lo, "ci_high": hi, "n": len(d), "n_hrd": int(y.sum())})
            acc = np.mean((d[m].values >= 0.5) == (y == 1))
            rows.append({"dataset": "SPECTRUM", "input": level, "method": m, "metric": "Accuracy (P>=0.5)",
                         "estimate": acc, "n": len(d)})
            sig = d[d["signature"].isin(["HRD-Dup", "HRD-Del", "FBI"])]
            g = [x[m].values for _, x in sig.groupby("signature") if len(x) > 1]
            if len(g) >= 2:
                rows.append({"dataset": "SPECTRUM", "input": level, "method": m, "metric": "Kruskal p (WGS subtypes)",
                             "estimate": kruskal(*g).pvalue, "n": len(sig)})
            for a_, b_ in (("HRD-Dup", "FBI"), ("HRD-Del", "FBI"), ("HRD-Dup", "HRD-Del")):
                ga, gb = sig.loc[sig.signature == a_, m], sig.loc[sig.signature == b_, m]
                if len(ga) > 1 and len(gb) > 1:
                    rows.append({"dataset": "SPECTRUM", "input": level, "method": m,
                                 "metric": f"Mann-Whitney p ({a_} vs {b_})", "estimate": mannwhitneyu(ga, gb).pvalue})
            if "myriad_gis" in d:
                gg = d[[m, "myriad_gis"]].dropna()
                if len(gg) >= 5:
                    r = spearmanr(gg[m], gg["myriad_gis"])
                    rows.append({"dataset": "SPECTRUM", "input": level, "method": m, "metric": "Spearman vs Myriad GIS",
                                 "estimate": r.correlation, "p_value": r.pvalue, "n": len(gg)})
        for m in ("Elastic net", "IdentifiHR"):
            diff, lo, hi, pv = paired_bootstrap_diff(auroc, y, d["HRD-XRes"].values, d[m].values, n_boot=nb)
            comps.append({"dataset": "SPECTRUM", "input": level, "comparison": f"HRD-XRes - {m}", "metric": "AUROC",
                          "difference": diff, "ci_low": lo, "ci_high": hi, "p_value": pv})
        r = spearmanr(d["HRD-XRes"], d["IdentifiHR"])
        comps.append({"dataset": "SPECTRUM", "input": level, "comparison": "HRD-XRes vs IdentifiHR agreement",
                      "metric": "Spearman", "difference": r.correlation, "p_value": r.pvalue})

    # site-level consistency (ICC) for every method
    multi = site.groupby("patient").filter(lambda g: len(g) >= 2)
    for m in METHODS:
        k = multi.groupby("patient").size()
        grand = multi[m].mean()
        msb = (k * (multi.groupby("patient")[m].mean() - grand) ** 2).sum() / (len(k) - 1)
        msw = ((multi[m] - multi.groupby("patient")[m].transform("mean")) ** 2).sum() / (len(multi) - len(k))
        k0 = (len(multi) - (k ** 2).sum() / len(multi)) / (len(k) - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        conc = multi.assign(c=multi[m] >= 0.5).groupby("patient")["c"].agg(lambda c: c.all() or (~c).all()).mean()
        rows.append({"dataset": "SPECTRUM", "input": "site pseudobulk, all cells", "method": m,
                     "metric": "ICC across sites", "estimate": icc, "n": len(multi)})
        rows.append({"dataset": "SPECTRUM", "input": "site pseudobulk, all cells", "method": m,
                     "metric": "Fraction patients with concordant site calls", "estimate": conc, "n": len(k)})

    # ---------- Visium section level
    for name, cohort in (("denisenko_section_tumour", "Denisenko"), ("stur_section_tumour", "Stur")):
        spots = pd.read_csv(RES / "spatial" / f"{name.split('_')[0]}_spots_spot_predictions.csv", index_col=0)
        resp = spots.groupby("section")["response"].first()
        mal = spots.groupby("section")["malignant_fraction"].mean()
        d = preds[name].join(resp).join(mal)
        for m in METHODS:
            if cohort == "Denisenko":
                d["ordinal"] = d["response"].map({"good": 0, "partial": 1, "poor": 2})
                dd = d[[m, "ordinal", "malignant_fraction"]].dropna()
                r = spearmanr(dd[m], dd["ordinal"])
                pr = partial_spearman(dd[m].values, dd["ordinal"].values, dd["malignant_fraction"].values)
                rows.append({"dataset": "Visium Denisenko (post-NACT)", "input": "section pseudobulk, tumour spots",
                             "method": m, "metric": "Spearman vs CRS (good<partial<poor); expected negative",
                             "estimate": r.correlation, "p_value": r.pvalue, "partial_rho_adj_malignant": pr.correlation,
                             "partial_p": pr.pvalue, "n": len(dd)})
            else:
                dd = d.assign(poor=(d["response"] == "poor").astype(float))[[m, "poor"]].dropna()
                est, lo, hi = bootstrap_ci(auroc, dd["poor"].values, -dd[m].values, n_boot=nb)
                rows.append({"dataset": "Visium Stur (pre-treatment)", "input": "section pseudobulk, tumour spots",
                             "method": m, "metric": "AUROC poor vs excellent (lower HRD in poor)", "estimate": est,
                             "ci_low": lo, "ci_high": hi,
                             "p_value": mannwhitneyu(dd.loc[dd.poor == 1, m], dd.loc[dd.poor == 0, m]).pvalue,
                             "n": len(dd)})

    res = pd.DataFrame(rows)
    res.to_csv(R / "identifihr_head_to_head.csv", index=False)
    pd.DataFrame(comps).to_csv(R / "identifihr_paired_comparisons.csv", index=False)
    show = res.pivot_table(index=["dataset", "input", "metric"], columns="method", values="estimate", aggfunc="first")
    log.info("Head-to-head (estimates):\n" + show.round(4).to_string())
    log.info("Paired comparisons:\n" + pd.DataFrame(comps).round(4).to_string(index=False))

# ====================================================================== N. adjusted neighbourhood test
if "N" in parts:
    log.info("Part N: spot P(HRD) vs neighbouring cell types, adjusted for malignant fraction")
    per_sec = []
    for name, cohort in (("denisenko_spots", "Denisenko"), ("stur_spots", "Stur")):
        spots = pd.read_csv(RES / "spatial" / f"{name}_spot_predictions.csv", index_col=0)
        fcols = [c for c in spots.columns if c.startswith("frac_")]
        for sec, s in spots.groupby("section"):
            for c in fcols:
                d = s[["p_hr", c, "malignant_fraction"]].dropna()
                if len(d) < 30 or d[c].std() == 0:
                    continue
                raw = spearmanr(d["p_hr"], d[c])
                par = partial_spearman(d["p_hr"].values, d[c].values, d["malignant_fraction"].values)
                per_sec.append({"cohort": cohort, "section": sec, "cell_type": c[5:], "n_spots": len(d),
                                "rho_raw": raw.correlation, "rho_adj_malignant": par.correlation, "p_adj": par.pvalue})
    ps = pd.DataFrame(per_sec)
    ps.to_csv(R / "neighbourhood_adjusted_per_section.csv", index=False)
    summ = []
    for (cohort, ct), g in list(ps.groupby(["cohort", "cell_type"])) + [(("Both", ct), g) for ct, g in ps.groupby("cell_type")]:
        npos = int((g["rho_adj_malignant"] > 0).sum())
        summ.append({"cohort": cohort, "cell_type": ct, "n_sections": len(g),
                     "mean_rho_raw": g["rho_raw"].mean(), "mean_rho_adj": g["rho_adj_malignant"].mean(),
                     "n_positive_adj": npos, "sign_test_p": binomtest(npos, len(g), 0.5).pvalue})
    summ = pd.DataFrame(summ)
    summ["sign_test_fdr"] = np.nan
    for coh, idx in summ.groupby("cohort").groups.items():
        summ.loc[idx, "sign_test_fdr"] = bh(summ.loc[idx, "sign_test_p"].values)
    summ = summ[~summ["cell_type"].str.contains("cancer|malignan", case=False)]  # malignant fraction is the covariate
    summ.to_csv(R / "neighbourhood_adjusted_summary.csv", index=False)
    log.info("Neighbourhood, adjusted for malignant fraction:\n" + summ.round(4).to_string(index=False))

log.info(f"Done. Results in {R}")
