#!/usr/bin/env python
"""10 - Single-cell validation.

A. SPECTRUM patient level: site pseudobulk and malignant metacell predictions aggregated per patient and
   tested against WGS-derived HR status (HRD vs HRP; Undetermined excluded), signature subtypes
   (HRD-Dup / HRD-Del / FBI) and Myriad GIS. Full model vs no-DA ablation vs baselines (paired bootstrap).
B. Metacell level: association with per-cell allelic imbalance (BAF) and chromosomal-instability module,
   globally and within patient.
C. Anatomical-site consistency within patients (ICC).
D. Per-cell resolution: kNN-pooled predictions for malignant cells; intratumoural HRD heterogeneity.
E. GSE165897: paired pre- vs post-NACT shift in predicted HRD / platinum resistance.

Outputs: Results/singlecell/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon

from hrdx.baselines import classical, oriented_hr_score
from hrdx.data import load_features, load_inputs, model_input, read_gmt
from hrdx.genes import read_h5ad_df, spectrum_files, spectrum_var_symbols
from hrdx.metrics import auroc, bootstrap_ci, paired_bootstrap_diff
from hrdx.singlecell import knn_pool
from hrdx.trainer import ensemble_predict, get_device, labels_to_source, load_ensemble
from hrdx.utils import H5adCSR, collapse_matrix, ensure_dir, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("10_validate_singlecell", cfg)
P, R = cfg["paths"]["processed"], ensure_dir(cfg["paths"]["results"] / "singlecell")
device = get_device(cfg["training"]["device"])
nb = cfg["evaluation"]["n_bootstrap"]
feat = load_features(P)
genes = feat["genes"]
gmt = read_gmt(inp(cfg, cfg["files"]["reactome_gmt"]))
ens = {v: load_ensemble(cfg["paths"]["models"], v, device) for v in ("full", "no_da")
       if (cfg["paths"]["models"] / "final" / v / "ensemble.json").exists()}
Xtr, labtr = load_inputs(P, "tcga")
src = labels_to_source(Xtr, labtr)
plabels = pd.read_csv(P / "spectrum" / "spectrum_patient_labels.csv", index_col=0)


def all_methods(X):
    out = {f"HRD-XRes ({v})": ensemble_predict(m, X, device) for v, m in ens.items()}
    out["Elastic net (TCGA)"] = enet_fit(X)
    out["HR pathway score"] = oriented_hr_score(Xtr, src.y, X, genes, gmt)
    return out


_enet_cache = {}


def enet_fit(X):
    key = (X.shape, float(np.round(X[:5].sum(), 3)))
    if key not in _enet_cache:
        _enet_cache[key] = classical("enet", Xtr, src.y, X, seed=0)
    return _enet_cache[key]


# ================================================================= A. patient-level validation
(R / "patient_predictions.csv").unlink(missing_ok=True)
pat_rows = []
rows, pat_tables, comps = [], {}, []
for level in ("spectrum_pseudobulk", "spectrum_metacells"):
    X, obs = load_inputs(P, level)
    for mname, p in all_methods(X).items():
        p.index = obs.index
        d = p.join(obs[["patient"]]).groupby("patient").mean(numeric_only=True).join(plabels, how="inner")
        d["n_units"] = obs.groupby("patient").size().reindex(d.index).values
        pat_tables[(level, mname)] = d
        pat_rows.append(d.assign(level=level, method=mname))
        y = d["hr_status_wgs"].values
        for score in ("p_hr", "pred_hrd"):
            if score not in d:
                continue
            est, lo, hi = bootstrap_ci(auroc, y, d[score].values, n_boot=nb)
            rows.append({"level": level, "method": mname, "score": score, "metric": "AUROC (WGS HR status)",
                         "estimate": est, "ci_low": lo, "ci_high": hi, "n_patients": int(np.isfinite(y).sum()),
                         "n_hrd": int(np.nansum(y))})
        sig = d[d["signature"].isin(["HRD-Dup", "HRD-Del", "FBI"])]
        groups = [g["p_hr"].values for _, g in sig.groupby("signature") if len(g) > 1]
        if len(groups) >= 2:
            rows.append({"level": level, "method": mname, "score": "p_hr", "metric": "Kruskal p (signature subtypes)",
                         "estimate": kruskal(*groups).pvalue, "n_patients": len(sig)})
            for a_, b_ in [("HRD-Dup", "FBI"), ("HRD-Del", "FBI"), ("HRD-Dup", "HRD-Del")]:
                ga, gb = sig.loc[sig["signature"] == a_, "p_hr"], sig.loc[sig["signature"] == b_, "p_hr"]
                if len(ga) > 1 and len(gb) > 1:
                    rows.append({"level": level, "method": mname, "score": "p_hr",
                                 "metric": f"Mann-Whitney p ({a_} vs {b_})", "estimate": mannwhitneyu(ga, gb).pvalue,
                                 "n_patients": len(ga) + len(gb)})
        g = d[["myriad_gis", "pred_hrd"]].dropna()
        if len(g) >= 5:
            r = spearmanr(g["myriad_gis"], g["pred_hrd"])
            rows.append({"level": level, "method": mname, "score": "pred_hrd", "metric": "Spearman vs Myriad GIS",
                         "estimate": r.correlation, "p_value": r.pvalue, "n_patients": len(g)})
    full = pat_tables.get((level, "HRD-XRes (full)"))
    if full is not None:
        for other in [m for (lv, m) in pat_tables if lv == level and m != "HRD-XRes (full)"]:
            o = pat_tables[(level, other)].reindex(full.index)
            dd, lo, hi, pv = paired_bootstrap_diff(auroc, full["hr_status_wgs"].values, full["p_hr"].values,
                                                   o["p_hr"].values, n_boot=nb)
            comps.append({"level": level, "comparison": f"full vs {other}", "metric": "AUROC",
                          "difference": dd, "ci_low": lo, "ci_high": hi, "p_value": pv})
pd.concat(pat_rows).rename_axis("patient").to_csv(R / "patient_predictions.csv")  # columns aligned across methods
pd.DataFrame(rows).to_csv(R / "patient_level_metrics.csv", index=False)
pd.DataFrame(comps).to_csv(R / "patient_level_comparisons.csv", index=False)
log.info("Patient-level validation:\n" + pd.DataFrame(rows).to_string(index=False))

# ================================================================= B. metacell-level genomic associations
X, obs = load_inputs(P, "spectrum_metacells")
pm = ensemble_predict(ens["full"], X, device)
pm.index = obs.index
mc = obs.join(pm)
mc.to_csv(R / "metacell_predictions.csv")
assoc = []
for score in ("p_hr", "pred_hrd"):
    for feat_col in [c for c in ("allelic_imbalance", "frac_arms_imbalanced", "CIN.module") if c in mc.columns]:
        d = mc[[score, feat_col, "patient"]].dropna()
        if len(d) < 20:
            continue
        r = spearmanr(d[score], d[feat_col])
        dc = d.copy()
        dc[[score, feat_col]] = dc[[score, feat_col]] - dc.groupby("patient")[[score, feat_col]].transform("mean")
        rw = spearmanr(dc[score], dc[feat_col])
        assoc.append({"score": score, "feature": feat_col, "n_metacells": len(d), "spearman_global": r.correlation,
                      "p_global": r.pvalue, "spearman_within_patient": rw.correlation, "p_within_patient": rw.pvalue})
pd.DataFrame(assoc).to_csv(R / "metacell_genomic_associations.csv", index=False)

# ================================================================= C. site consistency
Xp, obsp = load_inputs(P, "spectrum_pseudobulk")
pp = ensemble_predict(ens["full"], Xp, device)
pp.index = obsp.index
sites = obsp[["patient", "site", "supersite"]].join(pp)
sites.to_csv(R / "site_predictions.csv")
multi = sites.groupby("patient").filter(lambda g: len(g) >= 2)
if len(multi):
    k = multi.groupby("patient").size()
    grand = multi["p_hr"].mean()
    msb = (k * (multi.groupby("patient")["p_hr"].mean() - grand) ** 2).sum() / (len(k) - 1)
    msw = ((multi["p_hr"] - multi.groupby("patient")["p_hr"].transform("mean")) ** 2).sum() / (len(multi) - len(k))
    k0 = (len(multi) - (k ** 2).sum() / len(multi)) / (len(k) - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    conc = multi.assign(call=multi["p_hr"] >= 0.5).groupby("patient")["call"].agg(lambda c: c.all() or (~c).all()).mean()
    pd.DataFrame([{"n_patients": len(k), "n_sites": len(multi), "ICC_p_hr": icc,
                   "fraction_patients_concordant_calls": conc}]).to_csv(R / "site_consistency.csv", index=False)
    log.info(f"Site consistency: ICC={icc:.3f}, concordant calls in {conc:.2%} of patients")

# ================================================================= D. per-cell (kNN-pooled) predictions
files = spectrum_files(cfg)
st = np.load(P / "features" / "cohort_stats_spectrum_metacells.npz")
stats = (st["mu"], st["sd"])
cells = pd.read_csv(P / "spectrum" / "spectrum_malignant_cells.csv", index_col=0)
esym = spectrum_var_symbols(files["epithelial"])
egenes, EA = collapse_matrix(esym.values)
reader = H5adCSR(files["epithelial"])
rng = np.random.default_rng(0)
cap, kk = cfg["singlecell"]["max_cells_per_sample"], cfg["singlecell"]["knn_smoothing_k"]
cell_preds = []
for i, (samp, sub) in enumerate(cells.groupby("site_sample")):
    if len(sub) > cap:
        sub = sub.iloc[np.sort(rng.choice(len(sub), cap, replace=False))]
    M = (reader.read_rows(sub["row"].values) @ EA).tocsr()
    pooled = knn_pool(M, k=kk, seed=i)
    a = ad.AnnData(X=pooled, var=pd.DataFrame(index=egenes))
    a.uns["value_type"] = "counts"
    Xc, _ = model_input(a, genes, stats=stats)
    p = ensemble_predict(ens["full"], Xc, device)
    p.index = sub.index
    cell_preds.append(sub.join(p))
reader.close()
cp = pd.concat(cell_preds)
cp.to_csv(R / "cell_predictions.csv")
cell_assoc = []
for f in [c for c in ("allelic_imbalance", "CIN.module") if c in cp.columns]:
    d = cp[["p_hr", f, "patient"]].dropna()
    if len(d) > 50:
        dc = d[["p_hr", f]] - d.groupby("patient")[["p_hr", f]].transform("mean")
        cell_assoc.append({"feature": f, "n_cells": len(d), "spearman_global": spearmanr(d["p_hr"], d[f]).correlation,
                           "spearman_within_patient": spearmanr(dc["p_hr"], dc[f]).correlation})
pd.DataFrame(cell_assoc).to_csv(R / "cell_genomic_associations.csv", index=False)
het = cp.groupby("patient").agg(n_cells=("p_hr", "size"), mean_p_hr=("p_hr", "mean"), sd_p_hr=("p_hr", "std"),
                                iqr_p_hr=("p_hr", lambda v: np.subtract(*np.percentile(v, [75, 25]))),
                                frac_hrd_high=("p_hr", lambda v: float((v >= 0.5).mean())))
het = het.join(plabels)
het.to_csv(R / "patient_cell_heterogeneity.csv")
hrows = []
for m in ("sd_p_hr", "iqr_p_hr", "frac_hrd_high"):
    d = het[[m, "hr_status_wgs"]].dropna()
    if d["hr_status_wgs"].nunique() == 2:
        hrows.append({"metric": m, "auroc_hr_status": auroc(d["hr_status_wgs"], d[m]),
                      "mannwhitney_p": mannwhitneyu(d.loc[d.hr_status_wgs == 1, m], d.loc[d.hr_status_wgs == 0, m]).pvalue})
pd.DataFrame(hrows).to_csv(R / "heterogeneity_vs_hr_status.csv", index=False)
log.info(f"Per-cell predictions: {len(cp)} malignant cells from {cp['patient'].nunique()} patients")

# ================================================================= E. GSE165897 pre vs post NACT
Xg, og = load_inputs(P, "gse165897_metacells")
pg = ensemble_predict(ens["full"], Xg, device)
pg.index = og.index
g = og.join(pg)
g.to_csv(R / "gse165897_metacell_predictions.csv")
pat = g.groupby(["patient", "phase"])[["p_hr", "pred_hrd", "p_plat"]].mean().unstack("phase")
pat.to_csv(R / "gse165897_patient_phase_means.csv")
prow = []
for s in ("p_hr", "pred_hrd", "p_plat"):
    if ("pre" in pat[s].columns) and ("post" in pat[s].columns):
        d = pat[s][["pre", "post"]].dropna()
        if len(d) >= 3:
            prow.append({"score": s, "n_patients": len(d), "mean_pre": d["pre"].mean(), "mean_post": d["post"].mean(),
                         "median_delta": (d["post"] - d["pre"]).median(), "wilcoxon_p": wilcoxon(d["pre"], d["post"]).pvalue})
pd.DataFrame(prow).to_csv(R / "gse165897_pre_post.csv", index=False)
log.info("Pre vs post NACT:\n" + pd.DataFrame(prow).to_string(index=False))
