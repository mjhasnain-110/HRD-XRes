#!/usr/bin/env python
"""07b - Published-signature baseline (Peng et al., Nat Commun 2014; 230-gene HRD signature) and a
read-depth sensitivity check for the single-cell allelic-imbalance association.

Peng score per profile = mean z(up-in-HRD genes) - mean z(down-in-HRD genes), genes z-scored within each
cohort on log expression, computed on the full measured gene space of every dataset (not the model genes).
Directions are fixed a priori (no fitting): higher score = more HRD = platinum sensitive = better survival.

Evaluated exactly where HRD-XRes was evaluated, with paired tests against it:
  A. TCGA-OV: HR status, HRD score, BRCA, platinum, OS (paired vs HRD-XRes cross-validated predictions)
  B. SPECTRUM: patient-level AUROC vs WGS HR status (site pseudobulk and malignant metacells)
  C. External bulk cohorts: platinum AUROC, OS/PFS C-index, Cox HR per SD, random-effects meta-analysis
  D. Spatial: section-level mean score in tumour spots vs chemotherapy response
  E. Read-depth check: within-patient partial Spearman of predicted HRD vs allelic-imbalance measures,
     adjusting for metacell read depth and cell number

Outputs: Results/baselines_peng/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import mannwhitneyu, norm, spearmanr

from hrdx.metrics import auroc, bootstrap_ci, cindex, paired_bootstrap_diff, spearman
from hrdx.utils import ensure_dir, get_logger, inp, load_config, to_log_expression

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("07b_peng_baseline", cfg)
P, RES = cfg["paths"]["processed"], cfg["paths"]["results"]
R = ensure_dir(RES / "baselines_peng")
nb = cfg["evaluation"]["n_bootstrap"]

sig = pd.read_csv(inp(cfg, "priors/peng2014_HRD_signature_230genes_current.tsv"), sep="\t")
sig = sig[sig["gene"].notna() & (sig["gene"].astype(str).str.len() > 0)]
UP = set(sig.loc[sig["direction_in_HRD"] == "up", "gene"])
DOWN = set(sig.loc[sig["direction_in_HRD"] == "down", "gene"])
log.info(f"Peng signature: {len(UP)} up / {len(DOWN)} down genes with current symbols")


def peng_score(a: ad.AnnData) -> tuple[pd.Series, dict]:
    vt = a.uns.get("value_type", "counts")
    genes = pd.Index(a.var_names)
    up, dn = [g for g in genes if g in UP], [g for g in genes if g in DOWN]
    idx = genes.get_indexer(up + dn)
    if vt == "counts":  # CPM on the full library, then subset
        import scipy.sparse as sp
        M = sp.csr_matrix(a.X, dtype=np.float32) if sp.issparse(a.X) else np.asarray(a.X, np.float32)
        lib = np.asarray(M.sum(axis=1)).ravel()
        lib[lib == 0] = 1.0
        sub = M[:, idx]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        L = np.log2(sub / lib[:, None] * 1e6 + 1.0)
    else:
        L = to_log_expression(a.X[:, idx], vt)
    sd = L.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (L - L.mean(axis=0)) / sd
    s = Z[:, :len(up)].mean(axis=1) - Z[:, len(up):].mean(axis=1)
    return pd.Series(s, index=a.obs_names), {"n_up": len(up), "n_down": len(dn),
                                               "coverage": (len(up) + len(dn)) / (len(UP) + len(DOWN))}


cov_rows = []
ORIENTS = ("prespecified", "tcga_oriented", "best_case")


def best_case(est, metric):
    """Most favourable direction for the signature, chosen with knowledge of the labels (optimistic)."""
    if not np.isfinite(est):
        return est
    return abs(est) if metric == "Spearman" else max(est, 1 - est)


# ================================================================= A. TCGA
tcga = ad.read_h5ad(P / "tcga" / "tcga_ov.h5ad")
s, cv = peng_score(tcga)
cov_rows.append({"dataset": "TCGA-OV", **cv})
lab = tcga.obs.copy()
for c in ["hrd_score", "hr_status", "brca_alt", "platinum_resistant", "os", "os_time"]:
    lab[c] = pd.to_numeric(lab[c], errors="coerce")
lab["peng"] = s
full = pd.read_csv(RES / "cv" / "predictions_full.csv", index_col=0).reindex(lab.index)
enet = pd.read_csv(RES / "cv" / "predictions_enet.csv", index_col=0).reindex(lab.index)

# TCGA-oriented signature: direction = sign of Spearman(score, HRD score) on the TRAINING folds only,
# using the same outer folds as HRD-XRes; for external / single-cell / spatial data, the sign from all of TCGA.
folds = pd.read_csv(RES / "cv" / "fold_assignment.csv").set_index("sample")["fold"].reindex(lab.index)
lab["peng_tcga_oriented"] = np.nan
fold_signs = {}
for k in sorted(folds.dropna().unique()):
    tr, te = folds != k, folds == k
    sgn = np.sign(spearman(lab.loc[tr, "peng"].values, lab.loc[tr, "hrd_score"].values)) or 1.0
    fold_signs[int(k)] = sgn
    lab.loc[te, "peng_tcga_oriented"] = sgn * lab.loc[te, "peng"]
ORIENT_SIGN = np.sign(spearman(lab["peng"].values, lab["hrd_score"].values)) or 1.0
log.info(f"TCGA-derived orientation of the Peng score: {'as published' if ORIENT_SIGN > 0 else 'REVERSED vs publication'} "
         f"(per-fold signs: {fold_signs})")
lab[["peng", "peng_tcga_oriented"]].to_csv(R / "tcga_peng_scores.csv")

tasks = [  # task, metric, fn, label, clinical sign (higher HRD -> ...), HRD-XRes column, event column
    ("hr_status", "AUROC", auroc, "hr_status", 1.0, "p_hr", None),
    ("hrd_score", "Spearman", spearman, "hrd_score", 1.0, "pred_hrd", None),
    ("brca_alt", "AUROC", auroc, "brca_alt", 1.0, "p_brca", None),
    ("platinum_resistant", "AUROC", auroc, "platinum_resistant", -1.0, "p_plat", None),
    ("os", "C-index", cindex, "os_time", -1.0, "risk", "os"),
]
rows = []
for task, metric, fn, ycol, csign, mcol, ecol in tasks:
    y = lab[ycol].values
    extra = (lab[ecol].values,) if ecol else None
    for orient in ("prespecified", "tcga_oriented"):
        score = csign * (lab["peng"] if orient == "prespecified" else lab["peng_tcga_oriented"]).values
        args = (y, lab[ecol].values, score) if ecol else (y, score)
        est, lo, hi = bootstrap_ci(fn, *args, n_boot=nb)
        rec = {"task": task, "metric": metric, "orientation": orient, "peng": est, "peng_ci_low": lo, "peng_ci_high": hi}
        for name, pred in (("HRD-XRes", full), ("enet", enet)):
            if mcol in pred:
                d, dlo, dhi, p = paired_bootstrap_diff(fn, y, pred[mcol].values, score, n_boot=nb, extra=extra)
                rec.update({f"{name}_minus_peng": d, f"{name}_diff_ci_low": dlo, f"{name}_diff_ci_high": dhi, f"{name}_p": p})
        rows.append(rec)
    pre = rows[-2]["peng"]
    rows.append({"task": task, "metric": metric, "orientation": "best_case", "peng": best_case(pre, metric),
                 "HRD-XRes_estimate": fn(*((y, lab[ecol].values, full[mcol].values) if ecol else (y, full[mcol].values)))
                 if mcol in full else np.nan})
tcga_res = pd.DataFrame(rows)
tcga_res.to_csv(R / "tcga_comparison.csv", index=False)
log.info("TCGA (cross-validated HRD-XRes / elastic net vs Peng signature, three orientations):\n"
         + tcga_res.round(3).to_string(index=False))

# ================================================================= B. SPECTRUM patient level
plab = pd.read_csv(P / "spectrum" / "spectrum_patient_labels.csv", index_col=0)
pp = pd.read_csv(RES / "singlecell" / "patient_predictions.csv", index_col=0)
rows = []
for level in ("spectrum_pseudobulk", "spectrum_metacells"):
    a = ad.read_h5ad(P / "spectrum" / f"{level}.h5ad")
    s, cv = peng_score(a)
    cov_rows.append({"dataset": level, **cv})
    pat = pd.DataFrame({"peng": s, "patient": a.obs["patient"].astype(str).values}).groupby("patient")["peng"].mean()
    d = plab.join(pat, how="inner")
    m = pp[(pp["level"] == level) & (pp["method"] == "HRD-XRes (full)")]["p_hr"]
    d = d.join(m.rename("hrdxres"), how="left")
    d.to_csv(R / f"{level}_patient_scores.csv")
    y = d["hr_status_wgs"].values
    for orient in ORIENTS:
        if orient == "best_case":
            pre = rows[-2]["peng_auroc"]
            rows.append({"level": level, "orientation": orient, "peng_auroc": best_case(pre, "AUROC"),
                         "hrdxres_auroc": auroc(y, d["hrdxres"].values)})
            continue
        sc_ = d["peng"].values * (1.0 if orient == "prespecified" else ORIENT_SIGN)
        est, lo, hi = bootstrap_ci(auroc, y, sc_, n_boot=nb)
        dd, dlo, dhi, p = paired_bootstrap_diff(auroc, y, d["hrdxres"].values, sc_, n_boot=nb)
        rows.append({"level": level, "orientation": orient, "n_patients": int(np.isfinite(y).sum()), "peng_auroc": est,
                     "peng_ci_low": lo, "peng_ci_high": hi, "hrdxres_auroc": auroc(y, d["hrdxres"].values),
                     "hrdxres_minus_peng": dd, "diff_ci_low": dlo, "diff_ci_high": dhi, "diff_p": p})
sc = pd.DataFrame(rows)
sc.to_csv(R / "spectrum_comparison.csv", index=False)
log.info("SPECTRUM patient-level AUROC vs WGS HR status:\n" + sc.round(3).to_string(index=False))

# ================================================================= C. external bulk cohorts
ext_cox = pd.read_csv(RES / "external_bulk" / "external_cox.csv")
rows, cox_rows = [], []
min_cov = cfg["features"]["min_external_gene_coverage"]
for f in sorted((P / "bulk_external").glob("*.h5ad")):
    a = ad.read_h5ad(f)
    s, cv = peng_score(a)
    cov_rows.append({"dataset": f.stem, **cv})
    if cv["coverage"] < min_cov:
        continue
    obs = a.obs.copy()
    obs["peng"] = s
    for orient, sgn in (("prespecified", 1.0), ("tcga_oriented", ORIENT_SIGN)):
        if "platinum_resistant" in obs:
            y = pd.to_numeric(obs["platinum_resistant"], errors="coerce").values
            est, lo, hi = bootstrap_ci(auroc, y, -sgn * obs["peng"].values, n_boot=nb)
            rows.append({"cohort": f.stem, "orientation": orient, "task": "platinum_resistant", "metric": "AUROC",
                         "peng": est, "ci_low": lo, "ci_high": hi, "n": int(np.isfinite(y).sum())})
        for ep in ("os", "pfs"):
            if ep not in obs or f"{ep}_time" not in obs:
                continue
            t = pd.to_numeric(obs[f"{ep}_time"], errors="coerce").values
            e = pd.to_numeric(obs[ep], errors="coerce").values
            r = -sgn * obs["peng"].values
            m = np.isfinite(t) & np.isfinite(e) & (t > 0)
            if m.sum() < 30 or e[m].sum() < 5:
                continue
            est, lo, hi = bootstrap_ci(cindex, t, e, r, n_boot=nb)
            rows.append({"cohort": f.stem, "orientation": orient, "task": ep, "metric": "C-index", "peng": est,
                         "ci_low": lo, "ci_high": hi, "n": int(m.sum())})
            z = (r[m] - r[m].mean()) / (r[m].std() + 1e-9)
            cph = CoxPHFitter().fit(pd.DataFrame({"T": t[m], "E": e[m], "risk_per_sd": z}), "T", "E")
            sm = cph.summary.loc["risk_per_sd"]
            cox_rows.append({"cohort": f.stem, "orientation": orient, "endpoint": ep, "HR": sm["exp(coef)"],
                             "log_hr": sm["coef"], "se": sm["se(coef)"], "p": sm["p"], "n": int(m.sum())})
ext = pd.DataFrame(rows)
ext.to_csv(R / "external_metrics.csv", index=False)
pcox = pd.DataFrame(cox_rows)
pcox.to_csv(R / "external_cox.csv", index=False)


def dl_meta(g):
    y, v = g["log_hr"].values, g["se"].values ** 2
    w = 1 / v
    fixed = np.sum(w * y) / w.sum()
    Q = np.sum(w * (y - fixed) ** 2)
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    wr = 1 / (v + tau2)
    mu, se = np.sum(wr * y) / wr.sum(), np.sqrt(1 / wr.sum())
    return {"k_cohorts": len(y), "n_total": int(g["n"].sum()), "pooled_HR": np.exp(mu), "ci_low": np.exp(mu - 1.96 * se),
            "ci_high": np.exp(mu + 1.96 * se), "p": 2 * norm.sf(abs(mu / se)), "I2": max(0.0, (Q - (len(y) - 1)) / Q) if Q > 0 else 0.0}


meta = []
for (orient, ep), g in pcox.groupby(["orientation", "endpoint"]):
    if len(g) >= 2:
        r_ = dl_meta(g)
        meta.append({"method": "Peng 2014 signature", "orientation": orient, "endpoint": ep, **r_})
        if orient == "prespecified":  # best case = the more favourable direction of the pooled effect
            hr, lo, hi = r_["pooled_HR"], r_["ci_low"], r_["ci_high"]
            if hr < 1:
                hr, lo, hi = 1 / hr, 1 / hi, 1 / lo
            meta.append({"method": "Peng 2014 signature", "orientation": "best_case", "endpoint": ep,
                         "k_cohorts": r_["k_cohorts"], "n_total": r_["n_total"], "pooled_HR": hr, "ci_low": lo,
                         "ci_high": hi, "p": r_["p"], "I2": r_["I2"]})
for ep, g in pcox[pcox["orientation"] == "prespecified"].groupby("endpoint"):
    h = ext_cox[(ext_cox["method"] == "HRD-XRes (full)") & (ext_cox["model"] == "univariable") &
                (ext_cox["endpoint"] == ep) & (ext_cox["cohort"].isin(g["cohort"]))]
    if len(h) >= 2:
        meta.append({"method": "HRD-XRes (full), same cohorts", "orientation": "-", "endpoint": ep, **dl_meta(h)})
meta = pd.DataFrame(meta)
meta.to_csv(R / "meta_analysis_comparison.csv", index=False)
log.info("External platinum / survival (Peng):\n" + ext.round(3).to_string(index=False))
log.info("Meta-analysis, Peng (three orientations) vs HRD-XRes on the same cohorts:\n" + meta.round(4).to_string(index=False))

# ================================================================= D. spatial
srows = []
for name, h5, kind in (("denisenko_spots", "denisenko_spots.h5ad", "ordinal"), ("stur_spots", "stur_spots.h5ad", "binary")):
    a = ad.read_h5ad(P / "spatial" / h5)
    spots = pd.read_csv(RES / "spatial" / f"{name}_spot_predictions.csv", index_col=0)
    a = a[a.obs_names.isin(spots.index)].copy()
    s, cv = peng_score(a)
    cov_rows.append({"dataset": name, **cv})
    spots = spots.join(s.rename("peng"))
    t = spots[spots["tumor_spot"].astype(str).str.lower() == "true"]
    sec = t.groupby("section").agg(peng=("peng", "mean"), response=("response", "first"))
    for orient, sgn in (("prespecified", 1.0), ("tcga_oriented", ORIENT_SIGN)):
        v = sgn * sec["peng"]
        if kind == "ordinal":
            o = sec["response"].map({"good": 0, "partial": 1, "poor": 2})
            r = spearmanr(v, o)  # expected negative: more HRD -> better response
            srows.append({"cohort": "Denisenko (post-NACT)", "orientation": orient,
                          "test": "Spearman vs CRS (good<partial<poor); expected negative",
                          "estimate": r.correlation, "p": r.pvalue, "n_sections": len(sec)})
        else:
            y = (sec["response"] == "poor").astype(float).values
            srows.append({"cohort": "Stur (pre-treatment)", "orientation": orient,
                          "test": "AUROC poor vs excellent (lower HRD in poor)", "estimate": auroc(y, -v.values),
                          "p": mannwhitneyu(v[y == 1], v[y == 0]).pvalue, "n_sections": len(sec)})
pd.DataFrame(srows).to_csv(R / "spatial_comparison.csv", index=False)
log.info("Spatial (Peng, mean in tumour spots):\n" + pd.DataFrame(srows).round(3).to_string(index=False))
pd.DataFrame(cov_rows).to_csv(R / "signature_gene_coverage.csv", index=False)

# ================================================================= E. read-depth check (allelic imbalance)
mc = pd.read_csv(RES / "singlecell" / "metacell_predictions.csv", index_col=0)
mca = ad.read_h5ad(P / "spectrum" / "spectrum_metacells.h5ad")
depth = pd.Series(np.asarray(mca.X.sum(axis=1)).ravel(), index=mca.obs_names)
mc["log_depth"] = np.log10(depth.reindex(mc.index).values + 1)
mc["log_cells"] = np.log10(pd.to_numeric(mc["n_cells"], errors="coerce") + 1)


def within_rank(df, col):
    r = df[col].rank()
    return r - r.groupby(df["patient"]).transform("mean")


rows = []
for score in ("p_hr", "pred_hrd"):
    for feat in [c for c in ("allelic_imbalance", "frac_arms_imbalanced", "CIN.module") if c in mc.columns]:
        d = mc[[score, feat, "patient", "log_depth", "log_cells"]].dropna()
        if len(d) < 50:
            continue
        X = np.column_stack([np.ones(len(d)), within_rank(d, "log_depth"), within_rank(d, "log_cells")])
        res = {}
        for c in (score, feat):
            y = within_rank(d, c).values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res[c] = y - X @ beta
        raw = spearmanr(within_rank(d, score), within_rank(d, feat))
        rho = np.corrcoef(res[score], res[feat])[0, 1]
        nn = len(d)
        tstat = rho * np.sqrt((nn - 4) / max(1e-12, 1 - rho ** 2))
        rows.append({"score": score, "feature": feat, "n_metacells": nn,
                     "within_patient_rho": raw.correlation,
                     "depth_adjusted_within_patient_rho": rho,
                     "p_adjusted": 2 * norm.sf(abs(tstat)),
                     "rho_feature_vs_depth": spearmanr(within_rank(d, feat), within_rank(d, "log_depth")).correlation})
dep = pd.DataFrame(rows)
dep.to_csv(R / "allelic_imbalance_depth_check.csv", index=False)
log.info("Allelic imbalance vs predicted HRD, within patient, adjusted for read depth and cell number:\n"
         + dep.round(3).to_string(index=False))
log.info(f"Done. Results in {R}")
