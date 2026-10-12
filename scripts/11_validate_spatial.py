#!/usr/bin/env python
"""11 - Spatial validation.

Spot-level predictions (HRD-XRes full and no-DA), spot deconvolution with the SPECTRUM reference,
tumour-spot selection, section-level HRD heterogeneity metrics (mean, SD, IQR, fraction HRD-high,
Moran's I on the Visium hexagonal grid), association with chemotherapy response:
  Denisenko (post-NACT, CRS good / partial / poor; coordinates available)
  Stur (pre-treatment, excellent vs poor responders): sample-level prediction of response
and spatial neighbourhood analysis (spot HRD vs local cell-type composition).

Outputs: Results/spatial/
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from hrdx.data import cpm_dense, load_features, load_inputs, marker_genes, model_input, nnls_deconvolve
from hrdx.metrics import auroc, bootstrap_ci
from hrdx.trainer import ensemble_predict, get_device, load_ensemble
from hrdx.utils import ensure_dir, get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("11_validate_spatial", cfg)
P, R = cfg["paths"]["processed"], ensure_dir(cfg["paths"]["results"] / "spatial")
device = get_device(cfg["training"]["device"])
nb = cfg["evaluation"]["n_bootstrap"]
feat = load_features(P)
ens = {v: load_ensemble(cfg["paths"]["models"], v, device) for v in ("full", "no_da")
       if (cfg["paths"]["models"] / "final" / v / "ensemble.json").exists()}
ref = ad.read_h5ad(P / "spectrum" / "spectrum_reference.h5ad")
refdf = pd.DataFrame(ref.X, index=ref.obs_names, columns=ref.var_names)
mal_re = re.compile(cfg["singlecell"]["malignant_regex"], re.I)
mal_types = [c for c in refdf.index if mal_re.search(c)]
log.info(f"Reference cell types: {list(refdf.index)}; malignant = {mal_types}")
thr = cfg["spatial"]["tumor_fraction_threshold"]


def hex_neighbors(obs: pd.DataFrame) -> sp.csr_matrix:
    """Visium hexagonal-grid adjacency from array coordinates (6 neighbours)."""
    key = {(r, c): i for i, (r, c) in enumerate(zip(obs["array_row"].astype(int), obs["array_col"].astype(int)))}
    rows, cols = [], []
    for (r, c), i in key.items():
        for dr, dc in ((0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            j = key.get((r + dr, c + dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(obs), len(obs)))


def morans_i(x: np.ndarray, W: sp.csr_matrix) -> float:
    z = x - x.mean()
    s0 = W.sum()
    if s0 == 0 or np.allclose(z, 0):
        return np.nan
    return float(len(x) / s0 * (z @ (W @ z)) / (z @ z))


def morans_perm_p(x, W, n=999, seed=0):
    obs_i = morans_i(x, W)
    rng = np.random.default_rng(seed)
    perm = np.array([morans_i(rng.permutation(x), W) for _ in range(n)])
    return obs_i, float((np.sum(perm >= obs_i) + 1) / (n + 1))


def process(name: str, h5: Path, has_coords: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, obs = load_inputs(P, name)
    a = ad.read_h5ad(h5)
    a = a[obs.index.astype(str)].copy()
    spots = obs.copy()
    for v, m in ens.items():
        p = ensemble_predict(m, X, device)
        p.index = spots.index
        spots = spots.join(p.add_suffix("" if v == "full" else f"__{v}"))
    common = refdf.columns.intersection(a.var_names)
    mk = [g for g in marker_genes(refdf[common]) if g in common]
    mix = pd.DataFrame(cpm_dense(a.X), columns=a.var_names, index=spots.index)[mk].values
    fr = pd.DataFrame(nnls_deconvolve(mix, refdf[mk].values), index=spots.index,
                      columns=[f"frac_{re.sub(r'[^A-Za-z0-9]+', '_', c)}" for c in refdf.index])
    spots = spots.join(fr)
    mcols = [f"frac_{re.sub(r'[^A-Za-z0-9]+', '_', c)}" for c in mal_types]
    spots["malignant_fraction"] = spots[mcols].sum(axis=1)
    spots["tumor_spot"] = spots["malignant_fraction"] >= thr
    spots.to_csv(R / f"{name}_spot_predictions.csv")

    sec_rows, nb_rows = [], []
    for sec, s in spots.groupby("section"):
        t = s[s["tumor_spot"]]
        rec = {"section": sec, "n_spots": len(s), "n_tumor_spots": len(t),
               "response": s["response"].iloc[0], "patient": s.get("patient", s["section"]).iloc[0]}
        if len(t) >= 10:
            v = t["p_hr"].values
            rec.update({"mean_p_hr": v.mean(), "sd_p_hr": v.std(), "iqr_p_hr": np.subtract(*np.percentile(v, [75, 25])),
                        "frac_hrd_high": float((v >= 0.5).mean()), "mean_pred_hrd": t["pred_hrd"].mean(),
                        "mean_p_plat": t["p_plat"].mean(), "mean_risk": t["risk"].mean()})
            if "p_hr__no_da" in t:
                rec["mean_p_hr__no_da"] = t["p_hr__no_da"].mean()
                rec["mean_p_plat__no_da"] = t["p_plat__no_da"].mean()
            if has_coords and {"array_row", "array_col"} <= set(t.columns):
                W = hex_neighbors(t)
                rec["morans_I_p_hr"], rec["morans_p"] = morans_perm_p(v, W, seed=0)
        sec_rows.append(rec)
        for c in [c for c in s.columns if c.startswith("frac_")]:
            d = s[["p_hr", c]].dropna()
            if d[c].std() > 0 and len(d) > 30:
                r = spearmanr(d["p_hr"], d[c])
                nb_rows.append({"section": sec, "response": rec["response"], "cell_type": c[5:],
                                "spearman": r.correlation, "p": r.pvalue})
    sec = pd.DataFrame(sec_rows)
    sec.to_csv(R / f"{name}_section_metrics.csv", index=False)
    nbd = pd.DataFrame(nb_rows)
    if len(nbd):
        summ = nbd.groupby("cell_type").agg(mean_spearman=("spearman", "mean"),
                                            n_sections=("spearman", "size"),
                                            frac_positive=("spearman", lambda v: float((v > 0).mean())))
        nbd.to_csv(R / f"{name}_neighbourhood_per_section.csv", index=False)
        summ.to_csv(R / f"{name}_neighbourhood_summary.csv")

    # section pseudobulk of tumour spots -> bulk-like prediction per section
    pb_rows, pb_idx = [], []
    for s_name, s in spots.groupby("section"):
        ids = s.index[s["tumor_spot"]] if s["tumor_spot"].sum() >= 10 else s.index
        pb_rows.append(sp.csr_matrix(a[ids].X).sum(axis=0))
        pb_idx.append(s_name)
    pb = ad.AnnData(X=sp.csr_matrix(np.vstack(pb_rows).astype(np.float32)), obs=pd.DataFrame(index=pb_idx),
                    var=pd.DataFrame(index=a.var_names))
    pb.uns["value_type"] = "counts"
    st = np.load(P / "features" / "cohort_stats_spectrum_pseudobulk.npz")
    Xpb, _ = model_input(pb, feat["genes"], stats=(st["mu"], st["sd"]))
    ppb = ensemble_predict(ens["full"], Xpb, device)
    ppb.index = pb_idx
    sec = sec.set_index("section").join(ppb.add_prefix("pseudobulk_"))
    sec.to_csv(R / f"{name}_section_metrics.csv")
    return spots, sec


tests = []
den_spots, den = process("denisenko_spots", P / "spatial" / "denisenko_spots.h5ad", True)
den["ordinal"] = den["response"].map({"good": 0, "partial": 1, "poor": 2})
for m in [c for c in den.columns if c.startswith(("mean_", "sd_", "iqr_", "frac_hrd", "morans_I", "pseudobulk_"))]:
    d = den[[m, "ordinal"]].dropna()
    if len(d) >= 5:
        r = spearmanr(d[m], d["ordinal"])
        groups = [g[m].values for _, g in den.dropna(subset=[m]).groupby("response")]
        tests.append({"cohort": "Denisenko (post-NACT)", "metric": m, "test": "Spearman vs CRS response (good<partial<poor)",
                      "estimate": r.correlation, "p_value": r.pvalue, "n_sections": len(d),
                      "kruskal_p": kruskal(*groups).pvalue if len(groups) > 1 and all(len(g) > 0 for g in groups) else np.nan})

stur_spots, stur = process("stur_spots", P / "spatial" / "stur_spots.h5ad", False)
stur["poor"] = (stur["response"] == "poor").astype(float)
for m in [c for c in stur.columns if c.startswith(("mean_", "sd_", "iqr_", "frac_hrd", "pseudobulk_"))]:
    d = stur[[m, "poor"]].dropna()
    if len(d) >= 6 and d["poor"].nunique() == 2:
        # pre-specified direction: HRD-type scores are expected to be LOWER in poor responders
        hrd_like = (m.startswith(("mean_p_hr", "mean_pred_hrd", "frac_hrd", "pseudobulk_p_hr", "pseudobulk_pred_hrd",
                                  "pseudobulk_p_brca")))
        sign = -1.0 if hrd_like else 1.0
        est, lo, hi = bootstrap_ci(auroc, d["poor"].values, sign * d[m].values, n_boot=nb)
        tests.append({"cohort": "Stur (pre-treatment)", "metric": m, "direction": "lower in poor responders" if hrd_like
                      else "higher in poor responders", "test": "AUROC poor vs excellent responders (pre-specified direction)",
                      "estimate": est, "ci_low": lo, "ci_high": hi, "n_sections": len(d),
                      "p_value": mannwhitneyu(d.loc[d.poor == 1, m], d.loc[d.poor == 0, m]).pvalue})
pd.DataFrame(tests).to_csv(R / "spatial_response_tests.csv", index=False)
log.info("Spatial response tests:\n" + pd.DataFrame(tests).to_string(index=False))
