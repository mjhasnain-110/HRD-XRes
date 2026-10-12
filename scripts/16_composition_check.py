#!/usr/bin/env python
"""16 - Model-free check of the spatial neighbourhood finding.

Question: are the cell types that sit next to HRD-high tumour regions in Visium (dendritic cells, mast cells,
endothelial cells, B and T cells) also more abundant in tumours that are HRD by WHOLE-GENOME SEQUENCING?
No prediction model is used here - only SPECTRUM cell-type annotations and WGS labels, so the test cannot be
circular.

SPECTRUM sorted cells into CD45+ (immune) and CD45- fractions, so raw "fraction of all cells" depends on the
sorting ratio. Main analysis therefore uses compartment-normalised fractions:
  immune cell types   -> fraction of all immune cells at that site
  stromal cell types  -> fraction of all non-immune, non-malignant cells at that site
Raw fractions are reported as a secondary analysis.

Tests: patient level (mean over sites) HRD vs HRP (Mann-Whitney) and WGS subtypes (Kruskal);
       site level, stratified by anatomical supersite (Stouffer combination of signed site-level tests).
Pre-specified direction from the spatial analysis: higher in HRD.

Run:  python scripts/16_composition_check.py > Logs/16.out 2>&1
Outputs: Results/composition/
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, norm

from hrdx.utils import ensure_dir, get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("16_composition_check", cfg)
P, RES = cfg["paths"]["processed"], cfg["paths"]["results"]
R = ensure_dir(RES / "composition")

IMMUNE = re.compile(r"T_cell|B_cell|Plasma|Dendritic|Mast|Monocyte|Macrophage|Myeloid|NK|Neutrophil|pDC|Lymph", re.I)
MALIG = re.compile(r"cancer|malignan|tumou?r|epithel", re.I)
SPATIAL_HITS = ["Dendritic_cell", "Mast_cell", "Endothelial_cell", "B_cell", "T_cell"]  # from step 15, part N


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        q = p[ok]
        o = np.argsort(q)
        a = q[o] * len(q) / (np.arange(len(q)) + 1)
        a = np.minimum.accumulate(a[::-1])[::-1]
        r = np.empty_like(a)
        r[o] = np.clip(a, 0, 1)
        out[ok] = r
    return out


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b))


obs = ad.read_h5ad(P / "spectrum" / "spectrum_pseudobulk.h5ad", backed="r").obs.copy()
obs.index = obs.index.astype(str)
labels = pd.read_csv(P / "spectrum" / "spectrum_patient_labels.csv", index_col=0)
labels.index = labels.index.astype(str)
obs["patient"] = obs["patient"].astype(str)
for c in ("hr_status_wgs", "signature"):
    if c in obs:
        obs = obs.drop(columns=c)
obs = obs.join(labels[["hr_status_wgs", "signature"]], on="patient")

fcols = [c for c in obs.columns if c.startswith("frac_")]
types = [c[5:] for c in fcols]
imm = [t for t in types if IMMUNE.search(t)]
mal = [t for t in types if MALIG.search(t)]
stro = [t for t in types if t not in imm and t not in mal]
log.info(f"Cell types: immune={imm}; stromal={stro}; malignant={mal}")

F = obs[fcols].astype(float).copy()
F.columns = types
imm_tot = F[imm].sum(axis=1)
stro_tot = F[stro].sum(axis=1)
norm_frac = pd.DataFrame(index=F.index)
for t in imm:
    norm_frac[t] = np.where(imm_tot > 0, F[t] / imm_tot, np.nan)
for t in stro:
    norm_frac[t] = np.where(stro_tot > 0, F[t] / stro_tot, np.nan)
norm_frac["immune_fraction_of_all"] = imm_tot
site = obs[["patient", "supersite", "n_cells", "hr_status_wgs", "signature"]].join(norm_frac)
site_raw = obs[["patient", "supersite", "hr_status_wgs", "signature"]].join(F)
site.to_csv(R / "site_composition_normalised.csv")

rows = []
for analysis, S in (("compartment-normalised", site), ("raw fraction of all cells", site_raw)):
    S = S[S["hr_status_wgs"].isin([0, 1])]
    feats = [c for c in S.columns if c not in ("patient", "supersite", "n_cells", "hr_status_wgs", "signature")]
    pat = S.groupby("patient")[feats].mean().join(labels[["hr_status_wgs", "signature"]])
    for f in feats:
        h, p_ = pat.loc[pat.hr_status_wgs == 1, f].dropna(), pat.loc[pat.hr_status_wgs == 0, f].dropna()
        if len(h) < 3 or len(p_) < 3:
            continue
        mw = mannwhitneyu(h, p_, alternative="two-sided")
        rec = {"analysis": analysis, "cell_type": f, "level": "patient (mean of sites)",
               "n_hrd": len(h), "n_hrp": len(p_), "median_hrd": h.median(), "median_hrp": p_.median(),
               "cliffs_delta_hrd_vs_hrp": cliffs_delta(h, p_), "p_two_sided": mw.pvalue,
               "p_one_sided_higher_in_hrd": mannwhitneyu(h, p_, alternative="greater").pvalue}
        sig = pat[pat["signature"].isin(["HRD-Dup", "HRD-Del", "FBI"])]
        g = [x[f].dropna().values for _, x in sig.groupby("signature")]
        g = [x for x in g if len(x) > 1]
        if len(g) >= 2:
            rec["kruskal_p_subtypes"] = kruskal(*g).pvalue
            for s_ in ("HRD-Dup", "HRD-Del", "FBI"):
                rec[f"median_{s_}"] = sig.loc[sig.signature == s_, f].median()
        # site level, stratified by supersite (each stratum = one anatomical location)
        zs, ws = [], []
        for ss, g2 in S.groupby("supersite"):
            a, b = g2.loc[g2.hr_status_wgs == 1, f].dropna(), g2.loc[g2.hr_status_wgs == 0, f].dropna()
            if len(a) >= 2 and len(b) >= 2:
                pg = mannwhitneyu(a, b, alternative="greater").pvalue
                zs.append(norm.isf(np.clip(pg, 1e-12, 1 - 1e-12)))
                ws.append(np.sqrt(len(a) + len(b)))
        if zs:
            z = np.sum(np.array(ws) * np.array(zs)) / np.sqrt(np.sum(np.array(ws) ** 2))
            rec.update({"site_strata": len(zs), "site_stouffer_z": z, "site_p_one_sided_higher_in_hrd": norm.sf(z)})
        rows.append(rec)

res = pd.DataFrame(rows)
for an, idx in res.groupby("analysis").groups.items():
    res.loc[idx, "fdr_two_sided"] = bh(res.loc[idx, "p_two_sided"].values)
res["spatial_hit"] = res["cell_type"].isin(SPATIAL_HITS)
res["direction_matches_spatial"] = np.where(res["spatial_hit"], res["cliffs_delta_hrd_vs_hrp"] > 0, np.nan)
res.to_csv(R / "composition_hrd_vs_hrp.csv", index=False)

cols = ["analysis", "cell_type", "n_hrd", "n_hrp", "median_hrd", "median_hrp", "cliffs_delta_hrd_vs_hrp",
        "p_two_sided", "fdr_two_sided", "p_one_sided_higher_in_hrd", "kruskal_p_subtypes",
        "site_strata", "site_p_one_sided_higher_in_hrd", "spatial_hit"]
cols = [c for c in cols if c in res.columns]
pd.set_option("display.width", 250)
log.info("Cell-type composition, WGS HRD vs HRP (Cliff's delta > 0 = higher in HRD):\n" +
         res[cols].round(4).to_string(index=False))
hits = res[(res.analysis == "compartment-normalised") & res.spatial_hit]
if len(hits):
    n_same = int((hits["cliffs_delta_hrd_vs_hrp"] > 0).sum())
    log.info(f"Spatial hits with the same direction in SPECTRUM (compartment-normalised): {n_same}/{len(hits)}")
log.info(f"Done. Results in {R}")
