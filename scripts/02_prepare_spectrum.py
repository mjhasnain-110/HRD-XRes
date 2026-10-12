#!/usr/bin/env python
"""02 - MSK SPECTRUM single-cell data (memory-safe chunked processing of the 11 GB atlas).

Outputs (Processed/spectrum/):
  spectrum_pseudobulk.h5ad    one profile per tumour site (CD45+ and CD45- fractions pooled), raw counts
  spectrum_reference.h5ad     cell-type reference profiles (mean CPM) for deconvolution
  spectrum_metacells.h5ad     malignant-cell metacells (~30 cells), raw counts + module/BAF summaries
  spectrum_malignant_cells.csv per-cell table of malignant cells (row index in the epithelial file)
  spectrum_patient_labels.csv  WGS-derived HR status and signature subtype per patient
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hrdx.genes import read_h5ad_df, spectrum_files, spectrum_var_symbols
from hrdx.singlecell import aggregate, metacell_labels
from hrdx.utils import H5adCSR, collapse_matrix, ensure_dir, get_logger, group_sum, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("02_prepare_spectrum", cfg)
out = ensure_dir(cfg["paths"]["processed"] / "spectrum")
scfg = cfg["singlecell"]
files = spectrum_files(cfg)
log.info(f"SPECTRUM files: { {k: v.name for k, v in files.items()} }")

# ---------------------------------------------------------------- patient labels
supp = pd.read_excel(inp(cfg, cfg["files"]["spectrum"]["supp_table"]), sheet_name=cfg["files"]["spectrum"]["supp_sheet"])
supp.columns = [c.strip() for c in supp.columns]
supp["patient_id"] = supp["patient_id"].astype(str).str.strip()
supp = supp.set_index("patient_id")
lab = pd.DataFrame(index=supp.index)
lab["hr_status_wgs"] = supp["consensus_hr_status"].map({"HRD": 1.0, "HRP": 0.0})
lab["signature"] = supp["consensus_signature"].astype(str)
lab["wgs_signature"] = supp.get("wgs_signature", pd.Series(index=supp.index, dtype=object)).astype(str)
lab["myriad_status"] = supp.get("myriad_signature", pd.Series(index=supp.index, dtype=object)).map({"HRD": 1.0, "HRP": 0.0})
lab["myriad_gis"] = pd.to_numeric(supp.get("myriad_gis_score", pd.Series(index=supp.index, dtype=float)), errors="coerce")
lab.to_csv(out / "spectrum_patient_labels.csv")
log.info(f"Patient labels: {len(lab)} patients; HR status counts:\n{supp['consensus_hr_status'].value_counts().to_string()}")


def attach_labels(obs: pd.DataFrame, pcol: str) -> pd.DataFrame:
    return obs.join(lab, on=pcol)


# ---------------------------------------------------------------- atlas: pseudobulk + reference
obs = read_h5ad_df(files["atlas"], "obs")
for c in obs.columns:
    if obs[c].dtype.name == "category":
        obs[c] = obs[c].astype(str)
obs["site_sample"] = obs["author_sample_id"].str.replace(r"_CD45[NP]", "", regex=True)
ct_col = "author_cell_type" if "author_cell_type" in obs.columns else "cell_type"
log.info(f"Atlas: {len(obs)} cells, {obs['donor_id'].nunique()} patients, {obs['site_sample'].nunique()} sites; "
         f"cell types ({ct_col}):\n{obs[ct_col].value_counts().to_string()}")

sym = spectrum_var_symbols(files["atlas"])
genes, A = collapse_matrix(sym.values)
reader = H5adCSR(files["atlas"])
assert reader.shape == (len(obs), len(sym)), f"matrix {reader.shape} vs obs/var {(len(obs), len(sym))}"
log.info(f"Atlas matrix from {reader.source}; integer counts: {reader.is_integer_valued()}")

s_codes, s_levels = pd.factorize(obs["site_sample"])
c_codes, c_levels = pd.factorize(obs[ct_col])
S = np.zeros((len(s_levels), len(genes)), dtype=np.float64)
C = np.zeros((len(c_levels), len(genes)), dtype=np.float64)
for a, b, M in reader.iter_chunks(25000):
    M = (M @ A).tocsr()
    S += group_sum(M, s_codes[a:b], len(s_levels)).toarray()
    C += group_sum(M, c_codes[a:b], len(c_levels)).toarray()
    if (a // 25000) % 5 == 0:
        log.info(f"  atlas rows {b}/{reader.shape[0]}")
reader.close()

site_meta = obs.groupby("site_sample").agg(
    patient=("donor_id", "first"), site=("author_tumor_site", "first"),
    supersite=("author_tumor_supersite", "first"), n_cells=("donor_id", "size")).reindex(s_levels)
comp = pd.crosstab(obs["site_sample"], obs[ct_col], normalize="index").reindex(s_levels)
comp.columns = [f"frac_{re.sub(r'[^A-Za-z0-9]+', '_', c)}" for c in comp.columns]
site_meta = attach_labels(pd.concat([site_meta, comp], axis=1), "patient")
pb = ad.AnnData(X=sp.csr_matrix(S.astype(np.float32)), obs=site_meta, var=pd.DataFrame(index=genes))
pb.uns["value_type"] = "counts"
pb.write_h5ad(out / "spectrum_pseudobulk.h5ad", compression="gzip")

n_ct = obs[ct_col].value_counts().reindex(c_levels)
lib = C.sum(axis=1, keepdims=True)
ref = ad.AnnData(X=(C / np.maximum(lib, 1) * 1e6).astype(np.float32),
                 obs=pd.DataFrame({"n_cells": n_ct.values}, index=c_levels), var=pd.DataFrame(index=genes))
ref = ref[ref.obs["n_cells"] >= 200].copy()
ref.write_h5ad(out / "spectrum_reference.h5ad", compression="gzip")
log.info(f"Pseudobulk: {pb.n_obs} sites; reference: {ref.n_obs} cell types")

# ---------------------------------------------------------------- malignant metacells
eobs = read_h5ad_df(files["epithelial"], "obs")
for c in eobs.columns:
    if eobs[c].dtype.name == "category":
        eobs[c] = eobs[c].astype(str)
mal_re, non_re = re.compile(scfg["malignant_regex"], re.I), re.compile(scfg["nonmalignant_regex"], re.I)
mal_col, is_mal = None, None
for col in ["cluster_label", "cell_type_super", "author_cell_type", "cell_type"]:
    if col not in eobs.columns:
        continue
    vals = eobs[col].astype(str)
    flag = vals.map(lambda v: bool(mal_re.search(v)) and not bool(non_re.search(v))).values
    if 0 < flag.sum() < len(flag):
        mal_col, is_mal = col, flag
        break
if mal_col is None:
    raise RuntimeError("Could not identify malignant cells; adjust singlecell.malignant_regex in config")
log.info(f"Malignant cells defined by '{mal_col}':\n{eobs[mal_col].value_counts().to_string()}\n"
         f"-> {is_mal.sum()} malignant cells")

pcol = "donor_id" if "donor_id" in eobs.columns else "patient_id"
eobs["site_sample"] = eobs["author_sample_id"].str.replace(r"_CD45[NP]", "", regex=True)
baf_cols = [c for c in eobs.columns if c.startswith("BAF_")]
mod_cols = [c for c in eobs.columns if c.endswith(".module") or c.endswith(".pathway") or c.startswith("HALLMARK_")]
if baf_cols:
    baf = eobs[baf_cols].apply(pd.to_numeric, errors="coerce").values
    eobs["allelic_imbalance"] = np.nanmean(np.abs(baf - 0.5), axis=1) * 2.0
    eobs["frac_arms_imbalanced"] = np.nanmean(np.abs(baf - 0.5) > 0.1, axis=1)
summary_cols = mod_cols + [c for c in ("allelic_imbalance", "frac_arms_imbalanced") if c in eobs.columns]

esym = spectrum_var_symbols(files["epithelial"])
egenes, EA = collapse_matrix(esym.values)
er = H5adCSR(files["epithelial"])
assert er.shape[0] == len(eobs)
log.info(f"Epithelial matrix from {er.source}; integer counts: {er.is_integer_valued()}")

mal = eobs[is_mal].copy()
mal["row"] = np.where(is_mal)[0]
mats, mobs, cell_tabs = [], [], []
for k, (samp, sub) in enumerate(mal.groupby("site_sample", sort=True)):
    if len(sub) < scfg["min_cells_per_sample"]:
        continue
    M = (er.read_rows(sub["row"].values) @ EA).tocsr()
    labels = metacell_labels(M, size=scfg["metacell_size"], seed=k)
    agg = aggregate(M, labels)
    sub = sub.assign(metacell=[f"{samp}::mc{l}" for l in labels])
    grp = sub.groupby("metacell")
    mo = pd.DataFrame({"site_sample": samp, "patient": sub[pcol].iloc[0],
                       "site": sub["author_tumor_site"].iloc[0], "supersite": sub["author_tumor_supersite"].iloc[0],
                       "n_cells": grp.size()})
    if summary_cols:
        mo = mo.join(grp[summary_cols].mean(numeric_only=True))
    mo = mo.reindex([f"{samp}::mc{l}" for l in range(labels.max() + 1)])
    mats.append(agg)
    mobs.append(mo)
    cell_tabs.append(sub[["row", "site_sample", pcol, "author_tumor_site", "metacell"] +
                         [c for c in ("allelic_imbalance", "frac_arms_imbalanced", "CIN.module") if c in sub.columns]])
    if k % 25 == 0:
        log.info(f"  metacells: processed {k + 1} samples")
er.close()

mobs = attach_labels(pd.concat(mobs), "patient")
mc = ad.AnnData(X=sp.vstack(mats).tocsr().astype(np.float32), obs=mobs, var=pd.DataFrame(index=egenes))
mc.uns["value_type"] = "counts"
mc.write_h5ad(out / "spectrum_metacells.h5ad", compression="gzip")
cells = pd.concat(cell_tabs).rename(columns={pcol: "patient"})
cells.to_csv(out / "spectrum_malignant_cells.csv")
overlap = len(set(mobs["patient"]) & set(lab.index))
log.info(f"Metacells: {mc.n_obs} from {mobs['site_sample'].nunique()} sites / {mobs['patient'].nunique()} patients "
         f"({overlap} patients matched to WGS labels)")
