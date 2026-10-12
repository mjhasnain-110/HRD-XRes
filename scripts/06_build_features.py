#!/usr/bin/env python
"""06 - Shared gene space, Reactome pathway mask and model-ready matrices for every dataset.

Gene space = genes expressed in TCGA-OV  AND  measured in every RNA-seq domain (SPECTRUM, GSE165897,
Visium)  AND  member of at least one retained Reactome pathway (size limits from config).
External microarray cohorts are projected into this space; missing genes are set to the cohort mean (0)
after standardisation and gene coverage is recorded.

Outputs: Processed/features/{features.json, pathway_mask.npy, pathways.csv}
         Processed/model_inputs/<dataset>.{npy, obs.csv, info.csv}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hrdx.data import build_pathway_mask, model_input, read_gmt, save_inputs
from hrdx.utils import ensure_dir, get_logger, inp, load_config, save_json

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("06_build_features", cfg)
P = cfg["paths"]["processed"]
fcfg = cfg["features"]
fdir = ensure_dir(P / "features")

domains = {
    "tcga": P / "tcga" / "tcga_ov.h5ad",
    "spectrum_metacells": P / "spectrum" / "spectrum_metacells.h5ad",
    "spectrum_pseudobulk": P / "spectrum" / "spectrum_pseudobulk.h5ad",
    "gse165897_metacells": P / "gse165897" / "gse165897_metacells.h5ad",
    "denisenko_spots": P / "spatial" / "denisenko_spots.h5ad",
    "stur_spots": P / "spatial" / "stur_spots.h5ad",
}
missing = [k for k, v in domains.items() if not v.exists()]
if missing:
    raise FileNotFoundError(f"Missing processed datasets {missing}; run scripts 01-05 first")
objs = {k: ad.read_h5ad(v) for k, v in domains.items()}

tcga = objs["tcga"]
X = sp.csr_matrix(tcga.X) if sp.issparse(tcga.X) else sp.csr_matrix(np.asarray(tcga.X))
if tcga.uns.get("value_type", "counts") == "counts":
    lib = np.asarray(X.sum(axis=1)).ravel()
    cpm = sp.diags(1e6 / np.maximum(lib, 1)) @ X
    expressed = np.asarray((cpm >= fcfg["min_cpm"]).mean(axis=0)).ravel() >= fcfg["min_frac_samples"]
else:
    expressed = np.asarray((X >= 1).mean(axis=0)).ravel() >= fcfg["min_frac_samples"]
genes = set(tcga.var_names[expressed])
log.info(f"TCGA expressed genes: {len(genes)}")
for k, a in objs.items():
    if k != "tcga":
        genes &= set(a.var_names)
        log.info(f"  after intersect with {k}: {len(genes)}")

gmt = read_gmt(inp(cfg, cfg["files"]["reactome_gmt"]))
mask, pathways = build_pathway_mask(sorted(genes), gmt, fcfg["pathway_min_size"], fcfg["pathway_max_size"])
in_pw = mask.sum(axis=1) > 0
model_genes = [g for g, keep in zip(sorted(genes), in_pw) if keep]
mask = mask[in_pw]
log.info(f"Pathway layer: {len(model_genes)} genes x {len(pathways)} Reactome pathways "
         f"({int(mask.sum())} connections)")

np.save(fdir / "pathway_mask.npy", mask.astype(np.float32))
pd.DataFrame({"pathway": pathways, "n_genes": mask.sum(axis=0).astype(int)}).to_csv(fdir / "pathways.csv", index=False)
save_json({"genes": model_genes, "pathways": pathways, "domains": list(domains.keys()),
           "target_domains": [d for d in domains if d != "tcga"]}, fdir / "features.json")

mdir = ensure_dir(P / "model_inputs")
for k, a in objs.items():
    Xm, cov, (mu, sd) = model_input(a, model_genes, return_stats=True)
    np.savez(fdir / f"cohort_stats_{k}.npz", mu=mu, sd=sd)
    save_inputs(mdir / k, Xm, a.obs, cov)
    log.info(f"model input {k}: {Xm.shape}, gene coverage {cov:.3f}")

cov_rows = []
for p in sorted((P / "bulk_external").glob("*.h5ad")):
    a = ad.read_h5ad(p)
    Xm, cov = model_input(a, model_genes)
    cov_rows.append({"cohort": p.stem, "n": a.n_obs, "coverage": cov})
    if cov < fcfg["min_external_gene_coverage"]:
        log.warning(f"external {p.stem}: gene coverage {cov:.2f} below threshold, excluded")
        continue
    save_inputs(mdir / f"ext_{p.stem}", Xm, a.obs, cov)
    log.info(f"model input ext_{p.stem}: {Xm.shape}, gene coverage {cov:.3f}")
pd.DataFrame(cov_rows).to_csv(fdir / "external_gene_coverage.csv", index=False)
