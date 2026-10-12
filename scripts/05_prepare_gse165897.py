#!/usr/bin/env python
"""05 - GSE165897 (Zhang et al. 2022): paired treatment-naive / post-NACT scRNA-seq.

Builds malignant (EOC) metacells per sample, used as an unlabelled alignment domain and for the
paired pre/post-chemotherapy analysis.
Outputs: Processed/gse165897/gse165897_metacells.h5ad, gse165897_cells.h5ad (malignant cells, counts)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hrdx.singlecell import aggregate, metacell_labels
from hrdx.utils import collapse_sum, ensure_dir, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("05_prepare_gse165897", cfg)
G = cfg["files"]["gse165897"]
out = ensure_dir(cfg["paths"]["processed"] / "gse165897")
scfg = cfg["singlecell"]

info = pd.read_csv(inp(cfg, G["cellinfo"]), sep="\t")
log.info(f"cellInfo columns: {list(info.columns)}")
for c in info.columns:
    log.info(f"  {c}: {info[c].astype(str).unique()[:6].tolist()}")


def pick(patterns, exclude=()):
    for p in patterns:
        for c in info.columns:
            if re.search(p, c, re.I) and not any(re.search(e, c, re.I) for e in exclude):
                return c
    return None


cell_col = pick([r"^cell$", r"cell.?id", r"barcode"]) or info.columns[0]
type_col = pick([r"cell.?type", r"cell.?subtype", r"celltype"])
phase_col = pick([r"treatment", r"phase", r"timepoint", r"time"])
pat_col = pick([r"patient"])
samp_col = pick([r"^sample$", r"sample"], exclude=[r"type"])
log.info(f"Using cell={cell_col}, type={type_col}, phase={phase_col}, patient={pat_col}, sample={samp_col}")
info = info.set_index(cell_col)
info.index = info.index.astype(str)
mal_re = re.compile(scfg["malignant_regex"], re.I)
info["is_malignant"] = info[type_col].astype(str).map(lambda v: bool(mal_re.search(v)))
ph = info[phase_col].astype(str).str.lower()
info["phase"] = np.where(ph.str.contains("naive|pre|before|treatment-na"), "pre",
                         np.where(ph.str.contains("post|after|nact"), "post", ph))
if samp_col is None:
    info["sample"] = info[pat_col].astype(str) + "_" + info["phase"]
    samp_col = "sample"
log.info(f"Malignant cells: {info['is_malignant'].sum()} / {len(info)}; phases: {info['phase'].value_counts().to_dict()}")

mal_ids = set(info.index[info["is_malignant"]])
parts, genes, cols_all = [], None, None
reader = pd.read_csv(inp(cfg, G["counts"]), sep="\t", index_col=0, chunksize=2000)
gene_names = []
for k, ch in enumerate(reader):
    if cols_all is None:
        cols_all = ch.columns.astype(str)
        keep = np.array([c in mal_ids for c in cols_all])
        log.info(f"Counts matrix: {len(cols_all)} cells; keeping {keep.sum()} malignant cells")
    parts.append(sp.csr_matrix(ch.values[:, keep].astype(np.float32)))
    gene_names.extend(ch.index.astype(str))
    if k % 5 == 0:
        log.info(f"  read {len(gene_names)} genes")
X = sp.vstack(parts).T.tocsr()
X, genes = collapse_sum(X, gene_names)
cells = cols_all[keep]
cobs = info.loc[cells, [pat_col, samp_col, "phase", type_col]].rename(columns={pat_col: "patient", samp_col: "sample"})
cobj = ad.AnnData(X=X, obs=cobs, var=pd.DataFrame(index=genes))
cobj.uns["value_type"] = "counts"
cobj.write_h5ad(out / "gse165897_cells.h5ad", compression="gzip")

mats, mobs = [], []
for k, (s, sub) in enumerate(cobs.groupby("sample", sort=True)):
    if len(sub) < scfg["min_cells_per_sample"]:
        continue
    idx = cobs.index.get_indexer(sub.index)
    M = X[idx]
    labels = metacell_labels(M, size=scfg["metacell_size"], seed=k)
    mats.append(aggregate(M, labels))
    mobs.append(pd.DataFrame({"sample": s, "patient": sub["patient"].iloc[0], "phase": sub["phase"].iloc[0],
                              "n_cells": np.bincount(labels)},
                             index=[f"{s}::mc{l}" for l in range(labels.max() + 1)]))
mc = ad.AnnData(X=sp.vstack(mats).tocsr(), obs=pd.concat(mobs), var=pd.DataFrame(index=genes))
mc.uns["value_type"] = "counts"
mc.write_h5ad(out / "gse165897_metacells.h5ad", compression="gzip")
log.info(f"Metacells: {mc.n_obs} from {mc.obs['sample'].nunique()} samples / {mc.obs['patient'].nunique()} patients")
