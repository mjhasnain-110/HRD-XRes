#!/usr/bin/env python
"""03 - Spatial transcriptomics.

Denisenko et al. 2024 (GSE211956): 8 post-NACT Visium sections with coordinates and chemotherapy
response score (good / partial / poor), plus 5 matched scRNA-seq samples.
Stur et al. 2022 (GSE189843): 12 pre-treatment Visium sections, excellent vs poor responders.

Outputs (Processed/spatial/): denisenko_spots.h5ad, stur_spots.h5ad, denisenko_sc.h5ad
"""
import gzip
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from hrdx.geo import parse_series_matrix
from hrdx.utils import collapse_sum, download, ensure_dir, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("03_prepare_spatial", cfg)
out = ensure_dir(cfg["paths"]["processed"] / "spatial")
qc = cfg["spatial"]


def series_meta(gse: str, matrix_dir: Path) -> pd.DataFrame:
    files = sorted(matrix_dir.glob("*series_matrix.txt.gz"))
    if not files:
        stub = gse[:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/matrix/{gse}_series_matrix.txt.gz"
        files = [download(url, matrix_dir / f"{gse}_series_matrix.txt.gz", log)]
    return pd.concat([parse_series_matrix(f)[0] for f in files])


def read_10x(prefix_files: dict) -> ad.AnnData:
    with gzip.open(prefix_files["matrix"], "rb") as fh:
        M = sp.csr_matrix(scipy.io.mmread(fh)).T.tocsr().astype(np.float32)  # barcodes x genes
    bc = pd.read_csv(prefix_files["barcodes"], header=None, sep="\t")[0].astype(str).values
    feat = pd.read_csv(prefix_files["features"], header=None, sep="\t")
    sym = feat[1].astype(str).values if feat.shape[1] > 1 else feat[0].astype(str).values
    X, genes = collapse_sum(M, sym)
    return ad.AnnData(X=X, obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=genes))


def spot_qc(a: ad.AnnData) -> ad.AnnData:
    counts = np.asarray(a.X.sum(axis=1)).ravel()
    ngenes = np.asarray((a.X > 0).sum(axis=1)).ravel()
    a.obs["total_counts"], a.obs["n_genes"] = counts, ngenes
    return a[(counts >= qc["min_counts"]) & (ngenes >= qc["min_genes"])].copy()


def read_positions(zpath: Path, dest: Path) -> pd.DataFrame:
    ensure_dir(dest)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    pos_file = next(iter(sorted(dest.rglob("tissue_positions*.csv"))), None)
    if pos_file is None:
        raise FileNotFoundError(f"No tissue_positions file in {zpath}")
    head = open(pos_file).readline()
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    pos = pd.read_csv(pos_file, header=0 if "barcode" in head else None)
    pos.columns = cols[: pos.shape[1]]
    return pos.set_index("barcode")


# ---------------------------------------------------------------- Denisenko (GSE211956)
D = cfg["files"]["denisenko"]
draw = inp(cfg, D["raw_dir"])
dmeta = series_meta("GSE211956", ensure_dir(inp(cfg, D["matrix_dir"])))
title = dmeta["title"].to_dict()
sections = []
for mfile in sorted(draw.glob("GSM*_SP*_matrix.mtx.gz")):
    gsm, sec = mfile.name.split("_")[:2]
    files = {k: draw / f"{gsm}_{sec}_{k}.tsv.gz" for k in ("barcodes", "features")}
    files["matrix"] = mfile
    a = read_10x(files)
    pos = read_positions(draw / f"{gsm}_{sec}_spatial.zip", out / "denisenko_spatial" / sec)
    a.obs = a.obs.join(pos)
    a = a[a.obs["in_tissue"].fillna(0).astype(int) == 1].copy()
    t = title.get(gsm, "")
    m = re.search(r"(good|partial|poor)_response_(P\d+)", t)
    a.obs["section"], a.obs["gsm"] = sec, gsm
    a.obs["response"] = m.group(1) if m else "unknown"
    a.obs["patient"] = m.group(2) if m else sec
    a.obs_names = [f"{sec}_{b}" for b in a.obs_names]
    a = spot_qc(a)
    a.obsm["spatial"] = a.obs[["pxl_col", "pxl_row"]].values.astype(float)
    sections.append(a)
    log.info(f"Denisenko {sec} ({a.obs['patient'].iloc[0]}, {a.obs['response'].iloc[0]}): {a.n_obs} spots")
den = ad.concat(sections, join="outer", fill_value=0)
den.X = sp.csr_matrix(den.X, dtype=np.float32)
den.uns["value_type"] = "counts"
den.obs["response_ordinal"] = den.obs["response"].map({"good": 0, "partial": 1, "poor": 2}).astype(float)
den.write_h5ad(out / "denisenko_spots.h5ad", compression="gzip")

# Denisenko scRNA-seq (dense genes x cells text matrices)
sc_parts = []
for f in sorted(draw.glob("GSM*_counts_*.txt.gz")):
    gsm, _, sid = f.name.replace(".txt.gz", "").split("_", 2)
    chunks = []
    genes = []
    for ch in pd.read_csv(f, sep="\t", index_col=0, chunksize=4000):
        chunks.append(sp.csr_matrix(ch.values.astype(np.float32)))
        genes.extend(ch.index.astype(str))
        cols = ch.columns
    X, g = collapse_sum(sp.vstack(chunks).T.tocsr(), genes)
    a = ad.AnnData(X=X, obs=pd.DataFrame({"sample": sid, "gsm": gsm}, index=[f"{sid}_{c}" for c in cols]),
                   var=pd.DataFrame(index=g))
    sc_parts.append(a)
    log.info(f"Denisenko scRNA {sid}: {a.n_obs} cells")
if sc_parts:
    dsc = ad.concat(sc_parts, join="outer", fill_value=0)
    dsc.X = sp.csr_matrix(dsc.X, dtype=np.float32)
    dsc.uns["value_type"] = "counts"
    dsc.write_h5ad(out / "denisenko_sc.h5ad", compression="gzip")

# ---------------------------------------------------------------- Stur (GSE189843)
S = cfg["files"]["stur"]
sraw = inp(cfg, S["raw_dir"])
smeta = series_meta("GSE189843", ensure_dir(inp(cfg, S["matrix_dir"])))
varying = [c for c in smeta.columns if smeta[c].astype(str).nunique() > 1]  # sample-specific fields only
text = smeta[varying].astype(str).apply(lambda r: " | ".join(r.values), axis=1).str.lower()
resp = {}
for gsm, t in text.items():
    poor = re.search(r"poor|\bpr\b|non.?respon", t)
    exc = re.search(r"excellent|\ber\b|good respon", t)
    if poor and not exc:
        resp[gsm] = "poor"
    elif exc and not poor:
        resp[gsm] = "excellent"
override = inp(cfg, S["label_override"])
if len(resp) < len(smeta) and override.exists():
    ov = pd.read_csv(override, sep="\t")
    resp.update(dict(zip(ov["gsm"].astype(str), ov["response"].astype(str).str.lower())))
log.info(f"Stur response labels resolved for {len(resp)}/{len(smeta)} samples: "
         f"{pd.Series(resp).value_counts().to_dict()}")
if len(resp) < len(smeta):
    tmpl = override if not override.exists() else override.with_suffix(".template.tsv")
    pd.DataFrame({"gsm": smeta.index, "title": smeta["title"], "response": ""}).to_csv(tmpl, sep="\t", index=False)
    raise SystemExit(f"Response labels for some Stur samples could not be parsed from GEO. "
                     f"Fill the 'response' column (excellent/poor) in {tmpl} using Stur et al. 2022 Table S1, "
                     f"save it as {override} and re-run this script.")

parts = []
for mfile in sorted(sraw.glob("GSM*_matrix_*.mtx.gz")):
    gsm = mfile.name.split("_")[0]
    sid = mfile.name.replace(".mtx.gz", "").split("_matrix_")[1]
    files = {"matrix": mfile, "barcodes": sraw / f"{gsm}_barcodes_{sid}.tsv.gz",
             "features": sraw / f"{gsm}_features_{sid}.tsv.gz"}
    a = read_10x(files)
    a.obs_names = [f"{sid}_{b}" for b in a.obs_names]
    a.obs["section"], a.obs["gsm"], a.obs["response"] = sid, gsm, resp[gsm]
    a = spot_qc(a)
    parts.append(a)
    log.info(f"Stur {sid} ({resp[gsm]}): {a.n_obs} spots")
stur = ad.concat(parts, join="outer", fill_value=0)
stur.X = sp.csr_matrix(stur.X, dtype=np.float32)
stur.obs["poor_response"] = (stur.obs["response"] == "poor").astype(float)
stur.uns["value_type"] = "counts"
stur.write_h5ad(out / "stur_spots.h5ad", compression="gzip")
log.info(f"Saved spatial objects to {out}")
