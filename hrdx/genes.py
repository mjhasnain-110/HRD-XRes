"""Ensembl -> symbol mapping (built from the downloaded datasets themselves) and SPECTRUM file discovery."""
from __future__ import annotations

import gzip
from pathlib import Path

import h5py
import pandas as pd

from .utils import ensure_dir, inp, strip_version

try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # pragma: no cover
    from anndata.experimental import read_elem


def read_h5ad_df(path: Path, key: str) -> pd.DataFrame:
    """Read obs / var / raw/var of an h5ad without touching the expression matrix."""
    with h5py.File(path, "r") as f:
        return read_elem(f[key])


def spectrum_files(cfg: dict) -> dict[str, Path]:
    """Identify SPECTRUM h5ad files from the CELLxGENE title table."""
    tab = pd.read_csv(inp(cfg, cfg["files"]["spectrum"]["url_table"]), sep="\t", header=None, names=["title", "url"])
    d = inp(cfg, cfg["files"]["spectrum"]["dir"])
    out = {}
    for _, r in tab.iterrows():
        p = d / r["url"].rsplit("/", 1)[-1]
        t = r["title"].lower()
        if "all cells" in t:
            out["atlas"] = p
        elif "epithelial" in t:
            out["epithelial"] = p
        elif "macrophage" in t:
            out["macrophages"] = p
        elif "dendritic" in t:
            out["dendritic"] = p
        elif "t cells" in t:
            out["tnk"] = p
    missing = [k for k in ("atlas", "epithelial") if k not in out or not out[k].exists()]
    if missing:
        raise FileNotFoundError(f"SPECTRUM files not found: {missing}")
    return out


def spectrum_var_symbols(path: Path, raw: bool = True) -> pd.Series:
    """Symbols for the columns of (raw/)X, indexed by Ensembl ID."""
    with h5py.File(path, "r") as f:
        has_raw = raw and "raw" in f and "var" in f["raw"]
    var = read_h5ad_df(path, "var")
    sym = var["feature_name"].astype(str) if "feature_name" in var.columns else pd.Series(var.index, index=var.index)
    if has_raw:
        rv = read_h5ad_df(path, "raw/var")
        if "feature_name" in rv.columns:
            return pd.Series(rv["feature_name"].astype(str).values, index=rv.index)
        return pd.Series(sym.reindex(rv.index).fillna(pd.Series(rv.index, index=rv.index)).values, index=rv.index)
    return pd.Series(sym.values, index=var.index)


def _features_tsv(path: Path) -> pd.Series:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = [l.rstrip("\n").split("\t") for l in fh if l.strip()]
    return pd.Series({r[0]: r[1] for r in rows if len(r) > 1 and r[0].startswith("ENSG")})


def build_gene_map(cfg: dict, logger=None) -> pd.Series:
    """Ensembl ID (unversioned) -> gene symbol, cached in Processed/gene_map.tsv."""
    cache = ensure_dir(cfg["paths"]["processed"]) / "gene_map.tsv"
    if cache.exists():
        s = pd.read_csv(cache, sep="\t", index_col=0).iloc[:, 0]
        return s
    maps = []
    try:
        files = spectrum_files(cfg)
        smallest = min(files.values(), key=lambda p: p.stat().st_size)
        s = spectrum_var_symbols(smallest)
        s.index = strip_version(s.index)
        maps.append(s)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning(f"SPECTRUM gene names unavailable for gene map: {exc}")
    for key in ("denisenko", "stur"):
        raw = inp(cfg, cfg["files"][key]["raw_dir"])
        for f in sorted(raw.glob("*features*.tsv.gz"))[:1]:
            s = _features_tsv(f)
            s.index = strip_version(s.index)
            maps.append(s)
    gm = pd.concat(maps)
    gm = gm[~gm.index.duplicated()]
    gm = gm[gm.astype(str).str.len() > 0]
    gm.rename("symbol").to_csv(cache, sep="\t")
    if logger:
        logger.info(f"Gene map: {len(gm)} Ensembl IDs -> {gm.nunique()} symbols")
    return gm
