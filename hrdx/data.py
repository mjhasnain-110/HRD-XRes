"""Feature space, pathway mask, model-input I/O and deconvolution helpers."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls

from .utils import ensure_dir, load_json, to_log_expression, zscore_cols, zscore_rows

# domains used for adversarial alignment (unlabelled target domains)
TARGET_DOMAINS = ["spectrum_metacells", "spectrum_pseudobulk", "gse165897_metacells",
                  "denisenko_spots", "stur_spots"]


def read_gmt(path: Path) -> dict[str, set]:
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                out[f"{parts[0]} [{parts[1]}]"] = {g for g in parts[2:] if g}
    return out


def build_pathway_mask(genes: list[str], gmt: dict[str, set], min_size: int, max_size: int):
    gidx = {g: i for i, g in enumerate(genes)}
    kept, seen = [], set()
    for name, members in gmt.items():
        m = sorted(members & gidx.keys())
        if min_size <= len(m) <= max_size:
            key = tuple(m)
            if key in seen:  # drop exact duplicates
                continue
            seen.add(key)
            kept.append((name, m))
    mask = np.zeros((len(genes), len(kept)), dtype=np.float32)
    for j, (_, m) in enumerate(kept):
        mask[[gidx[g] for g in m], j] = 1.0
    return mask, [k for k, _ in kept]


def model_input(adata: ad.AnnData, genes: list[str], value_type: str | None = None,
                cohort_standardise: bool = True, stats: tuple | None = None, return_stats: bool = False):
    """AnnData (raw counts or log values) -> model-ready matrix in the shared gene space.

    1. log-transform on the full gene set (CPM uses the full library size),
    2. restrict to model genes (missing genes filled after standardisation with 0 = cohort mean),
    3. per-sample z-score, 4. per-cohort gene-wise z-score.
    Returns the matrix and the fraction of model genes present.
    """
    vt = value_type or adata.uns.get("value_type", "counts")
    present = pd.Index(adata.var_names).get_indexer(genes)
    have = present >= 0
    X = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    if vt == "counts":
        M = sp.csr_matrix(adata.X, dtype=np.float32) if sp.issparse(adata.X) else np.asarray(adata.X, np.float32)
        lib = np.asarray(M.sum(axis=1)).ravel()
        lib[lib == 0] = 1.0
        sub = M[:, present[have]]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        X[:, have] = np.log2(sub / lib[:, None] * 1e6 + 1.0)
    else:
        sub = adata.X[:, present[have]]
        X[:, have] = to_log_expression(sub, vt)
    Xh = zscore_rows(X[:, have])
    mu = sd = None
    if stats is not None:  # apply externally supplied cohort statistics (full gene space)
        mu, sd = stats
        Xh = ((Xh - mu[have]) / sd[have]).astype(np.float32)
    elif cohort_standardise:
        Xh, mh, sh = zscore_cols(Xh)
        mu, sd = np.zeros(len(genes), np.float32), np.ones(len(genes), np.float32)
        mu[have], sd[have] = mh, sh
    X[:, :] = 0.0
    X[:, have] = Xh
    if return_stats:
        return X, float(have.mean()), (mu, sd)
    return X, float(have.mean())


def save_inputs(path: Path, X: np.ndarray, obs: pd.DataFrame, coverage: float) -> None:
    ensure_dir(path.parent)
    np.save(path.with_suffix(".npy"), X.astype(np.float32))
    o = obs.copy()
    o.index = o.index.astype(str)
    o.to_csv(path.with_suffix(".obs.csv"))
    pd.Series({"coverage": coverage, "n": X.shape[0]}).to_csv(path.with_suffix(".info.csv"))


def load_inputs(processed: Path, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    base = processed / "model_inputs" / name
    X = np.load(base.with_suffix(".npy"))
    obs = pd.read_csv(base.with_suffix(".obs.csv"), index_col=0)
    return X, obs


def available_inputs(processed: Path) -> list[str]:
    d = processed / "model_inputs"
    return sorted(p.name[:-4] for p in d.glob("*.npy")) if d.exists() else []


def load_features(processed: Path) -> dict:
    f = load_json(processed / "features" / "features.json")
    f["mask"] = np.load(processed / "features" / "pathway_mask.npy")
    return f


def nnls_deconvolve(mix_cpm: np.ndarray, ref_cpm: np.ndarray, genes_idx: np.ndarray | None = None) -> np.ndarray:
    """Non-negative least squares deconvolution on log-free CPM (samples x genes; celltypes x genes).

    Returns cell-type fractions (rows sum to 1).
    """
    if genes_idx is not None:
        mix_cpm, ref_cpm = mix_cpm[:, genes_idx], ref_cpm[:, genes_idx]
    A = ref_cpm.T  # genes x celltypes
    scale = A.max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    A = A / scale
    out = np.zeros((mix_cpm.shape[0], ref_cpm.shape[0]))
    for i in range(mix_cpm.shape[0]):
        b = mix_cpm[i] / scale.ravel()
        coef, _ = nnls(A, b, maxiter=2000)
        s = coef.sum()
        out[i] = coef / s if s > 0 else coef
    return out


def marker_genes(ref_cpm: pd.DataFrame, n_per_type: int = 50, min_cpm: float = 10.0) -> list[str]:
    """Cell-type specific genes: highest log fold-change of each type vs the max of the others."""
    L = np.log2(ref_cpm + 1.0)
    genes = []
    for ct in L.index:
        others = L.drop(index=ct).max(axis=0)
        lfc = (L.loc[ct] - others)[ref_cpm.loc[ct] >= min_cpm]
        genes.extend(lfc.sort_values(ascending=False).index[:n_per_type].tolist())
    return list(dict.fromkeys(genes))


def cpm_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib[lib == 0] = 1
        return (sp.diags(1e6 / lib) @ X).toarray()
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    return X / lib * 1e6
