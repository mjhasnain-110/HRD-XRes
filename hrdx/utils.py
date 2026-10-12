"""Shared utilities for HRD-XRes: configuration, logging, I/O, normalisation."""
from __future__ import annotations

import gzip
import json
import logging
import os
import random
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# configuration / logging
# ---------------------------------------------------------------------------
def load_config(path: str | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key, val in cfg["paths"].items():
        p = Path(os.path.expanduser(str(val)))
        cfg["paths"][key] = p if p.is_absolute() else PROJECT_ROOT / p
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg


def inp(cfg: dict, rel: str) -> Path:
    """Resolve a path relative to the inputs directory (or project root for config/)."""
    if rel.startswith("config/"):
        return PROJECT_ROOT / rel
    return cfg["paths"]["inputs"] / rel


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if cfg is not None:
        fh = logging.FileHandler(cfg["paths"]["logs"] / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(p: Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def save_json(obj, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def download(url: str, dest: Path, logger: logging.Logger | None = None) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    if logger:
        logger.info(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def open_text(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


# ---------------------------------------------------------------------------
# gene handling
# ---------------------------------------------------------------------------
def strip_version(ids) -> np.ndarray:
    out = []
    for g in ids:
        g = str(g)
        if g.startswith("ENSG"):
            g = g.split(".")[0]
        out.append(g)
    return np.asarray(out)


def collapse_matrix(names) -> tuple[np.ndarray, sp.csr_matrix]:
    """Return unique names and an (n_in x n_unique) 0/1 matrix that sums duplicate columns."""
    names = pd.Index(pd.Series(names).astype(str))
    uniq, inv = np.unique(names.values, return_inverse=True)
    A = sp.csr_matrix((np.ones(len(names), dtype=np.float32), (np.arange(len(names)), inv)),
                      shape=(len(names), len(uniq)))
    return uniq, A


def collapse_sum(X, names):
    """Sum columns of X (cells x genes) that share a name."""
    uniq, A = collapse_matrix(names)
    if sp.issparse(X):
        return (sp.csr_matrix(X) @ A).tocsr(), uniq
    return np.asarray(X @ A), uniq


def collapse_max_mean(df: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
    """Probe-level matrix (probes x samples) -> gene-level, keeping the probe with highest mean."""
    d = df.copy()
    d["__sym"] = symbols.reindex(d.index).values
    d = d[d["__sym"].notna() & (d["__sym"].astype(str).str.len() > 0)]
    d["__mean"] = d.drop(columns="__sym").mean(axis=1)
    d = d.sort_values("__mean", ascending=False).drop_duplicates("__sym")
    return d.set_index("__sym").drop(columns="__mean")


def is_valid_symbol(s: str) -> bool:
    s = str(s)
    return len(s) > 0 and s not in ("nan", "---", "") and not s.startswith("ENSG")


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def log_cpm(X, scale: float = 1e6) -> np.ndarray:
    """counts (samples x genes, dense or sparse) -> log2(CPM + 1) dense float32."""
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib[lib == 0] = 1.0
        Xn = sp.diags(scale / lib) @ X
        Xn = Xn.toarray()
    else:
        X = np.asarray(X, dtype=np.float32)
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        Xn = X / lib * scale
    return np.log2(Xn + 1.0).astype(np.float32)


def to_log_expression(X, value_type: str) -> np.ndarray:
    if value_type == "counts":
        return log_cpm(X)
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    if value_type == "log2":
        return X.astype(np.float32)
    if value_type == "linear":
        return np.log2(np.clip(X, 0, None) + 1.0).astype(np.float32)
    raise ValueError(value_type)


def zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def zscore_cols(X: np.ndarray, floor: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, floor)
    return ((X - mu) / sd).astype(np.float32), mu.ravel(), sd.ravel()


# ---------------------------------------------------------------------------
# chunked CSR reader for very large h5ad files (avoids loading 11 GB in memory)
# ---------------------------------------------------------------------------
class H5adCSR:
    def __init__(self, path: Path, prefer_raw: bool = True):
        self.f = h5py.File(path, "r")
        if prefer_raw and "raw" in self.f and "X" in self.f["raw"]:
            grp, self.source = self.f["raw/X"], "raw"
        else:
            grp, self.source = self.f["X"], "X"
        if isinstance(grp, h5py.Dataset):
            raise ValueError(f"{path}: dense matrix storage is not supported by the chunked reader")
        enc = grp.attrs.get("encoding-type", b"")
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        if enc != "csr_matrix":
            raise ValueError(f"{path}: expected csr_matrix, found {enc}")
        self.shape = tuple(int(v) for v in grp.attrs["shape"])
        self.indptr = grp["indptr"][:].astype(np.int64)
        self.data = grp["data"]
        self.indices = grp["indices"]

    def iter_chunks(self, chunk: int = 20000):
        for a in range(0, self.shape[0], chunk):
            b = min(a + chunk, self.shape[0])
            yield a, b, self.rows_range(a, b)

    def rows_range(self, a: int, b: int) -> sp.csr_matrix:
        lo, hi = self.indptr[a], self.indptr[b]
        data = self.data[lo:hi].astype(np.float32)
        idx = self.indices[lo:hi]
        ptr = self.indptr[a:b + 1] - lo
        return sp.csr_matrix((data, idx, ptr), shape=(b - a, self.shape[1]))

    def read_rows(self, rows) -> sp.csr_matrix:
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return sp.csr_matrix((0, self.shape[1]), dtype=np.float32)
        order = np.argsort(rows, kind="stable")
        srt = rows[order]
        runs = np.split(srt, np.where(np.diff(srt) != 1)[0] + 1)
        mats = [self.rows_range(int(r[0]), int(r[-1]) + 1) for r in runs]
        M = sp.vstack(mats).tocsr()
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return M[inv]

    def is_integer_valued(self, n: int = 100000) -> bool:
        d = self.data[: min(n, self.data.shape[0])]
        return bool(np.allclose(d, np.round(d)))

    def close(self):
        self.f.close()


def group_sum(M: sp.csr_matrix, groups: np.ndarray, n_groups: int) -> sp.csr_matrix:
    """Sum rows of M by integer group labels (-1 = ignore)."""
    keep = groups >= 0
    G = sp.csr_matrix((np.ones(keep.sum(), dtype=np.float32), (groups[keep], np.where(keep)[0])),
                      shape=(n_groups, M.shape[0]))
    return (G @ M).tocsr()
