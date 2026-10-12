"""Single-cell helpers: metacell construction and per-cell kNN aggregation."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def _embed(X: sp.csr_matrix, n_comp: int = 20, n_hvg: int = 2000, seed: int = 0) -> np.ndarray:
    X = sp.csr_matrix(X, dtype=np.float32)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1
    L = sp.diags(1e4 / lib) @ X
    L.data = np.log1p(L.data)
    mean = np.asarray(L.mean(axis=0)).ravel()
    sq = np.asarray(L.multiply(L).mean(axis=0)).ravel()
    var = sq - mean ** 2
    hv = np.argsort(var)[::-1][: min(n_hvg, L.shape[1])]
    L = L[:, hv]
    n_comp = max(1, min(n_comp, L.shape[0] - 1, L.shape[1] - 1))
    return TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(L)


def metacell_labels(X: sp.csr_matrix, size: int = 30, seed: int = 0) -> np.ndarray:
    """Partition cells of one sample into ~size-cell metacells (k-means in PCA space)."""
    n = X.shape[0]
    k = max(1, int(round(n / size)))
    if k == 1:
        return np.zeros(n, dtype=int)
    Z = _embed(X, seed=seed)
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(Z)


def aggregate(X: sp.csr_matrix, labels: np.ndarray) -> sp.csr_matrix:
    k = labels.max() + 1
    G = sp.csr_matrix((np.ones(len(labels), np.float32), (labels, np.arange(len(labels)))), shape=(k, len(labels)))
    return (G @ X).tocsr()


def knn_pool(X: sp.csr_matrix, k: int = 15, seed: int = 0) -> sp.csr_matrix:
    """For every cell, sum the counts of itself and its k-1 nearest neighbours (same sample)."""
    n = X.shape[0]
    if n <= 2:
        return sp.csr_matrix(X)
    k = min(k, n)
    Z = _embed(X, seed=seed)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, ind = nn.kneighbors(Z)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((np.ones(n * k, np.float32), (rows, ind.ravel())), shape=(n, n))
    return (W @ X).tocsr()
