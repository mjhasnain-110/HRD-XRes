"""Baselines: classical machine learning, knowledge-based pathway scores, deconvolution + regression."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

BINARY = {"hr": "p_hr", "brca": "p_brca", "plat": "p_plat"}


def _clf(kind: str, seed: int, k: int):
    if kind == "enet":
        est = LogisticRegressionCV(Cs=8, penalty="elasticnet", solver="saga", l1_ratios=[0.1, 0.5, 0.9],
                                   cv=3, max_iter=3000, scoring="roc_auc", class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
    elif kind == "rf":
        est = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced_subsample",
                                     min_samples_leaf=2, n_jobs=-1, random_state=seed)
    else:
        est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=seed)
    return make_pipeline(SelectKBest(f_classif, k=k), est)


def _reg(kind: str, seed: int, k: int):
    if kind == "enet":
        est = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000, n_jobs=-1, random_state=seed)
    elif kind == "rf":
        est = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=seed)
    else:
        est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                            l2_regularization=1.0, random_state=seed)
    return make_pipeline(SelectKBest(f_regression, k=k), est)


def cox_risk(Ztr, time, event, Zte, penalizer=0.1) -> np.ndarray:
    m = np.isfinite(time) & np.isfinite(event) & (time > 0)
    df = pd.DataFrame(Ztr[m], columns=[f"f{i}" for i in range(Ztr.shape[1])])
    df["T"], df["E"] = time[m], event[m]
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df, duration_col="T", event_col="E", show_progress=False)
    return np.asarray(cph.predict_log_partial_hazard(pd.DataFrame(Zte, columns=df.columns[:-2]))).ravel()


def classical(kind: str, Xtr: np.ndarray, ytr: dict, Xte: np.ndarray, seed: int = 0, k: int = 2000) -> pd.DataFrame:
    k = min(k, Xtr.shape[1])
    out = {}
    for key, col in BINARY.items():
        y = ytr[key]
        m = np.isfinite(y)
        if m.sum() >= 20 and len(np.unique(y[m])) == 2:
            out[col] = _clf(kind, seed, k).fit(Xtr[m], y[m].astype(int)).predict_proba(Xte)[:, 1]
    y = ytr["hrd"]
    m = np.isfinite(y)
    if m.sum() >= 20:
        out["pred_hrd"] = _reg(kind, seed, k).fit(Xtr[m], y[m]).predict(Xte)
    pca = PCA(n_components=min(20, Xtr.shape[0] - 1), random_state=seed).fit(Xtr)
    out["risk"] = cox_risk(pca.transform(Xtr), ytr["time"], ytr["event"], pca.transform(Xte))
    return pd.DataFrame(out)


def pathway_scores(X: np.ndarray, genes: list[str], gmt: dict, patterns: dict) -> pd.DataFrame:
    """Mean standardised expression of genes in pathways matching each regex (knowledge baselines)."""
    gidx = {g: i for i, g in enumerate(genes)}
    out = {}
    for label, pat in patterns.items():
        members = set()
        for name, gs in gmt.items():
            if re.search(pat, name, re.I):
                members |= gs
        idx = [gidx[g] for g in members if g in gidx]
        out[label] = X[:, idx].mean(axis=1) if idx else np.full(X.shape[0], np.nan)
        out[f"{label}__n_genes"] = len(idx)
    return pd.DataFrame(out)


KNOWLEDGE_PATTERNS = {
    "hr_pathway_score": r"Homologous Recombination|HDR through|Fanconi",
    "dna_repair_score": r"^DNA Repair \[",
    "proliferation_score": r"^Cell Cycle, Mitotic \[|^Mitotic G1|^G2/M Checkpoints",
}


def oriented_hr_score(Xtr: np.ndarray, ytr: dict, X: np.ndarray, genes: list[str], gmt: dict) -> pd.DataFrame:
    """HR pathway score baseline with the direction of each output learned on the training (TCGA) data only."""
    from .metrics import cindex, spearman
    s_tr = pathway_scores(Xtr, genes, gmt, KNOWLEDGE_PATTERNS)["hr_pathway_score"].values
    s = pathway_scores(X, genes, gmt, KNOWLEDGE_PATTERNS)["hr_pathway_score"].values
    out = {}
    for key, col in (("hr", "p_hr"), ("hrd", "pred_hrd"), ("brca", "p_brca"), ("plat", "p_plat")):
        r = spearman(s_tr, ytr[key])
        out[col] = s * (np.sign(r) if np.isfinite(r) and r != 0 else 1.0)
    c = cindex(ytr["time"], ytr["event"], s_tr)
    out["risk"] = s * (1.0 if not np.isfinite(c) or c >= 0.5 else -1.0)
    return pd.DataFrame(out)
