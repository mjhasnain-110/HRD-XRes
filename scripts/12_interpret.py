#!/usr/bin/env python
"""12 - Interpretation.

A. Pathway attributions: integrated gradients on the Reactome pathway layer for each prediction head,
   averaged over the final ensemble and TCGA samples; enrichment of DNA-repair pathways among top ranks.
B. Gene attributions (integrated gradients on the input) for HR status and platinum heads.
C. Malignant cell-state correlates: metacell predictions vs SPECTRUM module/hallmark scores (within-patient,
   BH-FDR).
D. Domain alignment quality: how separable the domains are in the latent space (5-fold logistic
   domain classifier, balanced accuracy; lower = better aligned) for full vs no-DA models.

Outputs: Results/interpretation/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.multitest import multipletests

from hrdx.data import load_features, load_inputs
from hrdx.trainer import ensemble_predict, get_device, load_ensemble
from hrdx.utils import ensure_dir, get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("12_interpret", cfg)
P, R = cfg["paths"]["processed"], ensure_dir(cfg["paths"]["results"] / "interpretation")
device = get_device(cfg["training"]["device"])
feat = load_features(P)
ens = load_ensemble(cfg["paths"]["models"], "full", device)
X, lab = load_inputs(P, "tcga")
xt = torch.as_tensor(X, dtype=torch.float32, device=device)
STEPS = 32


def ig_pathways(model, head):
    model.eval()
    with torch.no_grad():
        a = model.pathway_activity(xt)
        a0 = model.pathway_activity(torch.zeros_like(xt[:1])).expand_as(a)
    total = torch.zeros_like(a)
    for alpha in torch.linspace(1.0 / STEPS, 1.0, STEPS, device=device):
        ai = (a0 + alpha * (a - a0)).detach().requires_grad_(True)
        out = model.heads[head](model.encode_from_pathways(ai)).sum()
        g, = torch.autograd.grad(out, ai)
        total += g
    return ((a - a0) * total / STEPS).detach().cpu().numpy()


def ig_genes(model, head):
    model.eval()
    total = torch.zeros_like(xt)
    for alpha in torch.linspace(1.0 / STEPS, 1.0, STEPS, device=device):
        xi = (alpha * xt).detach().requires_grad_(True)
        out = model(xi)[head].sum()
        g, = torch.autograd.grad(out, xi)
        total += g
    return (xt * total / STEPS).detach().cpu().numpy()


heads = [h for h in ("hr", "hrd", "plat", "brca", "cox") if h in ens[0].heads]
repair_pat = r"Homologous Recombination|HDR|Fanconi|DNA Double-Strand|DNA Repair|Nucleotide Excision|Mismatch|BRCA|Base Excision"
rows, enr = [], []
for h in heads:
    attr = np.mean([ig_pathways(m, h) for m in ens], axis=0)
    df = pd.DataFrame({"pathway": feat["pathways"], "mean_abs_attribution": np.abs(attr).mean(0),
                       "mean_signed_attribution": attr.mean(0)})
    # direction w.r.t. the label: correlation of per-sample attribution with prediction
    df = df.sort_values("mean_abs_attribution", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df.insert(0, "head", h)
    rows.append(df)
    is_rep = df["pathway"].str.contains(repair_pat, case=False, regex=True)
    if is_rep.sum() > 2:
        u = mannwhitneyu(df.loc[is_rep, "rank"], df.loc[~is_rep, "rank"], alternative="less")
        enr.append({"head": h, "n_repair_pathways": int(is_rep.sum()), "median_rank_repair": df.loc[is_rep, "rank"].median(),
                    "median_rank_other": df.loc[~is_rep, "rank"].median(), "p_one_sided": u.pvalue,
                    "repair_in_top50": int(is_rep[df["rank"] <= 50].sum())})
pd.concat(rows).to_csv(R / "pathway_attributions.csv", index=False)
pd.DataFrame(enr).to_csv(R / "dna_repair_enrichment.csv", index=False)
log.info("DNA-repair enrichment among pathway attributions:\n" + pd.DataFrame(enr).to_string(index=False))

gene_rows = []
for h in [h for h in ("hr", "plat") if h in heads]:
    attr = np.mean([ig_genes(m, h) for m in ens], axis=0)
    g = pd.DataFrame({"gene": feat["genes"], "mean_abs_attribution": np.abs(attr).mean(0),
                      "mean_signed_attribution": attr.mean(0)}).sort_values("mean_abs_attribution", ascending=False)
    g.insert(0, "head", h)
    gene_rows.append(g)
pd.concat(gene_rows).to_csv(R / "gene_attributions.csv", index=False)

# ---------------------------------------------------------------- C. cell-state correlates
Xm, om = load_inputs(P, "spectrum_metacells")
pm = ensemble_predict(ens, Xm, device)
pm.index = om.index
mc = om.join(pm)
mod_cols = [c for c in om.columns if c.endswith(".module") or c.endswith(".pathway") or c.startswith("HALLMARK_")]
cs = []
for c in mod_cols:
    d = mc[["p_hr", "p_plat", c, "patient"]].dropna()
    if len(d) < 30 or d[c].std() == 0:
        continue
    dc = d[["p_hr", "p_plat", c]] - d.groupby("patient")[["p_hr", "p_plat", c]].transform("mean")
    for s in ("p_hr", "p_plat"):
        r = spearmanr(dc[s], dc[c])
        cs.append({"score": s, "module": c, "n_metacells": len(d), "spearman_within_patient": r.correlation, "p": r.pvalue})
cs = pd.DataFrame(cs)
if len(cs):
    cs["fdr"] = np.nan
    for s, g in cs.groupby("score"):
        cs.loc[g.index, "fdr"] = multipletests(g["p"].fillna(1), method="fdr_bh")[1]
    cs.sort_values(["score", "p"]).to_csv(R / "cell_state_correlates.csv", index=False)

# ---------------------------------------------------------------- D. domain alignment
names = ["tcga"] + feat["target_domains"]
rng = np.random.default_rng(0)
data = []
for i, n in enumerate(names):
    Xd, _ = load_inputs(P, n)
    if Xd.shape[0] > 3000:
        Xd = Xd[rng.choice(Xd.shape[0], 3000, replace=False)]
    data.append((Xd, np.full(Xd.shape[0], i)))
Xall = np.vstack([d[0] for d in data])
yall = np.concatenate([d[1] for d in data])
al = []
for v in ("full", "no_da"):
    if not (cfg["paths"]["models"] / "final" / v / "ensemble.json").exists():
        continue
    members = load_ensemble(cfg["paths"]["models"], v, device)
    with torch.no_grad():
        Z = np.mean([m.encode(torch.as_tensor(Xall, dtype=torch.float32, device=device)).cpu().numpy() for m in members], axis=0)
    pred = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"), Z, yall,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    al.append({"variant": v, "domain_classifier_balanced_accuracy": balanced_accuracy_score(yall, pred),
               "chance_level": 1 / len(names), "n_domains": len(names)})
    np.save(R / f"latent_{v}.npy", Z)
np.save(R / "latent_domain_labels.npy", yall)
pd.DataFrame(al).to_csv(R / "domain_alignment.csv", index=False)
log.info("Domain alignment:\n" + pd.DataFrame(al).to_string(index=False))
