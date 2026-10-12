#!/usr/bin/env python
"""17 - In-silico mixing test: does non-malignant RNA shift predicted HRD?

Explains the spatial neighbourhood association (step 15, part N). For every SPECTRUM patient, the pure
malignant-cell profile (sum of malignant metacells) is mixed with the average expression profile of one
non-malignant cell type (SPECTRUM reference) at 0-50 % of the RNA:
        mixed = (1 - f) * malignant_CPM + f * celltype_CPM
and scored with frozen models. Standardisation uses the statistics of the UNMIXED profiles for every
mixture (fixed), so any shift is caused by the added RNA only.

Decision rule:
  - If adding dendritic / mast / endothelial RNA raises P(HRD), and the cell types that raise P(HRD) are the
    same ones that correlate with P(HRD) in Visium, the spatial association is explained by spot admixture.
  - If P(HRD) does not rise (or the ranking does not match), admixture cannot explain it.

Run:  python scripts/17_mixing_test.py > Logs/17.out 2>&1      (add --no_enet to skip the elastic net)
Outputs: Results/mixing/
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, wilcoxon

from hrdx.data import load_features, load_inputs, model_input
from hrdx.metrics import auroc
from hrdx.trainer import ensemble_predict, get_device, labels_to_source, load_ensemble
from hrdx.utils import ensure_dir, get_logger, load_config

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--no_enet", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
log = get_logger("17_mixing_test", cfg)
P, RES = cfg["paths"]["processed"], cfg["paths"]["results"]
R = ensure_dir(RES / "mixing")
device = get_device(cfg["training"]["device"])
FRACS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
MAL = re.compile(cfg["singlecell"]["malignant_regex"], re.I)


def clean(c):
    return re.sub(r"[^A-Za-z0-9]+", "_", c)


feat = load_features(P)
genes = feat["genes"]
models = {"HRD-XRes": load_ensemble(cfg["paths"]["models"], "full", device)}
try:
    models["HRD-XRes (no_da)"] = load_ensemble(cfg["paths"]["models"], "no_da", device)
except FileNotFoundError:
    pass
labels = pd.read_csv(P / "spectrum" / "spectrum_patient_labels.csv", index_col=0)
labels.index = labels.index.astype(str)

# ---------------------------------------------------------------- pure malignant profiles (CPM)
mc = ad.read_h5ad(P / "spectrum" / "spectrum_metacells.h5ad")
codes, pats = pd.factorize(mc.obs["patient"].astype(str))
G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(pats), len(codes)))
M = np.asarray((G @ sp.csr_matrix(mc.X)).todense(), dtype=np.float64)
M = M / M.sum(axis=1, keepdims=True) * 1e6
mal = pd.DataFrame(M, index=pats, columns=mc.var_names.astype(str))
mal = mal.T.groupby(level=0).sum().T
log.info(f"Malignant profiles: {mal.shape[0]} patients")

# ---------------------------------------------------------------- reference cell-type profiles (CPM)
ref = ad.read_h5ad(P / "spectrum" / "spectrum_reference.h5ad")
refdf = pd.DataFrame(np.asarray(ref.X), index=ref.obs_names.astype(str), columns=ref.var_names.astype(str))
refdf = refdf.T.groupby(level=0).sum().T
nonmal = [c for c in refdf.index if not MAL.search(c)]
common = mal.columns.intersection(refdf.columns)
mal, refdf = mal[common], refdf[common]
mal = mal.div(mal.sum(axis=1), axis=0) * 1e6
refdf = refdf.div(refdf.sum(axis=1), axis=0) * 1e6
log.info(f"Non-malignant reference types: {nonmal}; shared genes: {len(common)}")


def to_input(df, stats=None):
    a = ad.AnnData(X=df.values.astype(np.float32), obs=pd.DataFrame(index=df.index.astype(str)),
                   var=pd.DataFrame(index=df.columns))
    a.uns["value_type"] = "counts"  # rows are CPM (sum 1e6) -> log2 CPM inside model_input
    if stats is None:
        return model_input(a, genes, return_stats=True)
    return model_input(a, genes, stats=stats)


X0, cov, stats = to_input(mal)
log.info(f"Model gene coverage: {cov:.3f}")

enet = None
if not args.no_enet:
    from hrdx.baselines import _clf
    Xtr, labtr = load_inputs(P, "tcga")
    y = labels_to_source(Xtr, labtr).y["hr"]
    m = np.isfinite(y)
    enet = _clf("enet", 0, min(2000, Xtr.shape[1])).fit(Xtr[m], y[m].astype(int))
    log.info("Elastic net fitted on TCGA")


def score(X):
    out = {k: ensemble_predict(v, X, device)["p_hr"].values for k, v in models.items()}
    if enet is not None:
        out["Elastic net"] = enet.predict_proba(X)[:, 1]
    return out


base = score(X0)
rows = []
for ct in nonmal:
    for f in FRACS:
        mix = (1 - f) * mal + f * refdf.loc[ct].values[None, :]
        Xm = to_input(mix, stats=stats)[0] if f > 0 else X0
        sc = score(Xm) if f > 0 else base
        for mname, p in sc.items():
            for pat, pv, p0 in zip(mal.index, p, base[mname]):
                rows.append({"cell_type": clean(ct), "fraction": f, "method": mname, "patient": pat,
                             "p_hrd": pv, "delta_vs_pure": pv - p0})
    log.info(f"  mixed {ct}")
df = pd.DataFrame(rows).join(labels[["hr_status_wgs"]], on="patient")
df.to_csv(R / "mixing_per_patient.csv", index=False)

# ---------------------------------------------------------------- summaries
summ = []
for (ct, mname, f), g in df.groupby(["cell_type", "method", "fraction"]):
    rec = {"cell_type": ct, "method": mname, "fraction": f, "mean_p_hrd": g["p_hrd"].mean(),
           "mean_delta": g["delta_vs_pure"].mean(), "frac_patients_increased": float((g["delta_vs_pure"] > 0).mean())}
    if f > 0 and np.any(g["delta_vs_pure"] != 0):
        rec["wilcoxon_p"] = wilcoxon(g["delta_vs_pure"]).pvalue
    lab = g.dropna(subset=["hr_status_wgs"])
    rec["auroc_wgs"] = auroc(lab["hr_status_wgs"].values, lab["p_hrd"].values)
    summ.append(rec)
summ = pd.DataFrame(summ)
summ.to_csv(R / "mixing_summary.csv", index=False)

# slope of P(HRD) per 10 % admixture (fractions <= 0.3, the realistic range for a Visium spot)
slopes = []
for (ct, mname), g in df[df.fraction <= 0.3].groupby(["cell_type", "method"]):
    per_pat = g.groupby("patient").apply(lambda x: np.polyfit(x["fraction"], x["p_hrd"], 1)[0] * 0.1)
    slopes.append({"cell_type": ct, "method": mname, "slope_per_10pct": per_pat.mean(),
                   "frac_patients_positive": float((per_pat > 0).mean()),
                   "wilcoxon_p": wilcoxon(per_pat).pvalue if np.any(per_pat != 0) else np.nan})
slopes = pd.DataFrame(slopes)

# does the admixture effect predict the spatial association?
nb_file = RES / "identifihr" / "neighbourhood_adjusted_summary.csv"
if nb_file.exists():
    nb = pd.read_csv(nb_file)
    nb = nb[nb.cohort == "Both"][["cell_type", "mean_rho_adj", "n_positive_adj", "n_sections"]]
    slopes = slopes.merge(nb, on="cell_type", how="left")
    conc = []
    for mname, g in slopes.dropna(subset=["mean_rho_adj"]).groupby("method"):
        if len(g) >= 4:
            r = spearmanr(g["slope_per_10pct"], g["mean_rho_adj"])
            conc.append({"method": mname, "n_cell_types": len(g), "spearman_slope_vs_spatial_rho": r.correlation,
                         "p": r.pvalue})
    conc = pd.DataFrame(conc)
    conc.to_csv(R / "mixing_vs_spatial_concordance.csv", index=False)
else:
    conc = pd.DataFrame()
slopes.to_csv(R / "mixing_slopes.csv", index=False)

pd.set_option("display.width", 250)
log.info("Mean change in P(HRD) vs pure malignant profile (rows: cell type; columns: fraction of added RNA):\n" +
         summ[summ.method == "HRD-XRes"].pivot(index="cell_type", columns="fraction", values="mean_delta").round(4).to_string())
log.info("AUROC vs WGS HR status as admixture increases (HRD-XRes):\n" +
         summ[summ.method == "HRD-XRes"].pivot(index="cell_type", columns="fraction", values="auroc_wgs").round(3).to_string())
log.info("Slope of P(HRD) per 10% admixture (<=30%), with the spatial association for comparison:\n" +
         slopes.sort_values(["method", "slope_per_10pct"], ascending=[True, False]).round(4).to_string(index=False))
if len(conc):
    log.info("Does the admixture effect explain the spatial pattern? (positive rho = yes):\n" + conc.round(4).to_string(index=False))
log.info(f"Done. Results in {R}")
