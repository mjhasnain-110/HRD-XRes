#!/usr/bin/env python
"""01 - TCGA-OV training data: expression + HRD / BRCA / platinum / survival / purity labels.

Outputs (Processed/tcga/):
  tcga_ov.h5ad       primary tumours x genes (gene symbols), raw counts (or RSEM), obs = labels
  tcga_labels.csv    label table
  label_summary.csv  label availability and class balance
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hrdx.genes import build_gene_map
from hrdx.utils import collapse_sum, ensure_dir, get_logger, inp, load_config, strip_version

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("01_prepare_tcga", cfg)
F = cfg["files"]["tcga"]
out = ensure_dir(cfg["paths"]["processed"] / "tcga")

# ---------------------------------------------------------------- DDR resource
names = [l.strip() for l in open(inp(cfg, F["ddr_score_names"])) if l.strip()]
scores = pd.read_csv(inp(cfg, F["ddr_scores"]), sep="\t", header=None, na_values=["NaN", "NA", ""])
assert scores.shape[1] == len(names), f"DDRscores has {scores.shape[1]} columns, Scores.tsv lists {len(names)}"
scores.columns = names
samples = pd.read_csv(inp(cfg, F["ddr_samples"]), sep="\t", header=None, names=["sample", "cancer"])
assert len(samples) == len(scores)
scores.index = samples["sample"].str[:15].values
scores["cancer"] = samples["cancer"].values
ddr = scores[scores["cancer"] == "OV"].copy()
ddr = ddr[~ddr.index.duplicated()]
log.info(f"DDR resource: {len(ddr)} OV samples with genomic scores")

genes = [l.strip().split("\t")[0] for l in open(inp(cfg, F["ddr_genes"])) if l.strip()]
alt = pd.read_csv(inp(cfg, F["ddr_gene_alterations"]), sep="\t", header=None, na_values=["NaN", "NA", ""])
if alt.shape == (len(genes) + 1, len(samples)) or alt.shape == (len(genes), len(samples) + 1):
    raise ValueError("Unexpected header in GeneAlterations.tsv; please inspect the file")
if alt.shape[0] != len(genes) and alt.shape[0] == len(genes) - 1:
    genes = genes[1:]  # Genes.tsv carried a header line
if alt.shape == (len(genes), len(samples)):
    alt.index, alt.columns = genes, samples["sample"].str[:15].values
    alt = alt.T
elif alt.shape == (len(samples), len(genes)):
    alt.index, alt.columns = samples["sample"].str[:15].values, genes
else:
    raise ValueError(f"GeneAlterations shape {alt.shape} incompatible with {len(genes)} genes / {len(samples)} samples")
alt = alt[~alt.index.duplicated()]
log.info(f"Gene alteration matrix: {alt.shape[0]} samples x {alt.shape[1]} DDR genes")

# ---------------------------------------------------------------- clinical
cdr = pd.read_excel(inp(cfg, F["cdr"]), sheet_name=0)
cdr = cdr.loc[:, ~cdr.columns.astype(str).str.startswith("Unnamed")]
cdr = cdr[cdr["type"] == "OV"].set_index("bcr_patient_barcode")
for c in ["OS", "OS.time", "PFI", "PFI.time", "age_at_initial_pathologic_diagnosis"]:
    cdr[c] = pd.to_numeric(cdr[c], errors="coerce")
clin = pd.read_csv(inp(cfg, F["clinical_pancan"]), sep="\t", low_memory=False, encoding="latin-1", index_col=0)
clin.index = clin.index.astype(str).str[:12]
clin = clin[~clin.index.duplicated()]
resp_col = next((c for c in clin.columns if "primary_therapy_outcome" in c.lower()), None)
log.info(f"Clinical: {len(cdr)} OV patients in CDR; primary therapy outcome column = {resp_col}")

# ---------------------------------------------------------------- expression
src = cfg["tcga"]["expression_source"]
if src == "star" and inp(cfg, F["star_counts"]).exists():
    log.info("Reading GDC STAR counts (UCSC Xena)")
    df = pd.read_csv(inp(cfg, F["star_counts"]), sep="\t", index_col=0)
    vals = df.values.astype(np.float32)
    if np.nanmax(vals) < 60:  # Xena stores log2(count + 1)
        vals = np.round(np.power(2.0, vals) - 1.0)
    ens = strip_version(df.index)
    gmap = build_gene_map(cfg, log)
    sym = pd.Series(ens).map(gmap).values
    keep = pd.notna(sym)
    X = sp.csr_matrix(vals[keep].T)
    X, var_names = collapse_sum(X, sym[keep])
    sample_ids = df.columns.astype(str)
    value_type = "counts"
else:
    log.info("Reading PanCanAtlas EB++ RSEM matrix")
    head = pd.read_csv(inp(cfg, F["ebpp"]), sep="\t", nrows=0)
    ov_patients = set(cdr.index)
    cols = [c for c in head.columns[1:] if c[:12] in ov_patients and c[13:15] == "01"]
    df = pd.read_csv(inp(cfg, F["ebpp"]), sep="\t", usecols=[head.columns[0]] + cols, index_col=0)
    sym = df.index.str.split("|").str[0]
    keep = (sym != "?") & df.notna().all(axis=1).values
    X, var_names = collapse_sum(sp.csr_matrix(np.clip(df.values[keep], 0, None).T.astype(np.float32)), sym[keep])
    sample_ids = df.columns.astype(str)
    value_type = "linear"

sample15 = pd.Index([s[:15] for s in sample_ids])
is_primary = np.array([s[13:15] == "01" for s in sample15])
order = pd.Series(np.arange(len(sample_ids)), index=sample_ids).sort_index()
first = ~pd.Index([sample_ids[i][:15] for i in order.values]).duplicated()
sel = np.array(sorted(i for i, f in zip(order.values, first) if f and is_primary[i]))
X = sp.csr_matrix(X)[sel]
obs = pd.DataFrame(index=sample15[sel])
obs["patient"] = obs.index.str[:12]
log.info(f"Expression: {X.shape[0]} primary tumours x {X.shape[1]} genes ({value_type})")

# sample quality: exclude samples flagged "Do_not_use" in PanCanAtlas annotations
qpath = inp(cfg, F["sample_quality"])
if qpath.exists():
    q = pd.read_csv(qpath, sep="\t", low_memory=False, encoding="latin-1")
    bc_col = next((c for c in q.columns if "aliquot" in c.lower()), next(c for c in q.columns if "barcode" in c.lower()))
    dn_col = next((c for c in q.columns if "do_not_use" in c.lower()), None)
    if dn_col is not None:
        bad = set(q.loc[q[dn_col].astype(str).str.upper().isin(["TRUE", "1"]), bc_col].astype(str).str[:15])
        drop = obs.index.isin(bad)
        log.info(f"Removing {drop.sum()} samples flagged Do_not_use")
        X, obs = X[~drop], obs[~drop]

# ---------------------------------------------------------------- labels
thr = cfg["tcga"]["hrd_threshold"]
lab = pd.DataFrame(index=obs.index)
lab["patient"] = obs["patient"]
for c in ["HRD_Score", "HRD_LOH", "HRD_TAI", "HRD_LST", "purity", "ploidy", "genome_doublings", "aneuploidy_score"]:
    lab[c.lower() if c != "HRD_Score" else "hrd_score"] = ddr[c].reindex(lab.index).values
scar_path = out / "hrd_scar_scores.csv"
if scar_path.exists():
    scar = pd.read_csv(scar_path, index_col=0)
    lab["hrd_score"] = scar["hrd_final"].reindex(lab.index).values
    lab["hrd_source"] = scar["hrd_source"].reindex(lab.index).fillna("none").values
    for c in ("scar_loh", "scar_tai", "scar_lst"):
        lab[c] = scar[c].reindex(lab.index).values
    log.info(f"HRD labels from 01a: {lab['hrd_source'].value_counts().to_dict()}")
else:
    log.warning("hrd_scar_scores.csv not found - only published DDR HRD scores used (run 01a first)")
abs_path = inp(cfg, "tcga/TCGA_mastercalls.abs_tables_JSedit.fixed.txt")
if abs_path.exists():
    at = pd.read_csv(abs_path, sep="\t", encoding="latin-1")
    scol = next(c for c in at.columns if c.lower() in ("sample", "array"))
    at.index = at[scol].astype(str).str[:15]
    at = at[~at.index.duplicated()]
    for c in ("purity", "ploidy"):
        if c in at.columns:
            v = pd.to_numeric(at[c], errors="coerce").reindex(lab.index)
            lab[c] = lab[c].fillna(v) if c in lab else v
    log.info(f"Purity available for {lab['purity'].notna().sum()} tumours (ABSOLUTE)")
lab["hr_status"] = np.where(lab["hrd_score"].notna(), (lab["hrd_score"] >= thr).astype(float), np.nan)
for g in ["BRCA1", "BRCA2"]:
    lab[f"{g.lower()}_alt"] = alt[g].reindex(lab.index).values if g in alt.columns else np.nan
b = lab[["brca1_alt", "brca2_alt"]]
lab["brca_alt"] = np.where(b.notna().any(axis=1), (b.fillna(0).max(axis=1) > 0).astype(float), np.nan)

c = cdr.reindex(lab["patient"])
lab["os"], lab["os_time"] = c["OS"].values, c["OS.time"].values
lab["pfi"], lab["pfi_time"] = c["PFI"].values, c["PFI.time"].values
lab["age"] = c["age_at_initial_pathologic_diagnosis"].values
stage_col = next((s for s in ["clinical_stage", "ajcc_pathologic_tumor_stage"] if s in c.columns), None)
lab["stage"] = c[stage_col].astype(str).values if stage_col else np.nan
lab["residual_disease"] = clin.reindex(lab["patient"])[
    next((x for x in clin.columns if "residual" in x.lower()), clin.columns[0])].values

# platinum response (first-line platinum-based therapy is standard for TCGA-OV)
win = cfg["tcga"]["platinum_window_days"]
outcome = clin.reindex(lab["patient"])[resp_col].astype(str).str.lower().values if resp_col else np.array([""] * len(lab))
pfi, pft = lab["pfi"].values, lab["pfi_time"].values
plat = np.full(len(lab), np.nan)
refractory = np.array([("progressive" in o) or ("stable" in o) for o in outcome])
early_prog = (pfi == 1) & (pft < win)
late = (pft >= win)
plat[late & ~refractory] = 0.0
plat[early_prog | refractory] = 1.0
lab["platinum_resistant"] = plat
lab["primary_therapy_outcome"] = outcome

for c in lab.columns:
    if lab[c].dtype == object:
        lab[c] = lab[c].astype(str).replace({"nan": "NA", "None": "NA"})
ad_obj = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float32), obs=lab, var=pd.DataFrame(index=var_names))
ad_obj.uns["value_type"] = value_type
ad_obj.write_h5ad(out / "tcga_ov.h5ad", compression="gzip")
lab.to_csv(out / "tcga_labels.csv")

summ = []
for col in ["hrd_score", "hr_status", "brca_alt", "brca1_alt", "brca2_alt", "platinum_resistant", "os", "pfi", "purity"]:
    v = lab[col]
    summ.append({"label": col, "n_available": int(v.notna().sum()),
                 "n_positive": int((v == 1).sum()) if col not in ("hrd_score", "purity") else np.nan,
                 "mean": float(v.mean()) if v.notna().any() else np.nan})
pd.DataFrame(summ).to_csv(out / "label_summary.csv", index=False)
log.info("Label summary:\n" + pd.DataFrame(summ).to_string(index=False))
log.info(f"Saved {out / 'tcga_ov.h5ad'}")
