#!/usr/bin/env python
"""Create a small synthetic copy of Inputs/hgsoc_hrd_data with the same file formats as the real data.

Used only to test the pipeline end to end (python tests/make_synthetic_inputs.py <test_root>).
"""
import gzip
import io
import sys
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

root = Path(sys.argv[1])
I = root / "Inputs" / "hgsoc_hrd_data"
rng = np.random.default_rng(1)
NG = 600
SYM = np.array([f"GENE{i}" for i in range(NG)])
ENS = np.array([f"ENSG{i:011d}" for i in range(NG)])
HR_GENES = SYM[:40]          # driven by HRD
PLAT_GENES = SYM[40:70]      # driven by platinum resistance
CANCER_GENES = SYM[70:110]   # malignant markers
FIB_GENES, T_GENES, MY_GENES = SYM[110:140], SYM[140:170], SYM[170:200]


def mk(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def profile(hrd=0.0, plat=0.0, ctype="cancer"):
    mu = np.full(NG, 5.0)
    mu[:40] *= np.exp(1.0 * hrd)
    mu[40:70] *= np.exp(0.8 * plat)
    for genes, ct in ((CANCER_GENES, "cancer"), (FIB_GENES, "fib"), (T_GENES, "t"), (MY_GENES, "my")):
        idx = np.isin(SYM, genes)
        mu[idx] *= 8.0 if ctype == ct else 0.3
    return mu


def counts(mu, depth):
    m = mu / mu.sum() * depth
    return rng.negative_binomial(5, 5 / (5 + m)).astype(np.float32)


# ---------------------------------------------------------------- Reactome GMT
mk(I / "priors")
lines = [f"HDR through Homologous Recombination (HRR)\tR-HSA-1\t" + "\t".join(HR_GENES[:25]),
         f"DNA Repair\tR-HSA-2\t" + "\t".join(HR_GENES[10:40]),
         f"Cell Cycle, Mitotic\tR-HSA-3\t" + "\t".join(SYM[200:240])]
for k in range(45):
    lines.append(f"Pathway {k}\tR-HSA-{100 + k}\t" + "\t".join(rng.choice(SYM, 25, replace=False)))
(I / "priors" / "ReactomePathways.gmt").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- TCGA
T = mk(I / "tcga")
n = 160
pats = [f"TCGA-{10 + i // 100:02d}-{1000 + i:04d}" for i in range(n)]
hrd_lat = rng.normal(size=n)
plat_lat = -0.6 * hrd_lat + rng.normal(scale=0.8, size=n)
purity = rng.uniform(0.4, 0.95, n)
cols = [f"{p}-01A" for p in pats] + [f"{pats[0]}-11A"]
mat = np.vstack([counts(profile(h, q) * pu + profile(0, 0, "fib") * (1 - pu), 3e6) for h, q, pu in zip(hrd_lat, plat_lat, purity)])
mat = np.vstack([mat, mat[0]])
star = pd.DataFrame(np.log2(mat.T + 1), index=[f"{e}.{3}" for e in ENS], columns=cols)
star.index.name = "Ensembl_ID"
star.to_csv(T / "TCGA-OV.star_counts.tsv.gz", sep="\t", compression="gzip")
names = ["mutLoad_silent", "mutLoad_nonsilent"] + [f"mutSig{i}" for i in range(1, 22)] + [
    "CNA_n_segs", "CNA_frac_altered ", "CNA_n_focal_amp_del", "aneuploidy_score", "aneuploidy_score_prime", "LOH_n_seg",
    "LOH_frac_altered ", "purity", "ploidy", "genome_doublings", "subclonal_frac", "HRD_TAI", "HRD_LST", "HRD_LOH",
    "HRD_Score", "eCARD", "PARPi7", "PARPi7_bin", "RPS", "tp53_score", "rppa_ddr_score"]
(T / "Scores.tsv").write_text("\n".join(names) + "\n")
other = [f"TCGA-ZZ-{2000 + i}-01" for i in range(40)]
samples = [f"{p}-01" for p in pats] + other
pd.DataFrame({"s": samples, "c": ["OV"] * n + ["BRCA"] * 40}).to_csv(T / "Samples.tsv", sep="\t", header=False, index=False)
S = rng.normal(size=(len(samples), len(names)))
hscore = np.concatenate([42 + 14 * hrd_lat + rng.normal(scale=4, size=n), rng.uniform(5, 30, 40)])
S[:, names.index("HRD_Score")] = hscore
S[:, names.index("purity")] = np.concatenate([purity, rng.uniform(0.3, 0.9, 40)])
S[5, 0] = np.nan
pd.DataFrame(S).to_csv(T / "DDRscores.tsv", sep="\t", header=False, index=False, na_rep="NaN")
genes = ["ATM", "BRCA1", "BRCA2", "RAD51C"]
(T / "Genes.tsv").write_text("\n".join(genes) + "\n")
brca = np.concatenate([(hrd_lat + rng.normal(scale=0.7, size=n) > 1.0).astype(float), np.zeros(40)])
GA = np.vstack([rng.binomial(1, 0.05, len(samples)), brca, np.zeros(len(samples)), rng.binomial(1, 0.02, len(samples))])
pd.DataFrame(GA).to_csv(T / "GeneAlterations.tsv", sep="\t", header=False, index=False)
os_time = rng.exponential(1200 * np.exp(0.4 * hrd_lat - 0.4 * plat_lat))
pfi_time = rng.exponential(500 * np.exp(-0.6 * plat_lat))
cdr = pd.DataFrame({"bcr_patient_barcode": pats, "type": "OV", "age_at_initial_pathologic_diagnosis": rng.integers(40, 80, n),
                    "clinical_stage": rng.choice(["Stage IIIC", "Stage IV"], n), "OS": rng.binomial(1, 0.7, n),
                    "OS.time": os_time.round(), "PFI": rng.binomial(1, 0.8, n), "PFI.time": pfi_time.round()})
cdr.to_excel(T / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
outcome = np.where(plat_lat > 1.2, "Progressive Disease", "Complete Remission/Response")
pd.DataFrame({"bcr_patient_barcode": pats, "acronym": "OV", "primary_therapy_outcome_success": outcome,
              "tumor_residual_disease": rng.choice(["No Macroscopic disease", "1-10 mm"], n)}).to_csv(
    T / "clinical_PANCAN_patient_with_followup.tsv", sep="\t", index=False)
pd.DataFrame({"aliquot_barcode": [f"{pats[3]}-01A-11R"], "Do_not_use": [True]}).to_csv(
    T / "merged_sample_quality_annotations.tsv", sep="\t", index=False)

# ---------------------------------------------------------------- SPECTRUM
SP = mk(I / "scrna" / "spectrum")
npat = 14
spat = [f"SPECTRUM-OV-{100 + i:03d}" for i in range(npat)]
p_hrd = np.array([1, 0] * (npat // 2))
obs_rows, X_rows = [], []
for pi, p in enumerate(spat):
    for site in ("RIGHT_ADNEXA", "OMENTUM"):
        for sort in ("CD45N", "CD45P"):
            types = ["Ovarian.cancer.cell"] * 50 + ["Fibroblast"] * 20 if sort == "CD45N" else ["T.cell"] * 30 + ["Myeloid.cell"] * 20
            for t in types:
                ctype = {"Ovarian.cancer.cell": "cancer", "Fibroblast": "fib", "T.cell": "t", "Myeloid.cell": "my"}[t]
                h = (1.5 * p_hrd[pi] - 0.75 + rng.normal(scale=0.5)) if ctype == "cancer" else 0.0
                X_rows.append(counts(profile(h, 0, ctype), 3000))
                obs_rows.append({"donor_id": p, "author_sample_id": f"{p}_S1_{sort}_{site}", "author_cell_type": t,
                                 "cell_type": t.lower(), "author_tumor_site": site, "author_tumor_supersite": site.split("_")[-1],
                                 "author_therapy": "pre-Rx", "BAF_1p": 0.5 + 0.3 * h * rng.uniform(), "BAF_2q": 0.5 + 0.2 * h,
                                 "CIN.module": h + rng.normal(scale=0.3), "HALLMARK_DNA_REPAIR": rng.normal()})
obs = pd.DataFrame(obs_rows, index=[f"c{i}" for i in range(len(obs_rows))])
for c in ["donor_id", "author_sample_id", "author_cell_type", "cell_type", "author_tumor_site", "author_tumor_supersite", "author_therapy"]:
    obs[c] = obs[c].astype("category")
Xc = sp.csr_matrix(np.vstack(X_rows))
var = pd.DataFrame({"feature_name": SYM}, index=ENS)


def write_cxg(a_obs, Xm, fname):
    norm = sp.csr_matrix(Xm.multiply(1e4 / np.maximum(Xm.sum(axis=1), 1)).log1p())
    a = ad.AnnData(X=norm, obs=a_obs.drop(columns=[c for c in a_obs.columns if c.startswith(("BAF", "CIN", "HALL"))]), var=var.copy())
    a.raw = ad.AnnData(X=Xm, var=var.copy())
    a.write_h5ad(SP / fname)
    return a


write_cxg(obs, Xc, "f8f7f2b0-b8a5-4087-8048-8d3f5b6a49dd.h5ad")
epi = obs[obs["author_cell_type"] == "Ovarian.cancer.cell"].copy()
eX = Xc[np.where(obs["author_cell_type"] == "Ovarian.cancer.cell")[0]]
epi["cell_type_super"] = "Ovarian.cancer.cell"
norm = sp.csr_matrix(eX.multiply(1e4 / np.maximum(eX.sum(axis=1), 1)).log1p())
a = ad.AnnData(X=norm, obs=epi, var=var.copy())
a.raw = ad.AnnData(X=eX, var=var.copy())
a.write_h5ad(SP / "c9e5378d-a8b6-4cb9-bbc5-63de145b1e2e.h5ad")
small = obs.iloc[:50].copy()
write_cxg(small, Xc[:50], "7106ee6e-02fd-476f-b56d-a6eb5494f6da.h5ad")
pd.DataFrame({"t": ["MSK SPECTRUM – All cells", "MSK SPECTRUM - Malignant and non-malignant epithelial cells",
                    "MSK SPECTRUM - Dendritic cells"],
              "u": ["https://datasets.cellxgene.cziscience.com/f8f7f2b0-b8a5-4087-8048-8d3f5b6a49dd.h5ad",
                    "https://datasets.cellxgene.cziscience.com/c9e5378d-a8b6-4cb9-bbc5-63de145b1e2e.h5ad",
                    "https://datasets.cellxgene.cziscience.com/7106ee6e-02fd-476f-b56d-a6eb5494f6da.h5ad"]}).to_csv(
    SP / "spectrum_h5ad_urls.tsv", sep="\t", header=False, index=False)
sig = np.where(p_hrd == 1, rng.choice(["HRD-Dup", "HRD-Del"], npat), "FBI").astype(object)
sig[0] = "Undetermined"
supp = pd.DataFrame({"patient_id": spat, "consensus_signature": sig,
                     "consensus_hr_status": np.where(sig == "Undetermined", "Undetermined", np.where(p_hrd == 1, "HRD", "HRP")),
                     "wgs_signature": sig, "myriad_signature": np.nan, "myriad_gis_score": np.nan})
supp.loc[:6, "myriad_gis_score"] = 30 + 25 * p_hrd[:7]
with pd.ExcelWriter(SP / "41586_2022_5496_MOESM4_ESM.xlsx") as w:
    supp.to_excel(w, sheet_name="Mutational signatures", index=False)
    supp.to_excel(w, sheet_name="WGS - SNV counts", index=False)


# ---------------------------------------------------------------- spatial
def write_10x(prefix_bc, prefix_feat, prefix_mtx, M, bcs):
    with gzip.open(prefix_mtx, "wb") as fh:
        scipy.io.mmwrite(fh, sp.coo_matrix(M.T))
    pd.Series(bcs).to_csv(prefix_bc, index=False, header=False, compression="gzip")
    pd.DataFrame({"a": ENS, "b": SYM, "c": "Gene Expression"}).to_csv(prefix_feat, sep="\t", index=False, header=False, compression="gzip")


def section(h, q, nspots=150):
    rows, cols = np.divmod(np.arange(nspots), 15)
    cols = cols * 2 + (rows % 2)
    tumour = rng.uniform(size=nspots) < 0.7
    M = np.vstack([counts(profile(h + 0.3 * rng.normal(), q, "cancer") * (0.8 if t else 0.1) + profile(0, 0, "fib"), 4000)
                   for t in tumour])
    bcs = [f"AAAC{i:012d}-1" for i in range(nspots)]
    return M, bcs, rows, cols


def series_matrix(path, gsms, titles, chars, platform="GPL24676", table=None):
    lines = [f'!Series_platform_id\t"{platform}"', "!Sample_title\t" + "\t".join(f'"{t}"' for t in titles),
             "!Sample_geo_accession\t" + "\t".join(f'"{g}"' for g in gsms),
             "!Sample_data_processing\t" + "\t".join('"poor and excellent responders processed"' for _ in gsms)]
    for row in chars:
        lines.append("!Sample_characteristics_ch1\t" + "\t".join(f'"{v}"' for v in row))
    lines.append("!series_matrix_table_begin")
    if table is not None:
        lines.append('"ID_REF"\t' + "\t".join(f'"{g}"' for g in gsms))
        for pid, vals in table.iterrows():
            lines.append(f'"{pid}"\t' + "\t".join(f"{v:.4f}" for v in vals))
    lines.append("!series_matrix_table_end")
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


D = mk(I / "spatial" / "GSE211956" / "raw")
mk(I / "spatial" / "GSE211956" / "matrix")
resp = ["poor", "good", "good", "partial", "good", "partial", "poor", "poor"]
gsms, titles = [], []
for k in range(5):
    g = f"GSM650610{5 + k}"
    sc = np.vstack([counts(profile(0, 0, "cancer"), 3000) for _ in range(40)])
    df = pd.DataFrame(sc.T.astype(int), index=SYM, columns=[f"BC{i}_3" for i in range(40)])
    txt = "\t".join(df.columns) + "\n" + df.to_csv(sep="\t", header=False)
    with gzip.open(D / f"{g}_counts_S{k}.txt.gz", "wt") as fh:
        fh.write(txt)
    gsms.append(g)
    titles.append(f"scRNA-seq sample S{k}")
for k in range(8):
    g = f"GSM65061{10 + k}"
    sec = f"SP{k + 1}"
    hrd = 1.0 if resp[k] == "good" else -0.5
    M, bcs, r, c = section(hrd, 1.0 if resp[k] == "poor" else -0.5)
    write_10x(D / f"{g}_{sec}_barcodes.tsv.gz", D / f"{g}_{sec}_features.tsv.gz", D / f"{g}_{sec}_matrix.mtx.gz", M, bcs)
    pos = pd.DataFrame({"b": bcs, "i": 1, "r": r, "c": c, "pr": r * 100, "pc": c * 100})
    buf = io.BytesIO()
    with zipfile.ZipFile(D / f"{g}_{sec}_spatial.zip", "w") as z:
        z.writestr("spatial/tissue_positions_list.csv", pos.to_csv(index=False, header=False))
        z.writestr("spatial/scalefactors_json.json", "{}")
    gsms.append(g)
    titles.append(f"Visium sample {resp[k]}_response_P{k + 1}")
series_matrix(I / "spatial" / "GSE211956" / "matrix" / "GSE211956_series_matrix.txt.gz", gsms, titles,
              [["tissue: Ovarian tumour"] * len(gsms)])

SR = mk(I / "spatial" / "GSE189843" / "raw")
mk(I / "spatial" / "GSE189843" / "matrix")
gsms, chars = [], []
for k in range(6):
    g, sid = f"GSM57084{85 + k}", f"II2147{2 + k}"
    poor = k % 2 == 0
    M, bcs, _, _ = section(-0.5 if poor else 1.0, 1.0 if poor else -0.5)
    write_10x(SR / f"{g}_barcodes_{sid}.tsv.gz", SR / f"{g}_features_{sid}.tsv.gz", SR / f"{g}_matrix_{sid}.mtx.gz", M, bcs)
    gsms.append(g)
    chars.append(f"response: {'poor' if poor else 'excellent'}")
series_matrix(I / "spatial" / "GSE189843" / "matrix" / "GSE189843_series_matrix.txt.gz", gsms,
              [f"HGSC_{i}" for i in range(6)], [["tissue: ovary"] * 6, chars])

# ---------------------------------------------------------------- GSE165897
G = mk(I / "scrna" / "GSE165897")
cells, info = [], []
for p in range(4):
    for ph in ("treatment-naive", "post-NACT"):
        for i in range(40):
            t = "EOC" if i < 30 else "Fibroblast"
            cells.append(counts(profile(0.5 if ph == "post-NACT" else 0, 0, "cancer" if t == "EOC" else "fib"), 3000))
            info.append({"cell": f"EOC{p}{ph[:3]}{i}", "sample": f"EOC{p}_{ph}", "patient_id": f"EOC{p}",
                         "treatment_phase": ph, "cell_type": t})
info = pd.DataFrame(info)
info.to_csv(G / "GSE165897_cellInfo_HGSOC.tsv.gz", sep="\t", index=False, compression="gzip")
df = pd.DataFrame(np.vstack(cells).T.astype(int), index=SYM, columns=info["cell"])
with gzip.open(G / "GSE165897_UMIcounts_HGSOC.tsv.gz", "wt") as fh:
    fh.write("\t".join(df.columns) + "\n" + df.to_csv(sep="\t", header=False))

# ---------------------------------------------------------------- bulk GEO + platform + curated
B = mk(I / "bulk_validation")
PL = mk(B / "platforms")
probes = [f"{i}_at" for i in range(NG)] + ["dup_at"]
psym = list(SYM) + [SYM[0]]
with gzip.open(PL / "GPL570.annot.gz", "wt") as fh:
    fh.write("!platform_table_begin\nID\tGene symbol\n" + "\n".join(f"{p}\t{s}" for p, s in zip(probes, psym)) + "\n!platform_table_end\n")


def bulk(nb, hrd, plat):
    M = np.vstack([np.log2(counts(profile(h, q), 2e6) + 1) for h, q in zip(hrd, plat)])
    return pd.DataFrame(np.hstack([M, M[:, :1]]).T * 1.0, index=probes)


nb_ = 60
h = rng.normal(size=nb_)
q = -0.6 * h + rng.normal(scale=0.8, size=nb_)
gs = [f"GSM15{i:04d}" for i in range(nb_)]
tab = bulk(nb_, h, q)
tab.columns = gs
plat = np.where(q > 0.5, "resistant", np.where(q > -0.3, "moderately sensitive", "highly sensitive"))
os_d = rng.exponential(900 * np.exp(-0.5 * q)).round()
mk(B / "GSE63885")
series_matrix(B / "GSE63885" / "GSE63885_series_matrix.txt.gz", gs, gs, [
    ["histophatological type of tumor: serous ovarian ca."] * nb_,
    [f"brca1 mutation: {'mutation' if x > 1 else 'no mutation'}" for x in h],
    [f"residual tumor size (r0 < 1 cm;  1 < r1 < 5 cm; r2 > 5cm): R{rng.integers(0, 3)}" for _ in gs],
    [f"clinical status post 1st line chemotherapy (cr - complete response; pr - partial response; sd - stable disease; p - progression): CR" for _ in gs],
    [f"platinium sensitivity (resistant: DFS<180 days; moderately sensitive: 180>DFS>732; highly sensitive: DFS>732 days): {p}" for p in plat],
    [f"dfs - disease-free survival [days]: {int(d / 2)}" for d in os_d],
    [f"os - overall survival [days]: {int(d)}" for d in os_d],
    [f"clinical status at last follow-up (awd - alive with disease; dod - dead from disease; ned - no evidence of disease): {'DOD' if rng.uniform() < 0.7 else 'NED'}" for _ in gs]],
    platform="GPL570", table=tab)
mk(B / "GSE51373")
series_matrix(B / "GSE51373" / "GSE51373_series_matrix.txt.gz", gs[:30], gs[:30], [
    ["tissue: Ovary"] * 30, ["Stage: IIIc"] * 30,
    [f"classification: chemotherapy {'resistant' if x > 0 else 'sensitive'}" for x in q[:30]]], platform="GPL570", table=tab.iloc[:, :30])
mk(B / "GSE140082")
series_matrix(B / "GSE140082" / "GSE140082_series_matrix.txt.gz", gs, gs, [
    ["tissue: ovarian tumour"] * nb_, [f"os time (months): {d / 30:.1f}" for d in os_d],
    [f"os status: {'dead' if rng.uniform() < 0.6 else 'alive'}" for _ in gs]], platform="GPL570", table=tab)
C = mk(B / "curatedOvarianData")
cur = pd.DataFrame(np.vstack([np.log2(counts(profile(x, y), 2e6) + 1) for x, y in zip(h, q)]).T, index=SYM, columns=gs)
cur.index.name = "gene"
cur.to_csv(C / "GSE32062.GPL6480_eset_expr.tsv.gz", sep="\t", compression="gzip")
pd.DataFrame({"sample": gs, "days_to_death": os_d, "vital_status": rng.choice(["deceased", "living"], nb_),
              "days_to_tumor_recurrence": (os_d / 2).round(), "recurrence_status": rng.choice(["recurrence", "norecurrence"], nb_),
              "debulking": rng.choice(["optimal", "suboptimal"], nb_)}).to_csv(C / "GSE32062.GPL6480_eset_pheno.tsv", sep="\t", index=False)
print("synthetic inputs written to", I)

# ---------------------------------------------------------------- ABSOLUTE segments / purity (for 01a)
CHR_LEN = {c: int(2.5e8 - c * 9e6) for c in range(1, 23)}
seg_rows, abs_rows = [], []
for p, h, pu in zip(pats, hrd_lat, purity):
    smp = f"{p}-01"
    n_events = int(np.clip(round(12 + 10 * h + rng.normal(scale=3)), 1, 45))
    for c in range(1, 23):
        L = CHR_LEN[c]
        cuts = np.sort(rng.choice(np.arange(1, 40), size=max(1, n_events // 6), replace=False)) * (L // 40)
        bounds = [1] + list(cuts) + [L]
        for a, b in zip(bounds[:-1], bounds[1:]):
            r = rng.uniform()
            minor, major = (0, 2) if r < 0.15 + 0.05 * h else ((1, 2) if r < 0.4 else (1, 1))
            seg_rows.append({"Chromosome": c, "Start": a, "End": b, "Num_Probes": 100, "Length": b - a,
                             "Modal_HSCN_1": minor, "Modal_HSCN_2": major, "Modal_Total_CN": minor + major, "Sample": smp})
    abs_rows.append({"array": smp, "sample": smp, "call status": "called", "purity": pu, "ploidy": 2.5})
pd.DataFrame(seg_rows).to_csv(T / "TCGA_mastercalls.abs_segtabs.fixed.txt", sep="\t", index=False)
pd.DataFrame(abs_rows).to_csv(T / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt", sep="\t", index=False)
# published DDR scores only for a subset of OV tumours (as in the real resource)
dd = pd.read_csv(T / "DDRscores.tsv", sep="\t", header=None)
hi = names.index("HRD_Score")
dd.iloc[np.arange(0, n, 2), [hi, names.index("purity")]] = np.nan
dd.to_csv(T / "DDRscores.tsv", sep="\t", header=False, index=False, na_rep="NaN")
print("segments written")
