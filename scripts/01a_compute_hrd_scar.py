#!/usr/bin/env python
"""01a - Genomic HRD scar scores (HRD-LOH, TAI, LST) for every TCGA-OV tumour from ABSOLUTE
allele-specific copy-number segments (TCGA_mastercalls.abs_segtabs.fixed.txt, hg19).

Definitions (scarHRD / Telli et al. 2016 convention):
  HRD-LOH  LOH segments (minor allele CN = 0) longer than 15 Mb, excluding chromosomes whose LOH covers
           >= 90% of the chromosome (Abkevich et al. 2012).
  TAI      allelic-imbalance segments (major != minor CN) that reach a telomere and do not cross the
           centromere (Birkbak et al. 2012).
  LST      after removing segments < 3 Mb, breakpoints between adjacent segments that are both >= 10 Mb,
           within a chromosome arm (Popova et al. 2012).
  HRD sum  LOH + TAI + LST

The published DDR scores (Knijnenburg et al. 2018) exist for only a subset of OV tumours. On that overlap
the recomputed scores are compared with the published ones and a linear calibration is fitted, so the
final label is on the published scale: published value where available, calibrated recomputed value
elsewhere (column hrd_source records which).

Outputs (Processed/tcga/): hrd_scar_scores.csv, hrd_scar_concordance.csv
Run this before 01_prepare_tcga.py, which picks the scores up automatically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from hrdx.utils import ensure_dir, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("01a_compute_hrd_scar", cfg)
F = cfg["files"]["tcga"]
out = ensure_dir(cfg["paths"]["processed"] / "tcga")
MB = 1e6

# hg19 centromere positions (UCSC gap table)
CENTRO = {1: (121535434, 124535434), 2: (92326171, 95326171), 3: (90504854, 93504854), 4: (49660117, 52660117),
          5: (46405641, 49405641), 6: (58830166, 61830166), 7: (58054331, 61054331), 8: (43838887, 46838887),
          9: (47367679, 50367679), 10: (39254935, 42254935), 11: (51644205, 54644205), 12: (34856694, 37856694),
          13: (16000000, 19000000), 14: (16000000, 19000000), 15: (17000000, 20000000), 16: (35335801, 38335801),
          17: (22263006, 25263006), 18: (15460898, 18460898), 19: (24681782, 27681782), 20: (26369569, 29369569),
          21: (11288129, 14288129), 22: (13000000, 16000000)}


def pick(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} in columns {list(cols)}")


# ---------------------------------------------------------------- load segments
seg_path = cfg["paths"]["inputs"] / "tcga" / "TCGA_mastercalls.abs_segtabs.fixed.txt"
seg = pd.read_csv(seg_path, sep="\t", low_memory=False, encoding="latin-1")
log.info(f"ABSOLUTE segments: {len(seg)} rows; columns: {list(seg.columns)}")
c_s = pick(seg.columns, "Sample", "sample", "ID")
c_chr = pick(seg.columns, "Chromosome", "chrom", "chr")
c_st = pick(seg.columns, "Start", "start", "loc.start")
c_en = pick(seg.columns, "End", "end", "loc.end")
c_a1 = pick(seg.columns, "Modal_HSCN_1", "nMinor", "minor_cn", "nB")
c_a2 = pick(seg.columns, "Modal_HSCN_2", "nMajor", "major_cn", "nA")
seg = seg[[c_s, c_chr, c_st, c_en, c_a1, c_a2]].copy()
seg.columns = ["sample", "chrom", "start", "end", "a1", "a2"]
seg["sample"] = seg["sample"].astype(str).str[:15]
seg["chrom"] = pd.to_numeric(seg["chrom"].astype(str).str.replace("chr", "", regex=False), errors="coerce")
seg = seg[seg["chrom"].between(1, 22)].dropna(subset=["a1", "a2", "start", "end"])
seg["minor"] = np.minimum(seg["a1"], seg["a2"]).round().astype(int)
seg["major"] = np.maximum(seg["a1"], seg["a2"]).round().astype(int)

samples = pd.read_csv(inp(cfg, F["ddr_samples"]), sep="\t", header=None, names=["sample", "cancer"])
cdr = pd.read_excel(inp(cfg, F["cdr"]), sheet_name=0)
ov_patients = set(cdr.loc[cdr["type"] == "OV", "bcr_patient_barcode"].astype(str))
seg = seg[seg["sample"].str[:12].isin(ov_patients) & (seg["sample"].str[13:15] == "01")]
log.info(f"OV primary tumours with ABSOLUTE segments: {seg['sample'].nunique()}")


def merge_same(df: pd.DataFrame) -> pd.DataFrame:
    """Merge adjacent segments with identical allele-specific state."""
    rows = []
    for r in df.itertuples(index=False):
        if rows and rows[-1]["major"] == r.major and rows[-1]["minor"] == r.minor:
            rows[-1]["end"] = r.end
        else:
            rows.append({"start": r.start, "end": r.end, "major": r.major, "minor": r.minor})
    return pd.DataFrame(rows)


def scores_for(s: pd.DataFrame) -> dict:
    loh = tai = lst = 0
    for ch, d in s.groupby("chrom"):
        d = merge_same(d.sort_values("start"))
        if d.empty:
            continue
        d["len"] = d["end"] - d["start"]
        chr_start, chr_end = d["start"].min(), d["end"].max()
        chr_len = chr_end - chr_start
        cen0, cen1 = CENTRO[int(ch)]

        # HRD-LOH
        is_loh = (d["minor"] == 0) & (d["major"] > 0)
        if d.loc[is_loh, "len"].sum() < 0.9 * chr_len:
            loh += int(((d["len"] > 15 * MB) & is_loh).sum())

        # TAI: allelic imbalance segment touching a telomere, not crossing the centromere
        ai = d["major"] != d["minor"]
        first, last = d.iloc[0], d.iloc[-1]
        if ai.iloc[0] and first["end"] <= cen1 and not (first["start"] < cen0 and first["end"] > cen1):
            tai += 1
        if len(d) > 1 and ai.iloc[-1] and last["start"] >= cen0 and not (last["start"] < cen0 and last["end"] > cen1):
            tai += 1

        # LST: per arm, drop < 3 Mb segments, re-merge, count adjacent >= 10 Mb pairs with gap <= 3 Mb
        for arm in (d[d["end"] <= cen1], d[d["start"] >= cen0]):
            a = arm[arm["len"] >= 3 * MB]
            if len(a) < 2:
                continue
            a = merge_same(a.sort_values("start"))
            a["len"] = a["end"] - a["start"]
            for i in range(len(a) - 1):
                if a["len"].iloc[i] >= 10 * MB and a["len"].iloc[i + 1] >= 10 * MB and \
                        a["start"].iloc[i + 1] - a["end"].iloc[i] <= 3 * MB:
                    lst += 1
    return {"scar_loh": loh, "scar_tai": tai, "scar_lst": lst, "scar_sum": loh + tai + lst}


res = pd.DataFrame({smp: scores_for(s) for smp, s in seg.groupby("sample")}).T
res.index.name = "sample"
log.info(f"Recomputed scar scores for {len(res)} tumours")

# ---------------------------------------------------------------- concordance with published DDR scores
names = [l.strip() for l in open(inp(cfg, F["ddr_score_names"])) if l.strip()]
ddr = pd.read_csv(inp(cfg, F["ddr_scores"]), sep="\t", header=None, names=names, na_values=["NaN", "NA", ""])
ddr.index = samples["sample"].str[:15].values
ddr = ddr[samples["cancer"].values == "OV"]
ddr = ddr[~ddr.index.duplicated()]
both = res.join(ddr[["HRD_Score", "HRD_LOH", "HRD_TAI", "HRD_LST"]], how="inner").dropna(subset=["HRD_Score"])
conc = []
for mine, pub in [("scar_sum", "HRD_Score"), ("scar_loh", "HRD_LOH"), ("scar_tai", "HRD_TAI"), ("scar_lst", "HRD_LST")]:
    d = both[[mine, pub]].dropna().astype(float)
    if len(d) > 10:
        conc.append({"component": pub, "n": len(d), "spearman": spearmanr(d[mine], d[pub]).correlation,
                     "pearson": pearsonr(d[mine], d[pub])[0], "mean_recomputed": d[mine].mean(), "mean_published": d[pub].mean()})
conc = pd.DataFrame(conc)
slope, intercept = np.polyfit(both["scar_sum"].astype(float), both["HRD_Score"].astype(float), 1)
thr = cfg["tcga"]["hrd_threshold"]
cal = intercept + slope * res["scar_sum"].astype(float)
agree = ((cal.reindex(both.index) >= thr) == (both["HRD_Score"] >= thr)).mean()
conc = pd.concat([conc, pd.DataFrame([{"component": "calibration", "n": len(both), "slope": slope,
                                       "intercept": intercept, "hr_status_agreement": agree}])])
conc.to_csv(out / "hrd_scar_concordance.csv", index=False)
log.info("Concordance with published scores:\n" + conc.to_string(index=False))
if conc.loc[conc["component"] == "HRD_Score", "spearman"].iloc[0] < 0.8:
    log.warning("Spearman < 0.8 between recomputed and published HRD sum - inspect hrd_scar_concordance.csv")

res["hrd_calibrated"] = cal
res["hrd_published"] = ddr["HRD_Score"].reindex(res.index)
res["hrd_final"] = res["hrd_published"].fillna(res["hrd_calibrated"])
res["hrd_source"] = np.where(res["hrd_published"].notna(), "published", "recomputed_calibrated")
res.to_csv(out / "hrd_scar_scores.csv")
log.info(f"Final HRD labels: {res['hrd_source'].value_counts().to_dict()}; "
         f"HRD (>= {thr}): {(res['hrd_final'] >= thr).sum()} / {len(res)}")
