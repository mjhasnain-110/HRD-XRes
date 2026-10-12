#!/usr/bin/env python
"""04 - External bulk validation cohorts.

GEO series matrices (GSE63885, GSE51373, GSE140082, GSE9891) are mapped probe -> gene symbol
(platform annotation downloaded automatically) and labels are parsed into a common schema:
  platinum_resistant, platinum_class, brca1_mut, os, os_time (days), pfs, pfs_time (days)
curatedOvarianData cohorts (exported by scripts/R/export_curatedOvarianData.R) are added when present.

Outputs: Processed/bulk_external/<cohort>.h5ad (value_type = log2) and cohort_summary.csv
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd

from hrdx.geo import find_field, load_platform_symbols, parse_series_matrix, to_log2
from hrdx.utils import collapse_max_mean, ensure_dir, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("04_prepare_bulk", cfg)
B = cfg["files"]["bulk"]
out = ensure_dir(cfg["paths"]["processed"] / "bulk_external")
platform_dir = ensure_dir(inp(cfg, B["platform_dir"]))
summary = []


def num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce").values


def save(name: str, expr: pd.DataFrame, lab: pd.DataFrame, source: str):
    expr = expr.loc[:, lab.index]
    expr = expr[np.isfinite(expr.values).all(axis=1)]
    a = ad.AnnData(X=expr.T.values.astype(np.float32), obs=lab, var=pd.DataFrame(index=expr.index.astype(str)))
    a.uns["value_type"] = "log2"
    a.uns["source"] = source
    a.write_h5ad(out / f"{name}.h5ad", compression="gzip")
    rec = {"cohort": name, "source": source, "n_samples": a.n_obs, "n_genes": a.n_vars}
    for c in ["platinum_resistant", "brca1_mut", "os", "pfs"]:
        if c in lab.columns:
            rec[f"n_{c}"] = int(lab[c].notna().sum())
            rec[f"pos_{c}"] = int((lab[c] == 1).sum())
    summary.append(rec)
    log.info(f"{name}: {rec}")


def generic_survival(meta: pd.DataFrame, ov: dict) -> pd.DataFrame:
    lab = pd.DataFrame(index=meta.index)
    tcol = ov.get("os_time") or find_field(meta, [r"overall survival", r"\bos\b.*(time|day|month)", r"^char::os",
                                                 r"survival.*(time|day|month)", r"time.*death", r"follow"],
                                           exclude=[r"progress", r"pfs", r"dfs", r"recur", r"status", r"event"])
    ecol = ov.get("os_event") or find_field(meta, [r"vital", r"os.*(status|event|censor)", r"death", r"dead",
                                                  r"survival.*status", r"status"],
                                           exclude=[r"progress", r"pfs", r"dfs", r"recur", r"time", r"day", r"month"])
    if tcol and ecol:
        t = num(meta[tcol])
        unit = ov.get("time_unit") or ("months" if "month" in tcol.lower() or np.nanmedian(t) < 200 else "days")
        lab["os_time"] = t * (30.4375 if unit == "months" else 1.0)
        ev = meta[ecol].astype(str).str.lower().str.strip()
        pos = [str(v).lower() for v in ov.get("event_values", [])] or ["1", "dead", "deceased", "dod", "death", "yes", "true"]
        lab["os"] = np.where(ev.isin(["nan", "", "na", "n/a"]), np.nan, ev.isin(pos).astype(float))
        log.info(f"  survival fields: time='{tcol}' ({unit}), event='{ecol}' values={sorted(ev.unique())[:8]}")
    pcol = find_field(meta, [r"(pfs|progression|recurrence|relapse).*(time|day|month)", r"^char::pfs"],
                      exclude=[r"status", r"event"])
    pev = find_field(meta, [r"(pfs|progression|recurrence|relapse).*(status|event)"])
    if pcol and pev:
        t = num(meta[pcol])
        unit = "months" if "month" in pcol.lower() or np.nanmedian(t) < 200 else "days"
        lab["pfs_time"] = t * (30.4375 if unit == "months" else 1.0)
        ev = meta[pev].astype(str).str.lower()
        lab["pfs"] = ev.isin(["1", "yes", "true", "progression", "recurrence", "relapse", "progressed"]).astype(float)
    return lab


def load_geo(gse: str):
    files = sorted((inp(cfg, B["dir"]) / gse).glob("*series_matrix.txt.gz"))
    metas, exprs = [], []
    for f in files:
        meta, expr, gpl = parse_series_matrix(f)
        sym = load_platform_symbols(gpl, platform_dir, log)
        expr.index = expr.index.astype(str)
        g = collapse_max_mean(to_log2(expr), sym)
        metas.append(meta)
        exprs.append(g)
        log.info(f"{gse} [{gpl}]: {expr.shape[1]} samples, {g.shape[0]} genes")
    common = sorted(set.intersection(*[set(e.index) for e in exprs]))
    return pd.concat(metas), pd.concat([e.loc[common] for e in exprs], axis=1)


curated_dir = inp(cfg, B["curated_dir"])
curated = sorted(curated_dir.glob("*_expr.tsv.gz")) if curated_dir.exists() else []
curated_names = {p.name.replace("_expr.tsv.gz", "").split(".")[0].replace("_eset", "") for p in curated}

for gse, ov in cfg["geo_cohorts"].items():
    ov = ov or {}
    if gse in curated_names:
        log.info(f"{gse}: using the curatedOvarianData version (harmonised clinical annotation)")
        continue
    if not (inp(cfg, B["dir"]) / gse).exists():
        log.warning(f"{gse}: folder not found, skipping")
        continue
    meta, expr = load_geo(gse)
    lab = pd.DataFrame(index=meta.index)
    if gse == "GSE63885":
        hist = meta[find_field(meta, [r"histo"])].astype(str).str.lower()
        meta, lab = meta[hist.str.contains("serous")], lab[hist.str.contains("serous").values]
        ps = meta[find_field(meta, [r"platin"])].astype(str).str.lower()
        lab["platinum_class"] = ps.values
        lab["platinum_resistant"] = np.where(ps.str.contains("resistant"), 1.0,
                                             np.where(ps.str.contains("sensitive"), 0.0, np.nan))
        b1 = meta[find_field(meta, [r"brca1"])].astype(str).str.lower()
        lab["brca1_mut"] = np.where(b1.str.contains("no mutation"), 0.0, np.where(b1.str.contains("mutation"), 1.0, np.nan))
        lab["os_time"] = num(meta[find_field(meta, [r"overall survival"])])
        st = meta[find_field(meta, [r"clinical status at last"])].astype(str).str.upper()
        lab["os"] = np.where(st.str.contains("DOD"), 1.0, np.where(st.isin(["NAN", ""]), np.nan, 0.0))
        lab["dfs_time"] = num(meta[find_field(meta, [r"disease-free", r"dfs"])])
        lab["residual"] = meta[find_field(meta, [r"residual"])].astype(str).str.extract(r"(R\d)")[0].values
        lab["first_line_response"] = meta[find_field(meta, [r"post 1st line"])].astype(str).str.extract(r"(CR|PR|SD|P)\s*$")[0].values
    elif gse == "GSE51373":
        cl = meta[find_field(meta, [r"classification"])].astype(str).str.lower()
        lab["platinum_resistant"] = np.where(cl.str.contains("resistant"), 1.0, np.where(cl.str.contains("sensitive"), 0.0, np.nan))
        lab["stage"] = meta[find_field(meta, [r"stage"])].astype(str).values
    elif gse == "GSE140082":
        keep = (meta["char::histology.serous"].str.lower() == "serous") & \
               (meta["char::newgrade"].str.lower() != "low.grade")
        meta = meta[keep]
        lab = pd.DataFrame(index=meta.index)
        lab["os_time"] = num(meta["char::final_ostm"])
        lab["os"] = num(meta["char::final_osid"])
        lab["pfs_time"] = num(meta["char::final_pfstm"])
        lab["pfs"] = num(meta["char::final_pfsid"])
        lab["debulking"] = meta["char::debulking_status"].values
        lab["stage"] = meta["char::figo_stage"].values
        lab["treatment"] = meta["char::treatment"].values
    else:
        lab = generic_survival(meta, ov)
        tissue = find_field(meta, [r"tissue", r"sample type", r"source_name"])
        if tissue:
            t = meta[tissue].astype(str).str.lower()
            normal = t.str.contains(r"normal|healthy|fallopian|ose\b")
            lab = lab[~normal.values]
        hist, grade = find_field(meta, [r"histolog"]), find_field(meta, [r"grade"])
        keep = pd.Series(True, index=lab.index)
        if hist:
            keep &= meta.loc[lab.index, hist].astype(str).str.lower().str.contains("serous").values
        if grade:
            keep &= ~meta.loc[lab.index, grade].astype(str).str.lower().str.contains("low").values
        lab = lab[keep.values]
        for pat, name in [(r"debulk", "debulking"), (r"figo|stage", "stage"), (r"treatment|arm", "treatment")]:
            col = find_field(meta, [pat])
            if col:
                lab[name] = meta.loc[lab.index, col].astype(str).values
        log.info(f"  {gse}: {len(lab)} serous, non-low-grade tumours kept")
        if "os" not in lab.columns:
            log.warning(f"{gse}: survival fields not detected; add os_time/os_event overrides under geo_cohorts in config")
            continue
    save(gse, expr, lab, "GEO")

for p in curated:
    name = p.name.replace("_expr.tsv.gz", "")
    expr = pd.read_csv(p, sep="\t", index_col=0)
    ph = pd.read_csv(p.with_name(f"{name}_pheno.tsv"), sep="\t", index_col=0)
    ph.index = ph.index.astype(str)
    expr.columns = expr.columns.astype(str)
    lab = pd.DataFrame(index=ph.index)
    lab["os_time"] = pd.to_numeric(ph.get("days_to_death"), errors="coerce")
    lab["os"] = ph.get("vital_status").map({"deceased": 1.0, "living": 0.0}) if "vital_status" in ph else np.nan
    if "days_to_tumor_recurrence" in ph:
        lab["pfs_time"] = pd.to_numeric(ph["days_to_tumor_recurrence"], errors="coerce")
        lab["pfs"] = ph["recurrence_status"].map({"recurrence": 1.0, "norecurrence": 0.0})
    for c in ["debulking", "tumorstage", "summarystage", "summarygrade", "age_at_initial_pathologic_diagnosis"]:
        if c in ph:
            lab[c] = ph[c].values
    expr = to_log2(expr)
    save(re.sub(r"[^A-Za-z0-9]+", "_", name.replace("_eset", "")), expr, lab, "curatedOvarianData")

pd.DataFrame(summary).to_csv(out / "cohort_summary.csv", index=False)
log.info("External cohorts:\n" + pd.DataFrame(summary).to_string(index=False))
