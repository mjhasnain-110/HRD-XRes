#!/usr/bin/env python
"""13b - Figure panels for the comparison with the published HRD signature (Peng et al. 2014).

Reads Results/baselines_peng/ (step 07b) plus the HRD-XRes results of steps 07, 09 and 10.
Panels (PDF + 300 dpi PNG) in Results/figures/:
  FigS_peng_A_tcga         TCGA cross-validation: HRD-XRes vs Peng (pre-specified, TCGA-oriented, best case)
  FigS_peng_B_spectrum     SPECTRUM patient-level AUROC vs WGS HR status
  FigS_peng_C_meta         Survival meta-analysis (pooled HR per SD) on the same cohorts
  FigS_peng_D_patients     Per-patient scores: HRD-XRes vs Peng, coloured by WGS HR status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from hrdx.utils import ensure_dir, get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("13b_peng_figures", cfg)
RES = cfg["paths"]["results"]
B = RES / "baselines_peng"
F = ensure_dir(RES / "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

COL = {"HRD-XRes": "#C0392B", "prespecified": "#AAB7B8", "tcga_oriented": "#2E86C1", "best_case": "#85C1E9"}
LAB = {"HRD-XRes": "HRD-XRes", "prespecified": "Peng 2014 (published direction)",
       "tcga_oriented": "Peng 2014 (TCGA-oriented)", "best_case": "Peng 2014 (best case)"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=300)
    plt.close(fig)
    log.info(f"figure {name}")


def err(est, lo, hi):
    est, lo, hi = (np.asarray(v, float) for v in (est, lo, hi))
    return np.nan_to_num(np.vstack([est - lo, hi - est]), nan=0.0).clip(min=0)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ================================================================= A. TCGA
def panel_tcga():
    t = pd.read_csv(B / "tcga_comparison.csv")
    m = pd.read_csv(RES / "cv" / "metrics_all_methods.csv") if (RES / "cv" / "metrics_all_methods.csv").exists() else None
    tasks = [("hr_status", "AUROC", "HR status\nAUROC"), ("hrd_score", "Spearman", "HRD score\nSpearman"),
             ("brca_alt", "AUROC", "BRCA1/2\nAUROC"), ("platinum_resistant", "AUROC", "Platinum resistance\nAUROC"),
             ("os", "C-index", "Overall survival\nC-index")]
    series = ["HRD-XRes", "prespecified", "tcga_oriented", "best_case"]
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (task, metric, _) in enumerate(tasks):
        sub = t[t["task"] == task].set_index("orientation")
        vals = {}
        if m is not None:
            r = m[(m["method"] == "full") & (m["task"] == task) & (m["metric"] == metric)]
            if len(r):
                vals["HRD-XRes"] = (r["estimate"].iloc[0], r["ci_low"].iloc[0], r["ci_high"].iloc[0])
        if "HRD-XRes" not in vals and "best_case" in sub.index:
            v = sub.loc["best_case", "HRD-XRes_estimate"]
            vals["HRD-XRes"] = (v, np.nan, np.nan)
        for o in ("prespecified", "tcga_oriented", "best_case"):
            if o in sub.index:
                r = sub.loc[o]
                vals[o] = (r["peng"], r.get("peng_ci_low", np.nan), r.get("peng_ci_high", np.nan))
        for j, sname in enumerate(series):
            if sname not in vals:
                continue
            est, lo, hi = vals[sname]
            x = i + (j - 1.5) * w
            ax.bar(x, est, w, color=COL[sname], yerr=err([est], [lo], [hi]), capsize=2,
                   label=LAB[sname] if i == 0 else None, edgecolor="none")
        if "tcga_oriented" in sub.index:
            p = sub.loc["tcga_oriented"].get("HRD-XRes_p", np.nan)
            top = np.nanmax([v[2] if np.isfinite(v[2]) else v[0] for v in vals.values()])
            ax.text(i, top + 0.04, stars(p), ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", lw=0.8, c="grey")
    ax.axhline(0.0, lw=0.6, c="black")
    ax.set_xticks(range(len(tasks)), [t_[2] for t_ in tasks])
    ax.set_ylabel("Cross-validated performance (95% CI)")
    ax.set_ylim(min(-0.3, ax.get_ylim()[0]), 1.08)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("TCGA-OV: HRD-XRes vs published HRD signature\n(stars: paired bootstrap, HRD-XRes vs TCGA-oriented Peng)",
                 fontsize=9)
    save(fig, "FigS_peng_A_tcga")


# ================================================================= B. SPECTRUM
def panel_spectrum():
    s = pd.read_csv(B / "spectrum_comparison.csv")
    pl = RES / "singlecell" / "patient_level_metrics.csv"
    hm = pd.read_csv(pl) if pl.exists() else None
    levels = [("spectrum_pseudobulk", "Tumour-site pseudobulk\n(all cell types)"),
              ("spectrum_metacells", "Malignant metacells")]
    series = ["HRD-XRes", "prespecified", "tcga_oriented", "best_case"]
    w = 0.2
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for i, (lvl, _) in enumerate(levels):
        sub = s[s["level"] == lvl].set_index("orientation")
        vals = {}
        if hm is not None:
            r = hm[(hm["level"] == lvl) & (hm["method"] == "HRD-XRes (full)") & (hm["score"] == "p_hr") &
                   (hm["metric"] == "AUROC (WGS HR status)")]
            if len(r):
                vals["HRD-XRes"] = (r["estimate"].iloc[0], r["ci_low"].iloc[0], r["ci_high"].iloc[0])
        if "HRD-XRes" not in vals and len(sub):
            vals["HRD-XRes"] = (sub["hrdxres_auroc"].iloc[0], np.nan, np.nan)
        for o in ("prespecified", "tcga_oriented", "best_case"):
            if o in sub.index:
                r = sub.loc[o]
                vals[o] = (r["peng_auroc"], r.get("peng_ci_low", np.nan), r.get("peng_ci_high", np.nan))
        for j, sname in enumerate(series):
            if sname in vals:
                est, lo, hi = vals[sname]
                ax.bar(i + (j - 1.5) * w, est, w, color=COL[sname], yerr=err([est], [lo], [hi]), capsize=2,
                       label=LAB[sname] if i == 0 else None, edgecolor="none")
        if "tcga_oriented" in sub.index:
            ax.text(i, 1.02, stars(sub.loc["tcga_oriented", "diff_p"]), ha="center")
    n = int(s["n_patients"].dropna().iloc[0]) if s["n_patients"].notna().any() else None
    ax.axhline(0.5, ls="--", lw=0.8, c="grey")
    ax.set_xticks(range(len(levels)), [l_[1] for l_ in levels])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC vs WGS HR status (95% CI)")
    ax.set_title(f"MSK SPECTRUM{f' ({n} patients)' if n else ''}", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[k], label=LAB[k]) for k in series], frameon=False, fontsize=7, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "FigS_peng_B_spectrum")


# ================================================================= C. survival meta-analysis
def panel_meta():
    m = pd.read_csv(B / "meta_analysis_comparison.csv")
    m["series"] = np.where(m["method"].str.startswith("HRD-XRes"), "HRD-XRes", m["orientation"])
    eps = [e for e in ("os", "pfs") if e in set(m["endpoint"])]
    fig, axes = plt.subplots(1, len(eps), figsize=(5.6 * len(eps), 2.9), squeeze=False)
    order = ["HRD-XRes", "tcga_oriented", "prespecified", "best_case"]
    for ax, ep in zip(axes[0], eps):
        g = m[m["endpoint"] == ep].set_index("series").reindex([o for o in order if o in set(m["series"])]).dropna(subset=["pooled_HR"])
        y = np.arange(len(g))[::-1]
        for yi, (sname, r) in zip(y, g.iterrows()):
            ax.errorbar(r["pooled_HR"], yi, xerr=[[r["pooled_HR"] - r["ci_low"]], [r["ci_high"] - r["pooled_HR"]]],
                        fmt="D" if sname == "HRD-XRes" else "s", color=COL[sname], capsize=3, ms=6)
            ax.text(1.02, (yi + 0.5) / len(g), f"{r['pooled_HR']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f})\np = {r['p']:.1e}",
                    transform=ax.transAxes, va="center", fontsize=7)
        ax.set_yticks(y, [LAB[s_] for s_ in g.index], fontsize=7)
        ax.set_ylim(-0.5, len(g) - 0.5)
        ax.axvline(1, ls="--", lw=0.8, c="grey")
        ax.set_xscale("log")
        lo_, hi_ = g["ci_low"].min(), g["ci_high"].max()
        ax.set_xlim(min(lo_, 0.9) * 0.92, max(hi_, 1.1) * 1.08)
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        ticks = [t_ for t_ in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0) if ax.get_xlim()[0] <= t_ <= ax.get_xlim()[1]]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        k, n = int(g["k_cohorts"].iloc[0]), int(g["n_total"].iloc[0])
        ax.set_title(f"{ep.upper()}: {k} cohorts, {n} patients", fontsize=9)
        ax.set_xlabel("Pooled hazard ratio per SD (random effects)")
    fig.subplots_adjust(right=0.8, wspace=0.9)
    save(fig, "FigS_peng_C_meta")


# ================================================================= D. per-patient scores
def panel_patients():
    sgn = 1.0
    ts = B / "tcga_peng_scores.csv"
    if ts.exists():
        t = pd.read_csv(ts, index_col=0).dropna()
        if len(t) > 2:
            sgn = float(np.sign(np.corrcoef(t["peng"], t["peng_tcga_oriented"])[0, 1])) or 1.0
    levels = [("spectrum_pseudobulk", "Tumour-site pseudobulk"), ("spectrum_metacells", "Malignant metacells")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (lvl, title) in zip(axes, levels):
        f = B / f"{lvl}_patient_scores.csv"
        if not f.exists():
            ax.axis("off")
            continue
        d = pd.read_csv(f, index_col=0).dropna(subset=["peng", "hrdxres"])
        d["peng_oriented"] = sgn * d["peng"]
        for status, col, lab in ((1.0, "#C0392B", "HRD (WGS)"), (0.0, "#2E86C1", "HRP (WGS)")):
            s_ = d[d["hr_status_wgs"] == status]
            ax.scatter(s_["peng_oriented"], s_["hrdxres"], c=col, s=28, alpha=0.85, label=lab, edgecolor="white", lw=0.5)
        u = d[d["hr_status_wgs"].isna()]
        if len(u):
            ax.scatter(u["peng_oriented"], u["hrdxres"], c="lightgrey", s=22, label="Undetermined", edgecolor="white", lw=0.5)
        r = spearmanr(d["peng_oriented"], d["hrdxres"])
        ax.axhline(0.5, ls="--", lw=0.8, c="grey")
        ax.set(xlabel="Peng 2014 score (TCGA-oriented)", ylabel="HRD-XRes predicted P(HRD)",
               title=f"{title}\nSpearman between scores = {r.correlation:.2f}")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
               (("#C0392B", "HRD (WGS)"), ("#2E86C1", "HRP (WGS)"), ("lightgrey", "Undetermined"))]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    save(fig, "FigS_peng_D_patients")


for fn in (panel_tcga, panel_spectrum, panel_meta, panel_patients):
    try:
        fn()
    except FileNotFoundError as exc:
        log.warning(f"{fn.__name__}: input missing ({exc.filename})")
log.info(f"Peng comparison figures written to {F}")
