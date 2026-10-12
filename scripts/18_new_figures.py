#!/usr/bin/env python
"""18 - Figures for analyses 14-17 (same style as 13_make_figures.py).

Main-text figures
  Fig4b_phrd_survival_forest     P(HRD) vs overall survival, 16 external cohorts (step 14 D)
  Fig7_model_concordance         HRD-XRes vs IdentifiHR vs elastic net (step 15 I)
  Fig8_spatial_niche             HRD-high regions and neighbouring cell types + two control analyses (15 N, 16, 17)
Supplementary figures
  FigS_robustness_genes          gene-dropout and platform patterns (14 A)
  FigS_calibration               calibration across datasets (14 C)
  FigS_admixture_auroc           accuracy vs WGS under admixture (17)

Run:  python scripts/18_new_figures.py > Logs/18.out 2>&1
Outputs: Results/figures/  (PDF + 300 dpi PNG)
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
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from scipy.stats import mannwhitneyu

from hrdx.utils import ensure_dir, get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("18_new_figures", cfg)
RES = cfg["paths"]["results"]
F = ensure_dir(RES / "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.grid": False, "legend.frameon": False})
# same identities as 13_make_figures.py; IdentifiHR added (palette validated for colour-vision deficiency)
PALETTE = {"full": "#C0392B", "no_da": "#E59866", "no_pathway": "#F5B7B1", "mlp": "#AAB7B8", "enet": "#2E86C1",
           "HRD-XRes": "#C0392B", "HRD-XRes (full)": "#C0392B", "HRD-XRes (no_da)": "#E59866",
           "Elastic net": "#2E86C1", "Elastic net (TCGA)": "#2E86C1", "IdentifiHR": "#B7950B",
           "HR pathway score": "#229954"}
LABELS = {"full": "HRD-XRes", "no_da": "w/o domain adaptation", "no_pathway": "w/o pathway layer", "mlp": "MLP",
          "enet": "Elastic net", "rf": "Random forest", "gbm": "Gradient boosting", "HRD-XRes (full)": "HRD-XRes", "HRD-XRes (no_da)": "w/o domain adaptation",
          "Elastic net (TCGA)": "Elastic net"}
GREY = "#7F8C8D"
CT_ORDER = ["Dendritic_cell", "Mast_cell", "Endothelial_cell", "B_cell", "T_cell", "Monocyte", "Fibroblast", "Plasma_cell"]


def nice(ct):
    return ct.replace("_", " ")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=300)
    plt.close(fig)
    log.info(f"figure {name}")


def plain_log_axis(ax, ticks=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))


def panel_label(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def panel(fn):
    try:
        fn()
    except FileNotFoundError as exc:
        log.warning(f"{fn.__name__}: input missing ({exc.filename})")


def pfmt(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


# ===================================================================== Fig 4b: P(HRD) vs overall survival
def fig_phrd_survival():
    per = pd.read_csv(RES / "robustness" / "phrd_survival_per_cohort.csv")
    meta = pd.read_csv(RES / "robustness" / "phrd_survival_meta.csv")
    d = per[(per.method == "HRD-XRes (full)") & (per.endpoint == "os")].copy()
    d["lo"], d["hi"] = np.exp(d.log_hr - 1.96 * d.se), np.exp(d.log_hr + 1.96 * d.se)
    d = d.sort_values("HR_per_SD_PHRD", ascending=False).reset_index(drop=True)
    m = meta[meta.endpoint == "os"].set_index("method")

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(d) + 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(d))[::-1] + 2
    w = 1 / d.se ** 2
    ax.errorbar(d.HR_per_SD_PHRD, y, xerr=[d.HR_per_SD_PHRD - d.lo, d.hi - d.HR_per_SD_PHRD], fmt="none",
                ecolor=GREY, elinewidth=1, capsize=0)
    ax.scatter(d.HR_per_SD_PHRD, y, s=12 + 60 * w / w.max(), marker="s", c=PALETTE["full"], zorder=3)
    if "HRD-XRes (full)" in m.index:
        r = m.loc["HRD-XRes (full)"]
        xs = [r.ci_low, r.pooled_HR, r.ci_high, r.pooled_HR]
        ax.fill(xs, [0.5, 0.75, 0.5, 0.25], c=PALETTE["full"])
        ax.text(1.02, 0.5, f"Pooled HR {r.pooled_HR:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f}), {pfmt(r.p)}, I² = {100 * r.I2:.0f}%",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_yticks(list(y) + [0.5], [f"{c} (n={n})" for c, n in zip(d.cohort, d.n)] + ["Random effects"])
    ax.axvline(1, ls="--", c=GREY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2.5)
    plain_log_axis(ax, ticks=(0.5, 1.0, 2.0))
    ax.set_xlabel("Hazard ratio per SD of predicted HRD (overall survival)")
    ax.set_title("HRD-XRes, 16 external cohorts", loc="left")
    panel_label(ax, "a")

    ax = axes[1]
    order = [k for k in ("HRD-XRes (full)", "HRD-XRes (no_da)", "Elastic net (TCGA)", "HR pathway score") if k in m.index]
    mm = m.loc[order]
    yy = np.arange(len(mm))[::-1]
    ax.errorbar(mm.pooled_HR, yy, xerr=[mm.pooled_HR - mm.ci_low, mm.ci_high - mm.pooled_HR], fmt="none",
                ecolor=GREY, elinewidth=1)
    ax.scatter(mm.pooled_HR, yy, s=60, marker="D", c=[PALETTE.get(k, GREY) for k in mm.index], zorder=3)
    for yi, (k, r) in zip(yy, mm.iterrows()):
        ax.text(1.02, yi, f"{r.pooled_HR:.2f}, {pfmt(r.p)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_yticks(yy, [LABELS.get(k, k) for k in mm.index])
    ax.axvline(1, ls="--", c=GREY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 1.4)
    ax.set_ylim(-0.6, len(mm) - 0.4)
    plain_log_axis(ax, ticks=(0.6, 0.8, 1.0, 1.2, 1.4))
    ax.set_xlabel("Pooled HR per SD")
    ax.set_title("Pooled estimate by model", loc="left")
    panel_label(ax, "b")
    save(fig, "Fig4b_phrd_survival_forest")


# ===================================================================== Fig 7: model concordance (IdentifiHR)
def fig_model_concordance():
    h = pd.read_csv(RES / "identifihr" / "identifihr_head_to_head.csv")
    models = ["HRD-XRes", "IdentifiHR", "Elastic net"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.2, 1, 1.1, 1]})

    # a. AUROC vs WGS
    ax = axes[0]
    inputs = [("patient pseudobulk, malignant cells", "Malignant cells"), ("site pseudobulk, all cells", "All cells (sites)")]
    width = 0.26
    for j, mname in enumerate(models):
        vals, lo, hi = [], [], []
        for inp, _ in inputs:
            r = h[(h.dataset == "SPECTRUM") & (h.input == inp) & (h.method == mname) & (h.metric == "AUROC vs WGS HR status")].iloc[0]
            vals.append(r.estimate); lo.append(r.estimate - r.ci_low); hi.append(r.ci_high - r.estimate)
        x = np.arange(len(inputs)) + (j - 1) * width
        ax.bar(x, vals, width * 0.92, yerr=[lo, hi], color=PALETTE[mname], capsize=2, label=mname)
        for xi, v in zip(x, vals):
            ax.text(xi, 0.505, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="white", rotation=90)
    ax.set_xticks(np.arange(len(inputs)), [b for _, b in inputs])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC vs WGS-defined HRD")
    ax.set_title("SPECTRUM, 37 patients", loc="left")
    ax.legend(loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.32))
    panel_label(ax, "a")

    # b. agreement HRD-XRes vs IdentifiHR
    ax = axes[1]
    pr = pd.read_csv(RES / "identifihr" / "spectrum_patient_malignant_predictions.csv", index_col=0)
    lab = pd.read_csv(cfg["paths"]["processed"] / "spectrum" / "spectrum_patient_labels.csv", index_col=0)
    pr.index, lab.index = pr.index.astype(str), lab.index.astype(str)
    d = pr.join(lab[["hr_status_wgs"]], how="inner").dropna(subset=["hr_status_wgs"])
    for v, c, l in ((1, PALETTE["full"], "HRD (WGS)"), (0, PALETTE["enet"], "HR-proficient (WGS)")):
        s = d[d.hr_status_wgs == v]
        ax.scatter(s["IdentifiHR"], s["HRD-XRes"], s=22, c=c, label=l, edgecolors="white", linewidths=0.5)
    rho = d[["IdentifiHR", "HRD-XRes"]].corr(method="spearman").iloc[0, 1]
    ax.plot([0, 1], [0, 1], ls=":", c=GREY, lw=0.8)
    ax.axhline(0.5, c=GREY, lw=0.5, ls="--"); ax.axvline(0.5, c=GREY, lw=0.5, ls="--")
    ax.text(0.03, 0.95, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top")
    ax.set_xlabel("IdentifiHR P(HRD)"); ax.set_ylabel("HRD-XRes P(HRD)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Patient-level agreement", loc="left")
    panel_label(ax, "b")

    # c. calibration-dependent behaviour on mixed-cell site profiles
    ax = axes[2]
    mets = [("Accuracy (P>=0.5)", "Accuracy at P = 0.5"), ("ICC across sites", "Site ICC"),
            ("Fraction patients with concordant site calls", "Concordant site calls")]
    for j, mname in enumerate(models):
        vals = []
        for met, _ in mets:
            r = h[(h.input == "site pseudobulk, all cells") & (h.method == mname) & (h.metric == met)]
            vals.append(r.estimate.iloc[0] if len(r) else np.nan)
        x = np.arange(len(mets)) + (j - 1) * width
        ax.bar(x, vals, width * 0.92, color=PALETTE[mname])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(len(mets)), [b for _, b in mets], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("All-cell site profiles (156 sites, 41 patients)", loc="left")
    panel_label(ax, "c")

    # d. spatial, section pseudobulk
    ax = axes[3]
    rows = []
    for ds, met, lab_ in (("Visium Denisenko (post-NACT)", "Spearman vs CRS", "Denisenko\nρ vs CRS"),
                          ("Visium Stur (pre-treatment)", "AUROC poor vs excellent", "Stur\nAUROC")):
        for mname in models:
            r = h[(h.dataset == ds) & (h.method == mname) & h.metric.str.startswith(met)]
            rows.append((lab_, mname, r.estimate.iloc[0] if len(r) else np.nan))
    sp_ = pd.DataFrame(rows, columns=["test", "method", "v"])
    for i, (t, g) in enumerate(sp_.groupby("test", sort=False)):
        for j, (_, r) in enumerate(g.iterrows()):
            x = i + (j - 1) * width
            ax.bar(x, r.v, width * 0.92, color=PALETTE[r.method])
            ax.text(x, r.v + (0.02 if r.v >= 0 else -0.02), f"{r.v:.2f}", ha="center",
                    va="bottom" if r.v >= 0 else "top", fontsize=6.5)
    ax.axhline(0, c="black", lw=0.6)
    ax.axhline(0.5, c=GREY, lw=0.6, ls="--", xmin=0.5)
    ax.set_xticks([0, 1], sp_["test"].unique())
    ax.set_ylim(-1, 1)
    ax.set_title("Visium section pseudobulk", loc="left")
    panel_label(ax, "d")
    save(fig, "Fig7_model_concordance")


# ===================================================================== Fig 8: spatial niche + controls
def fig_spatial_niche():
    per = pd.read_csv(RES / "identifihr" / "neighbourhood_adjusted_per_section.csv")
    summ = pd.read_csv(RES / "identifihr" / "neighbourhood_adjusted_summary.csv")
    slopes = pd.read_csv(RES / "mixing" / "mixing_slopes.csv")
    mix = pd.read_csv(RES / "mixing" / "mixing_summary.csv")
    cts = [c for c in CT_ORDER if c in set(per.cell_type)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1.4, 1.1, 1, 1.1]})

    # a. per-section adjusted correlation
    ax = axes[0]
    rng = np.random.default_rng(0)
    both = summ[summ.cohort == "Both"].set_index("cell_type")
    for i, ct in enumerate(cts):
        for coh, mk, col in (("Denisenko", "o", PALETTE["full"]), ("Stur", "^", "#E59866")):
            v = per[(per.cell_type == ct) & (per.cohort == coh)]["rho_adj_malignant"]
            ax.scatter(i + rng.normal(0, 0.07, len(v)), v, s=14, marker=mk, c=col, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
        if ct in both.index:
            b = both.loc[ct]
            ax.plot([i - 0.3, i + 0.3], [b.mean_rho_adj] * 2, c="black", lw=1.5)
            star = "*" if b.sign_test_fdr < 0.05 else ""
            ax.text(i, 0.33, f"{int(b.n_positive_adj)}/{int(b.n_sections)}{star}",
                    ha="center", fontsize=7)
    ax.axhline(0, c=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(len(cts)), [nice(c) for c in cts], rotation=40, ha="right")
    ax.set_ylim(-0.3, 0.4)
    ax.set_ylabel("Spot P(HRD) vs cell-type fraction\n(partial ρ, adjusted for malignant fraction)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", c=PALETTE["full"], label="Denisenko (8 sections)"),
                       Line2D([], [], marker="^", ls="", c="#E59866", label="Stur (12 sections)"),
                       Line2D([], [], c="black", lw=1.5, label="mean")], fontsize=7, loc="lower left", ncol=3)
    ax.set_title("Neighbourhood of HRD-high tumour spots", loc="left")
    panel_label(ax, "a")

    # b. admixture curves
    ax = axes[1]
    mm = mix[mix.method == "HRD-XRes"]
    cols = plt.get_cmap("tab10")
    for k, ct in enumerate(cts):
        g = mm[mm.cell_type == ct].sort_values("fraction")
        if len(g):
            hit = ct in ("Dendritic_cell", "Mast_cell")
            ax.plot(100 * g.fraction, g.mean_delta, lw=2 if hit else 1, c=cols(k), label=nice(ct),
                    marker="o" if hit else None, ms=4)
    ax.axhline(0, c=GREY, lw=0.8, ls="--")
    ax.set_xlabel("Non-malignant RNA added (%)")
    ax.set_ylabel("Change in P(HRD) vs pure malignant profile")
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    ax.set_title("In-silico admixture (41 patients)", loc="left")
    panel_label(ax, "b")

    # c. admixture slope vs spatial association
    ax = axes[2]
    s = slopes[slopes.method == "HRD-XRes"].dropna(subset=["mean_rho_adj"])
    ax.axhline(0, c=GREY, lw=0.6); ax.axvline(0, c=GREY, lw=0.6)
    ax.scatter(s.slope_per_10pct, s.mean_rho_adj, s=36, c=PALETTE["full"], zorder=3)
    for _, r in s.iterrows():
        ax.annotate(nice(r.cell_type), (r.slope_per_10pct, r.mean_rho_adj), xytext=(4, 3),
                    textcoords="offset points", fontsize=7)
    ax.text(0.98, 0.98, "If admixture caused the spatial\npattern, points would lie here", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=GREY)
    ax.set_xlabel("Admixture effect on P(HRD)\n(change per 10% added RNA)")
    ax.set_ylabel("Spatial association (mean partial ρ)")
    ax.set_title("Admixture does not explain the pattern", loc="left")
    panel_label(ax, "c")

    # d. whole-tumour abundance by WGS status (model-free)
    ax = axes[3]
    comp = pd.read_csv(RES / "composition" / "site_composition_normalised.csv", index_col=0)
    show = [c for c in ("Dendritic_cell", "Mast_cell", "Endothelial_cell") if c in comp.columns]
    pat = comp.groupby("patient")[show + ["hr_status_wgs"]].mean()
    pat = pat[pat.hr_status_wgs.isin([0, 1])]
    for i, ct in enumerate(show):
        for off, v, col in ((-0.18, 1, PALETTE["full"]), (0.18, 0, PALETTE["enet"])):
            vals = pat.loc[pat.hr_status_wgs == v, ct].dropna()
            bp = ax.boxplot(vals, positions=[i + off], widths=0.3, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=col, alpha=0.35, edgecolor=col)
            for k_ in ("medians", "whiskers", "caps"):
                for l_ in bp[k_]:
                    l_.set(color=col)
            ax.scatter(i + off + rng.normal(0, 0.03, len(vals)), vals, s=8, c=col)
    ticks = []
    for ct in show:
        a, b = pat.loc[pat.hr_status_wgs == 1, ct].dropna(), pat.loc[pat.hr_status_wgs == 0, ct].dropna()
        ticks.append(f"{nice(ct)}\n{pfmt(mannwhitneyu(a, b).pvalue)}")
    ax.set_xticks(range(len(show)), ticks)
    ax.set_ylabel("Fraction within compartment")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", c=PALETTE["full"], label="HRD (WGS)"),
                       Line2D([], [], marker="s", ls="", c=PALETTE["enet"], label="HR-proficient (WGS)")],
              fontsize=7, loc="upper left")
    ax.set_title("Whole-tumour abundance (SPECTRUM)", loc="left")
    panel_label(ax, "d")
    save(fig, "Fig8_spatial_niche")


# ===================================================================== Supplementary
def figS_robustness():
    s = pd.read_csv(RES / "robustness" / "gene_dropout_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, (met, sd, ylab) in zip(axes[:2], (("auroc_hr", "auroc_sd", "HRD status AUROC"),
                                              ("spearman_hrd", "spearman_sd", "HRD score Spearman ρ"))):
        r = s[s.scenario == "random"].copy()
        r["level"] = pd.to_numeric(r["level"])
        r[met] = pd.to_numeric(r[met])
        for mname in ("full", "no_da", "no_pathway", "mlp", "enet"):
            g = r[r.method == mname].sort_values("level")
            if len(g):
                ax.errorbar(100 * g.level, g[met], yerr=g[sd].fillna(0), c=PALETTE[mname], lw=2 if mname == "full" else 1,
                            capsize=2, label=LABELS[mname], marker="o", ms=3)
        ax.set_xlabel("Input genes removed at random (%)")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].set_title("Random gene loss (TCGA outer folds)", loc="left")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    ax = axes[2]
    p = s[s.scenario == "platform_pattern"].pivot(index="level", columns="method", values="auroc_hr").astype(float)
    fr = s[(s.scenario == "platform_pattern") & (s.method == "full")].set_index("level")["frac_dropped"]
    p = p.loc[fr.sort_values().index]
    for mname in ("full", "no_da", "no_pathway", "mlp", "enet"):
        if mname in p:
            ax.scatter(p[mname], range(len(p)), c=PALETTE[mname], s=18, label=LABELS[mname])
    ax.set_yticks(range(len(p)), [f"{c} ({100 * fr[c]:.0f}% missing)" for c in p.index], fontsize=7)
    ax.set_xlabel("HRD status AUROC (TCGA) with platform's missing genes")
    ax.set_title("Real platform gene-loss patterns", loc="left")
    panel_label(ax, "c")
    save(fig, "FigS_robustness_genes")


def figS_calibration():
    c = pd.read_csv(RES / "robustness" / "calibration.csv")
    c["label"] = c.method.map(lambda m: LABELS.get(m, m))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (met, title, ref) in zip(axes, (("brier", "Brier score (lower is better)", None),
                                            ("calibration_slope", "Calibration slope (1 = ideal)", 1.0))):
        y = np.arange(len(c))[::-1]
        ax.barh(y, c[met], color=[PALETTE.get(m, GREY) for m in c.method], height=0.7)
        ax.set_yticks(y, [f"{d} | {l}" for d, l in zip(c.dataset, c.label)], fontsize=7)
        if ref is not None:
            ax.axvline(ref, ls="--", c=GREY, lw=0.8)
        ax.set_title(title, loc="left")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    save(fig, "FigS_calibration")


def figS_admixture_auroc():
    mix = pd.read_csv(RES / "mixing" / "mixing_summary.csv")
    mm = mix[mix.method == "HRD-XRes"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    cols = plt.get_cmap("tab10")
    for k, ct in enumerate([c for c in CT_ORDER if c in set(mm.cell_type)]):
        g = mm[mm.cell_type == ct].sort_values("fraction")
        ax.plot(100 * g.fraction, g.auroc_wgs, c=cols(k), label=nice(ct), marker="o", ms=3, lw=1)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, ls="--", c=GREY, lw=0.8)
    ax.set_xlabel("Non-malignant RNA added (%)")
    ax.set_ylabel("AUROC vs WGS-defined HRD")
    ax.set_title("Discrimination under admixture (HRD-XRes)", loc="left")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    save(fig, "FigS_admixture_auroc")


for fn in (fig_phrd_survival, fig_model_concordance, fig_spatial_niche, figS_robustness, figS_calibration,
           figS_admixture_auroc):
    panel(fn)
log.info(f"Done. Figures in {F}")