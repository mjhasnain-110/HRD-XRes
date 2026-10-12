#!/usr/bin/env python
"""13 - Publication figure panels (PDF + 300 dpi PNG) from the result tables.

Outputs: Results/figures/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from sklearn.metrics import roc_curve

from hrdx.utils import ensure_dir, get_logger, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("13_make_figures", cfg)
RES = cfg["paths"]["results"]
F = ensure_dir(RES / "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
PALETTE = {"full": "#C0392B", "no_da": "#E59866", "no_pathway": "#F5B7B1", "mlp": "#AAB7B8", "single_task": "#D98880",
           "enet": "#2E86C1", "rf": "#5DADE2", "gbm": "#85C1E9", "deconv": "#7D3C98",
           "hr_pathway_score": "#229954", "proliferation_score": "#82E0AA", "purity_only": "#7F8C8D"}
LABELS = {"full": "HRD-XRes", "no_da": "w/o domain adaptation", "no_pathway": "w/o pathway layer", "mlp": "MLP",
          "single_task": "single-task", "enet": "Elastic net", "rf": "Random forest", "gbm": "Gradient boosting",
          "deconv": "Deconvolution", "hr_pathway_score": "HR pathway score", "proliferation_score": "Proliferation",
          "purity_only": "Purity"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=300)
    plt.close(fig)
    log.info(f"figure {name}")


def plain_log_axis(ax):
    lo, hi = ax.get_xlim()
    ticks = [t for t in (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0) if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def panel(fn):
    try:
        fn()
    except FileNotFoundError as exc:
        log.warning(f"{fn.__name__}: input missing ({exc.filename})")


def fig_cv_benchmark():
    m = pd.read_csv(RES / "cv" / "metrics_all_methods.csv")
    tasks = [("hr_status", "AUROC"), ("hrd_score", "Spearman"), ("brca_alt", "AUROC"),
             ("platinum_resistant", "AUROC"), ("os", "C-index")]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 4.2), sharey=False)
    for ax, (t, met) in zip(axes, tasks):
        d = m[(m.task == t) & (m.metric == met)].copy()
        d = d[d["method"].isin(PALETTE)].sort_values("estimate")
        y = np.arange(len(d))
        err = np.vstack([d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]])
        ax.barh(y, d["estimate"], xerr=np.nan_to_num(err), color=[PALETTE[x] for x in d["method"]], capsize=2)
        ax.set_yticks(y, [LABELS[x] for x in d["method"]])
        ax.axvline(0.5 if met != "Spearman" else 0, ls="--", c="grey", lw=0.8)
        ax.set_title(f"{t.replace('_', ' ')}\n{met}")
        lo = 0.3 if met != "Spearman" else min(0, d["estimate"].min() - 0.05)
        ax.set_xlim(lo, 1)
    save(fig, "Fig2_cv_benchmark")


def fig_cv_roc():
    lab = pd.read_csv(cfg["paths"]["processed"] / "tcga" / "tcga_labels.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (ycol, scol, title) in zip(axes, [("hr_status", "p_hr", "HR status"), ("brca_alt", "p_brca", "BRCA1/2"),
                                              ("platinum_resistant", "p_plat", "Platinum resistance")]):
        for meth in ("full", "no_da", "enet", "rf", "deconv"):
            f = RES / "cv" / f"predictions_{meth}.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f, index_col=0)
            if scol not in p:
                continue
            d = lab[[ycol]].join(p[[scol]]).dropna()
            fpr, tpr, _ = roc_curve(d[ycol], d[scol])
            ax.plot(fpr, tpr, color=PALETTE[meth], lw=2 if meth == "full" else 1, label=LABELS[meth])
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
        ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "Fig2_cv_roc")


def fig_external():
    m = pd.read_csv(RES / "external_bulk" / "external_metrics.csv")
    d = m[(m.metric == "AUROC") & (m.score.isin(["P(resistant)", "P(BRCA)"]))]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    d = d.assign(label=d["cohort"] + "\n" + d["task"].str.replace("platinum_resistant", "platinum resistance")
                 .str.replace("brca1_mut", "BRCA1 mutation"))
    labs = list(dict.fromkeys(d["label"]))
    meths = list(dict.fromkeys(d["method"]))
    w = 0.8 / len(meths)
    for i, mm in enumerate(meths):
        s = d[d.method == mm].set_index("label").reindex(labs)
        x = np.arange(len(labs)) + i * w
        ax.bar(x, s["estimate"], w, yerr=np.nan_to_num(np.vstack([s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]])),
               capsize=2, label=mm)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xticks(np.arange(len(labs)) + 0.4 - w / 2, labs)
    ax.set(ylabel="AUROC (95% CI)", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7)
    save(fig, "Fig3_external_classification")

    meta = pd.read_csv(RES / "external_bulk" / "meta_analysis.csv")
    cox = pd.read_csv(RES / "external_bulk" / "external_cox.csv")
    c = cox[(cox.method == "HRD-XRes (full)") & (cox.model == "univariable")]
    for ep in c["endpoint"].unique():
        g = c[c.endpoint == ep]
        mm = meta[(meta.method == "HRD-XRes (full)") & (meta.endpoint == ep)]
        fig, ax = plt.subplots(figsize=(5, 0.4 * (len(g) + 2) + 1))
        y = np.arange(len(g))[::-1] + 1
        ax.errorbar(g["HR"], y, xerr=[g["HR"] - g["ci_low"], g["ci_high"] - g["HR"]], fmt="s", c="k", capsize=2)
        ax.set_yticks(list(y) + [0], [f"{a} (n={n})" for a, n in zip(g["cohort"], g["n"])] + ["Pooled (random effects)"])
        if len(mm):
            r = mm.iloc[0]
            ax.errorbar([r["pooled_HR"]], [0], xerr=[[r["pooled_HR"] - r["ci_low"]], [r["ci_high"] - r["pooled_HR"]]],
                        fmt="D", c=PALETTE["full"], capsize=3)
            ax.set_title(f"{ep.upper()}: pooled HR {r['pooled_HR']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f}), "
                         f"p={r['p']:.1e}, I²={r['I2']:.0%}", fontsize=8)
        ax.axvline(1, ls="--", c="grey", lw=0.8)
        ax.set_xscale("log")
        plain_log_axis(ax)
        ax.set_xlabel("Hazard ratio per SD of HRD-XRes risk")
        save(fig, f"Fig3_forest_{ep}")

    km = pd.read_csv(RES / "external_bulk" / "external_km_curves.csv")
    km = km[km.method == "HRD-XRes (full)"]
    combos = km[["cohort", "endpoint"]].drop_duplicates().values
    n = len(combos)
    if n:
        ncol = min(5, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        for ax, (co, ep) in zip(axes.ravel(), combos):
            s = km[(km.cohort == co) & (km.endpoint == ep)]
            for grp, col in (("high risk", PALETTE["full"]), ("low risk", "#2E86C1")):
                g = s[s.group == grp]
                ax.step(g["time"] / 30.4375, g["survival"], where="post", c=col, label=grp)
            ax.set(xlabel="Months", ylabel="Survival", ylim=(0, 1.02))
            ax.set_title(f"{co} {ep.upper()}\nlog-rank p={s['logrank_p'].iloc[0]:.1e}", fontsize=8)
        axes[0][0].legend(frameon=False)
        save(fig, "Fig3_km")


def fig_singlecell():
    pp = pd.read_csv(RES / "singlecell" / "patient_predictions.csv", index_col=0, low_memory=False)
    d = pp[(pp.method == "HRD-XRes (full)")].copy()
    for c in ("hr_status_wgs", "p_hr"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["hr_status_wgs", "p_hr"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, level in zip(axes[:2], ("spectrum_pseudobulk", "spectrum_metacells")):
        s = d[d.level == level]
        groups = [s.loc[s.hr_status_wgs == v, "p_hr"].values for v in (0, 1)]
        ax.boxplot(groups, widths=0.5, showfliers=False)
        for i, g in enumerate(groups, start=1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(g)), g, s=12, c=PALETTE["full"], alpha=0.8)
        ax.set_xticks([1, 2], ["HRP (WGS)", "HRD (WGS)"])
        ax.set(ylabel="Predicted P(HRD), patient mean", title=level.replace("spectrum_", "SPECTRUM ").replace("_", " "))
    pm = pd.read_csv(RES / "singlecell" / "patient_level_metrics.csv")
    a = pm[(pm.metric == "AUROC (WGS HR status)") & (pm.score == "p_hr")]
    ax = axes[2]
    a = a.assign(label=a["method"] + "\n" + a["level"].str.replace("spectrum_", ""))
    y = np.arange(len(a))
    ax.barh(y, a["estimate"], xerr=np.nan_to_num(np.vstack([a["estimate"] - a["ci_low"], a["ci_high"] - a["estimate"]])),
            color=[PALETTE["full"] if "full" in m else ("#E59866" if "no_da" in m else "#2E86C1") for m in a["method"]], capsize=2)
    ax.set_yticks(y, a["label"], fontsize=6)
    ax.axvline(0.5, ls="--", c="grey", lw=0.8)
    ax.set(xlabel="AUROC vs WGS HR status", xlim=(0.3, 1))
    save(fig, "Fig4_singlecell_validation")

    mc = pd.read_csv(RES / "singlecell" / "metacell_predictions.csv", index_col=0)
    if "frac_arms_imbalanced" in mc:
        dep = RES / "baselines_peng" / "allelic_imbalance_depth_check.csv"
        note = ""
        if dep.exists():
            dc = pd.read_csv(dep)
            r = dc[(dc.score == "p_hr") & (dc.feature == "frac_arms_imbalanced")]
            if len(r):
                note = (f"\nwithin-patient rho = {r['within_patient_rho'].iloc[0]:.2f}; "
                        f"depth-adjusted rho = {r['depth_adjusted_within_patient_rho'].iloc[0]:.2f}")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        hb = ax.hexbin(mc["frac_arms_imbalanced"], mc["p_hr"], gridsize=30, cmap="Reds", mincnt=1, bins="log")
        fig.colorbar(hb, ax=ax, label="Metacells (log)")
        ax.set(xlabel="Fraction of chromosome arms in allelic imbalance", ylabel="Predicted P(HRD)")
        ax.set_title(f"SPECTRUM malignant metacells (n={len(mc)}){note}", fontsize=8)
        save(fig, "Fig4_metacell_allelic_imbalance")


def fig_spatial():
    sp_ = pd.read_csv(RES / "spatial" / "denisenko_spots_spot_predictions.csv", index_col=0)
    secs = sorted(sp_["section"].unique())
    fig, axes = plt.subplots(2, int(np.ceil(len(secs) / 2)), figsize=(3 * np.ceil(len(secs) / 2), 6), squeeze=False)
    for ax, s in zip(axes.ravel(), secs):
        d = sp_[sp_.section == s]
        sc = ax.scatter(d["array_col"], -d["array_row"], c=d["p_hr"], s=4, cmap="RdBu_r", vmin=0, vmax=1)
        ax.scatter(d.loc[~d.tumor_spot, "array_col"], -d.loc[~d.tumor_spot, "array_row"], s=4, c="lightgrey")
        ax.set_title(f"{s} ({d['patient'].iloc[0]}, {d['response'].iloc[0]})", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes.ravel()[len(secs):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="P(HRD) (tumour spots)")
    fig.savefig(F / "Fig5_spatial_maps.pdf")
    fig.savefig(F / "Fig5_spatial_maps.png", dpi=300)
    plt.close(fig)

    den = pd.read_csv(RES / "spatial" / "denisenko_spots_section_metrics.csv", index_col=0)
    stur = pd.read_csv(RES / "spatial" / "stur_spots_section_metrics.csv", index_col=0)
    tests = pd.read_csv(RES / "spatial" / "spatial_response_tests.csv")

    def stat(cohort, metric):
        r = tests[(tests.cohort.str.startswith(cohort)) & (tests.metric == metric)]
        return r.iloc[0] if len(r) else None

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    rng = np.random.default_rng(1)
    for ax, (m, lab) in zip(axes[:2], (("mean_p_hr", "Mean P(HRD) in tumour spots"),
                                       ("frac_hrd_high", "Fraction of HRD-high tumour spots"))):
        for i, r_ in enumerate(["good", "partial", "poor"]):
            v = den.loc[den.response == r_, m].dropna()
            ax.scatter(i + rng.normal(0, 0.04, len(v)), v, c=PALETTE["full"], s=30)
        st = stat("Denisenko", m)
        ttl = f"Denisenko (post-NACT)\nSpearman rho = {st['estimate']:.2f}, p = {st['p_value']:.3f}" if st is not None else "Denisenko"
        ax.set_xticks(range(3), ["good", "partial", "poor"])
        ax.set(xlabel="Chemotherapy response score", ylabel=lab)
        ax.set_title(ttl, fontsize=8)
    ax = axes[2]
    for i, r_ in enumerate(["excellent", "poor"]):
        v = stur.loc[stur.response == r_, "mean_p_hr"].dropna()
        ax.scatter(i + rng.normal(0, 0.04, len(v)), v, c=PALETTE["full"], s=30)
    st = stat("Stur", "mean_p_hr")
    ax.set_xticks([0, 1], ["excellent", "poor"])
    ax.set(xlabel="Response to NACT", ylabel="Mean P(HRD) in tumour spots")
    ax.set_title("Stur (pre-treatment, primary test)\n" + (f"AUROC = {st['estimate']:.2f}, p = {st['p_value']:.3f}"
                                                            if st is not None else ""), fontsize=8)
    save(fig, "Fig5_spatial_response")


def fig_interpret():
    pa = pd.read_csv(RES / "interpretation" / "pathway_attributions.csv")
    enr_f = RES / "interpretation" / "dna_repair_enrichment.csv"
    enr = pd.read_csv(enr_f) if enr_f.exists() else pd.DataFrame()
    repair = r"Homologous Recombination|HDR|Fanconi|DNA Double-Strand|DNA Repair|Nucleotide Excision|Mismatch|BRCA|Base Excision|D-loop|Nonhomologous|NHEJ"
    heads = [h for h in ("hrd", "hr") if h in set(pa["head"])]
    fig, axes = plt.subplots(1, 1 + len(heads), figsize=(4.0 + 3.6 * len(heads), 4.6),
                             gridspec_kw={"width_ratios": [1.6] + [1] * len(heads)})
    d = pa[pa["head"] == heads[0]].head(25)[::-1]
    is_rep = d["pathway"].str.contains(repair, case=False, regex=True)
    ax = axes[0]
    ax.barh(range(len(d)), d["mean_abs_attribution"], color=np.where(is_rep, PALETTE["full"], "#BFC5C9"))
    ax.set_yticks(range(len(d)), [p_.split(" [")[0][:60] for p_ in d["pathway"]], fontsize=6.5)
    ax.set_xlabel("Mean |integrated gradient|")
    ax.set_title(f"Top 25 pathways, {'HRD-score' if heads[0] == 'hrd' else 'HR-status'} head", fontsize=9)
    ax.legend(handles=[Patch(color=PALETTE["full"], label="DNA-repair pathway"), Patch(color="#BFC5C9", label="other")],
              frameon=False, fontsize=7, loc="lower right")
    for ax, h in zip(axes[1:], heads):
        g = pa[pa["head"] == h]
        rep_mask = g["pathway"].str.contains(repair, case=False, regex=True)
        n = len(g)
        for msk, col, lab in ((rep_mask, PALETTE["full"], "DNA-repair"), (~rep_mask, "#7F8C8D", "other")):
            r_ = np.sort(g.loc[msk, "rank"].values) / n
            ax.step(r_, np.arange(1, len(r_) + 1) / max(len(r_), 1), where="post", c=col, label=f"{lab} (n={msk.sum()})")
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
        e = enr[enr["head"] == h] if len(enr) else pd.DataFrame()
        ptxt = f"\nenrichment p = {e['p_one_sided'].iloc[0]:.3f}" if len(e) else ""
        ax.set(xlabel="Attribution rank (fraction of pathways)", ylabel="Cumulative fraction",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.set_title(f"{'HRD-score' if h == 'hrd' else 'HR-status'} head{ptxt}", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "Fig6_pathway_attributions")
    al = pd.read_csv(RES / "interpretation" / "domain_alignment.csv")
    fig, ax = plt.subplots(figsize=(3.2, 3))
    ax.bar(al["variant"].map(LABELS), al["domain_classifier_balanced_accuracy"],
           color=[PALETTE.get(v, "grey") for v in al["variant"]])
    ax.axhline(al["chance_level"].iloc[0], ls="--", c="grey", lw=0.8)
    ax.set(ylabel="Domain classifier balanced accuracy\n(lower = better aligned)")
    save(fig, "Fig6_domain_alignment")


for fn in (fig_cv_benchmark, fig_cv_roc, fig_external, fig_singlecell, fig_spatial, fig_interpret):
    panel(fn)
log.info(f"Figures written to {F}")
