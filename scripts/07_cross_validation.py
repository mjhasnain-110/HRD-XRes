#!/usr/bin/env python
"""07 - Nested cross-validation on TCGA-OV: HRD-XRes, ablations and baselines.

Outer loop: stratified K-fold (by HR status). Inner loop: hyper-parameter grid search for the full model
(inner K-fold). Every variant (full, no_da, no_pathway, mlp, single_task) is then trained with the selected
hyper-parameters as a multi-seed ensemble and evaluated on the untouched outer test fold.
Baselines: elastic net, random forest, gradient boosting, knowledge pathway scores, purity,
deconvolution (SPECTRUM reference) + regression.

All checkpoints are kept: Models/cv/<variant>/fold<k>/seed<s>/{best.pt,last.pt,history.csv,meta.json}
Inner-loop checkpoints: Models/cv/inner/fold<k>/cfg<c>_in<j>/...
Results: Results/cv/
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from hrdx.baselines import KNOWLEDGE_PATTERNS, classical, cox_risk, pathway_scores
from hrdx.data import load_features, load_inputs, marker_genes, nnls_deconvolve, read_gmt, cpm_dense
from hrdx.metrics import auroc, cindex, evaluate_predictions, paired_bootstrap_diff, spearman
from hrdx.trainer import (VARIANTS, ensemble_predict, get_device, labels_to_source, load_checkpoint,
                          stratified_holdout, train_model)
from hrdx.utils import ensure_dir, get_logger, inp, load_config, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--variants", default=",".join(VARIANTS))
ap.add_argument("--skip-baselines", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config)
log = get_logger("07_cross_validation", cfg)
P, R, M = cfg["paths"]["processed"], ensure_dir(cfg["paths"]["results"] / "cv"), ensure_dir(cfg["paths"]["models"] / "cv")
T, MC = cfg["training"], cfg["model"]
device = get_device(T["device"])
log.info(f"Device: {device}")

feat = load_features(P)
mask, genes = feat["mask"], feat["genes"]
X, lab = load_inputs(P, "tcga")
src = labels_to_source(X, lab)
rng = np.random.default_rng(T["seed"])
targets = []
for d in feat["target_domains"]:
    Xt, _ = load_inputs(P, d)
    if Xt.shape[0] > 20000:
        Xt = Xt[rng.choice(Xt.shape[0], 20000, replace=False)]
    targets.append(Xt)
    log.info(f"target domain {d}: {Xt.shape[0]} profiles")

grid = [dict(zip(T["grid"], v)) for v in itertools.product(*T["grid"].values())]
strat = np.where(np.isfinite(src.y["hr"]), src.y["hr"], -1).astype(int)
outer = StratifiedKFold(T["outer_folds"], shuffle=True, random_state=T["seed"])
folds = list(outer.split(X, strat))
fold_of = np.empty(len(lab), dtype=int)
for k, (_, te) in enumerate(folds):
    fold_of[te] = k
pd.DataFrame({"sample": lab.index, "fold": fold_of}).to_csv(R / "fold_assignment.csv", index=False)
if (R / "metrics_per_fold.csv").exists():
    (R / "metrics_per_fold.csv").unlink()


def train_or_load(train, val, hp, variant, seed, out_dir):
    if (out_dir / "best.pt").exists():
        model, meta = load_checkpoint(out_dir / "best.pt", device)
        return model, meta
    return train_model(train, val, targets, mask, hp, MC, T, variant, seed, out_dir, device, log)


# ---------------------------------------------------------------- inner model selection
selected = {}
sel_path = R / "selected_hyperparameters.csv"
sel_rows = []
for k, (tr, te) in enumerate(folds):
    inner = StratifiedKFold(T["inner_folds"], shuffle=True, random_state=T["seed"] + k)
    rows = []
    for c, hp in enumerate(grid):
        scores = []
        for j, (itr, iva) in enumerate(inner.split(tr, strat[tr])):
            od = M / "inner" / f"fold{k}" / f"cfg{c}_in{j}"
            _, meta = train_or_load(src.subset(tr[itr]), src.subset(tr[iva]), hp, "full", T["seed"] + 100 * k + j, od)
            scores.append(meta["best_val_composite"])
        rows.append({"fold": k, "config": c, **hp, "inner_mean": float(np.mean(scores)), "inner_sd": float(np.std(scores))})
    rdf = pd.DataFrame(rows)
    best = rdf.sort_values("inner_mean", ascending=False).iloc[0]
    selected[k] = {h: best[h] for h in T["grid"]}
    sel_rows.append({"fold": k, **selected[k], "inner_mean": best["inner_mean"]})
    rdf.to_csv(R / f"inner_grid_fold{k}.csv", index=False)
    log.info(f"Fold {k}: selected {selected[k]} (inner composite {best['inner_mean']:.4f})")
pd.DataFrame(sel_rows).to_csv(sel_path, index=False)

# ---------------------------------------------------------------- outer evaluation of DL variants
preds = {}
for variant in args.variants.split(","):
    parts = []
    for k, (tr, te) in enumerate(folds):
        tr_i, va_i = stratified_holdout(src.y["hr"][tr], T["early_stop_fraction"], T["seed"] + k)
        models = []
        for s in range(T["n_seeds_cv"]):
            od = M / variant / f"fold{k}" / f"seed{s}"
            m, _ = train_or_load(src.subset(tr[tr_i]), src.subset(tr[va_i]), selected[k], variant, T["seed"] + 1000 * s + k, od)
            models.append(m)
        p = ensemble_predict(models, X[te], device)
        p.index = lab.index[te]
        p["fold"] = k
        parts.append(p)
    preds[variant] = pd.concat(parts).loc[lab.index]
    preds[variant].to_csv(R / f"predictions_{variant}.csv")
    log.info(f"{variant}: outer-fold predictions saved")

# ---------------------------------------------------------------- baselines
if not args.skip_baselines:
    gmt = read_gmt(inp(cfg, cfg["files"]["reactome_gmt"]))
    kscores = pathway_scores(X, genes, gmt, KNOWLEDGE_PATTERNS)
    kscores.index = lab.index
    kscores.to_csv(R / "knowledge_scores.csv")

    # deconvolution with the SPECTRUM reference
    ref = ad.read_h5ad(P / "spectrum" / "spectrum_reference.h5ad")
    tcga = ad.read_h5ad(P / "tcga" / "tcga_ov.h5ad")
    refdf = pd.DataFrame(ref.X, index=ref.obs_names, columns=ref.var_names)
    common = refdf.columns.intersection(tcga.var_names)
    mk = [g for g in marker_genes(refdf[common]) if g in common]
    mix = cpm_dense(tcga[lab.index, :].X) if tcga.uns.get("value_type") == "counts" else np.asarray(tcga[lab.index].X)
    mix = pd.DataFrame(mix, columns=tcga.var_names)[mk].values
    fracs = pd.DataFrame(nnls_deconvolve(mix, refdf[mk].values), index=lab.index,
                         columns=[f"frac_{c}" for c in refdf.index])
    fracs.to_csv(R / "tcga_deconvolution_fractions.csv")
    log.info(f"Deconvolution: {len(mk)} marker genes, {fracs.shape[1]} cell types")

    for kind in ["enet", "rf", "gbm"]:
        parts = []
        for k, (tr, te) in enumerate(folds):
            p = classical(kind, X[tr], src.subset(tr).y, X[te], seed=T["seed"] + k)
            p.index = lab.index[te]
            p["fold"] = k
            parts.append(p)
        preds[kind] = pd.concat(parts).loc[lab.index]
        preds[kind].to_csv(R / f"predictions_{kind}.csv")
        log.info(f"baseline {kind} done")

    # single-score baselines (knowledge pathway scores, purity): the direction of each score is learned on the
    # training folds only (sign of its Spearman correlation with the label), then applied to the test fold
    single = {"hr_pathway_score": kscores["hr_pathway_score"].values, "proliferation_score": kscores["proliferation_score"].values,
              "purity_only": lab["purity"].values}
    targets_map = [("hr", "p_hr"), ("hrd", "pred_hrd"), ("brca", "p_brca"), ("plat", "p_plat")]
    for sname, score in single.items():
        parts = []
        for k, (tr, te) in enumerate(folds):
            p = pd.DataFrame(index=lab.index[te])
            for key, col in targets_map:
                r = spearman(score[tr], src.y[key][tr])
                p[col] = score[te] * (np.sign(r) if np.isfinite(r) and r != 0 else 1.0)
            r = cindex(src.y["time"][tr], src.y["event"][tr], score[tr])
            p["risk"] = score[te] * (1.0 if not np.isfinite(r) or r >= 0.5 else -1.0)
            p["fold"] = k
            parts.append(p)
        preds[sname] = pd.concat(parts).loc[lab.index]
        preds[sname].to_csv(R / f"predictions_{sname}.csv")

    parts = []
    F = fracs.values
    for k, (tr, te) in enumerate(folds):
        p = pd.DataFrame(index=lab.index[te])
        for key, col in [("hr", "p_hr"), ("brca", "p_brca"), ("plat", "p_plat")]:
            y = src.y[key][tr]
            m = np.isfinite(y)
            if m.sum() > 20 and len(np.unique(y[m])) == 2:
                p[col] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(F[tr][m], y[m]).predict_proba(F[te])[:, 1]
        p["risk"] = cox_risk(F[tr], src.y["time"][tr], src.y["event"][tr], F[te], penalizer=0.5)
        p["fold"] = k
        parts.append(p)
    preds["deconv"] = pd.concat(parts).loc[lab.index]
    preds["deconv"].to_csv(R / "predictions_deconv.csv")

# ---------------------------------------------------------------- metrics
nb = cfg["evaluation"]["n_bootstrap"]
allm = []
for name, p in preds.items():
    mt = evaluate_predictions(p.drop(columns=["fold"], errors="ignore"), lab, n_boot=nb, seed=T["seed"])
    mt.insert(0, "method", name)
    allm.append(mt)
    if "fold" in p:  # per-fold estimates (variability across folds)
        for k in sorted(p["fold"].unique()):
            idx = p.index[p["fold"] == k]
            fm = evaluate_predictions(p.loc[idx].drop(columns=["fold"]), lab.loc[idx], n_boot=0)
            fm.insert(0, "fold", k)
            fm.insert(0, "method", name)
            fm.to_csv(R / "metrics_per_fold.csv", mode="a", header=not (R / "metrics_per_fold.csv").exists(), index=False)
metrics = pd.concat(allm)
metrics.to_csv(R / "metrics_all_methods.csv", index=False)
log.info("Cross-validated performance:\n" + metrics.pivot_table(index="method", columns=["task", "metric"], values="estimate").round(3).to_string())

# paired comparisons: full model vs every other method
comp = []
full = preds.get("full")
if full is not None:
    tests = [("hr_status", "AUROC", auroc, "hr_status", "p_hr", None),
             ("hrd_score", "Spearman", spearman, "hrd_score", "pred_hrd", None),
             ("brca_alt", "AUROC", auroc, "brca_alt", "p_brca", None),
             ("platinum_resistant", "AUROC", auroc, "platinum_resistant", "p_plat", None),
             ("os", "C-index", cindex, "os_time", "risk", "os")]
    for name, p in preds.items():
        if name == "full":
            continue
        for task, metric, fn, ycol, scol, ecol in tests:
            if scol not in p.columns:
                continue
            d, lo, hi, pv = paired_bootstrap_diff(fn, lab[ycol].values, full[scol].values, p[scol].values,
                                                  n_boot=nb, seed=T["seed"], extra=(lab[ecol].values,) if ecol else None)
            comp.append({"comparison": f"full vs {name}", "task": task, "metric": metric,
                         "difference": d, "ci_low": lo, "ci_high": hi, "p_value": pv})
    pd.DataFrame(comp).to_csv(R / "paired_comparisons.csv", index=False)

    # purity-adjusted added value of the prediction (likelihood-ratio test)
    import statsmodels.api as sm
    rows = []
    for ycol, scol in [("hr_status", "p_hr"), ("brca_alt", "p_brca"), ("platinum_resistant", "p_plat")]:
        d = pd.DataFrame({"y": lab[ycol], "pur": lab["purity"], "s": full[scol]}).dropna()
        if d["y"].nunique() < 2 or len(d) < 30:
            continue
        m0 = sm.Logit(d["y"], sm.add_constant(d[["pur"]])).fit(disp=0)
        m1 = sm.Logit(d["y"], sm.add_constant(d[["pur", "s"]])).fit(disp=0)
        lr = 2 * (m1.llf - m0.llf)
        from scipy.stats import chi2
        rows.append({"label": ycol, "n": len(d), "auroc_purity_only": auroc(d["y"], m0.predict()),
                     "auroc_purity_plus_model": auroc(d["y"], m1.predict()),
                     "LR_stat": lr, "p_value": chi2.sf(lr, 1), "model_coef": m1.params["s"]})
    d = pd.DataFrame({"h": lab["hrd_score"], "pur": lab["purity"], "s": full["pred_hrd"]}).dropna()
    rh = sm.OLS(d["h"].rank(), sm.add_constant(d["pur"].rank())).fit().resid
    rs = sm.OLS(d["s"].rank(), sm.add_constant(d["pur"].rank())).fit().resid
    rows.append({"label": "hrd_score (partial Spearman | purity)", "n": len(d),
                 "partial_spearman": float(np.corrcoef(rh, rs)[0, 1])})
    pd.DataFrame(rows).to_csv(R / "purity_adjusted_analysis.csv", index=False)

    # threshold sensitivity: AUROC of p_hr for alternative HRD score cut-offs
    rows = []
    for thr in cfg["tcga"]["hrd_threshold_sensitivity"]:
        y = np.where(lab["hrd_score"].notna(), (lab["hrd_score"] >= thr).astype(float), np.nan)
        rows.append({"threshold": thr, "n_hrd": int(np.nansum(y)), "auroc_p_hr": auroc(y, full["p_hr"]),
                     "auroc_pred_hrd": auroc(y, full["pred_hrd"])})
    pd.DataFrame(rows).to_csv(R / "threshold_sensitivity.csv", index=False)
save_json({"selected_hyperparameters": selected, "n_samples": int(len(lab)), "variants": list(preds)}, R / "cv_summary.json")
log.info(f"Cross-validation complete. Results in {R}")
