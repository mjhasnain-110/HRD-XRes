#!/usr/bin/env python
"""14 - Robustness and "what can the linear model not do" analyses.

Part A  Gene-dropout robustness (TCGA outer CV folds, frozen fold models; no retraining of HRD-XRes)
        - random gene loss 0-50 % (missing genes set to the cohort mean = 0, as for external cohorts)
        - the real missing-gene pattern of every external microarray cohort
        HRD-XRes variants (full, no_da, no_pathway, mlp) vs elastic net refitted on each training fold.
Part B  Elastic net on the spatial cohorts (Denisenko, Stur) with exactly the tests used for HRD-XRes in 11.
Part C  Calibration of P(HRD) in TCGA (outer-fold predictions) and SPECTRUM (patient level).
Part D  P(HRD) itself vs survival in the external cohorts (per-cohort Cox + random-effects meta-analysis).
Part E  Summary of elastic-net vs HRD-XRes for SPECTRUM subtypes / Myriad GIS (already computed in step 10).

Run (one command, ~30-90 min on CPU):
  nohup python scripts/14_robustness.py > Logs/14.out 2>&1 &
Single parts:   python scripts/14_robustness.py --parts B,D
Outputs: Results/robustness/
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import mannwhitneyu, norm, rankdata, spearmanr
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline

from hrdx.data import available_inputs, load_inputs
from hrdx.metrics import auroc, bootstrap_ci, spearman
from hrdx.trainer import get_device, labels_to_source, load_checkpoint, predict
from hrdx.utils import ensure_dir, get_logger, load_config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--parts", default="A,B,C,D,E")
ap.add_argument("--n_jobs", type=int, default=4)
ap.add_argument("--repeats", type=int, default=5, help="random repeats per gene-dropout level")
args = ap.parse_args()
parts = set(args.parts.upper().split(","))

cfg = load_config(args.config)
log = get_logger("14_robustness", cfg)
P, RES = cfg["paths"]["processed"], cfg["paths"]["results"]
R = ensure_dir(RES / "robustness")
M = cfg["paths"]["models"]
device = get_device(cfg["training"]["device"])
nb = cfg["evaluation"]["n_bootstrap"]
SEED = cfg["training"]["seed"]
K_FEAT = 2000

Xtr, labtr = load_inputs(P, "tcga")
src = labels_to_source(Xtr, labtr)
y_hr, y_hrd = src.y["hr"], src.y["hrd"]


def enet_hr(seed=0):
    est = LogisticRegressionCV(Cs=8, penalty="elasticnet", solver="saga", l1_ratios=[0.1, 0.5, 0.9], cv=3,
                               max_iter=3000, scoring="roc_auc", class_weight="balanced",
                               random_state=seed, n_jobs=args.n_jobs)
    return make_pipeline(SelectKBest(f_classif, k=min(K_FEAT, Xtr.shape[1])), est)


def enet_hrd(seed=0):
    est = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000, n_jobs=args.n_jobs, random_state=seed)
    return make_pipeline(SelectKBest(f_regression, k=min(K_FEAT, Xtr.shape[1])), est)


def fit_enet(rows):
    """Elastic net for HRD status and HRD score fitted on the given TCGA rows (same settings as the baseline)."""
    m1 = rows & np.isfinite(y_hr)
    m2 = rows & np.isfinite(y_hrd)
    clf = enet_hr().fit(Xtr[m1], y_hr[m1].astype(int))
    reg = enet_hrd().fit(Xtr[m2], y_hrd[m2])
    return clf, reg


def enet_predict(models, X):
    clf, reg = models
    return pd.DataFrame({"p_hr": clf.predict_proba(X)[:, 1], "pred_hrd": reg.predict(X)})


def meta_dl(y, se):
    v = se ** 2
    w = 1 / v
    fixed = np.sum(w * y) / w.sum()
    Q = np.sum(w * (y - fixed) ** 2)
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    wr = 1 / (v + tau2)
    mu, s = np.sum(wr * y) / wr.sum(), np.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - (len(y) - 1)) / Q) if Q > 0 else 0.0
    return {"pooled_HR": np.exp(mu), "ci_low": np.exp(mu - 1.96 * s), "ci_high": np.exp(mu + 1.96 * s),
            "p": 2 * norm.sf(abs(mu / s)), "I2": I2, "tau2": tau2}


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)


# ====================================================================== A. gene-dropout robustness
if "A" in parts:
    log.info("Part A: gene-dropout robustness")
    fa = pd.read_csv(RES / "cv" / "fold_assignment.csv").set_index("sample").reindex(labtr.index)
    fold = fa["fold"].values.astype(int)
    variants = [v for v in ("full", "no_da", "no_pathway", "mlp") if (M / "cv" / v).exists()]
    fold_models = {}
    for v in variants:
        for k in sorted(set(fold)):
            paths = sorted((M / "cv" / v / f"fold{k}").glob("seed*/best.pt"))
            fold_models[(v, k)] = [load_checkpoint(p, device)[0] for p in paths]
        log.info(f"  loaded fold models for {v}: {[len(fold_models[(v, k)]) for k in sorted(set(fold))]} seeds/fold")
    enet_models = {}
    for k in sorted(set(fold)):
        enet_models[k] = fit_enet(fold != k)
        log.info(f"  elastic net fitted for fold {k}")

    def score_all(mask_fn, tag, level, rep):
        """mask_fn(rng) -> boolean vector of genes to drop; same mask for every method."""
        preds = {m: np.full(len(labtr), np.nan) for m in variants + ["enet"]}
        hrd = {m: np.full(len(labtr), np.nan) for m in variants + ["enet"]}
        rng = np.random.default_rng(SEED + rep)
        drop = mask_fn(rng)
        for k in sorted(set(fold)):
            te = np.where(fold == k)[0]
            Xte = Xtr[te].copy()
            Xte[:, drop] = 0.0
            for v in variants:
                ps = [predict(mm, Xte, device) for mm in fold_models[(v, k)]]
                p = sum(ps) / len(ps)
                preds[v][te], hrd[v][te] = p["p_hr"].values, p["pred_hrd"].values
            p = enet_predict(enet_models[k], Xte)
            preds["enet"][te], hrd["enet"][te] = p["p_hr"].values, p["pred_hrd"].values
        out = []
        for m in preds:
            out.append({"scenario": tag, "level": level, "repeat": rep, "method": m,
                        "frac_genes_dropped": float(drop.mean()),
                        "auroc_hr": auroc(y_hr, preds[m]), "spearman_hrd": spearman(y_hrd, hrd[m])})
        return out, preds

    rows = []
    G = Xtr.shape[1]
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        reps = 1 if frac == 0 else args.repeats
        for rep in range(reps):
            r, _ = score_all(lambda rng, f=frac: rng.random(G) < f, "random", frac, rep)
            rows += r
        log.info(f"  random dropout {frac:.0%} done")

    # real missing-gene pattern of each external cohort
    for name in [n for n in available_inputs(P) if n.startswith("ext_")]:
        Xe, _ = load_inputs(P, name)
        miss = np.all(Xe == 0, axis=0)
        if miss.mean() < 0.01:
            continue
        r, _ = score_all(lambda rng, mm=miss: mm, "platform_pattern", name[4:], 0)
        rows += r
        log.info(f"  platform pattern {name[4:]}: {miss.mean():.1%} genes missing")

    df = pd.DataFrame(rows)
    df.to_csv(R / "gene_dropout_raw.csv", index=False)
    summ = (df.groupby(["scenario", "level", "method"])
              .agg(frac_dropped=("frac_genes_dropped", "mean"), auroc_hr=("auroc_hr", "mean"),
                   auroc_sd=("auroc_hr", "std"), spearman_hrd=("spearman_hrd", "mean"),
                   spearman_sd=("spearman_hrd", "std")).reset_index())
    base = summ[(summ.scenario == "random") & (summ.level == 0.0)].set_index("method")
    summ["auroc_loss_vs_full_genes"] = summ["auroc_hr"] - summ["method"].map(base["auroc_hr"])
    summ["spearman_loss_vs_full_genes"] = summ["spearman_hrd"] - summ["method"].map(base["spearman_hrd"])
    summ.to_csv(R / "gene_dropout_summary.csv", index=False)

    # paired test at 30 % and 50 % random loss: does HRD-XRes lose less than the elastic net?
    tests = []
    for lvl in (0.3, 0.5):
        d = df[(df.scenario == "random") & (df.level == lvl)]
        for metric in ("auroc_hr", "spearman_hrd"):
            loss = {m: d[d.method == m][metric].values - base.loc[m, metric]
                    for m in d.method.unique()}
            for v in variants:
                diff = loss[v] - loss["enet"]
                tests.append({"level": lvl, "metric": metric, "comparison": f"{v} vs enet (loss difference)",
                              "mean_diff": diff.mean(), "n_repeats": len(diff),
                              "repeats_favouring_hrdxres": int((diff > 0).sum())})
    pd.DataFrame(tests).to_csv(R / "gene_dropout_tests.csv", index=False)
    log.info("Gene dropout summary (random):\n" +
             summ[summ.scenario == "random"].pivot(index="level", columns="method", values="auroc_hr").round(3).to_string())
    log.info("Gene dropout summary (platform patterns):\n" +
             summ[summ.scenario == "platform_pattern"].pivot(index="level", columns="method", values="auroc_hr").round(3).to_string())

# ====================================================================== B. elastic net on spatial data
if "B" in parts:
    log.info("Part B: elastic net on the Visium cohorts")
    enet_all = fit_enet(np.ones(len(labtr), bool))
    spatial_rows = []
    for name, cohort in (("denisenko_spots", "Denisenko (post-NACT)"), ("stur_spots", "Stur (pre-treatment)")):
        X, obs = load_inputs(P, name)
        spots = pd.read_csv(RES / "spatial" / f"{name}_spot_predictions.csv", index_col=0)
        if len(spots) != len(obs):
            raise ValueError(f"{name}: spot table ({len(spots)}) and model input ({len(obs)}) differ; rerun 11 first")
        pe = enet_predict(enet_all, X)
        spots["p_hr__enet"], spots["pred_hrd__enet"] = pe["p_hr"].values, pe["pred_hrd"].values
        spots.to_csv(R / f"{name}_spot_predictions_with_enet.csv")
        methods = {"HRD-XRes (full)": "p_hr", "HRD-XRes (no_da)": "p_hr__no_da", "Elastic net (TCGA)": "p_hr__enet"}
        methods = {k: c for k, c in methods.items() if c in spots}
        sec_rows = []
        for sec, s in spots.groupby("section"):
            t = s[s["tumor_spot"].astype(bool)]
            if len(t) < 10:
                continue
            rec = {"section": sec, "response": s["response"].iloc[0], "n_tumor_spots": len(t),
                   "mean_malignant_fraction": s["malignant_fraction"].mean()}
            for mname, c in methods.items():
                rec[f"{mname}|mean_p_hr"] = t[c].mean()
                rec[f"{mname}|frac_hrd_high"] = float((t[c] >= 0.5).mean())
            sec_rows.append(rec)
        sec = pd.DataFrame(sec_rows)
        sec.to_csv(R / f"{name}_section_metrics_with_enet.csv", index=False)

        for mname in methods:
            for metric in ("mean_p_hr", "frac_hrd_high"):
                col = f"{mname}|{metric}"
                if cohort.startswith("Denisenko"):
                    d = sec.assign(ordinal=sec["response"].map({"good": 0, "partial": 1, "poor": 2}))[[col, "ordinal", "mean_malignant_fraction"]].dropna()
                    if len(d) < 5:
                        continue
                    r = spearmanr(d[col], d["ordinal"])
                    pr = partial_spearman(d[col].values, d["ordinal"].values, d["mean_malignant_fraction"].values)
                    spatial_rows.append({"cohort": cohort, "method": mname, "metric": metric,
                                         "test": "Spearman vs CRS (good<partial<poor); expected negative",
                                         "estimate": r.correlation, "p_value": r.pvalue,
                                         "partial_rho_adj_malignant_fraction": pr.correlation,
                                         "partial_p": pr.pvalue, "n_sections": len(d)})
                else:
                    d = sec.assign(poor=(sec["response"] == "poor").astype(float))[[col, "poor"]].dropna()
                    if len(d) < 6 or d["poor"].nunique() < 2:
                        continue
                    est, lo, hi = bootstrap_ci(auroc, d["poor"].values, -d[col].values, n_boot=nb)
                    spatial_rows.append({"cohort": cohort, "method": mname, "metric": metric,
                                         "test": "AUROC poor vs excellent (pre-specified: lower HRD in poor)",
                                         "estimate": est, "ci_low": lo, "ci_high": hi,
                                         "p_value": mannwhitneyu(d.loc[d.poor == 1, col], d.loc[d.poor == 0, col]).pvalue,
                                         "n_sections": len(d)})
    sp_df = pd.DataFrame(spatial_rows)
    sp_df.to_csv(R / "spatial_enet_vs_hrdxres.csv", index=False)
    log.info("Spatial, elastic net vs HRD-XRes:\n" + sp_df.round(3).to_string(index=False))

# ====================================================================== C. calibration
if "C" in parts:
    log.info("Part C: calibration of P(HRD)")

    def calib(y, p, bins=5):
        m = np.isfinite(y) & np.isfinite(p)
        y, p = y[m].astype(int), np.clip(p[m], 1e-4, 1 - 1e-4)
        lg = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg, y)
        q = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
        ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
        return {"n": len(y), "brier": float(np.mean((p - y) ** 2)), "calibration_in_the_large": float(y.mean() - p.mean()),
                "calibration_slope": float(lr.coef_[0, 0]), "calibration_intercept": float(lr.intercept_[0]),
                "ECE": float(ece), "auroc": auroc(y, p)}

    rows = []
    yser = pd.Series(y_hr, index=labtr.index)
    for name in ("full", "no_da", "no_pathway", "mlp", "enet", "rf", "gbm"):
        f = RES / "cv" / f"predictions_{name}.csv"
        if f.exists():
            p = pd.read_csv(f, index_col=0)
            if "p_hr" in p:
                rows.append({"dataset": "TCGA (outer CV)", "method": name,
                             **calib(yser.reindex(p.index).values, p["p_hr"].values)})
    pp = RES / "singlecell" / "patient_predictions.csv"
    if pp.exists():
        d = pd.read_csv(pp, index_col=0)
        for (level, mname), g in d.groupby(["level", "method"]):
            if "p_hr" in g and g["p_hr"].between(0, 1).all():
                rows.append({"dataset": f"SPECTRUM {level.replace('spectrum_', '')}", "method": mname,
                             **calib(g["hr_status_wgs"].values.astype(float), g["p_hr"].values, bins=4)})
    cal = pd.DataFrame(rows)
    cal.to_csv(R / "calibration.csv", index=False)
    log.info("Calibration (slope 1 and intercept 0 = perfect):\n" + cal.round(3).to_string(index=False))

# ====================================================================== D. P(HRD) vs survival externally
if "D" in parts:
    log.info("Part D: predicted HRD probability vs external survival")
    ep_file = RES / "external_bulk" / "external_predictions.csv"
    d = pd.read_csv(ep_file, index_col=0, low_memory=False)
    cox_rows = []
    for (cohort, mname), g in d.groupby(["cohort", "method"]):
        if "pred__p_hr" not in g or g["pred__p_hr"].isna().all():
            continue
        s = g["pred__p_hr"].astype(float).values
        if s.min() >= 0 and s.max() <= 1:  # probability -> logit, so "per SD" is comparable across methods
            s = np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
        for ep in ("os", "pfs"):
            if ep not in g or f"{ep}_time" not in g:
                continue
            t, e = g[f"{ep}_time"].astype(float).values, g[ep].astype(float).values
            m = np.isfinite(t) & np.isfinite(e) & np.isfinite(s) & (t > 0)
            if m.sum() < 20 or e[m].sum() < 10 or np.std(s[m]) == 0:
                continue
            z = (s[m] - s[m].mean()) / s[m].std()
            df = pd.DataFrame({"T": t[m], "E": e[m], "phrd_per_sd": z})
            cph = CoxPHFitter().fit(df, "T", "E")
            r = cph.summary.loc["phrd_per_sd"]
            cox_rows.append({"cohort": cohort, "method": mname, "endpoint": ep, "n": int(m.sum()),
                             "events": int(e[m].sum()), "HR_per_SD_PHRD": float(np.exp(r["coef"])),
                             "log_hr": float(r["coef"]), "se": float(r["se(coef)"]), "p": float(r["p"])})
    cox = pd.DataFrame(cox_rows)
    cox.to_csv(R / "phrd_survival_per_cohort.csv", index=False)
    meta = []
    for (mname, ep), g in cox.groupby(["method", "endpoint"]):
        if len(g) >= 2:
            meta.append({"method": mname, "endpoint": ep, "k_cohorts": len(g), "n_total": int(g["n"].sum()),
                         **meta_dl(g["log_hr"].values, g["se"].values)})
    meta = pd.DataFrame(meta)
    meta.to_csv(R / "phrd_survival_meta.csv", index=False)
    log.info("P(HRD) per SD vs survival (HR < 1 = higher predicted HRD, longer survival):\n" +
             meta.round(4).to_string(index=False))

# ====================================================================== E. summary of step-10 subtype tests
if "E" in parts:
    f = RES / "singlecell" / "patient_level_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        keep = d["metric"].str.contains("Kruskal|Mann-Whitney|Myriad|AUROC", regex=True)
        out = d[keep].pivot_table(index=["level", "metric", "score"], columns="method", values="estimate", aggfunc="first")
        out.to_csv(R / "singlecell_methods_side_by_side.csv")
        log.info("SPECTRUM, all methods side by side (AUROC / Spearman = higher better; p values = lower better):\n" +
                 out.round(4).to_string())

log.info(f"Done. Results in {R}")
