#!/usr/bin/env python
"""09 - External bulk validation (models frozen; no external label is ever used for training).

For every external cohort: platinum response, BRCA1 status and survival (OS / PFS) are evaluated for
HRD-XRes (full), its no-DA ablation, an elastic-net baseline trained on all TCGA and the HR pathway score.
Survival: Harrell's C, univariable and multivariable Cox (per SD of the model risk), KM median split with
log-rank test, and random-effects meta-analysis of hazard ratios across cohorts.

Outputs: Results/external_bulk/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import kruskal, norm

from hrdx.baselines import classical, oriented_hr_score
from hrdx.data import available_inputs, load_features, load_inputs, read_gmt
from hrdx.metrics import auroc, bootstrap_ci, cindex
from hrdx.trainer import ensemble_predict, get_device, labels_to_source, load_ensemble
from hrdx.utils import ensure_dir, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("09_validate_bulk", cfg)
P, R = cfg["paths"]["processed"], ensure_dir(cfg["paths"]["results"] / "external_bulk")
device = get_device(cfg["training"]["device"])
nb = cfg["evaluation"]["n_bootstrap"]
feat = load_features(P)
gmt = read_gmt(inp(cfg, cfg["files"]["reactome_gmt"]))

ens = {}
for v in ("full", "no_da"):
    try:
        ens[v] = load_ensemble(cfg["paths"]["models"], v, device)
    except FileNotFoundError:
        log.warning(f"final ensemble '{v}' not found")
Xtr, labtr = load_inputs(P, "tcga")
src = labels_to_source(Xtr, labtr)

cohorts = [n for n in available_inputs(P) if n.startswith("ext_")]
log.info(f"External cohorts: {cohorts}")
data = {n: load_inputs(P, n) for n in cohorts}
# elastic-net baseline: fitted once on all TCGA, applied to every cohort
stack = np.vstack([data[n][0] for n in cohorts])
enet_all = classical("enet", Xtr, src.y, stack, seed=0)
bounds = np.cumsum([0] + [data[n][0].shape[0] for n in cohorts])
enet = {n: enet_all.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True) for i, n in enumerate(cohorts)}

rows, km_rows, cox_rows, all_preds = [], [], [], []
for name in cohorts:
    X, obs = data[name]
    cname = name[4:]
    methods = {f"HRD-XRes ({v})": ensemble_predict(m, X, device) for v, m in ens.items()}
    methods["Elastic net (TCGA)"] = enet[name]
    methods["HR pathway score"] = oriented_hr_score(Xtr, src.y, X, feat["genes"], gmt)
    for mname, p in methods.items():
        p = p.reset_index(drop=True)
        p.index = obs.index
        out = p.add_prefix("pred__").join(obs)
        out.insert(0, "method", mname)
        out.insert(0, "cohort", cname)
        all_preds.append(out)
        if "platinum_resistant" in obs:
            y = obs["platinum_resistant"].values
            for score, desc in [("p_plat", "P(resistant)"), ("p_hr", "1 - P(HRD)")]:
                if score in p:
                    s = p[score].values if score == "p_plat" else -p[score].values
                    est, lo, hi = bootstrap_ci(auroc, y, s, n_boot=nb)
                    rows.append({"cohort": cname, "method": mname, "task": "platinum_resistant", "score": desc,
                                 "metric": "AUROC", "estimate": est, "ci_low": lo, "ci_high": hi,
                                 "n": int(np.isfinite(y).sum()), "n_pos": int(np.nansum(y))})
            if "platinum_class" in obs and "p_plat" in p:
                groups = [p.loc[obs["platinum_class"] == g, "p_plat"].values for g in obs["platinum_class"].dropna().unique()]
                groups = [g for g in groups if len(g) > 2]
                if len(groups) > 2:
                    rows.append({"cohort": cname, "method": mname, "task": "platinum_class (3 levels)",
                                 "score": "P(resistant)", "metric": "Kruskal p", "estimate": kruskal(*groups).pvalue})
        if "brca1_mut" in obs and "p_brca" in p and np.nansum(obs["brca1_mut"].values) >= 5:
            y = obs["brca1_mut"].values
            est, lo, hi = bootstrap_ci(auroc, y, p["p_brca"].values, n_boot=nb)
            rows.append({"cohort": cname, "method": mname, "task": "brca1_mut", "score": "P(BRCA)", "metric": "AUROC",
                         "estimate": est, "ci_low": lo, "ci_high": hi, "n": int(np.isfinite(y).sum()), "n_pos": int(np.nansum(y))})
        for ep in ("os", "pfs"):
            if ep not in obs or f"{ep}_time" not in obs or "risk" not in p:
                continue
            t, e, r = obs[f"{ep}_time"].values.astype(float), obs[ep].values.astype(float), p["risk"].values
            m = np.isfinite(t) & np.isfinite(e) & np.isfinite(r) & (t > 0)
            if m.sum() < 30 or e[m].sum() < 5:
                continue
            est, lo, hi = bootstrap_ci(cindex, t, e, r, n_boot=nb)
            rows.append({"cohort": cname, "method": mname, "task": ep, "score": "risk", "metric": "C-index",
                         "estimate": est, "ci_low": lo, "ci_high": hi, "n": int(m.sum()), "n_pos": int(e[m].sum())})
            z = (r[m] - r[m].mean()) / (r[m].std() + 1e-9)
            df = pd.DataFrame({"T": t[m], "E": e[m], "risk_per_sd": z})
            cph = CoxPHFitter().fit(df, "T", "E")
            s = cph.summary.loc["risk_per_sd"]
            cox_rows.append({"cohort": cname, "method": mname, "endpoint": ep, "model": "univariable",
                             "HR": s["exp(coef)"], "ci_low": s["exp(coef) lower 95%"], "ci_high": s["exp(coef) upper 95%"],
                             "p": s["p"], "log_hr": s["coef"], "se": s["se(coef)"], "n": int(m.sum())})
            covs = [c for c in ("debulking", "residual", "summarystage", "tumorstage", "stage", "treatment") if c in obs and obs[c].notna().sum() > 0.7 * m.sum()]
            if covs:
                cov = pd.get_dummies(obs.loc[m, covs].astype(str), drop_first=True, dtype=float)
                cov = cov.loc[:, cov.std() > 0]
                dfm = pd.concat([df.reset_index(drop=True), cov.reset_index(drop=True)], axis=1)
                try:
                    cphm = CoxPHFitter(penalizer=0.01).fit(dfm, "T", "E")
                    s = cphm.summary.loc["risk_per_sd"]
                    cox_rows.append({"cohort": cname, "method": mname, "endpoint": ep, "model": f"adjusted ({'+'.join(covs)})",
                                     "HR": s["exp(coef)"], "ci_low": s["exp(coef) lower 95%"], "ci_high": s["exp(coef) upper 95%"],
                                     "p": s["p"], "log_hr": s["coef"], "se": s["se(coef)"], "n": int(m.sum())})
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"{cname} {mname} {ep}: adjusted Cox failed ({exc})")
            hi_grp = r[m] >= np.median(r[m])
            lr = logrank_test(t[m][hi_grp], t[m][~hi_grp], e[m][hi_grp], e[m][~hi_grp])
            for grp, sel in (("high risk", hi_grp), ("low risk", ~hi_grp)):
                km = KaplanMeierFitter().fit(t[m][sel], e[m][sel])
                sf = km.survival_function_.reset_index()
                sf.columns = ["time", "survival"]
                km_rows.append(sf.assign(cohort=cname, method=mname, endpoint=ep, group=grp, logrank_p=lr.p_value))

res = pd.DataFrame(rows)
res.to_csv(R / "external_metrics.csv", index=False)
cox = pd.DataFrame(cox_rows)
cox.to_csv(R / "external_cox.csv", index=False)
if km_rows:
    pd.concat(km_rows).to_csv(R / "external_km_curves.csv", index=False)
pd.concat(all_preds).to_csv(R / "external_predictions.csv")

# random-effects meta-analysis (DerSimonian-Laird) of univariable log HR across cohorts
meta = []
for (mname, ep), g in cox[cox["model"] == "univariable"].groupby(["method", "endpoint"]):
    if len(g) < 2:
        continue
    y, v = g["log_hr"].values, g["se"].values ** 2
    w = 1 / v
    fixed = np.sum(w * y) / w.sum()
    Q = np.sum(w * (y - fixed) ** 2)
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    wr = 1 / (v + tau2)
    mu, se = np.sum(wr * y) / wr.sum(), np.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - (len(y) - 1)) / Q) if Q > 0 else 0.0
    meta.append({"method": mname, "endpoint": ep, "k_cohorts": len(y), "n_total": int(g["n"].sum()),
                 "pooled_HR": np.exp(mu), "ci_low": np.exp(mu - 1.96 * se), "ci_high": np.exp(mu + 1.96 * se),
                 "p": 2 * norm.sf(abs(mu / se)), "tau2": tau2, "I2": I2})
pd.DataFrame(meta).to_csv(R / "meta_analysis.csv", index=False)
log.info("External validation:\n" + res.to_string(index=False))
log.info("Meta-analysis:\n" + pd.DataFrame(meta).to_string(index=False))
