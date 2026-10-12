#!/usr/bin/env python
"""08 - Final models trained on all TCGA-OV samples.

Hyper-parameters: the configuration selected most often across outer folds in 07 (ties broken by the
inner-loop score). For each variant (full and no_da, the latter for the domain-adaptation ablation in
external validation) an ensemble of n_seeds_final networks is trained, each with its own stratified
early-stopping split.

Saved: Models/final/<variant>/seed<s>/{best.pt,last.pt,history.csv,meta.json}
       Models/final/<variant>/ensemble.json          (ensemble = the deployed "best model")
       Models/final/<variant>/best_single_model.pt   (single network with the best validation score)
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hrdx.data import load_features, load_inputs
from hrdx.trainer import get_device, labels_to_source, stratified_holdout, train_model
from hrdx.utils import ensure_dir, get_logger, load_config, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--variants", default="full,no_da")
args = ap.parse_args()
cfg = load_config(args.config)
log = get_logger("08_train_final", cfg)
P, T, MC = cfg["paths"]["processed"], cfg["training"], cfg["model"]
device = get_device(T["device"])

sel = pd.read_csv(cfg["paths"]["results"] / "cv" / "selected_hyperparameters.csv")
keys = list(T["grid"])
sel["key"] = sel[keys].astype(str).agg("|".join, axis=1)
counts = sel.groupby("key").agg(n=("fold", "size"), score=("inner_mean", "mean")).sort_values(["n", "score"], ascending=False)
best = sel[sel["key"] == counts.index[0]].iloc[0]
hp = {k: float(best[k]) for k in keys}
log.info(f"Final hyper-parameters: {hp} (selected in {int(counts.iloc[0]['n'])}/{len(sel)} folds)")

feat = load_features(P)
X, lab = load_inputs(P, "tcga")
src = labels_to_source(X, lab)
rng = np.random.default_rng(T["seed"])
targets = []
for d in feat["target_domains"]:
    Xt, _ = load_inputs(P, d)
    if Xt.shape[0] > 20000:
        Xt = Xt[rng.choice(Xt.shape[0], 20000, replace=False)]
    targets.append(Xt)

for variant in args.variants.split(","):
    root = ensure_dir(cfg["paths"]["models"] / "final" / variant)
    members, rows = [], []
    for s in range(T["n_seeds_final"]):
        seed = T["seed"] + 7919 * (s + 1)
        tr, va = stratified_holdout(src.y["hr"], T["early_stop_fraction"], seed)
        od = root / f"seed{s}"
        if (od / "best.pt").exists() and (od / "meta.json").exists():  # resume
            import json
            meta = json.load(open(od / "meta.json"))
            log.info(f"{variant} seed{s}: existing checkpoint reused")
        else:
            _, meta = train_model(src.subset(tr), src.subset(va), targets, feat["mask"], hp, MC, T, variant, seed, od, device, log)
        members.append(f"seed{s}/best.pt")
        rows.append({"seed_index": s, "seed": seed, **{k: meta[k] for k in ("best_val_composite", "best_step", "stopped_step")}})
    tab = pd.DataFrame(rows)
    tab.to_csv(root / "members.csv", index=False)
    top = tab.sort_values("best_val_composite", ascending=False).iloc[0]
    shutil.copy(root / f"seed{int(top['seed_index'])}" / "best.pt", root / "best_single_model.pt")
    save_json({"variant": variant, "hyperparameters": hp, "members": members,
               "best_single": f"seed{int(top['seed_index'])}/best.pt",
               "genes": len(feat["genes"]), "pathways": len(feat["pathways"])}, root / "ensemble.json")
    log.info(f"{variant}: ensemble of {len(members)} saved; best single = seed{int(top['seed_index'])} "
             f"(val composite {top['best_val_composite']:.4f})")
