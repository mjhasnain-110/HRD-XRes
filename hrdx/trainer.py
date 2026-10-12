"""Training loop, checkpointing (best + last), prediction and ensembling for HRD-XRes."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import auroc, cindex, composite_score, spearman
from .model import HEADS, HRDXRes, cox_ph_loss, grl_schedule, masked_bce, masked_mse
from .utils import ensure_dir, save_json, set_seed

LABEL_KEYS = ("hrd", "hr", "brca", "plat", "time", "event", "purity")

VARIANTS = {
    "full":        {"use_pathway": True,  "use_da": True,  "heads": list(HEADS)},
    "no_da":       {"use_pathway": True,  "use_da": False, "heads": list(HEADS)},
    "no_pathway":  {"use_pathway": False, "use_da": True,  "heads": list(HEADS)},
    "mlp":         {"use_pathway": False, "use_da": False, "heads": list(HEADS)},
    "single_task": {"use_pathway": True,  "use_da": True,  "heads": ["hr", "hrd"]},
}


@dataclass
class SourceData:
    X: np.ndarray
    y: dict = field(default_factory=dict)
    ids: np.ndarray | None = None

    def subset(self, idx):
        return SourceData(self.X[idx], {k: v[idx] for k, v in self.y.items()},
                          None if self.ids is None else self.ids[idx])

    def __len__(self):
        return self.X.shape[0]


def labels_to_source(X: np.ndarray, lab: pd.DataFrame) -> SourceData:
    def col(c):
        return lab[c].values.astype(np.float32) if c in lab.columns else np.full(len(lab), np.nan, np.float32)
    y = {"hrd": col("hrd_score"), "hr": col("hr_status"), "brca": col("brca_alt"),
         "plat": col("platinum_resistant"), "time": col("os_time"), "event": col("os"),
         "purity": col("purity")}
    return SourceData(X.astype(np.float32), y, lab.index.values)


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_model(n_genes, mask, n_domains, mcfg, hp, variant):
    v = VARIANTS[variant]
    return HRDXRes(
        n_genes=n_genes,
        pathway_mask=mask if v["use_pathway"] else None,
        n_domains=n_domains if v["use_da"] else 1,
        hidden_dim=mcfg["hidden_dim"], latent_dim=mcfg["latent_dim"], domain_hidden=mcfg["domain_hidden"],
        dropout=hp["dropout"], heads=v["heads"], n_pathway_units=mask.shape[1],
    )


@torch.no_grad()
def predict(model: HRDXRes, X: np.ndarray, device=None, batch: int = 2048) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    model.eval()
    outs = {h: [] for h in model.heads.keys()}
    for a in range(0, X.shape[0], batch):
        xb = torch.as_tensor(X[a:a + batch], dtype=torch.float32, device=device)
        o = model(xb)
        for h in outs:
            outs[h].append(o[h].float().cpu().numpy())
    res = {}
    for h, parts in outs.items():
        v = np.concatenate(parts) if parts else np.zeros(0)
        if h == "hrd":
            res["pred_hrd"] = v * float(model.hrd_sd) + float(model.hrd_mu)
        elif h in ("hr", "brca", "plat"):
            res[f"p_{h}"] = 1.0 / (1.0 + np.exp(-v))
        elif h == "cox":
            res["risk"] = v
        elif h == "purity":
            res["pred_purity"] = np.clip(v, 0.0, 1.0)
    return pd.DataFrame(res)


def val_metrics(pred: pd.DataFrame, y: dict) -> dict:
    m = {}
    if "p_hr" in pred:
        m["auroc_hr"] = auroc(y["hr"], pred["p_hr"])
    if "pred_hrd" in pred:
        m["spearman_hrd"] = spearman(y["hrd"], pred["pred_hrd"])
    if "p_brca" in pred:
        m["auroc_brca"] = auroc(y["brca"], pred["p_brca"])
    if "p_plat" in pred:
        m["auroc_plat"] = auroc(y["plat"], pred["p_plat"])
    if "risk" in pred:
        m["cindex"] = cindex(y["time"], y["event"], pred["risk"])
    m["composite"] = composite_score(m)
    return m


def _pos_weight(y: np.ndarray, device) -> torch.Tensor:
    y = y[np.isfinite(y)]
    if len(y) == 0 or y.sum() == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(float(np.clip((len(y) - y.sum()) / y.sum(), 0.2, 5.0)), device=device)


def save_checkpoint(path: Path, model: HRDXRes, meta: dict) -> None:
    init = {"n_genes": model.pathway.weight.shape[1] if hasattr(model.pathway, "mask") else model.pathway.in_features,
            "mask_shape": list(model.pathway.mask.shape[::-1]) if hasattr(model.pathway, "mask") else None,
            "n_domains": model.disc[-1].out_features if model.disc is not None else 1,
            "hidden_dim": model.fc1.out_features, "latent_dim": model.fc2.out_features,
            "domain_hidden": model.disc[0].out_features if model.disc is not None else 64,
            "dropout": model.drop.p, "heads": list(model.heads.keys()),
            "n_pathway_units": model.norm0.normalized_shape[0]}
    torch.save({"init": init, "state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[HRDXRes, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    ini = ck["init"]
    mask = np.zeros(tuple(ini["mask_shape"]), dtype=np.float32) if ini["mask_shape"] else None
    model = HRDXRes(n_genes=ini["n_genes"], pathway_mask=mask, n_domains=ini["n_domains"],
                    hidden_dim=ini["hidden_dim"], latent_dim=ini["latent_dim"],
                    domain_hidden=ini["domain_hidden"], dropout=ini["dropout"], heads=ini["heads"],
                    n_pathway_units=ini["n_pathway_units"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck["meta"]


def train_model(train: SourceData, val: SourceData, targets: list[np.ndarray], mask: np.ndarray,
                hp: dict, mcfg: dict, tcfg: dict, variant: str, seed: int,
                out_dir: Path | None = None, device=None, logger=None) -> tuple[HRDXRes, dict]:
    """Train one model. Saves best.pt (best validation composite), last.pt and history.csv."""
    set_seed(seed)
    device = device or get_device(tcfg.get("device", "auto"))
    v = VARIANTS[variant]
    n_domains = 1 + len(targets)
    model = build_model(train.X.shape[1], mask, n_domains, mcfg, hp, variant).to(device)

    hrd = train.y["hrd"][np.isfinite(train.y["hrd"])]
    if len(hrd):
        model.hrd_mu.fill_(float(hrd.mean()))
        model.hrd_sd.fill_(float(max(hrd.std(), 1e-3)))

    pw = {k: _pos_weight(train.y[k], device) for k in ("hr", "brca", "plat")}
    w = mcfg["task_weights"]
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["max_steps"], eta_min=hp["lr"] * 0.05)

    Xs = torch.as_tensor(train.X, dtype=torch.float32, device=device)
    Ys = {k: torch.as_tensor(val_, dtype=torch.float32, device=device) for k, val_ in train.y.items()}
    Ys["hrd_z"] = (Ys["hrd"] - model.hrd_mu) / model.hrd_sd
    Xt = [torch.as_tensor(t, dtype=torch.float32, device=device) for t in targets] if v["use_da"] else []

    rng = np.random.default_rng(seed)
    bs = tcfg["batch_size"]
    n = len(train)
    best_score, best_state, bad, history = -np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, tcfg["max_steps"] + 1):
        model.train()
        idx = torch.as_tensor(rng.choice(n, size=min(bs, n), replace=False), device=device)
        lambd = hp["lambda_domain"] * grl_schedule(step / tcfg["max_steps"], tcfg["grl_gamma"])
        out = model(Xs[idx], lambd=lambd, return_domain=bool(Xt))
        loss = torch.zeros((), device=device)
        parts = {}
        if "hrd" in out:
            parts["hrd"] = masked_mse(out["hrd"], Ys["hrd_z"][idx])
        for k in ("hr", "brca", "plat"):
            if k in out:
                parts[k] = masked_bce(out[k], Ys[k][idx], pw[k])
        if "purity" in out:
            parts["purity"] = masked_mse(out["purity"], Ys["purity"][idx])
        if "cox" in out:
            t, e = Ys["time"][idx], Ys["event"][idx]
            m = torch.isfinite(t) & torch.isfinite(e) & (t > 0)
            parts["cox"] = cox_ph_loss(out["cox"][m], t[m], e[m]) if m.sum() > 2 else out["cox"].sum() * 0.0
        for k, val_ in parts.items():
            loss = loss + w.get(k, 1.0) * val_
        if Xt:
            per = max(2, bs // len(Xt))
            xt_parts, dom = [], []
            for d, T in enumerate(Xt, start=1):
                ti = torch.as_tensor(rng.choice(T.shape[0], size=min(per, T.shape[0]), replace=False), device=device)
                xt_parts.append(T[ti])
                dom.append(torch.full((len(ti),), d, dtype=torch.long, device=device))
            out_t = model(torch.cat(xt_parts), lambd=lambd, return_domain=True)
            logits = torch.cat([out["domain"], out_t["domain"]])
            labels = torch.cat([torch.zeros(len(idx), dtype=torch.long, device=device), torch.cat(dom)])
            # class-balanced domain loss
            counts = torch.bincount(labels, minlength=logits.shape[1]).float().clamp(min=1)
            parts["domain"] = F.cross_entropy(logits, labels, weight=(counts.sum() / counts) / logits.shape[1])
            loss = loss + parts["domain"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % tcfg["eval_every"] == 0 or step == tcfg["max_steps"]:
            pv = predict(model, val.X, device)
            vm = val_metrics(pv, val.y)
            rec = {"step": step, "loss": float(loss.item()), "lambda": lambd,
                   **{f"loss_{k}": float(x.item()) for k, x in parts.items()}, **vm}
            history.append(rec)
            if vm["composite"] > best_score + 1e-4:
                best_score, bad = vm["composite"], 0
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            else:
                bad += 1
            if bad >= tcfg["patience"]:
                break

    last_state = copy.deepcopy(model.state_dict())
    meta = {"variant": variant, "hp": hp, "seed": seed, "best_val_composite": best_score,
            "best_step": int(best_step) if best_state is not None else step,
            "stopped_step": step, "train_n": n, "val_n": len(val), "seconds": round(time.time() - t0, 1)}
    if out_dir is not None:
        out_dir = ensure_dir(out_dir)
        save_checkpoint(out_dir / "last.pt", model, {**meta, "checkpoint": "last"})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    if out_dir is not None:
        save_checkpoint(out_dir / "best.pt", model, {**meta, "checkpoint": "best"})
        save_json(meta, out_dir / "meta.json")
    model.eval()
    if logger:
        logger.info(f"[{variant}] seed={seed} hp={hp} best_val={best_score:.4f} at step {meta['best_step']} "
                    f"({meta['seconds']}s)")
    return model, meta


def ensemble_predict(models: list[HRDXRes], X: np.ndarray, device=None) -> pd.DataFrame:
    preds = [predict(m, X, device) for m in models]
    return sum(preds) / len(preds)


def stratified_holdout(y_strat: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_tr, idx_va = [], []
    strat = np.where(np.isfinite(y_strat), y_strat, -1)
    for g in np.unique(strat):
        ids = np.where(strat == g)[0]
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * frac)))
        idx_va.extend(ids[:k])
        idx_tr.extend(ids[k:])
    return np.sort(np.array(idx_tr)), np.sort(np.array(idx_va))


def load_ensemble(models_root: Path, variant: str = "full", device="cpu") -> list[HRDXRes]:
    """Load the final ensemble (all seeds' best checkpoints) for a variant."""
    import json
    manifest = Path(models_root) / "final" / variant / "ensemble.json"
    with open(manifest) as fh:
        paths = json.load(fh)["members"]
    return [load_checkpoint(Path(models_root) / "final" / variant / p, device)[0] for p in paths]
