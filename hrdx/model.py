"""HRD-XRes network.

Architecture
------------
x (genes, standardised)
  -> MaskedLinear(genes -> pathways)          knowledge-guided layer (Reactome membership)
  -> LayerNorm + GELU + Dropout
  -> Linear(pathways -> hidden) + LN + GELU + Dropout
  -> Linear(hidden -> latent) + LN             shared latent z
  -> task heads: hrd (regression), hr / brca / plat (logits), cox (risk), purity (regression)
  -> GRL(z) -> domain discriminator            domain-adversarial alignment
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class MaskedLinear(nn.Module):
    """Linear layer whose connectivity is fixed by a binary (in x out) mask."""

    def __init__(self, mask: np.ndarray):
        super().__init__()
        n_in, n_out = mask.shape
        self.register_buffer("mask", torch.tensor(mask.T, dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))
        fan_in = self.mask.sum(dim=1).clamp(min=1.0)
        with torch.no_grad():
            std = (2.0 / fan_in).sqrt().unsqueeze(1)
            self.weight.copy_(torch.randn(n_out, n_in) * std)
            self.weight.mul_(self.mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


HEADS = ("hrd", "hr", "brca", "plat", "cox", "purity")


class HRDXRes(nn.Module):
    def __init__(self, n_genes: int, pathway_mask: np.ndarray | None, n_domains: int,
                 hidden_dim: int = 256, latent_dim: int = 64, domain_hidden: int = 64,
                 dropout: float = 0.3, heads=HEADS, n_pathway_units: int | None = None):
        super().__init__()
        if pathway_mask is not None:
            self.pathway = MaskedLinear(pathway_mask)
            width = pathway_mask.shape[1]
        else:  # ablation: dense first layer of matched width
            width = n_pathway_units or 512
            self.pathway = nn.Linear(n_genes, width)
        self.uses_pathways = pathway_mask is not None
        self.norm0 = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(width, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.heads = nn.ModuleDict({h: nn.Linear(latent_dim, 1) for h in heads})
        self.disc = nn.Sequential(nn.Linear(latent_dim, domain_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(domain_hidden, n_domains)) if n_domains > 1 else None
        # label scaling for the hrd regression head (set from training data)
        self.register_buffer("hrd_mu", torch.tensor(0.0))
        self.register_buffer("hrd_sd", torch.tensor(1.0))

    def pathway_activity(self, x):
        return F.gelu(self.norm0(self.pathway(x)))

    def encode_from_pathways(self, a):
        h = F.gelu(self.norm1(self.fc1(self.drop(a))))
        return self.norm2(self.fc2(self.drop(h)))

    def encode(self, x):
        return self.encode_from_pathways(self.pathway_activity(x))

    def forward(self, x, lambd: float = 0.0, return_domain: bool = False):
        z = self.encode(x)
        out = {h: head(z).squeeze(-1) for h, head in self.heads.items()}
        out["z"] = z
        if return_domain and self.disc is not None:
            out["domain"] = self.disc(grad_reverse(z, lambd))
        return out


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow), averaged over events."""
    if event.sum() < 1:
        return risk.sum() * 0.0
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, dim=0)
    return -((r - log_cum) * e).sum() / e.sum()


def masked_bce(logit, y, pos_weight):
    m = torch.isfinite(y)
    if m.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[m], y[m], pos_weight=pos_weight)


def masked_mse(pred, y):
    m = torch.isfinite(y)
    if m.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[m], y[m])


def grl_schedule(progress: float, gamma: float = 10.0) -> float:
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
