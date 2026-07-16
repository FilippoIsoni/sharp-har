"""SupCon, gradient reversal (GRL) with its §6-C2 ramp, label-smoothed
cross-entropy. Ref. §5.3, §6-C2, §8.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss, L_out formulation of Khosla et al.
    2020 (eq. 2): mean over anchors with >= 1 positive of
    -1/|P(i)| * sum_{p in P(i)} log(exp(z_i.z_p/tau) / sum_{a != i}
    exp(z_i.z_a/tau)). Ref. §5.3, §6-C3 (tau = 0.1 from the config).

    `feats`: (B, d), ALREADY L2-normalized (ProjectionHead's contract);
    `labels`: (B,) activity labels — with 2 views per window (§3) the
    caller passes the view-expanded batch, so B = 512 for the standard
    phase-A batch of 256 windows (§4.2). Computed in float32 even under
    autocast: exp/log on fp16 similarity logits is not stable. Anchors
    without positives are excluded from the mean (never happens with the
    P×K sampler + 2 views: every anchor's twin view is a positive).
    """
    assert feats.ndim == 2 and labels.shape == feats.shape[:1], (
        f"expected feats (B, d) and labels (B,), got {tuple(feats.shape)} / {tuple(labels.shape)}"
    )
    feats = feats.float()
    sim = feats @ feats.T / tau
    # Stability: subtract the row max before exponentiating.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(len(feats), dtype=torch.bool, device=feats.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

    n_pos = pos_mask.sum(dim=1)
    has_pos = n_pos > 0
    if not bool(has_pos.any()):
        return feats.sum() * 0.0  # degenerate batch: keep the graph alive, contribute nothing
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1)[has_pos] / n_pos[has_pos]
    return -mean_log_prob_pos.mean()


class _ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad, None


class GRL(nn.Module):
    """Gradient Reversal Layer: identity in forward, reverses and scales
    the gradient by `lambda_` in backward. Used by the AR-set adversary
    in configs C2/C4. The RAMP weighting sits on the loss term
    (L = L_att + λ(p)·L_env, taken literally from §6-C2), so lambda_
    stays 1.0 in this project. Ref. §5.3, §8."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReverseGrad.apply(x, self.lambda_)


def grl_lambda(epoch: int, lambda_max: float = 1.0, ramp_epochs: int = 20) -> float:
    """The fixed §6-C2 ramp: p = min(epoch / ramp_epochs, 1),
    λ(p) = λ_max · (2 / (1 + exp(−10 p)) − 1) — λ ≈ λ_max from
    `ramp_epochs` on, INDEPENDENT of early stopping (otherwise C2
    collapses into C1). λ_max is the single tuning knob (→ 0.5 if the
    activity accuracy collapses); C4's contingency (b) delays the ramp
    by raising ramp_epochs to 30. Ref. §6-C2, §6-C4."""
    p = min(epoch / ramp_epochs, 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def ce_with_label_smoothing(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.1
) -> torch.Tensor:
    """Cross-entropy with label smoothing, a thin wrapper around
    torch.nn.functional.cross_entropy so the smoothing value always
    comes from the run config (§6-C1: 0.1 for C1/C2; C0 uses 0.0).
    Unweighted by design — declared choice, §4.1. Ref. §5.3."""
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
