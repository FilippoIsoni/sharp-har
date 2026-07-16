"""SupCon, gradient reversal (GRL), label-smoothed cross-entropy.
Ref. §5.3, §8. CE (day 2) and supcon_loss (day 3) are implemented; GRL
stays stubbed until day 4 per the gated-implementation rule.
"""
from __future__ import annotations

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


class GRL(nn.Module):
    """Gradient Reversal Layer: identity in forward, reverses and scales
    the gradient by `lambda_` in backward. Used by the AR-set adversary
    in configs C2/C4. Ref. §5.3, §8."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        raise NotImplementedError("day 2/4 — §5.3, §8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 2/4 — §5.3, §8")


def ce_with_label_smoothing(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.1
) -> torch.Tensor:
    """Cross-entropy with label smoothing, a thin wrapper around
    torch.nn.functional.cross_entropy so the smoothing value always
    comes from the run config (§6-C1: 0.1 for C1/C2; C0 uses 0.0).
    Unweighted by design — declared choice, §4.1. Ref. §5.3."""
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
