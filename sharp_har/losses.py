"""SupCon, gradient reversal (GRL), label-smoothed cross-entropy.
Ref. §5.3, §8. The CE helper is implemented (day 2, needed by the smoke
gate); supcon_loss and GRL stay stubbed until day 3/4 per the
gated-implementation rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020). `feats`
    L2-normalized, shape (B, d); `labels` shape (B,). Ref. §5.3."""
    raise NotImplementedError("day 3 — §5.3 (Khosla et al.)")


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
