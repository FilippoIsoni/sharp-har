"""[STUB] SupCon, gradient reversal (GRL), label-smoothed cross-entropy.
Ref. v5 pipeline §5.3, §8. Do not implement before the day-2 gate
(§10.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


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
    torch.nn.functional.cross_entropy. Ref. §5.3."""
    raise NotImplementedError("day 2 — §5.3")
