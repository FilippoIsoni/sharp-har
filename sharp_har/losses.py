"""[STUB] SupCon, gradient reversal (GRL), cross-entropy con label
smoothing. Rif. pipeline v5 §5.3, §8. Non implementare prima del gate
giorno 2 (§10.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020). `feats`
    normalizzate L2, shape (B, d); `labels` shape (B,). Rif. §5.3."""
    raise NotImplementedError("giorno 3 — §5.3 (Khosla et al.)")


class GRL(nn.Module):
    """Gradient Reversal Layer: identità in forward, inverte e scala il
    gradiente per `lambda_` in backward. Usata dall'adversary AR-set nei
    config C2/C4. Rif. §5.3, §8."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2/4 — §5.3, §8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("giorno 2/4 — §5.3, §8")


def ce_with_label_smoothing(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.1
) -> torch.Tensor:
    """Cross-entropy con label smoothing, wrapper sottile su
    torch.nn.functional.cross_entropy. Rif. §5.3."""
    raise NotImplementedError("giorno 2 — §5.3")
