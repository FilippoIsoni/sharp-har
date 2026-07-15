"""[STUB] Classification/projection/adversary heads. Ref. v5 pipeline
§5.3. All parametrized on d_enc, no hardcoded numbers. Do not implement
before the day-2 gate (§10.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivityHead(nn.Module):
    """Linear classifier d_enc -> n_att. Ref. §5.3."""

    def __init__(self, d_enc: int, n_att: int) -> None:
        super().__init__()
        raise NotImplementedError("day 2 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 2 — §5.3")


class ProjectionHead(nn.Module):
    """Projection head for SupCon: d_enc -> d_enc -> 128, ReLU in
    between, L2 normalization on output. Ref. §5.3."""

    def __init__(self, d_enc: int, out_dim: int = 128) -> None:
        super().__init__()
        raise NotImplementedError("day 3 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 3 — §5.3")


class ARSetHead(nn.Module):
    """AR-set adversary: GRL -> d_enc -> d_enc/2 -> n_arset, ReLU,
    dropout 0.3. Used by configs C2/C4 (adversary.type: grl). Ref. §5.3."""

    def __init__(self, d_enc: int, n_arset: int, lambda_: float = 1.0, dropout: float = 0.3) -> None:
        super().__init__()
        raise NotImplementedError("day 2/4 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 2/4 — §5.3")
