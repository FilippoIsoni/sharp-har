"""Classification/projection/adversary heads. Ref. §5.3. All
parametrized on d_enc, no hardcoded numbers. ActivityHead is
implemented (day 2, needed by the smoke gate); the projection and
adversary heads stay stubbed until their day (3/4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivityHead(nn.Module):
    """Linear classifier d_enc -> n_att (C1/C2 end-to-end). Ref. §5.3."""

    def __init__(self, d_enc: int, n_att: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_enc, n_att)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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
