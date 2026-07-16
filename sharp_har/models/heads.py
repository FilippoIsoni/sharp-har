"""Classification/projection/adversary heads. Ref. §5.3. All
parametrized on d_enc, no hardcoded numbers. ActivityHead (day 2) and
ProjectionHead (day 3, SupCon phase A) are implemented; the adversary
head stays stubbed until day 4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivityHead(nn.Module):
    """Linear classifier d_enc -> n_att (C1/C2 end-to-end). Ref. §5.3."""

    def __init__(self, d_enc: int, n_att: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_enc, n_att)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ProjectionHead(nn.Module):
    """Projection head for SupCon: MLP d_enc -> d_enc -> 128, ReLU in
    between, L2-normalized output — the input losses.supcon_loss
    expects. Discarded after phase-A pretraining (§5.3)."""

    def __init__(self, d_enc: int, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_enc, d_enc),
            nn.ReLU(inplace=True),
            nn.Linear(d_enc, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class ARSetHead(nn.Module):
    """AR-set adversary: GRL -> d_enc -> d_enc/2 -> n_arset, ReLU,
    dropout 0.3. Used by configs C2/C4 (adversary.type: grl). Ref. §5.3."""

    def __init__(self, d_enc: int, n_arset: int, lambda_: float = 1.0, dropout: float = 0.3) -> None:
        super().__init__()
        raise NotImplementedError("day 2/4 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 2/4 — §5.3")
