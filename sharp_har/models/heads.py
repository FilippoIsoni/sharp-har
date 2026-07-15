"""[STUB] Head di classificazione/proiezione/adversary. Rif. pipeline v5
§5.3. Tutte parametrizzate su d_enc, nessun numero cablato. Non
implementare prima del gate giorno 2 (§10.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivityHead(nn.Module):
    """Classificatore lineare d_enc -> n_att. Rif. §5.3."""

    def __init__(self, d_enc: int, n_att: int) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("giorno 2 — §5.3")


class ProjectionHead(nn.Module):
    """Testa di proiezione per SupCon: d_enc -> d_enc -> 128, ReLU
    intermedia, normalizzazione L2 in uscita. Rif. §5.3."""

    def __init__(self, d_enc: int, out_dim: int = 128) -> None:
        super().__init__()
        raise NotImplementedError("giorno 3 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("giorno 3 — §5.3")


class ARSetHead(nn.Module):
    """Adversary per l'AR-set: GRL -> d_enc -> d_enc/2 -> n_arset, ReLU,
    dropout 0.3. Usata dai config C2/C4 (adversary.type: grl). Rif. §5.3."""

    def __init__(self, d_enc: int, n_arset: int, lambda_: float = 1.0, dropout: float = 0.3) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2/4 — §5.3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("giorno 2/4 — §5.3")
