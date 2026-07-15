"""[STUB] Rete di riproduzione C0, fedele al paper SHARP. Rif. pipeline
v5 §5.1. Non implementare prima del gate giorno 2 (§10.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SharpLike(nn.Module):
    """Riproduzione dell'architettura del paper SHARP, usata solo dalla
    run C0 (protocollo P1, config `configs/c0_sharp.yaml`). Rif. §5.1."""

    def __init__(self, d_enc: int = 256) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2 — §5.1 (fedele al paper SHARP)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("giorno 2 — §5.1 (fedele al paper SHARP)")
