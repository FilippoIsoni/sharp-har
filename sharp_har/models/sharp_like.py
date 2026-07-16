"""[STUB] C0 reproduction network, faithful to the SHARP paper. Ref. v5
pipeline §5.1. Do not implement before the day-2 gate (§10.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SharpLike(nn.Module):
    """Reproduction of the SHARP paper's architecture, used only by the
    C0 run (P1 protocol, config `configs/c0_sharp.yaml`). Ref. §5.1."""

    def __init__(self, d_enc: int = 256) -> None:
        super().__init__()
        raise NotImplementedError("day 4 — §5.1 (C0 time-box, faithful to the SHARP paper)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 4 — §5.1 (C0 time-box, faithful to the SHARP paper)")
