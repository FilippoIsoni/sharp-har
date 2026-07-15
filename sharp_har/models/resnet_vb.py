"""[STUB] V-B backbone (asymmetric ResNet over the time/velocity axes).
Ref. v5 pipeline §5.2. Do not implement before the day-2 gate (§10.1):
escalation (b) depends on the gate's outcome.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNetVB(nn.Module):
    """V-B backbone. Stem conv3x3 stride(2,1) + maxpool3x3 stride(2,1);
    layer2 stride(2,2); layer3-4 stride(2,1); channels 32/64/128/256; GAP
    -> 256-d feature. Expected final map ~11x50 (time x velocity).

    Escalation (b), conditioned on the day-2 gate's outcome: optional
    stride(2,2) at layer3 -> final map ~11x25. Ref. §5.2.
    """

    def __init__(self, d_enc: int = 256, escalation_b: bool = False) -> None:
        super().__init__()
        raise NotImplementedError("day 2 — §5.2 (escalation at the gate)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("day 2 — §5.2 (escalation at the gate)")
