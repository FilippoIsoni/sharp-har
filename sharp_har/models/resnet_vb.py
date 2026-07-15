"""[STUB] Backbone V-B (ResNet asimmetrico su assi tempo/velocità). Rif.
pipeline v5 §5.2. Non implementare prima del gate giorno 2 (§10.1):
l'escalation (b) dipende dall'esito del gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNetVB(nn.Module):
    """Backbone V-B. Stem conv3x3 stride(2,1) + maxpool3x3 stride(2,1);
    layer2 stride(2,2); layer3-4 stride(2,1); canali 32/64/128/256; GAP
    -> feature 256-d. Mappa finale attesa ~11x50 (tempo x velocità).

    Escalation (b), condizionata all'esito del gate giorno 2: stride(2,2)
    opzionale a layer3 -> mappa finale ~11x25. Rif. §5.2.
    """

    def __init__(self, d_enc: int = 256, escalation_b: bool = False) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2 — §5.2 (escalation al gate)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("giorno 2 — §5.2 (escalation al gate)")
