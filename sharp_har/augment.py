"""[STUB] Augmentation Doppler CSI, ordine fisso. Rif. pipeline v5 §3.
Non implementare prima del gate giorno 2 (§10.1).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Ordine fisso, applicato dopo la standardizzazione (x - mu) / sigma.
# Riempimento delle maschere = 0. Vietati: flip velocità, flip temporale.
AUGMENTATION_ORDER = (
    "time_shift",
    "time_masking",
    "velocity_masking",
    "amplitude_scaling",
    "gaussian_noise",
)


def apply(x: torch.Tensor, cfg: dict[str, Any], rng: np.random.Generator) -> torch.Tensor:
    """Applica la pipeline di augmentation in AUGMENTATION_ORDER a un
    singolo sample già standardizzato. Chiamata due volte (viste
    indipendenti) per generare le coppie positive di SupCon. Rif. §3."""
    raise NotImplementedError("giorno 2 — §3")
