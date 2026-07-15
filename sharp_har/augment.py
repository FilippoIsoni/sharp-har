"""[STUB] Doppler CSI augmentation, fixed order. Ref. v5 pipeline §3.
Do not implement before the day-2 gate (§10.1).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Fixed order, applied after standardization (x - mu) / sigma. Masked
# regions are filled with 0. Forbidden: velocity flip, time flip.
AUGMENTATION_ORDER = (
    "time_shift",
    "time_masking",
    "velocity_masking",
    "amplitude_scaling",
    "gaussian_noise",
)


def apply(x: torch.Tensor, cfg: dict[str, Any], rng: np.random.Generator) -> torch.Tensor:
    """Applies the augmentation pipeline in AUGMENTATION_ORDER to a
    single already-standardized sample. Called twice (independent views)
    to generate SupCon's positive pairs. Ref. §3."""
    raise NotImplementedError("day 2 — §3")
