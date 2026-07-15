"""[STUB] Linear probe, single recipe for C1-lin/C2-lin/phase B
C3-C4/diagnostic probes. Ref. v5 pipeline §5.3, §7. Do not implement
before the day-2 gate (§10.1).
"""
from __future__ import annotations

import numpy as np


def linear_probe(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Frozen encoder, features cached to disk (no augmentation). Linear
    head d_enc -> n_classes, Adam lr 1e-3 wd 1e-4 batch 256 max 30
    epochs, early stopping on val macro-F1 patience 5. Single recipe for
    every linear-probe evaluation in the project (C1-lin, C2-lin,
    C3/C4 phase B, diagnostic probes). Ref. §5.3, §7.
    """
    raise NotImplementedError("day 3+ — §5.3, §7")
