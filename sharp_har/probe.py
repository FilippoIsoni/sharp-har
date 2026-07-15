"""[STUB] Linear probe, ricetta unica per C1-lin/C2-lin/fase B C3-C4/probe
diagnostici. Rif. pipeline v5 §5.3, §7. Non implementare prima del gate
giorno 2 (§10.1).
"""
from __future__ import annotations

import numpy as np


def linear_probe(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Encoder congelato, feature cachate su disco (senza augmentation).
    Testa lineare d_enc -> n_classi, Adam lr 1e-3 wd 1e-4 batch 256 max
    30 epoche, early stopping su val macro-F1 patience 5. Ricetta unica
    per tutte le valutazioni a probe lineare del progetto (C1-lin,
    C2-lin, fase B di C3/C4, probe diagnostici). Rif. §5.3, §7.
    """
    raise NotImplementedError("giorno 3+ — §5.3, §7")
