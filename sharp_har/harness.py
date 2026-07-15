"""[STUB] Interfaccia unica checkpoint -> CSV per tutti gli stream di
valutazione. Rif. pipeline v5 §0.7, §2.1, §9. Non implementare prima del
gate giorno 2 (§10.1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal


def evaluate(
    checkpoint: str | Path, split_file: str | Path, set_name: Literal["val", "test"]
) -> Path:
    """Valuta un checkpoint su un set (val/test) di uno split file e
    scrive un CSV di risultati per-sample. Fusione antenne per media
    softmax -> argmax; macro-F1 calcolata per set solo sulle classi
    presenti nel ground truth. Ogni invocazione con set_name="test" va
    loggata (§0.7): il test set non si consuma per errore durante lo
    sviluppo. Include il wrapper per l'eval C0 in stile repo SHARP
    originale. Rif. §0.7, §2.1, §9.
    """
    raise NotImplementedError("giorno 2+ — §0.7, §2.1, §9")
