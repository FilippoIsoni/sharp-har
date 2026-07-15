"""[STUB] Dataset Doppler CSI: windowing completo, normalizzazione, antenne.
Rif. pipeline v5 §1.2–1.4. Non implementare prima del gate giorno 2 (§10.1):
l'architettura di training può cambiare l'input pipeline (escalation §5.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class DopplerDataset(Dataset):
    """Dataset per una rotazione di split (train/val/test).

    Windowing: finestra 340 (tempo) x 100 (velocità), train_stride 100,
    eval_stride 340, scarta l'ultima finestra incompleta. Ogni (finestra,
    antenna) è un sample indipendente in train. Normalizzazione
    `(x - mu) / sigma` con mu/sigma letti dal file di split (mai
    ricalcolati qui). Label (attività, persona, ambiente, ar_set,
    trace-id, antenna) ereditate dalla trace di appartenenza. La fusione
    antenne (media softmax) NON avviene qui: è responsabilità di
    harness.py in fase di valutazione. Rif. §1.2–1.4.
    """

    def __init__(self, split_file: str | Path, set_name: str, **kwargs: Any) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2 — §1.2–1.4")

    def __len__(self) -> int:
        raise NotImplementedError("giorno 2 — §1.2–1.4")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("giorno 2 — §1.2–1.4")
