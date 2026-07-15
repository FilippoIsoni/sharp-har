"""[STUB] Loop di training comune a C1–C4. Rif. pipeline v5 §8. Non
implementare prima del gate giorno 2 (§10.1).
"""
from __future__ import annotations

from typing import Any


def train_run(cfg: dict[str, Any]) -> None:
    """Esegue una run di training da una config (uno dei file
    `configs/c*.yaml` caricato e validato). AdamW, scheduler cosine con
    warmup 5 epoche, AMP, grad clip 1.0, epoca = 400 step.

    Checkpoint completo per resume ad ogni epoca: pesi + optimizer +
    scheduler + stato GradScaler + epoca + config + stati RNG
    (torch/cuda/numpy/python). Scrive `last.ckpt` ad ogni epoca e
    `best.ckpt` (più i checkpoint a 40/50/60 in fase A SupCon per
    C3/C4). Resume automatico se un `last.ckpt` è già presente. Rif. §8.
    """
    raise NotImplementedError("giorno 2+ — §8")
