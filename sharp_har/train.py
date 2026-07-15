"""[STUB] Training loop shared by C1–C4. Ref. v5 pipeline §8. Do not
implement before the day-2 gate (§10.1).
"""
from __future__ import annotations

from typing import Any


def train_run(cfg: dict[str, Any]) -> None:
    """Runs a training run from a config (one of the loaded and
    validated `configs/c*.yaml` files). AdamW, cosine scheduler with a
    5-epoch warmup, AMP, grad clip 1.0, epoch = 400 steps.

    Full checkpoint for resuming, saved every epoch: weights + optimizer
    + scheduler + GradScaler state + epoch + config + RNG states
    (torch/cuda/numpy/python). Writes `last.ckpt` every epoch and
    `best.ckpt` (plus the checkpoints at 40/50/60 in SupCon phase A for
    C3/C4). Automatic resume if a `last.ckpt` already exists. Ref. §8.
    """
    raise NotImplementedError("day 2+ — §8")
