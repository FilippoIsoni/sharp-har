"""[STUB] Doppler CSI dataset: full windowing, normalization, antennas.
Ref. v5 pipeline §1.2–1.4. Do not implement before the day-2 gate
(§10.1): the training architecture may change the input pipeline
(escalation §5.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class DopplerDataset(Dataset):
    """Dataset for a split rotation (train/val/test).

    Windowing: 340 (time) x 100 (velocity) window, train_stride 100,
    eval_stride 340, discards the final incomplete window. Each
    (window, antenna) is an independent sample in train. Normalization
    `(x - mu) / sigma` with mu/sigma read from the split file (never
    recomputed here). Labels (activity, subject, environment, ar_set,
    trace-id, antenna) inherited from the parent trace. Antenna fusion
    (softmax averaging) does NOT happen here: it is harness.py's
    responsibility at evaluation time. Ref. §1.2–1.4.
    """

    def __init__(self, split_file: str | Path, set_name: str, **kwargs: Any) -> None:
        super().__init__()
        raise NotImplementedError("day 2 — §1.2–1.4")

    def __len__(self) -> int:
        raise NotImplementedError("day 2 — §1.2–1.4")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("day 2 — §1.2–1.4")
