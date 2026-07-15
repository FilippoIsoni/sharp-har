"""[STUB] Single checkpoint -> CSV interface for every evaluation
stream. Ref. v5 pipeline §0.7, §2.1, §9. Do not implement before the
day-2 gate (§10.1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal


def evaluate(
    checkpoint: str | Path, split_file: str | Path, set_name: Literal["val", "test"]
) -> Path:
    """Evaluates a checkpoint on a set (val/test) of a split file and
    writes a per-sample results CSV. Antenna fusion by softmax averaging
    -> argmax; macro-F1 computed per set only over the classes present
    in the ground truth. Every invocation with set_name="test" must be
    logged (§0.7): the test set doesn't get consumed by mistake during
    development. Includes the SHARP-repo-style evaluation wrapper for
    C0. Ref. §0.7, §2.1, §9.
    """
    raise NotImplementedError("day 2+ — §0.7, §2.1, §9")
