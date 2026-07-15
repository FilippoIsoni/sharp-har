"""Day 1 — minimal windowing: window enumeration + μ/σ accumulation.
Ref. giorno1_inventory_splits_SPEC.md §3.

Not the full dataset (that's day 2+ material, see sharp_har/data.py).
Here it's only used for the expected counts and to compute global μ/σ
on the train set of a rotation, before any augmentation.

Note (§1.4): μ/σ over overlapping windows weighs the central frames
~3.4x more than the edges of the trace; the effect is negligible for
two global scalars — the single code path is accepted, no correction
applied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .inventory import load_trace
from .utils import get_logger

logger = get_logger(__name__)

WINDOW_TIME_STEPS = 340
TRAIN_STRIDE = 100
EVAL_STRIDE = 340

# Expected volumes at a 6ms hop (sanity check §1.2): if the real counts
# diverge a lot, the assumed hop is wrong — revisit before freezing.
EXPECTED_WINDOWS_TRAIN_STRIDE = 197
EXPECTED_WINDOWS_EVAL_STRIDE = 58


def iter_windows(trace_array: np.ndarray, stride: int, win: int = WINDOW_TIME_STEPS) -> Iterator[np.ndarray]:
    """Yields windows (win, n_velocity) from a trace (n_frame, n_velocity).
    Discards the final incomplete window."""
    n_frame = trace_array.shape[0]
    for start in range(0, n_frame - win + 1, stride):
        yield trace_array[start : start + win]


def count_windows(n_frame: int, win: int = WINDOW_TIME_STEPS, stride: int = TRAIN_STRIDE) -> int:
    """Number of complete windows extractable from a trace of n_frame
    frames, given win and stride. Used to populate the expected volumes
    (§1.2)."""
    if n_frame < win:
        return 0
    return (n_frame - win) // stride + 1


def accumulate_moments(
    file_list: list[str | Path], stride: int = TRAIN_STRIDE, win: int = WINDOW_TIME_STEPS
) -> tuple[float, float]:
    """μ, σ as two global scalars over all train windows of all the
    files passed in (typically all 4 antennas of the current rotation's
    train set), computed after windowing, before any augmentation (§1.4).

    Running accumulation (sum, sum of squares, count) so we don't have
    to keep every window in RAM.
    """
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0
    for fp in file_list:
        arr = load_trace(fp)
        for window in iter_windows(arr, stride=stride, win=win):
            total_sum += float(window.sum())
            total_sumsq += float(np.square(window, dtype=np.float64).sum())
            total_count += window.size

    if total_count == 0:
        raise ValueError("no windows accumulated: file_list is empty or traces are too short")

    mu = total_sum / total_count
    variance = total_sumsq / total_count - mu**2
    sigma = float(np.sqrt(max(variance, 0.0)))
    return mu, sigma
