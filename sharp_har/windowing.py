"""Giorno 1 — windowing minimale: enumerazione finestre + accumulo μ/σ.
Rif. giorno1_inventory_splits_SPEC.md §3.

Non è il dataset completo (quello è materia del giorno 2+, vedi
sharp_har/data.py). Qui serve solo per i conteggi attesi e per calcolare
μ/σ globali sul train di una rotazione, prima di qualsiasi augmentation.

Nota (§1.4): μ/σ su finestre sovrapposte pesa i frame centrali ~3.4x i
bordi rispetto ai bordi della trace; effetto trascurabile per due scalari
globali — si accetta il code path unico, nessuna correzione applicata.
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

# Volumi attesi a hop 6ms (sanity §1.2): se i conteggi reali divergono
# molto, l'hop assunto è sbagliato — rivedere prima di congelare gli split.
EXPECTED_WINDOWS_TRAIN_STRIDE = 197
EXPECTED_WINDOWS_EVAL_STRIDE = 58


def iter_windows(trace_array: np.ndarray, stride: int, win: int = WINDOW_TIME_STEPS) -> Iterator[np.ndarray]:
    """Yield finestre (win, n_velocity) da una trace (n_frame, n_velocity).
    Scarta la finestra incompleta finale."""
    n_frame = trace_array.shape[0]
    for start in range(0, n_frame - win + 1, stride):
        yield trace_array[start : start + win]


def count_windows(n_frame: int, win: int = WINDOW_TIME_STEPS, stride: int = TRAIN_STRIDE) -> int:
    """Numero di finestre complete estraibili da una trace di n_frame
    frame, dato win e stride. Usato per popolare i volumi attesi (§1.2)."""
    if n_frame < win:
        return 0
    return (n_frame - win) // stride + 1


def accumulate_moments(
    file_list: list[str | Path], stride: int = TRAIN_STRIDE, win: int = WINDOW_TIME_STEPS
) -> tuple[float, float]:
    """μ, σ come due scalari globali su tutte le finestre di train di
    tutti i file passati (tipicamente tutte le 4 antenne del train della
    rotazione corrente), calcolate dopo il windowing, prima di qualsiasi
    augmentation (§1.4).

    Accumulo running (somma, somma dei quadrati, conteggio) per non
    tenere tutte le finestre in RAM.
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
        raise ValueError("nessuna finestra accumulata: file_list vuota o trace troppo corte")

    mu = total_sum / total_count
    variance = total_sumsq / total_count - mu**2
    sigma = float(np.sqrt(max(variance, 0.0)))
    return mu, sigma
