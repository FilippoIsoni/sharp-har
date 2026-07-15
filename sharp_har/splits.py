"""Day 1 — building and freezing the P2-office and P1-SHARP splits.
Ref. giorno1_inventory_splits_SPEC.md §4.

All stochasticity uses seed 42 (§0.5). The splits written here are
frozen artifacts: once committed they are never touched again (§0.1).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from .inventory import trace_table
from .utils import get_logger, write_json
from .windowing import EVAL_STRIDE, TRAIN_STRIDE, accumulate_moments

logger = get_logger(__name__)

SPLIT_SEED = 42
VAL_FRACTION = 0.15
RARE_CELL_THRESHOLD = 4  # (ar_set, attivita) cells with < 4 traces: pin 1 into train

# P1-SHARP (§4.2): SHARP reproduction. Train = bedroom AR-1a. Test = the
# generalization set. AR-3 and AR-4 are the dataset's NLOS environments
# (no extra campaign filter needed). Verify against the real contingency
# table before freezing (§2.3).
P1_TRAIN_AR_SET = "AR-1"
P1_TRAIN_CAMPAGNA = "a"
P1_TEST_SPECS: list[tuple[str, str | None]] = [
    ("AR-1", "b"), ("AR-1", "c"), ("AR-1", "d"), ("AR-1", "e"),
    ("AR-2", "a"),
    ("AR-3", None), ("AR-4", None),
    ("AR-5", None), ("AR-6", None), ("AR-7", None), ("AR-8", None), ("AR-9", None),
]


def _assert_disjoint(train: list[str], val: list[str], test: list[str]) -> None:
    """No trace-id appears in more than one list (§6: disjointness assert)."""
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert not (s_train & s_val), f"train/val leakage: {sorted(s_train & s_val)[:5]}"
    assert not (s_train & s_test), f"train/test leakage: {sorted(s_train & s_test)[:5]}"
    assert not (s_val & s_test), f"val/test leakage: {sorted(s_val & s_test)[:5]}"


def _stratified_val_split(
    train_pool: pd.DataFrame, val_fraction: float, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Stratifies by (ar_set, attivita): for each cell with >= RARE_CELL_THRESHOLD
    traces, pins 1 trace into train, then splits the rest train/val
    according to val_fraction. Returns (train_ids, val_ids, pinned_ids)."""
    train_ids: list[str] = []
    val_ids: list[str] = []
    pinned_ids: list[str] = []

    for (_ar_set, _attivita), cell in train_pool.groupby(["ar_set", "attivita"]):
        ids = cell["trace_id"].tolist()
        rng.shuffle(ids)

        if len(ids) < RARE_CELL_THRESHOLD:
            # rare cell: pin 1 trace into train, the rest also falls back
            # to train (no val from a cell that's too small, §2.2/§9)
            pinned_ids.append(ids[0])
            train_ids.extend(ids)
            continue

        n_val = round(len(ids) * val_fraction)
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return train_ids, val_ids, pinned_ids


def build_p2_office(
    inventory_df: pd.DataFrame,
    out_path: str | Path = "splits/p2_office.json",
    seed: int = SPLIT_SEED,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Primary rotation (§4.1). Test = all AR-8 (office) traces. Train =
    the rest. Val = 15% of train, stratified by (ar_set, attivita), with
    rare cells pinned. Raises if the per-cell coverage assert fails, in
    which case the JSON is NOT written (§4.1)."""
    rng = random.Random(seed)
    traces = trace_table(inventory_df)

    test_traces = traces[traces["ar_set"] == "AR-8"]
    train_pool = traces[traces["ar_set"] != "AR-8"]

    train_ids, val_ids, pinned_ids = _stratified_val_split(train_pool, VAL_FRACTION, rng)
    test_ids = test_traces["trace_id"].tolist()

    # blocking assert: every (ar_set, attivita) cell in train has >= 1 trace in train
    train_cells = set(map(tuple, train_pool[["ar_set", "attivita"]].drop_duplicates().values))
    covered_cells = set(
        map(tuple, train_pool[train_pool["trace_id"].isin(train_ids)][["ar_set", "attivita"]].drop_duplicates().values)
    )
    missing_cells = train_cells - covered_cells
    assert not missing_cells, (
        f"(ar_set, attivita) cells with no trace in train: {sorted(missing_cells)} — "
        "build_p2_office aborted, no JSON written."
    )

    _assert_disjoint(train_ids, val_ids, test_ids)

    train_files = inventory_df.loc[inventory_df["trace_id"].isin(train_ids), "filepath"].tolist()
    mu, sigma = accumulate_moments(train_files, stride=TRAIN_STRIDE)

    if labels is None:
        labels = sorted(inventory_df["attivita"].unique())

    payload = {
        "protocol": "P2-office",
        "axes": {"time": 340, "velocity": 100, "layout": "time_x_velocity", "stft_hop_s": 0.006},
        "window": {"train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE},
        "classes": {"n_att": len(labels), "labels": labels, "c0_paper_set": None},
        "split_seed": seed,
        "pinned_train_traces": sorted(pinned_ids),
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "norm": {"mu": mu, "sigma": sigma},
    }
    write_json(out_path, payload)
    logger.info(
        "p2_office.json written: train=%d val=%d test=%d pinned=%d",
        len(train_ids), len(val_ids), len(test_ids), len(pinned_ids),
    )
    return payload


def build_p1_sharp(
    inventory_df: pd.DataFrame,
    out_path: str | Path = "splits/p1_sharp.json",
    seed: int = SPLIT_SEED,
    c0_paper_set: str = "TODO: verify arXiv (5 classes) vs extended TMC (8 classes)",
) -> dict[str, Any]:
    """SHARP reproduction for C0 (§4.2). Train = bedroom AR-1a. Val = 20%
    of AR-1a by trace. Test = the SHARP generalization set."""
    rng = random.Random(seed)
    traces = trace_table(inventory_df)

    train_pool = traces[
        (traces["ar_set"] == P1_TRAIN_AR_SET) & (traces["campagna"] == P1_TRAIN_CAMPAGNA)
    ]
    train_ids = train_pool["trace_id"].tolist()
    rng.shuffle(train_ids)
    n_val = round(len(train_ids) * 0.20)
    val_ids = train_ids[:n_val]
    train_ids = train_ids[n_val:]

    test_mask = pd.Series(False, index=traces.index)
    for ar_set, campagna in P1_TEST_SPECS:
        if campagna is None:
            test_mask |= traces["ar_set"] == ar_set
        else:
            test_mask |= (traces["ar_set"] == ar_set) & (traces["campagna"] == campagna)
    test_ids = traces.loc[test_mask, "trace_id"].tolist()

    _assert_disjoint(train_ids, val_ids, test_ids)

    train_files = inventory_df.loc[inventory_df["trace_id"].isin(train_ids), "filepath"].tolist()
    mu, sigma = accumulate_moments(train_files, stride=TRAIN_STRIDE)

    payload = {
        "protocol": "P1-sharp",
        "axes": {"time": 340, "velocity": 100, "layout": "time_x_velocity", "stft_hop_s": 0.006},
        "window": {"train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE},
        "classes": {"n_att": None, "labels": None, "c0_paper_set": c0_paper_set},
        "split_seed": seed,
        "pinned_train_traces": [],
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "norm": {"mu": mu, "sigma": sigma},
    }
    write_json(out_path, payload)
    logger.info(
        "p1_sharp.json written: train=%d val=%d test=%d", len(train_ids), len(val_ids), len(test_ids)
    )
    return payload
