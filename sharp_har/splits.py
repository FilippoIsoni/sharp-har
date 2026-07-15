"""Day 1 — building and freezing the P2-lab and P1-SHARP splits.
Ref. v5.1 pipeline §2.1-2.2.

All stochasticity uses seed 42 (§0.5). The splits written here are
frozen artifacts: once committed they are never touched again (§0.1).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from .inventory import C0_PAPER_CLASSES, trace_table
from .utils import get_logger, write_json
from .windowing import EVAL_STRIDE, TRAIN_STRIDE, accumulate_moments

logger = get_logger(__name__)

SPLIT_SEED = 42
VAL_FRACTION = 0.15
RARE_CELL_THRESHOLD = 4  # (ar_set, attivita) cells with < 4 traces: pin 1 into train

# P1-SHARP (v5.1, §2.1): SHARP reproduction, faithful to the TMC paper.
# Train = bedroom S1 (AR-1), ALL campaigns a/b/c (repo README trains on
# S1a+S1b+S1c). Test = S2 (other day), S3 (other person), S4-S5 (NLOS),
# S6 (living room), S7 (laboratory). Verify against the real contingency
# table before freezing (§2.3).
P1_TRAIN_AR_SET = "AR-1"
P1_TEST_SPECS: list[tuple[str, str | None]] = [
    ("AR-2", None), ("AR-3", None), ("AR-4", None),
    ("AR-5", None), ("AR-6", None), ("AR-7", None),
]

# P2 primary rotation (v5.1, §2.2): leave-S7-out — laboratory, monitor M4,
# person P3: the paper's hardest generalization target (environment, day
# and person never seen in training).
P2_PRIMARY_TEST_AR_SET = "AR-7"


def _assert_disjoint(train: list[str], val: list[str], test: list[str]) -> None:
    """No trace-id appears in more than one list (§6: disjointness assert)."""
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert not (s_train & s_val), f"train/val leakage: {sorted(s_train & s_val)[:5]}"
    assert not (s_train & s_test), f"train/test leakage: {sorted(s_train & s_test)[:5]}"
    assert not (s_val & s_test), f"val/test leakage: {sorted(s_val & s_test)[:5]}"


def _stratified_val_split(
    train_pool: pd.DataFrame, val_fraction: float, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Stratifies by (ar_set, attivita) (§2.2): cells with >=
    RARE_CELL_THRESHOLD traces split train/val within the cell; for each
    rare cell 1 trace is pinned into train and the remainder degrades to
    AR-set-level stratification. The degradation matters: in the v5.1
    dataset every cell has 1-3 traces (one trace per campaign, max 3
    campaigns per set), so without it val would be empty.
    Returns (train_ids, val_ids, pinned_ids)."""
    train_ids: list[str] = []
    val_ids: list[str] = []
    pinned_ids: list[str] = []
    leftover_by_set: dict[str, list[str]] = {}

    for (ar_set, _attivita), cell in train_pool.groupby(["ar_set", "attivita"]):
        ids = cell["trace_id"].tolist()
        rng.shuffle(ids)

        if len(ids) < RARE_CELL_THRESHOLD:
            # rare cell: pin 1 trace into train, the rest is stratified
            # at AR-set level below (§2.2)
            pinned_ids.append(ids[0])
            train_ids.append(ids[0])
            leftover_by_set.setdefault(ar_set, []).extend(ids[1:])
            continue

        n_val = round(len(ids) * val_fraction)
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    for ar_set in sorted(leftover_by_set):
        ids = leftover_by_set[ar_set]
        rng.shuffle(ids)
        n_val = round(len(ids) * val_fraction)
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return train_ids, val_ids, pinned_ids


def build_p2_rotation(
    inventory_df: pd.DataFrame,
    test_ar_set: str = P2_PRIMARY_TEST_AR_SET,
    protocol: str = "P2-lab",
    out_path: str | Path = "splits/p2_lab.json",
    seed: int = SPLIT_SEED,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Leave-one-set-out rotation (§2.2). Primary rotation (defaults):
    test = all AR-7 (laboratory, S7) traces, train = S1-S6. The E2
    extension reuses this with test_ar_set="AR-6" (living room),
    protocol="P2-living", its own out_path and its own mu/sigma. Val =
    15% of train, stratified by (ar_set, attivita), with rare cells
    pinned. Raises if the per-cell coverage assert fails, in which case
    the JSON is NOT written."""
    rng = random.Random(seed)
    traces = trace_table(inventory_df)

    test_traces = traces[traces["ar_set"] == test_ar_set]
    train_pool = traces[traces["ar_set"] != test_ar_set]
    assert not test_traces.empty, f"no traces found for test set {test_ar_set!r}"

    train_ids, val_ids, pinned_ids = _stratified_val_split(train_pool, VAL_FRACTION, rng)
    test_ids = test_traces["trace_id"].tolist()

    # blocking assert: an empty val makes early stopping / checkpoint
    # selection impossible (§2.2 val = 15% of train) — never freeze it
    assert val_ids, (
        "empty val set after stratification — check the rare-cell "
        "degradation; build_p2_rotation aborted, no JSON written."
    )

    # blocking assert: every (ar_set, attivita) cell in train has >= 1 trace in train
    train_cells = set(map(tuple, train_pool[["ar_set", "attivita"]].drop_duplicates().values))
    covered_cells = set(
        map(tuple, train_pool[train_pool["trace_id"].isin(train_ids)][["ar_set", "attivita"]].drop_duplicates().values)
    )
    missing_cells = train_cells - covered_cells
    assert not missing_cells, (
        f"(ar_set, attivita) cells with no trace in train: {sorted(missing_cells)} — "
        "build_p2_rotation aborted, no JSON written."
    )

    _assert_disjoint(train_ids, val_ids, test_ids)

    train_files = inventory_df.loc[inventory_df["trace_id"].isin(train_ids), "filepath"].tolist()
    mu, sigma = accumulate_moments(train_files, stride=TRAIN_STRIDE)

    if labels is None:
        labels = sorted(inventory_df["attivita"].unique())

    payload = {
        "protocol": protocol,
        "axes": {"time": 340, "velocity": 100, "layout": "time_x_velocity", "stft_hop_s": 0.006},
        "window": {"train_stride": TRAIN_STRIDE, "eval_stride": EVAL_STRIDE},
        "classes": {"n_att": len(labels), "labels": labels, "c0_paper_set": C0_PAPER_CLASSES},
        "split_seed": seed,
        "pinned_train_traces": sorted(pinned_ids),
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "norm": {"mu": mu, "sigma": sigma},
    }
    write_json(out_path, payload)
    logger.info(
        "%s written (%s): train=%d val=%d test=%d pinned=%d",
        out_path, protocol, len(train_ids), len(val_ids), len(test_ids), len(pinned_ids),
    )
    return payload


def build_p1_sharp(
    inventory_df: pd.DataFrame,
    out_path: str | Path = "splits/p1_sharp.json",
    seed: int = SPLIT_SEED,
    c0_paper_set: list[str] | None = None,
) -> dict[str, Any]:
    """SHARP reproduction for C0 (v5.1, §2.1). Train = bedroom S1 (AR-1),
    all campaigns a/b/c as in the paper/repo. Val = 20% of S1 by trace
    (declared deviation: the repo has no hold-out). Test = S2-S7."""
    if c0_paper_set is None:
        c0_paper_set = C0_PAPER_CLASSES
    rng = random.Random(seed)
    traces = trace_table(inventory_df)

    train_pool = traces[traces["ar_set"] == P1_TRAIN_AR_SET]
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
