"""Day 2 — Doppler CSI dataset: full windowing, normalization, antennas.
Ref. §1.2–1.4 (windowing, antennas, normalization), §8.5 (lazy per-trace
loading from local staging).

The frozen split JSON is the contract: window geometry, strides, mu/sigma
and the class list are read from it and never recomputed here (§0.3).
Trace lengths come from the frozen day-1 inventory
(reports/inventory.csv), so building the window index requires no data
reads; a blocking assert at load time catches any drift between the
staged files and what day 1 inventoried.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .inventory import (
    DUPLICATE_SOURCE_MARKER,
    DUPLICATE_TRACE_SUFFIX,
    list_files,
    load_trace,
    parse_filename,
)
from .utils import get_logger, read_json

logger = get_logger(__name__)

N_ANTENNAS = 4
SET_NAMES = ("train", "val", "test")


def build_file_index(stage_dir: str | Path) -> tuple[dict[str, dict[int, Path]], dict[str, dict[str, str]]]:
    """Scans the staged files and rebuilds the same trace-id space as the
    day-1 inventory: repetition numbers folded into trace_id, and the
    known dual-archive duplicates (doppler_traces vs doppler_traces_S4_S5)
    kept as distinct traces via DUPLICATE_TRACE_SUFFIX. Ref. day-1 freeze,
    inventory.resolve_duplicate_streams.

    Returns (files, meta): files maps trace_id -> {antenna: filepath},
    meta maps trace_id -> {"set_raw": ..., "attivita": ...}. Raises on
    any (trace_id, antenna) collision that does not match the known
    one-file-per-archive pattern — investigate, don't guess.
    """
    candidates: dict[tuple[str, int], list[Path]] = {}
    parsed_by_path: dict[Path, dict[str, Any]] = {}
    for fp in list_files(stage_dir):
        try:
            p = parse_filename(fp.name)
        except ValueError:
            continue  # non-matching files were already audited on day 1
        parsed_by_path[fp] = p
        trace_id = f"{p['set_raw']}_{p['attivita']}{p['repetition']}"
        candidates.setdefault((trace_id, p["stream"]), []).append(fp)

    files: dict[str, dict[int, Path]] = {}
    meta: dict[str, dict[str, str]] = {}

    def _register(trace_id: str, antenna: int, fp: Path) -> None:
        files.setdefault(trace_id, {})[antenna] = fp
        p = parsed_by_path[fp]
        meta[trace_id] = {"set_raw": p["set_raw"], "attivita": p["attivita"]}

    for (trace_id, antenna), paths in candidates.items():
        if len(paths) == 1:
            _register(trace_id, antenna, paths[0])
            continue
        from_supplementary = [DUPLICATE_SOURCE_MARKER in str(fp) for fp in paths]
        assert len(paths) == 2 and sum(from_supplementary) == 1, (
            f"unexpected duplicate staged files for ({trace_id!r}, antenna {antenna}): "
            f"{[str(p) for p in paths]} — not the known dual-archive case, resolve manually."
        )
        for fp, is_alt in zip(paths, from_supplementary):
            _register(trace_id + DUPLICATE_TRACE_SUFFIX if is_alt else trace_id, antenna, fp)

    return files, meta


class DopplerDataset(Dataset):
    """Dataset for one set (train/val/test) of a frozen split rotation.

    Windowing (§1.2): windows of `axes.time` x `axes.velocity` steps
    (340x100), stride `window.train_stride` (100) for train and
    `window.eval_stride` (340, disjoint) for val/test, both read from the
    split file; the final incomplete window is discarded. Each
    (window, antenna) is an independent sample (§1.3) — antenna fusion
    (softmax averaging) is harness.py's responsibility at evaluation
    time, NOT done here; `trace_id`/`antenna`/`window_start` are returned
    with each sample so the harness can group. Normalization (§1.4):
    `(x - mu) / sigma` with the split file's train-only scalars, then the
    optional `transform` (augmentation, §3) is applied.

    I/O (§8.5): lazy per-file loading from local staging with an LRU
    cache of decoded float32 traces. The default (unbounded) cache warms
    to ~2.5 GB for all 408 streams — fine on Colab RAM, but every
    DataLoader worker process holds its own copy: keep num_workers small
    or pass a bounded `cache_size`.

    Args:
        split_file: frozen split JSON (e.g. splits/p2_lab.json).
        set_name: "train", "val" or "test".
        stage_dir: local staging root (paths.yaml stage_dir on Colab).
        inventory_csv: frozen day-1 inventory, source of per-trace
            lengths for the window index.
        arset_map: frozen set_raw -> AR-set map
            (reports/name_to_arset.json); dict accepted for tests.
        labels: explicit class list, overriding the split file's. Needed
            for P1, whose split records labels as null (the C0 letter ->
            paper-class mapping is a pending deliberate decision).
        transform: applied to the normalized (1, time, velocity) float32
            tensor; must return a tensor of the same shape.
        cache_size: LRU entries for decoded traces (None = unbounded).
    """

    def __init__(
        self,
        split_file: str | Path,
        set_name: str,
        stage_dir: str | Path,
        inventory_csv: str | Path = "reports/inventory.csv",
        arset_map: str | Path | dict[str, str] = "reports/name_to_arset.json",
        labels: list[str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cache_size: int | None = None,
    ) -> None:
        super().__init__()
        assert set_name in SET_NAMES, f"set_name must be one of {SET_NAMES}, got {set_name!r}"
        self.set_name = set_name
        self.transform = transform

        split = read_json(split_file)
        self._win = int(split["axes"]["time"])
        self._n_velocity = int(split["axes"]["velocity"])
        self.stride = int(
            split["window"]["train_stride"] if set_name == "train" else split["window"]["eval_stride"]
        )
        self.mu = float(split["norm"]["mu"])
        self.sigma = float(split["norm"]["sigma"])

        labels = labels if labels is not None else split["classes"]["labels"]
        if labels is None:
            raise ValueError(
                f"{split_file} records no class labels (P1: the C0 letter->class mapping "
                "is a pending decision) — pass labels= explicitly."
            )
        self.labels: list[str] = list(labels)
        self.label_to_idx = {lab: i for i, lab in enumerate(self.labels)}

        # AR-set label space (§2.2): the adversary is defined on the
        # rotation's TRAIN domains, so derive it from the train list
        # regardless of set_name — identical across the three instances.
        # Samples from the held-out domain (test) get ar_set = -1: the
        # GRL loss never sees them, the sentinel only marks "not a train
        # domain" for downstream diagnostics.
        if isinstance(arset_map, (str, Path)):
            arset_map = read_json(arset_map)
        self._arset_map: dict[str, str] = dict(arset_map)
        unmapped = {t.split("_", 1)[0] for t in split["train"]} - set(self._arset_map)
        assert not unmapped, (
            f"set(s) {sorted(unmapped)} in the split's train list are missing from the "
            "name_to_arset map — split and map are frozen artifacts and must agree."
        )
        train_arsets = {self._arset_map[t.split("_", 1)[0]] for t in split["train"]}
        self.arset_labels: list[str] = sorted(train_arsets)
        self.arset_to_idx = {a: i for i, a in enumerate(self.arset_labels)}

        trace_ids: list[str] = sorted(split[set_name])

        files, meta = build_file_index(stage_dir)
        missing = [t for t in trace_ids if t not in files]
        assert not missing, (
            f"{len(missing)} trace(s) of {set_name} not found under {stage_dir}: {missing[:5]} — "
            "staging incomplete or inconsistent with the frozen split."
        )
        bad_antennas = {t: sorted(files[t]) for t in trace_ids if sorted(files[t]) != list(range(N_ANTENNAS))}
        assert not bad_antennas, (
            f"traces without exactly antennas 0..{N_ANTENNAS - 1}: {bad_antennas} — "
            "staging inconsistent with the day-1 inventory."
        )
        self._files = {t: files[t] for t in trace_ids}
        self._meta = {t: meta[t] for t in trace_ids}

        # Per-trace lengths from the frozen inventory (no data reads).
        inv = pd.read_csv(inventory_csv, usecols=["trace_id", "n_frame"])
        frames = inv.groupby("trace_id")["n_frame"].agg(["nunique", "first"])
        assert (frames["nunique"] == 1).all(), (
            "inventory.csv records different n_frame across streams of the same trace: "
            f"{frames[frames['nunique'] > 1].index.tolist()} — day-1 artifact corrupted?"
        )
        n_frame = frames["first"].to_dict()
        not_inventoried = [t for t in trace_ids if t not in n_frame]
        assert not not_inventoried, (
            f"trace(s) missing from {inventory_csv}: {not_inventoried[:5]} — "
            "split and inventory disagree; both are frozen, investigate."
        )
        self._n_frame = {t: int(n_frame[t]) for t in trace_ids}

        # Sample index: deterministic order (sorted traces, antennas
        # ascending, starts ascending); shuffling is the DataLoader's job.
        self._samples: list[tuple[str, int, int]] = [
            (t, antenna, start)
            for t in trace_ids
            for antenna in range(N_ANTENNAS)
            for start in range(0, self._n_frame[t] - self._win + 1, self.stride)
        ]
        assert self._samples, f"no complete windows in {set_name} — wrong stage_dir or split?"

        # Plain-dict LRU (not functools.lru_cache: a wrapped bound method
        # would make the dataset unpicklable for spawn-based DataLoaders).
        self._cache_size = cache_size
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        logger.info(
            "%s: %d traces, %d (window, antenna) samples (win=%d, stride=%d)",
            set_name, len(trace_ids), len(self._samples), self._win, self.stride,
        )

    @property
    def n_att(self) -> int:
        return len(self.labels)

    @property
    def n_arset(self) -> int:
        return len(self.arset_labels)

    def _load(self, path: Path) -> np.ndarray:
        """LRU-cached decode of one file-stream to float32 (halves cache
        RAM vs the on-disk float64; training runs in float32/AMP anyway)."""
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        arr = load_trace(path).astype(np.float32)
        self._cache[path] = arr
        if self._cache_size is not None and len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trace_id, antenna, start = self._samples[index]
        arr = self._load(self._files[trace_id][antenna])
        expected = (self._n_frame[trace_id], self._n_velocity)
        assert arr.shape == expected, (
            f"{self._files[trace_id][antenna]}: shape {arr.shape} != inventoried {expected} — "
            "staged data drifted from the frozen day-1 inventory."
        )
        window = arr[start : start + self._win]
        x = torch.from_numpy((window - self.mu) / self.sigma)[None]  # (1, time, velocity) float32
        if self.transform is not None:
            x = self.transform(x)
        m = self._meta[trace_id]
        return {
            "x": x,
            "y": self.label_to_idx[m["attivita"]],
            "ar_set": self.arset_to_idx.get(self._arset_map[m["set_raw"]], -1),
            "trace_id": trace_id,
            "antenna": antenna,
            "window_start": start,
        }
