"""Day 3 — P×K sampler with distinct-trace constraint + minimum window
offset on reuse. Ref. §4.2. Used by configs C3/C4 (sampler: pxk).

Per class, per batch (§4.2): (1) round-robin over the train AR-sets in
which the class exists (induces "same activity, different domain"
positives — deliberately oversamples trace-poor sets, declared §4.2);
(2) inside the current set, one trace without replacement (per-set
queues, reshuffled when exhausted); (3) from the trace: uniform window,
uniform antenna; (4) hard constraint: at most one window per trace per
class in the batch while the class has >= K traces; below K, reuse is
allowed only with |Δstart| >= window length (disjoint windows —
at stride 100 two "different" windows share up to ~71% of their
content). A trace with no disjoint start left is skipped in favour of
the next round-robin trace.

Deterministic per-epoch reseed (utils.epoch_seed, §4.2/§8.2): iterating
the same (seed, epoch) twice yields identical batches, so resuming from
an epoch boundary reproduces the sequence without persisting sampler
state. Mandatory logging (§4.2): per-epoch mean batch composition
(distinct AR-sets and unique traces per class, trace reuses with the
minimum observed offset), kept in `last_epoch_stats` and logged.
"""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from torch.utils.data import Sampler

from .utils import epoch_seed, get_logger

logger = get_logger(__name__)

# Bail-out threshold for the per-class sampling loop: generous — reuse
# rejections are common by design when traces < K, an infinite loop is not.
_MAX_ATTEMPTS_PER_PICK = 200


class PKSampler(Sampler[list[int]]):
    """Batch sampler yielding `num_batches` lists of P*K dataset indices
    per epoch (epoch = fixed steps, §8.1). P = n_att of the train set
    (asserted), K = ⌊batch/P⌋ = 32 for the standard 256-window batch;
    the 2 SupCon views per window (512 views) are the training loop's
    job, not the sampler's. Ref. §4.2.

    Args:
        dataset: the TRAIN DopplerDataset of the frozen rotation.
        p: number of classes; must equal dataset.n_att (§4.2).
        k: samples per class per batch.
        seed: run seed (42); reseeded per epoch via utils.epoch_seed.
        num_batches: batches per epoch (train.epoch_steps, §8.1).
        epoch: initial epoch; update per epoch with set_epoch().
    """

    def __init__(self, dataset: Any, p: int, k: int, seed: int, num_batches: int, epoch: int = 0) -> None:
        super().__init__()
        assert p == dataset.n_att, f"P must be the train n_att ({dataset.n_att}), got {p} (§4.2)"
        assert k >= 1 and num_batches >= 1
        self._p, self._k, self._seed, self._num_batches = p, k, seed, num_batches
        self._epoch = epoch
        self._win = dataset.window_len
        self.class_names: list[str] = list(dataset.labels)

        info = dataset.trace_info
        # (trace, antenna, start) -> dataset index, plus per-trace starts
        # (identical across the 4 antennas by construction) and antennas.
        self._index_of: dict[tuple[str, int, int], int] = {}
        starts: dict[str, set[int]] = {}
        antennas: dict[str, set[int]] = {}
        for idx, (trace, antenna, start) in enumerate(dataset.samples):
            self._index_of[(trace, antenna, start)] = idx
            starts.setdefault(trace, set()).add(start)
            antennas.setdefault(trace, set()).add(antenna)
        self._starts = {t: sorted(s) for t, s in starts.items()}
        self._antennas = {t: sorted(a) for t, a in antennas.items()}

        # class -> AR-set -> sorted trace list (the deterministic base
        # order every per-epoch shuffle starts from).
        by_class: dict[int, dict[str, list[str]]] = {}
        for trace in sorted(self._starts):
            m = info[trace]
            by_class.setdefault(m["y"], {}).setdefault(m["ar_set"], []).append(trace)
        assert sorted(by_class) == list(range(p)), (
            f"train classes {sorted(by_class)} != 0..{p - 1} — a class has no train trace, "
            "which the rare-cell pin guarantee (§2.2) rules out; investigate."
        )
        self._by_class = by_class
        self._arsets = {c: sorted(sets) for c, sets in by_class.items()}
        self._n_traces = {c: sum(len(v) for v in sets.values()) for c, sets in by_class.items()}

        self.last_epoch_stats: dict[str, Any] | None = None

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch's iteration (§4.2): the whole epoch is
        a pure function of (seed, epoch)."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._num_batches

    def _pick_start(
        self, rng: np.random.Generator, trace: str, used: dict[str, list[int]]
    ) -> tuple[int, int] | None:
        """Uniform start for `trace`; on reuse, uniform over the starts
        disjoint (|Δstart| >= window length) from every start of the
        trace already in the batch. Returns (start, min_offset_observed)
        with min_offset = 0 on first use, or None if no valid start."""
        prev = used.get(trace)
        if not prev:
            return int(rng.choice(self._starts[trace])), 0
        valid = [s for s in self._starts[trace] if min(abs(s - u) for u in prev) >= self._win]
        if not valid:
            return None
        start = int(rng.choice(valid))
        return start, min(abs(start - u) for u in prev)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(epoch_seed(self._seed, self._epoch))
        # Per-(class, AR-set) without-replacement queues, consumed from the
        # end; refilled with a fresh permutation when exhausted. Round-robin
        # pointers persist across the batches of the epoch.
        queues = {
            c: {a: list(rng.permutation(traces)) for a, traces in sets.items()}
            for c, sets in self._by_class.items()
        }
        rr = {c: 0 for c in self._by_class}

        n_batches = 0
        per_class: dict[int, dict[str, float]] = {
            c: {"arsets": 0.0, "traces": 0.0, "reuses": 0.0} for c in self._by_class
        }
        reuse_offsets: list[int] = []

        for _ in range(self._num_batches):
            batch: list[int] = []
            for c in sorted(self._by_class):
                arsets = self._arsets[c]
                used: dict[str, list[int]] = {}
                picks = 0
                attempts = 0
                while picks < self._k:
                    attempts += 1
                    assert attempts <= _MAX_ATTEMPTS_PER_PICK * self._k, (
                        f"P×K sampler stalled on class {self.class_names[c]!r} "
                        f"({self._n_traces[c]} traces, K={self._k}): not enough disjoint "
                        "windows — discuss before touching the constraint (§4.2)."
                    )
                    arset = arsets[rr[c] % len(arsets)]
                    rr[c] += 1
                    queue = queues[c][arset]
                    if not queue:
                        queue.extend(rng.permutation(self._by_class[c][arset]))
                    trace = queue.pop()
                    if trace in used and self._n_traces[c] >= self._k:
                        continue  # hard constraint (4): no reuse while traces >= K
                    picked = self._pick_start(rng, trace, used)
                    if picked is None:
                        continue  # no disjoint start left: next round-robin trace
                    start, min_offset = picked
                    if trace in used:
                        per_class[c]["reuses"] += 1
                        reuse_offsets.append(min_offset)
                    antenna = int(rng.choice(self._antennas[trace]))
                    batch.append(self._index_of[(trace, antenna, start)])
                    used.setdefault(trace, []).append(start)
                    picks += 1
                per_class[c]["arsets"] += len(
                    {arset for arset, traces in self._by_class[c].items() if any(t in used for t in traces)}
                )
                per_class[c]["traces"] += len(used)
            n_batches += 1
            yield batch

        self.last_epoch_stats = {
            "epoch": self._epoch,
            "num_batches": n_batches,
            "per_class": {
                self.class_names[c]: {
                    "mean_distinct_arsets": s["arsets"] / n_batches,
                    "mean_unique_traces": s["traces"] / n_batches,
                    "n_reuses": int(s["reuses"]),
                    "n_train_traces": self._n_traces[c],
                }
                for c, s in per_class.items()
            },
            "n_reuses_total": int(sum(s["reuses"] for s in per_class.values())),
            "min_reuse_offset": int(min(reuse_offsets)) if reuse_offsets else None,
        }
        logger.info(
            "P×K epoch %d: %d batches, %d trace reuses (min offset %s; window %d)",
            self._epoch, n_batches, self.last_epoch_stats["n_reuses_total"],
            self.last_epoch_stats["min_reuse_offset"], self._win,
        )
