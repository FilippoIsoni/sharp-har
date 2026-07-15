"""[STUB] P×K sampler with distinct-trace constraint + offset. Ref. v5
pipeline §4.2. Used by configs C3/C4 (sampler: pxk). Do not implement
before the day-2 gate (§10.1).
"""
from __future__ import annotations

from typing import Iterator

from torch.utils.data import Sampler


class PKSampler(Sampler[list[int]]):
    """Sampler for P×K batches: P = n_att (number of activities), K = 32
    samples per activity, round-robin over AR-sets. Constraints: at most
    one window per trace per class in the batch; reusing the same trace
    is only allowed with a window offset `|Δstart| >= 340`. Deterministic
    reseed per epoch: `seed_epoch = f(seed, epoch)`. Logs the composition
    of every batch (trace-id, AR-set, offset). Ref. §4.2.
    """

    def __init__(self, dataset, p: int, k: int, seed: int, epoch: int = 0) -> None:
        super().__init__()
        raise NotImplementedError("day 2/3 — §4.2")

    def set_epoch(self, epoch: int) -> None:
        raise NotImplementedError("day 2/3 — §4.2")

    def __iter__(self) -> Iterator[list[int]]:
        raise NotImplementedError("day 2/3 — §4.2")

    def __len__(self) -> int:
        raise NotImplementedError("day 2/3 — §4.2")
