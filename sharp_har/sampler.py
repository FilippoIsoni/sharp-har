"""[STUB] Sampler P×K con vincolo trace-distinte + offset. Rif. pipeline
v5 §4.2. Usato dai config C3/C4 (sampler: pxk). Non implementare prima
del gate giorno 2 (§10.1).
"""
from __future__ import annotations

from typing import Iterator

from torch.utils.data import Sampler


class PKSampler(Sampler[list[int]]):
    """Sampler per batch P×K: P = n_att (numero di attività), K = 32
    sample per attività, round-robin sugli AR-set. Vincoli: al massimo
    una finestra per trace per classe nel batch; il riuso della stessa
    trace è ammesso solo con offset di finestra `|Δstart| >= 340`. Riseed
    deterministico per epoca: `seed_epoch = f(seed, epoca)`. Logga la
    composizione di ogni batch (trace-id, AR-set, offset). Rif. §4.2.
    """

    def __init__(self, dataset, p: int, k: int, seed: int, epoch: int = 0) -> None:
        super().__init__()
        raise NotImplementedError("giorno 2/3 — §4.2")

    def set_epoch(self, epoch: int) -> None:
        raise NotImplementedError("giorno 2/3 — §4.2")

    def __iter__(self) -> Iterator[list[int]]:
        raise NotImplementedError("giorno 2/3 — §4.2")

    def __len__(self) -> int:
        raise NotImplementedError("giorno 2/3 — §4.2")
