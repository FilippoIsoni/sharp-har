"""Day 3 — linear probe, single frozen recipe for every linear-probe
evaluation in the project: C1-lin, C2-lin, C3/C4 phase B, §7 diagnostic
probes. Ref. §5.3, §7.

Recipe (frozen, §5.3): encoder FROZEN, features cached once to disk by
harness.cache_features (no augmentation, declared note i). Linear head
d_enc -> n_classes, Adam lr 1e-3 wd 1e-4 (Adam vs the core's AdamW is
intentional and declared, §5.3 note ii), batch 256, max 30 epochs,
early stopping on val macro-F1 computed by the common harness fusion
(softmax averaging over antennas, §1.3), patience 5, best checkpoint.
Cost: minutes. Test evaluation of the selected head goes through
harness.evaluate_features (logged, §0.7) — never through this module.

§7 diagnostics reuse linear_probe unchanged with different targets
(y = AR-set index or person) and compare against majority_baseline —
the majority class, NOT 1/n (classes are unbalanced).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .harness import fuse_windows, macro_f1
from .utils import epoch_seed, get_logger

logger = get_logger(__name__)


def _as_arrays(features: str | Path | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    if isinstance(features, (str, Path)):
        return np.load(features, allow_pickle=False)
    return features


def majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the majority class — the §7
    reference line for the AR-set/person probes (not 1/n: unbalanced)."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def linear_probe(
    train_features: str | Path | Mapping[str, np.ndarray],
    val_features: str | Path | Mapping[str, np.ndarray],
    *,
    target: str = "y",
    n_classes: int | None = None,
    seed: int = 42,
    device: str | None = None,
    batch_size: int = 256,
    max_epochs: int = 30,
    lr: float = 1e-3,
    wd: float = 1e-4,
    patience: int = 5,
) -> dict[str, Any]:
    """Trains the single-recipe linear probe on cached features
    (harness.cache_features .npz, or an equivalent array mapping with
    `features`, the `target` label array, `trace_id`, `window_start`).

    `target`: label key — "y" for activity probes/phase B, "ar_set" for
    the §7 domain probe (any integer label array works, e.g. a person
    encoding). Selection = val macro-F1 with antenna fusion from the
    common harness (§5.3). Returns the BEST head as numpy arrays
    ("weight" (C, d), "bias" (C,)) ready for harness.evaluate_features,
    plus best_val_macro_f1, best_epoch and the per-epoch history.
    """
    tr, va = _as_arrays(train_features), _as_arrays(val_features)
    x_tr = torch.from_numpy(np.asarray(tr["features"], dtype=np.float32))
    y_tr = torch.from_numpy(np.asarray(tr[target], dtype=np.int64))
    x_va = torch.from_numpy(np.asarray(va["features"], dtype=np.float32))
    y_va = np.asarray(va[target], dtype=np.int64)
    assert y_tr.min() >= 0 and y_va.min() >= 0, (
        f"negative label in target {target!r} (held-out-domain sentinel?) — "
        "probes are defined on train-domain labels (§7)."
    )
    n = n_classes if n_classes is not None else int(max(int(y_tr.max()), int(y_va.max()))) + 1

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    head = nn.Linear(x_tr.shape[1], n).to(dev)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    x_va_dev = x_va.to(dev)

    best: dict[str, Any] = {"best_val_macro_f1": -1.0, "best_epoch": 0}
    history: list[dict[str, float]] = []
    epochs_no_improve = 0
    gen = torch.Generator()
    for epoch in range(1, max_epochs + 1):
        head.train()
        gen.manual_seed(epoch_seed(seed, epoch))
        loss_sum = 0.0
        perm = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            logits = head(x_tr[idx].to(dev))
            loss = F.cross_entropy(logits, y_tr[idx].to(dev))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            probs = torch.softmax(head(x_va_dev), dim=1).cpu().numpy()
        fused = fuse_windows(probs, y_va, va["trace_id"].astype(object), va["window_start"])
        val_f1 = macro_f1(fused["y_true"], fused["y_pred"])
        history.append({"epoch": epoch, "train_loss": loss_sum / len(x_tr), "val_macro_f1": val_f1})

        if val_f1 > best["best_val_macro_f1"]:
            best.update(
                best_val_macro_f1=val_f1, best_epoch=epoch,
                weight=head.weight.detach().cpu().numpy().copy(),
                bias=head.bias.detach().cpu().numpy().copy(),
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    logger.info(
        "linear probe [%s]: best val macro-F1 %.4f at epoch %d/%d",
        target, best["best_val_macro_f1"], best["best_epoch"], len(history),
    )
    best["history"] = history
    return best
