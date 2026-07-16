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

The notebook-facing runners live here too: probe_encoder (one frozen
checkpoint -> cached features -> probe -> persisted head, the C1-lin/
C2-lin and §7 path) and select_phase_b (the §6-C3/C4 grid -> selected
checkpoint, the operational definition of "plateau"). Both touch train
and val ONLY — test evaluation of the persisted heads happens in the
single final session via harness.evaluate_features (§0.7).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .harness import cache_features, fuse_windows, macro_f1
from .inventory import AR_SET_METADATA
from .utils import epoch_seed, get_git_hash, get_logger, write_json

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


# ------------------------------------------------------------------ runners

def _probe_arrays(npz_path: str | Path, target: str) -> Mapping[str, np.ndarray]:
    """Arrays for one probe from a harness.cache_features .npz. "y" and
    "ar_set" are stored in the cache; "persona" (§7 person probe) is
    derived per sample from arset_name via inventory.AR_SET_METADATA
    (index over the sorted person names — a fixed AR-set property, so no
    harness/cache change is needed)."""
    data = np.load(npz_path, allow_pickle=False)
    if target != "persona":
        return data
    persons = sorted({m["persona"] for m in AR_SET_METADATA.values()})
    label = np.array(
        [persons.index(AR_SET_METADATA[str(a)]["persona"]) for a in data["arset_name"]],
        dtype=np.int64,
    )
    return {
        **{k: data[k] for k in ("features", "trace_id", "window_start")},
        "persona": label,
    }


def probe_encoder(
    checkpoint: str | Path,
    split_file: str | Path,
    *,
    stage_dir: str | Path,
    repo_dir: str | Path = ".",
    target: str = "y",
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """One frozen encoder -> §5.3 probe, end to end on train/val ONLY:
    caches train+val features via harness.cache_features (the .npz is
    reused if already on disk — computed once, every probe shares it,
    §5.3), trains the single-recipe linear probe, and persists next to
    the checkpoint:

    - ``probe_head_<stem>[_<target>].npz`` ("weight"/"bias") — the input
      of harness.evaluate_features in the final test session (§0.7);
    - ``probe_<stem>[_<target>].json`` — metrics, probe history, and the
      §7 majority baseline of the val target (audit trail).

    target="y" is C1-lin/C2-lin and phase B; "ar_set"/"persona" are the
    §7 diagnostics (val is in-domain, so train-domain labels are defined
    on it). Besides the selection metric (fused val macro-F1) the summary
    reports the fused val accuracy of the best head — §7's metric, to be
    read against val_majority_baseline. Never touches the test set.
    Ref. §5.3, §6 (unified linear-probe evaluation), §7, §0.7.
    """
    checkpoint = Path(checkpoint)
    features: dict[str, Path] = {}
    for set_name in ("train", "val"):
        path = checkpoint.parent / f"features_{checkpoint.stem}_{set_name}.npz"
        if not path.exists():
            path = cache_features(
                checkpoint, split_file, set_name,
                stage_dir=stage_dir, repo_dir=repo_dir, device=device,
            )
        features[set_name] = path

    tr, va = _probe_arrays(features["train"], target), _probe_arrays(features["val"], target)
    result = linear_probe(tr, va, target=target, seed=seed, device=device)

    # Fused val accuracy of the best head (§7 reads accuracy, not F1).
    logits = va["features"] @ result["weight"].T + result["bias"]
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)).astype(np.float32)
    fused = fuse_windows(probs, va[target], va["trace_id"].astype(object), va["window_start"])
    val_accuracy = float((fused["y_pred"] == fused["y_true"]).mean())

    suffix = "" if target == "y" else f"_{target}"
    head_path = checkpoint.parent / f"probe_head_{checkpoint.stem}{suffix}.npz"
    np.savez(
        head_path, weight=result["weight"], bias=result["bias"],
        target=np.array(target), checkpoint=np.array(str(checkpoint)),
        git_hash=np.array(get_git_hash(repo_dir)),
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "target": target,
        "seed": seed,
        "git_hash": get_git_hash(repo_dir),
        "best_val_macro_f1": result["best_val_macro_f1"],
        "best_epoch": result["best_epoch"],
        "val_accuracy": val_accuracy,
        "val_majority_baseline": majority_baseline(np.asarray(va[target])),
        "head_path": str(head_path),
        "features": {k: str(v) for k, v in features.items()},
        "history": result["history"],
    }
    summary_path = checkpoint.parent / f"probe_{checkpoint.stem}{suffix}.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    logger.info(
        "probe_encoder [%s, %s]: val macro-F1 %.4f, val accuracy %.4f "
        "(majority baseline %.4f) -> %s",
        checkpoint.stem, target, summary["best_val_macro_f1"], val_accuracy,
        summary["val_majority_baseline"], head_path,
    )
    return summary


def select_phase_b(
    run_dir: str | Path,
    split_file: str | Path,
    checkpoint_epochs: Sequence[int],
    *,
    stage_dir: str | Path,
    repo_dir: str | Path = ".",
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Phase B for C3/C4 (§6): every pre-committed grid checkpoint
    (train.checkpoint_epochs = ⌈2H/3⌉, ⌈5H/6⌉, H) gets the §5.3 probe
    via probe_encoder; the best fused val macro-F1 selects the
    checkpoint — the operational definition of "plateau", identical for
    C3 and C4. Ties break toward the earliest epoch (deterministic,
    declared). Persists ``phase_b_selection.json`` in run_dir; ONLY the
    selected checkpoint+head go to the final test session
    (harness.cache_features + evaluate_features, §0.7).
    Ref. §6-C3/C4, §5.3, §0.7.
    """
    run_dir = Path(run_dir)
    for n in checkpoint_epochs:
        assert (run_dir / f"epoch{n}.ckpt").exists(), (
            f"missing grid checkpoint epoch{n}.ckpt in {run_dir} — phase B runs only "
            "after phase A has produced the full pre-committed grid (§6-C3)."
        )
    candidates = [
        probe_encoder(
            run_dir / f"epoch{n}.ckpt", split_file,
            stage_dir=stage_dir, repo_dir=repo_dir, seed=seed, device=device,
        ) | {"epoch": int(n)}
        for n in checkpoint_epochs
    ]
    best = max(candidates, key=lambda c: c["best_val_macro_f1"])  # first max = earliest epoch

    selection = {
        "run_dir": str(run_dir),
        "split_file": str(split_file),
        "checkpoint_epochs": [int(n) for n in checkpoint_epochs],
        "seed": seed,
        "git_hash": get_git_hash(repo_dir),
        "selected_epoch": best["epoch"],
        "selected_checkpoint": best["checkpoint"],
        "selected_head": best["head_path"],
        "selected_val_macro_f1": best["best_val_macro_f1"],
        # Per-checkpoint probe history stays in the probe_<stem>.json files.
        "candidates": [
            {k: c[k] for k in (
                "epoch", "checkpoint", "head_path",
                "best_val_macro_f1", "val_accuracy", "best_epoch",
            )}
            for c in candidates
        ],
    }
    write_json(run_dir / "phase_b_selection.json", selection)
    logger.info(
        "phase B [%s]: selected epoch %d (val macro-F1 %.4f) over grid %s -> %s",
        run_dir.name, best["epoch"], best["best_val_macro_f1"],
        [int(n) for n in checkpoint_epochs], run_dir / "phase_b_selection.json",
    )
    return selection
