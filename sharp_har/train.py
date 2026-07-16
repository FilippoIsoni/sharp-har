"""Day 2 — training loop shared by the core runs. Ref. §8 (optimization,
checkpoint/resume), §6 (per-run configs), §0.4-0.5 (run artifacts, seed).

Day-2 scope: the CE path (C1-style; C0 once its class mapping is
decided). SupCon losses, the GRL adversary, the P×K sampler and the
sharp_like backbone raise NotImplementedError with their pipeline day —
gated implementation, nothing downstream of the throughput gate is
filled in early.

Colab free disconnects: every epoch ends with a complete `last.ckpt`
(weights + optimizer + scheduler + GradScaler + epoch + config + RNG
states) written atomically, and `train_run` resumes from it
automatically. Batch order is reproducible across resumes because the
shuffle generator is reseeded per epoch as seed_epoch = f(seed, epoch)
(§8.2), so no sampler state needs saving.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DopplerDataset
from .harness import fuse_windows, macro_f1, predict
from .losses import ce_with_label_smoothing
from .models import build_backbone
from .models.heads import ActivityHead
from .utils import epoch_seed as _epoch_seed
from .utils import get_git_hash, get_logger, set_seed, write_json

logger = get_logger(__name__)

GRAD_CLIP_NORM = 1.0  # §8.1 default when the config omits train.grad_clip; essential with GRL later


@torch.no_grad()
def _fused_val_macro_f1(
    backbone: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device, amp: bool
) -> float:
    """In-loop selection metric (§6-C1): antenna fusion by softmax
    averaging per (trace_id, window_start), then argmax, then macro-F1
    (§1.3, §9) — computed by the SAME harness functions used for
    reporting (day 3), so selection and reporting cannot drift.
    Reporting/test evaluation still goes only through harness.evaluate."""
    res = predict(backbone, head, loader, device, amp)
    fused = fuse_windows(res["probs"], res["y"], res["trace_id"], res["window_start"])
    return macro_f1(fused["y_true"], fused["y_pred"])


def _build_model(cfg: dict[str, Any]) -> tuple[nn.Module, nn.Module]:
    return build_backbone(cfg), ActivityHead(cfg["d_enc"], cfg["n_att"])


def _build_optimizer(cfg: dict[str, Any], params: Any) -> torch.optim.Optimizer:
    o = cfg["optim"]
    if o["name"] == "adamw":
        return torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["wd"])
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["wd"])
    raise ValueError(f"unknown optimizer {o['name']!r}")


def _build_scheduler(
    cfg: dict[str, Any], optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup (linear, warmup_epochs) then cosine decay over the FULL
    phase horizon — scheduled on the horizon even if early stopping cuts
    the run short, standard and declared (§8.1). scheduler "none" keeps
    LR flat after warmup (C0)."""
    o = cfg["optim"]
    warmup_steps = o["warmup_epochs"] * cfg["train"]["epoch_steps"]
    kind = o["scheduler"]
    if kind not in ("cosine", "none"):
        raise ValueError(f"unknown scheduler {kind!r}")

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if kind == "none":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(states: dict[str, Any]) -> None:
    # Generator states are consumed as CPU ByteTensors — even the CUDA
    # ones (torch.cuda.set_rng_state_all rejects CUDA tensors with "RNG
    # state must be a torch.ByteTensor"). Coerce defensively in case the
    # checkpoint was loaded with a non-CPU map_location.
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"].detach().to("cpu", torch.uint8))
    if states["cuda"] is not None and torch.cuda.is_available():
        cuda_states = states["cuda"]
        if isinstance(cuda_states, torch.Tensor):
            cuda_states = [cuda_states]
        torch.cuda.set_rng_state_all([s.detach().to("cpu", torch.uint8) for s in cuda_states])


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    """Colab can disconnect mid-write: save to a temp file and replace,
    so last.ckpt is never left corrupted (§8.2)."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train_run(
    cfg: dict[str, Any],
    stage_dir: str | Path,
    ckpt_dir: str | Path,
    repo_dir: str | Path = ".",
    device: str | None = None,
    num_workers: int = 2,
) -> dict[str, Any]:
    """Runs (or resumes) one config-defined training run. Ref. §8.

    cfg is one of configs/c*.yaml, loaded as a dict — the config fully
    describes the run, no per-run flags (§0.4). The smoke gate passes a
    modified COPY of the config (fewer epochs/steps), never edits the
    file. Optional cfg keys: "seed" (default 42, E1/E3 extensions use
    43/44), "labels" (explicit class list, required for P1/C0).

    Artifacts under ckpt_dir/<cfg.name>/: last.ckpt (every epoch,
    atomic), best.ckpt (val-selected), epoch{N}.ckpt for the phase-A
    grid (train.checkpoint_epochs), run_meta.json (config + seed + git
    hash, §0.4), history.csv (per-epoch loss/metric/lr/s-per-step — the
    day-2 gate reads s_per_step from here).
    """
    if cfg["loss"]["type"] != "ce":
        raise NotImplementedError(
            f"loss {cfg['loss']['type']!r} — phase-A wiring is day 4 (§6-C3/C4); "
            "supcon_loss/ProjectionHead/PKSampler/augment are implemented (day 3)."
        )
    if (cfg.get("adversary") or {}).get("type"):
        raise NotImplementedError("GRL adversary — day 4 (§5.3, §8; C2/C4)")
    if cfg["train"]["sampler"] != "uniform":
        raise NotImplementedError(f"sampler {cfg['train']['sampler']!r} wiring — day 4 (§4.2, P×K)")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.backends.cudnn.benchmark = True  # §8.1; declared non-bit-exactness (§0.5)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["train"]["amp"]) and dev.type == "cuda"
    repo_dir = Path(repo_dir)

    run_dir = Path(ckpt_dir) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        s: DopplerDataset(
            repo_dir / cfg["split_file"], s, stage_dir,
            inventory_csv=repo_dir / "reports" / "inventory.csv",
            arset_map=repo_dir / "reports" / "name_to_arset.json",
            labels=cfg.get("labels"),
        )
        for s in ("train", "val")
    }
    assert datasets["train"].n_att == cfg["n_att"], (
        f"cfg.n_att={cfg['n_att']} != dataset n_att={datasets['train'].n_att} — "
        "config and frozen split disagree."
    )

    batch_size = cfg["train"]["batch_size"]
    grad_clip = float(cfg["train"].get("grad_clip", GRAD_CLIP_NORM))
    epoch_steps = cfg["train"]["epoch_steps"]
    max_epochs = cfg["train"]["max_epochs"]
    patience = cfg["eval"]["patience"]
    grid_epochs = set(cfg["train"].get("checkpoint_epochs") or [])
    total_steps = max_epochs * epoch_steps

    assert len(datasets["train"]) >= batch_size, (
        f"train set ({len(datasets['train'])} samples) smaller than one batch "
        f"({batch_size}) — with drop_last the loader would be empty."
    )
    shuffle_gen = torch.Generator()
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True, generator=shuffle_gen,
        num_workers=num_workers, pin_memory=dev.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=dev.type == "cuda",
    )

    backbone, head = _build_model(cfg)
    backbone.to(dev)
    head.to(dev)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = _build_optimizer(cfg, params)
    scheduler = _build_scheduler(cfg, optimizer, total_steps)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    label_smoothing = float(cfg["loss"].get("label_smoothing") or 0.0)

    start_epoch = 1
    best_metric = -1.0
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    last_path = run_dir / "last.ckpt"
    if last_path.exists():  # automatic resume, §8.2
        # map_location="cpu", NOT dev: the RNG state tensors must stay
        # CPU ByteTensors (see _restore_rng); load_state_dict copies the
        # weights onto the module's device and the optimizer casts its
        # state to the params' device by itself.
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        _restore_rng(ckpt["rng"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        history = ckpt["history"]
        logger.info("resumed %s from %s at epoch %d", cfg["name"], last_path, start_epoch)
    else:
        write_json(run_dir / "run_meta.json", {
            "config": cfg, "seed": seed, "git_hash": get_git_hash(repo_dir),
            "device": str(dev), "torch": torch.__version__,
        })

    for epoch in range(start_epoch, max_epochs + 1):
        backbone.train()
        head.train()
        shuffle_gen.manual_seed(_epoch_seed(seed, epoch))
        batches = iter(train_loader)
        loss_sum, t0 = 0.0, time.time()

        for _step in range(epoch_steps):
            try:
                batch = next(batches)
            except StopIteration:  # epoch = fixed steps (§8.1), cycle the loader
                batches = iter(train_loader)
                batch = next(batches)
            x = batch["x"].to(dev, non_blocking=True)
            y = batch["y"].to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = ce_with_label_smoothing(head(backbone(x)), y, label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()

        train_seconds = time.time() - t0
        val_f1 = _fused_val_macro_f1(backbone, head, val_loader, dev, amp)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / epoch_steps,
            "val_macro_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
            "s_per_step": train_seconds / epoch_steps,
            "epoch_seconds": train_seconds,
        }
        history.append(row)
        logger.info(
            "%s epoch %d/%d: loss %.4f, val macro-F1 %.4f, %.3f s/step",
            cfg["name"], epoch, max_epochs, row["train_loss"], val_f1, row["s_per_step"],
        )

        improved = val_f1 > best_metric
        if improved:
            best_metric = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        state = {
            "backbone": backbone.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": _rng_states(),
            "epoch": epoch, "config": cfg, "seed": seed,
            "best_metric": best_metric, "epochs_no_improve": epochs_no_improve,
            "history": history,
        }
        _atomic_save(state, last_path)
        if improved:
            _atomic_save(state, run_dir / "best.ckpt")
        if epoch in grid_epochs:  # phase-A selection grid (§6-C3)
            _atomic_save(state, run_dir / f"epoch{epoch}.ckpt")

        _write_history_csv(run_dir / "history.csv", history)

        if patience is not None and epochs_no_improve >= patience:
            logger.info("%s: early stop at epoch %d (patience %d)", cfg["name"], epoch, patience)
            break

    return {"run_dir": run_dir, "best_val_macro_f1": best_metric, "history": history}


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    cols = list(history[0])
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in history]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
