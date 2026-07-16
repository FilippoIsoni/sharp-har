"""Day 2/4 — training loop shared by the core runs C0–C4. Ref. §8
(optimization, checkpoint/resume), §6 (per-run configs), §4 (batch
composition), §3 (augmentation), §0.4-0.5 (run artifacts, seed).

Two loss paths, both config-driven (§0.4):
- "ce" (C0/C1/C2): uniform shuffling, "ce" augmentation profile in the
  dataset transform, ActivityHead, in-loop selection on fused val
  macro-F1 (harness functions, so selection ≡ reporting).
- "supcon" (C3/C4 phase A): P×K batch sampler (§4.2), 2 augmented views
  per (window, antenna) built in the DataLoader workers (§3),
  ProjectionHead + supcon_loss, NO early stopping/selection in-loop —
  selection is phase B's linear-probe grid (§6-C3). No gradient
  accumulation, forbidden in phase A (§4.2).

The GRL adversary (C2, C4) follows §6-C2 literally:
L = L_task + β·λ(p)·L_env with the fixed ramp λ(p) (losses.grl_lambda,
β = 1 for C2), plus the mandatory monitoring — AR-set head accuracy on
train batches, per-epoch mean, in history.csv.

Colab free disconnects: every epoch ends with a complete `last.ckpt`
(weights + heads + optimizer + scheduler + GradScaler + epoch + config
+ RNG states) written atomically, and `train_run` resumes from it
automatically. Batch order AND augmentation streams are reproducible
across resumes: shuffle generator, P×K sampler and Augmenter are all
reseeded per epoch as f(seed, epoch) (§8.2), per worker for the
transform, so no sampler/augmenter state needs saving. Changing
num_workers changes the augmentation stream (declared).
"""
from __future__ import annotations

import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info

from .augment import Augmenter, TwoViewAugmenter
from .data import DopplerDataset
from .harness import fuse_windows, macro_f1, predict
from .losses import ce_with_label_smoothing, grl_lambda, supcon_loss
from .models import build_backbone
from .models.heads import ActivityHead, ARSetHead, ProjectionHead
from .sampler import PKSampler
from .utils import epoch_seed as _epoch_seed
from .utils import get_git_hash, get_logger, set_seed, write_json

logger = get_logger(__name__)

GRAD_CLIP_NORM = 1.0  # §8.1 default when the config omits train.grad_clip; essential with GRL
_TRANSFORM_SEED_OFFSET = 7919  # decouple the augmentation stream from the shuffle stream


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


def _reseed_transform_worker(worker_id: int, base_seed: int) -> None:
    """DataLoader worker_init_fn: forked workers would otherwise clone
    the SAME augmentation generator and draw identical streams (§0.5)."""
    dataset = get_worker_info().dataset
    if getattr(dataset, "transform", None) is not None:
        dataset.transform.reseed(base_seed + worker_id + 1)


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
    43/44), "labels" (explicit class list, required for P1/C0),
    "escalation_b", adversary "lambda_max"/"beta"/"ramp_epochs".

    Artifacts under ckpt_dir/<cfg.name>/: last.ckpt (every epoch,
    atomic), best.ckpt (val-selected; CE runs only — phase-A selection
    is phase B's probe grid), epoch{N}.ckpt for the phase-A grid
    (train.checkpoint_epochs), run_meta.json (config + seed + git hash,
    §0.4), history.csv (per-epoch loss/metric/lr/s-per-step — the gates
    read s_per_step from here).
    """
    loss_type = cfg["loss"]["type"]
    adversary_cfg = cfg.get("adversary") or {}
    use_grl = bool(adversary_cfg.get("type"))
    if use_grl:
        assert adversary_cfg["type"] == "grl", f"unknown adversary {adversary_cfg['type']!r}"
        assert adversary_cfg.get("target", "ar_set") == "ar_set", "adversary target must be ar_set (§2.2)"

    if loss_type in ("supcon", "supcon+grl"):
        # §6-C3/C4 phase A: P×K batches, no early stopping/selection in-loop.
        use_grl = use_grl or loss_type == "supcon+grl"
        assert cfg["train"]["sampler"] == "pxk", "SupCon phase A requires the P×K sampler (§4.2)"
        assert cfg["eval"]["patience"] is None, "no early stopping in phase A (§6-C3)"
        is_supcon = True
    elif loss_type == "ce":
        assert cfg["train"]["sampler"] == "uniform", "CE runs use uniform shuffling (§4.1)"
        is_supcon = False
    else:
        raise ValueError(f"unknown loss type {loss_type!r}")

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

    # Augmentation (§3): train only, after standardization, profile by
    # loss path. Reseeded per (epoch, worker) — see the epoch loop.
    # train.augment: false opts out — C0 reproduces the SHARP repo,
    # which does not use the §3 set (§6-C0 "faithful").
    train_ds = datasets["train"]
    if not cfg["train"].get("augment", True):
        assert not is_supcon, "SupCon phase A cannot run without augmentation: the 2 views ARE augmentations (§3)"
    elif is_supcon:
        train_ds.transform = TwoViewAugmenter(seed, train_ds.window_len, train_ds.velocity_bins)
    else:
        train_ds.transform = Augmenter("ce", seed, train_ds.window_len, train_ds.velocity_bins)

    batch_size = cfg["train"]["batch_size"]
    grad_clip = float(cfg["train"].get("grad_clip", GRAD_CLIP_NORM))
    epoch_steps = cfg["train"]["epoch_steps"]
    max_epochs = cfg["train"]["max_epochs"]
    patience = cfg["eval"]["patience"]
    grid_epochs = set(cfg["train"].get("checkpoint_epochs") or [])
    total_steps = max_epochs * epoch_steps

    shuffle_gen = torch.Generator()
    pk_sampler: PKSampler | None = None
    if is_supcon:
        p = datasets["train"].n_att
        k = batch_size // p  # P = n_att, K = ⌊batch/P⌋ (§4.2)
        assert k >= 1, f"batch_size {batch_size} < n_att {p}"
        pk_sampler = PKSampler(datasets["train"], p=p, k=k, seed=seed, num_batches=epoch_steps)
        if p * k != batch_size:
            logger.info("P×K batch is %d windows (batch_size %d, P=%d, K=%d)", p * k, batch_size, p, k)
    else:
        assert len(datasets["train"]) >= batch_size, (
            f"train set ({len(datasets['train'])} samples) smaller than one batch "
            f"({batch_size}) — with drop_last the loader would be empty."
        )

    def make_train_loader(epoch: int) -> DataLoader:
        """Per-epoch loader: reseeds shuffle/sampler/augmenter streams as
        pure functions of (seed, epoch) so an epoch-boundary resume
        reproduces both batch order and augmentation (§8.2)."""
        base = _epoch_seed(seed, epoch)
        if train_ds.transform is not None:
            train_ds.transform.reseed(base + _TRANSFORM_SEED_OFFSET)  # num_workers=0 path
        init = partial(_reseed_transform_worker, base_seed=base + _TRANSFORM_SEED_OFFSET)
        if pk_sampler is not None:
            pk_sampler.set_epoch(epoch)
            return DataLoader(
                train_ds, batch_sampler=pk_sampler, num_workers=num_workers,
                pin_memory=dev.type == "cuda", worker_init_fn=init,
            )
        shuffle_gen.manual_seed(base)
        return DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_gen,
            num_workers=num_workers, pin_memory=dev.type == "cuda", drop_last=True,
            worker_init_fn=init,
        )

    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=dev.type == "cuda",
    )

    backbone = build_backbone(cfg).to(dev)
    feature_dim = getattr(backbone, "feature_dim", cfg["d_enc"])
    head: nn.Module = (
        ProjectionHead(feature_dim) if is_supcon else ActivityHead(feature_dim, cfg["n_att"])
    ).to(dev)
    arset_head: ARSetHead | None = None
    if use_grl:
        arset_head = ARSetHead(feature_dim, datasets["train"].n_arset).to(dev)
    lambda_max = float(adversary_cfg.get("lambda_max") or 1.0)
    beta = float(adversary_cfg.get("beta") or 1.0)
    ramp_epochs = int(adversary_cfg.get("ramp_epochs") or 20)
    tau = float(cfg["loss"].get("tau") or 0.1)

    params = list(backbone.parameters()) + list(head.parameters())
    if arset_head is not None:
        params += list(arset_head.parameters())
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
        if arset_head is not None:
            arset_head.load_state_dict(ckpt["adversary"])
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
        if arset_head is not None:
            arset_head.train()
        train_loader = make_train_loader(epoch)
        batches = iter(train_loader)
        lam = grl_lambda(epoch, lambda_max, ramp_epochs) if use_grl else 0.0
        loss_sum, arset_hits, arset_count, t0 = 0.0, 0, 0, time.time()

        for _step in range(epoch_steps):
            try:
                batch = next(batches)
            except StopIteration:  # epoch = fixed steps (§8.1), cycle the loader
                batches = iter(train_loader)
                batch = next(batches)
            x = batch["x"].to(dev, non_blocking=True)
            y = batch["y"].to(dev, non_blocking=True)

            if is_supcon:
                # (B, 2, 1, T, V) -> (2B, 1, T, V); labels follow the views.
                x = x.flatten(0, 1)
                y = y.repeat_interleave(2)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                feat = backbone(x)
                if is_supcon:
                    loss = supcon_loss(head(feat), y, tau=tau)
                else:
                    loss = ce_with_label_smoothing(head(feat), y, label_smoothing)
                if arset_head is not None:
                    ar = batch["ar_set"].to(dev, non_blocking=True)
                    if is_supcon:
                        ar = ar.repeat_interleave(2)
                    env_logits = arset_head(feat)
                    loss = loss + beta * lam * F.cross_entropy(env_logits, ar)
                    arset_hits += int((env_logits.argmax(dim=1) == ar).sum())
                    arset_count += len(ar)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()

        train_seconds = time.time() - t0
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / epoch_steps,
        }
        if not is_supcon:
            # Phase A has no in-loop selection (§6-C3): the val metric
            # only exists on the CE path.
            row["val_macro_f1"] = _fused_val_macro_f1(backbone, head, val_loader, dev, amp)
        if use_grl:
            # §6-C2 mandatory monitoring: AR-set head accuracy on train
            # batches (per-epoch mean) must fall toward the majority
            # baseline as λ ramps; if it stays high, the GRL isn't biting.
            row["arset_train_acc"] = arset_hits / max(arset_count, 1)
            row["grl_lambda"] = lam
        row["lr"] = scheduler.get_last_lr()[0]
        row["s_per_step"] = train_seconds / epoch_steps
        row["epoch_seconds"] = train_seconds
        history.append(row)
        logger.info(
            "%s epoch %d/%d: loss %.4f%s%s, %.3f s/step",
            cfg["name"], epoch, max_epochs, row["train_loss"],
            f", val macro-F1 {row['val_macro_f1']:.4f}" if "val_macro_f1" in row else "",
            f", arset acc {row['arset_train_acc']:.3f} (λ={lam:.3f})" if use_grl else "",
            row["s_per_step"],
        )

        improved = not is_supcon and row["val_macro_f1"] > best_metric
        if improved:
            best_metric = row["val_macro_f1"]
            epochs_no_improve = 0
        elif not is_supcon:
            epochs_no_improve += 1

        state = {
            "backbone": backbone.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": _rng_states(),
            "epoch": epoch, "config": cfg, "seed": seed,
            "best_metric": best_metric, "epochs_no_improve": epochs_no_improve,
            "history": history,
        }
        if arset_head is not None:
            state["adversary"] = arset_head.state_dict()
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
