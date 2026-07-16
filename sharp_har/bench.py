"""Day 3 — phase-A memory/throughput measurement: one full-batch
forward+backward (512 SupCon views) with the REAL P×K sampler, real
augmentation and the real optimizer step. Ref. §10.1 (day 3), §4.2, §8.

Replaces the day-2 gate's declared 2×-per-step approximation
(reports/gate_day2.json) with a measured number BEFORE any phase-A run:
projected phase A > 8 h → escalation §5.2 first (§10.1). Also the §5.2
memory check ("checkpointing probably not needed — verify on day 3"):
peak allocated/reserved memory is part of the report.

The 2 SupCon views are built per sample in the step loop (augment §3,
"supcon_view" profile), so the measured s/step INCLUDES augmentation
cost on the main process — an honest upper bound for the day-4 loop.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import augment
from .data import DopplerDataset
from .losses import supcon_loss
from .models import build_backbone
from .models.heads import ProjectionHead
from .sampler import PKSampler
from .utils import epoch_seed, get_git_hash, get_logger, set_seed

logger = get_logger(__name__)

PHASE_A_MAX_HOURS = 8.0  # pre-committed go/no-go rule (§10.1)


def phase_a_step_bench(
    cfg: dict[str, Any],
    stage_dir: str | Path,
    *,
    repo_dir: str | Path = ".",
    warmup_steps: int = 5,
    measure_steps: int = 20,
    num_workers: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    """Measures the real phase-A step on `cfg` (a c3_supcon.yaml copy —
    never edit the file): P×K batch of train.batch_size windows -> 2
    augmented views each -> encoder + projection head -> supcon_loss ->
    backward + clipped AdamW step under AMP (§4.2, §8.1). Returns the
    gate dict (s/step, peak memory, projected phase-A hours vs the 8 h
    rule, sampler composition stats) for reports/gate_day3_phase_a.json.

    No gradient accumulation anywhere — forbidden in phase A (§4.2)."""
    cfg = copy.deepcopy(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.backends.cudnn.benchmark = True  # §8.1
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["train"]["amp"]) and dev.type == "cuda"
    repo_dir = Path(repo_dir)

    dataset = DopplerDataset(
        repo_dir / cfg["split_file"], "train", stage_dir,
        inventory_csv=repo_dir / "reports" / "inventory.csv",
        arset_map=repo_dir / "reports" / "name_to_arset.json",
        labels=cfg.get("labels"),
    )
    p = dataset.n_att
    k = cfg["train"]["batch_size"] // p  # P = n_att, K = ⌊256/P⌋ = 32 (§4.2)
    n_steps = warmup_steps + measure_steps
    sampler = PKSampler(dataset, p=p, k=k, seed=seed, num_batches=n_steps, epoch=1)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=dev.type == "cuda"
    )

    aug_cfg = augment.augment_cfg("supcon_view", dataset.window_len, dataset.velocity_bins)
    rng = np.random.default_rng(epoch_seed(seed, 1))

    backbone = build_backbone(cfg).to(dev)
    head = ProjectionHead(cfg["d_enc"]).to(dev)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["wd"])
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    tau = float(cfg["loss"]["tau"])
    backbone.train()
    head.train()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = 0.0
    for step, batch in enumerate(loader):
        x, y = batch["x"], batch["y"]
        # 2 independent views per (window, antenna) sample (§3): the
        # SupCon batch is 2 × batch_size views with duplicated labels.
        views = torch.cat([
            torch.stack([augment.apply(x[i], aug_cfg, rng) for i in range(len(x))]),
            torch.stack([augment.apply(x[i], aug_cfg, rng) for i in range(len(x))]),
        ]).to(dev, non_blocking=True)
        labels = torch.cat([y, y]).to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, enabled=amp):
            loss = supcon_loss(head(backbone(views)), labels, tau=tau)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == warmup_steps - 1:  # measurement starts AFTER warmup
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    s_per_step = (time.time() - t0) / measure_steps

    horizon_steps = cfg["train"]["max_epochs"] * cfg["train"]["epoch_steps"]
    projected_hours = horizon_steps * s_per_step / 3600
    result: dict[str, Any] = {
        "git_hash": get_git_hash(repo_dir),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev),
        "torch": torch.__version__,
        "config_name": cfg.get("name"),
        "p": p, "k": k,
        "batch_windows": p * k,
        "batch_views": 2 * p * k,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "s_per_step": s_per_step,
        "augmentation_in_step": True,  # declared: aug cost included in s/step
        "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3) if dev.type == "cuda" else None,
        "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2**30, 3) if dev.type == "cuda" else None,
        "projected_phase_a_hours": round(projected_hours, 2),
        "rule_phase_a_max_hours": PHASE_A_MAX_HOURS,
        "phase_a_pass": projected_hours <= PHASE_A_MAX_HOURS,
        "sampler_epoch_stats": sampler.last_epoch_stats,
    }
    logger.info(
        "phase-A step: %.3f s/step (%d views) -> projected %.2f h (rule <= %.0f h) -> %s",
        s_per_step, 2 * p * k, projected_hours, PHASE_A_MAX_HOURS,
        "PASS" if result["phase_a_pass"] else "FAIL -> escalation §5.2 BEFORE any phase-A run",
    )
    return result
