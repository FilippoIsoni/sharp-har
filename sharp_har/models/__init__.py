"""Model factory shared by train.py, harness.py and bench.py, so a
checkpoint's config rebuilds the exact same backbone everywhere.
Ref. §5.1-5.2.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .resnet_vb import ResNetVB
from .sharp_like import SharpLike


def build_backbone(cfg: dict[str, Any]) -> nn.Module:
    """Builds the backbone described by a run config (§0.4: the config
    fully describes the run). Raises on unknown names — never guesses."""
    if cfg["backbone"] == "resnet_vb":
        return ResNetVB(d_enc=cfg["d_enc"], escalation_b=cfg.get("escalation_b", False))
    if cfg["backbone"] == "sharp_like":
        return SharpLike(d_enc=cfg["d_enc"])  # d_enc unused: feature size fixed by geometry (§5.1)
    raise ValueError(f"unknown backbone {cfg['backbone']!r}")
