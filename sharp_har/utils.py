"""Base utilities shared across the whole package: seed, yaml/json I/O,
git hash for reproducibility, logging. Ref. SETUP_REPO_SPEC.md §1,
giorno1_inventory_splits_SPEC.md §0.5.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any

import yaml


def set_seed(seed: int = 42) -> None:
    """Fixes the seed for every known source of stochasticity (§0.5: a
    single seed = 42 for every stochastic choice in the project).

    Always sets random and numpy; torch/cuda only if torch is installed,
    so day-1 modules (pure I/O + split logic, no training) don't pick up
    a hard dependency.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def epoch_seed(seed: int, epoch: int) -> int:
    """Deterministic per-epoch reseed, seed_epoch = f(seed, epoch)
    (§4.2, §8.2): shared by the train-loop shuffle generator and the P×K
    sampler, so resuming from an epoch boundary reproduces the batch
    sequence without persisting any sampler state."""
    return seed * 100_003 + epoch


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Reads a YAML file and returns its content as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Writes a dict as YAML, creating intermediate directories if
    needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: str | Path) -> Any:
    """Reads a JSON file and returns its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Writes an object as JSON, creating intermediate directories if
    needed. Used for frozen artifacts (splits/*.json) and reports
    (reports/*.json) — never for data or checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_git_hash(repo_dir: str | Path = ".") -> str:
    """Returns the current HEAD commit hash for reproducibility (§0.4:
    YAML config, seed, git hash). Returns "unknown" if not inside a git
    repo or if git isn't available — must never raise."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a console logger with a timestamp/level/name format, with
    no duplicate handlers if called again with the same name."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
