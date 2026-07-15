"""Utility di base condivise da tutto il package: seed, I/O yaml/json,
git hash per la riproducibilità, logging. Rif. SETUP_REPO_SPEC.md §1,
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
    """Fissa il seed per ogni sorgente di stocasticità nota (§0.5: seed
    unico = 42 per ogni scelta stocastica del progetto).

    Imposta random e numpy sempre; torch/cuda solo se torch è
    installato, per non introdurre una dipendenza hard nei moduli del
    giorno 1 (puro I/O + logica di split, niente training).
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


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Legge un file YAML e ne ritorna il contenuto come dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Scrive un dict come YAML, creando le cartelle intermedie se
    necessario."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: str | Path) -> Any:
    """Legge un file JSON e ne ritorna il contenuto."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Scrive un oggetto come JSON, creando le cartelle intermedie se
    necessario. Usato per gli artefatti congelati (splits/*.json) e i
    report (reports/*.json) — mai per dati o checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_git_hash(repo_dir: str | Path = ".") -> str:
    """Ritorna l'hash del commit HEAD corrente per la riproducibilità
    (§0.4: config YAML, seed, git hash). Ritorna "unknown" se non ci si
    trova in un repo git o se git non è disponibile — non deve mai
    sollevare eccezione."""
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
    """Ritorna un logger su console con formato timestamp/livello/nome,
    senza handler duplicati se richiamato più volte con lo stesso name."""
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
