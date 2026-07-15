"""Giorno 1 — inventario dei file Doppler CSI staggiati in locale.
Rif. giorno1_inventory_splits_SPEC.md §1.1, §2.

Produce reports/inventory.csv (una riga per file-stream = coppia
trace/antenna) e reports/name_to_arset.json.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger, write_json

logger = get_logger(__name__)

# Regex parametrico per il naming atteso `{Set}{campagna}_{Attività}_stream_{0-3}.txt`.
# La copia su Drive può usare una nomenclatura propria (es. suffisso S4_S5):
# ispezionare i nomi reali con list_files()/print_naming_patterns() nel
# notebook PRIMA di fidarsi di questo pattern, e correggerlo qui se serve (§2.1).
FILENAME_PATTERN = re.compile(
    r"^(?P<set_num>[A-Za-z]+\d+)(?P<campagna>[a-z])?_(?P<attivita>[A-Za-z]+)_stream_(?P<stream>[0-3])\.txt$"
)

EXPECTED_VELOCITY_BINS = 100
EXPECTED_TIME_STEPS = 340
NAN_EXCLUSION_THRESHOLD = 0.05  # §2.3: policy NaN, stop se le trace escluse superano il 5%

# Metadati (persona, ambiente, hardware) per AR-set, dal paper/dataset SHARP.
# Placeholder deliberato: va popolato con i valori reali del paper prima di
# usare build_inventory in produzione. Finché una entry manca, la colonna
# corrispondente resta "unknown" (§2.2: non inventare).
AR_SET_METADATA: dict[str, dict[str, str]] = {}


def list_files(stage_dir: str | Path, pattern: str = "**/*.txt") -> list[Path]:
    """Elenca ricorsivamente i file staggiati in locale. Prima cella del
    notebook: l'umano ispeziona l'output prima di confermare il regex
    (§2.1 punto 1)."""
    return sorted(Path(stage_dir).glob(pattern))


def print_naming_patterns(files: list[Path], n_examples: int = 30) -> None:
    """Stampa n_examples nomi di esempio e i pattern distinti (cifre
    sostituite da '#') per l'ispezione umana del regex (§2.1 punto 1)."""
    for f in files[:n_examples]:
        print(f.name)
    stems = sorted({re.sub(r"\d+", "#", f.name) for f in files})
    print(f"\n{len(stems)} pattern distinti (cifre sostituite da '#'):")
    for s in stems:
        print(" ", s)


def parse_filename(name: str) -> dict[str, Any]:
    """Estrae set_raw, campagna, attivita, stream da un nome file secondo
    FILENAME_PATTERN. Solleva ValueError se il nome non combacia — il
    chiamante deve loggare ed escludere, non indovinare (§2.1 punto 2)."""
    m = FILENAME_PATTERN.match(name)
    if m is None:
        raise ValueError(f"nome file non riconosciuto dal pattern atteso: {name!r}")
    d = m.groupdict()
    campagna = d["campagna"] or ""
    return {
        "set_raw": f"{d['set_num']}{campagna}",
        "set_num": d["set_num"],
        "campagna": campagna,
        "attivita": d["attivita"],
        "stream": int(d["stream"]),
    }


def build_ar_map(set_raw_values: list[str], out_path: str | Path) -> dict[str, str]:
    """Costruisce la mappa set_raw -> AR-set (AR-1…AR-9) e la salva come
    artefatto in reports/name_to_arset.json (§2.1 punto 3). Non
    hardcodare questa mappa altrove nel codice."""
    ar_map: dict[str, str] = {}
    for set_raw in sorted(set(set_raw_values)):
        m = re.match(r"^[A-Za-z]+(\d+)", set_raw)
        if m is None:
            logger.warning("set_raw non mappabile a un AR-set: %r", set_raw)
            continue
        ar_map[set_raw] = f"AR-{int(m.group(1))}"
    write_json(out_path, ar_map)
    return ar_map


def load_trace(filepath: str | Path) -> np.ndarray:
    """Carica un singolo file-stream Doppler (pickle numpy, shape
    (N_frame, 100)). Solleva se il file non è leggibile — nessun
    fallback silenzioso su dati corrotti."""
    with open(filepath, "rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr)


def build_inventory(
    stage_dir: str | Path,
    out_dir: str | Path = "reports",
    stft_hop_s: float | None = 0.006,
) -> pd.DataFrame:
    """Scansiona i file Doppler staggiati e produce reports/inventory.csv,
    una riga per file-stream (§2.2). Scrive anche name_to_arset.json.

    `stft_hop_s`: se non verificato sui metadati reali, passare None —
    verrà registrato come NaN e va segnalato come blocker (§2.1).
    """
    files = list_files(stage_dir)
    rows: list[dict[str, Any]] = []
    parsed_ok: list[dict[str, Any]] = []
    for fp in files:
        try:
            parsed = parse_filename(fp.name)
        except ValueError as exc:
            logger.warning("scarto file non parsabile: %s", exc)
            continue
        parsed_ok.append({**parsed, "filepath": str(fp)})

    ar_map = build_ar_map(
        [p["set_raw"] for p in parsed_ok], Path(out_dir) / "name_to_arset.json"
    )

    for p in parsed_ok:
        fp = Path(p["filepath"])
        try:
            arr = load_trace(fp)
            shape_0, shape_1 = (arr.shape + (None, None))[:2]
            has_nan = bool(np.isnan(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False
            n_frame = shape_0
            dtype = str(arr.dtype)
        except Exception as exc:  # dati illeggibili: logga, non inventare valori
            logger.error("impossibile leggere %s: %s", fp, exc)
            shape_0 = shape_1 = n_frame = None
            has_nan = None
            dtype = "unreadable"

        ar_set = ar_map.get(p["set_raw"], "unknown")
        meta = AR_SET_METADATA.get(ar_set, {})
        rows.append(
            {
                "filepath": p["filepath"],
                "trace_id": f"{p['set_raw']}_{p['attivita']}",
                "ar_set": ar_set,
                "campagna": p["campagna"],
                "attivita": p["attivita"],
                "persona": meta.get("persona", "unknown"),
                "ambiente": meta.get("ambiente", "unknown"),
                "hardware": meta.get("hardware", "unknown"),
                "stream_antenna": p["stream"],
                "shape_0": shape_0,
                "shape_1": shape_1,
                "dtype": dtype,
                "has_nan": has_nan,
                "n_frame": n_frame,
                "stft_hop_s": stft_hop_s if stft_hop_s is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(out_dir) / "inventory.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("inventory.csv scritto: %d righe (%s)", len(df), out_path)
    return df


def trace_table(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Collassa l'inventario per-stream a una riga per trace_id (unità di
    split, §0.2). Assume ar_set/campagna/attivita coerenti tra i 4 stream
    della stessa trace."""
    agg = (
        inventory_df.groupby("trace_id")
        .agg(
            ar_set=("ar_set", "first"),
            campagna=("campagna", "first"),
            attivita=("attivita", "first"),
            persona=("persona", "first"),
            ambiente=("ambiente", "first"),
            n_streams=("stream_antenna", "nunique"),
        )
        .reset_index()
    )
    return agg


def assert_axes(inventory_df: pd.DataFrame) -> None:
    """Verifica assi (§2.3): shape_1 == 100 (bin di velocità) per tutti i
    file. Solleva AssertionError con l'elenco dei file non conformi —
    blocker esplicito, assi trasposti o ND diverso da investigare."""
    bad = inventory_df[inventory_df["shape_1"] != EXPECTED_VELOCITY_BINS]
    assert bad.empty, (
        f"{len(bad)} file con shape_1 != {EXPECTED_VELOCITY_BINS}: "
        f"{bad['filepath'].tolist()[:10]}"
    )


def assert_coverage(inventory_df: pd.DataFrame, expected_ar_sets: list[str] | None = None) -> set[str]:
    """Verifica copertura AR-1…AR-9 dopo l'unione dei due zip (§2.3).
    Ritorna i set attesi mancanti; il chiamante decide se è un blocker
    (di norma sì)."""
    if expected_ar_sets is None:
        expected_ar_sets = [f"AR-{i}" for i in range(1, 10)]
    present = set(inventory_df["ar_set"].unique())
    missing = set(expected_ar_sets) - present
    if missing:
        logger.error("AR-set mancanti dopo l'unione degli zip: %s", sorted(missing))
    return missing


def build_contingency_table(inventory_df: pd.DataFrame, out_path: str | Path) -> pd.DataFrame:
    """Tabella di contingenza attività × AR-set, conteggio di *trace* (non
    stream) — reports/contingency.csv (§2.3)."""
    traces = trace_table(inventory_df)
    table = pd.crosstab(traces["ar_set"], traces["attivita"])
    table.to_csv(out_path)
    return table


def apply_nan_policy(
    inventory_df: pd.DataFrame, out_dir: str | Path = "reports", threshold: float = NAN_EXCLUSION_THRESHOLD
) -> pd.DataFrame:
    """Esclude le trace con almeno uno stream NaN, le logga in
    reports/excluded_traces.csv con il motivo. Solleva se la frazione di
    trace escluse supera `threshold` (§2.3: stop, non procedere)."""
    traces = trace_table(inventory_df)
    nan_trace_ids = set(inventory_df.loc[inventory_df["has_nan"] == True, "trace_id"])  # noqa: E712
    excluded = traces[traces["trace_id"].isin(nan_trace_ids)].copy()
    excluded["reason"] = "nan_in_stream"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    excluded.to_csv(Path(out_dir) / "excluded_traces.csv", index=False)

    frac = len(excluded) / len(traces) if len(traces) else 0.0
    assert frac <= threshold, (
        f"{frac:.1%} delle trace escluse per NaN, oltre la soglia {threshold:.0%}: "
        "stop, decidere l'imputazione prima di procedere."
    )
    logger.info("trace escluse per NaN: %d/%d (%.1f%%)", len(excluded), len(traces), 100 * frac)
    clean_trace_ids = set(traces["trace_id"]) - nan_trace_ids
    return inventory_df[inventory_df["trace_id"].isin(clean_trace_ids)].copy()


def decide_classes(inventory_df: pd.DataFrame) -> dict[str, Any]:
    """Registra le attività osservate (atteso 7 attività + empty => n_att
    8) e il set di classi del paper per C0 (§2.3, da verificare a mano
    sulla versione finale del dataset)."""
    labels = sorted(inventory_df["attivita"].unique())
    return {
        "n_att": len(labels),
        "labels": labels,
        "c0_paper_set": "TODO: verificare arXiv (5 classi) vs TMC esteso (8 classi)",
    }
