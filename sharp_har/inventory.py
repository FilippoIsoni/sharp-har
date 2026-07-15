"""Day 1 — inventory of the Doppler CSI files staged locally.
Ref. giorno1_inventory_splits_SPEC.md §1.1, §2.

Produces reports/inventory.csv (one row per file-stream = trace/antenna
pair) and reports/name_to_arset.json.
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

# Parametric regex for the expected naming `{Set}{campaign}_{Activity}_stream_{0-3}.txt`
# (e.g. S1a_W_stream_0.txt). Some activities carry a repetition number,
# either appended directly (S4a_C1_stream_0.txt) or underscore-separated
# (S6b_J_0_stream_0.txt, an inconsistency in the raw naming) — both are
# captured in `repetition` and folded into trace_id so that repeated
# recordings of the same activity stay distinct traces (day-1 real-data
# run, confirmed on Drive `DATASET_SHARP`). The Drive copy follows the
# SHARP repo layout (`doppler_traces/S1a … S7a`, v5.1); file names inside
# each set directory must still be inspected with
# list_files()/print_naming_patterns() in the notebook BEFORE trusting
# this pattern — fix it here if needed (§2.1).
FILENAME_PATTERN = re.compile(
    r"^(?P<set_num>[A-Za-z]+\d+)(?P<campagna>[a-z])?_(?P<attivita>[A-Za-z]+)(?:_?(?P<repetition>\d+))?_stream_(?P<stream>[0-3])\.txt$"
)

EXPECTED_VELOCITY_BINS = 100
EXPECTED_TIME_STEPS = 340
NAN_EXCLUSION_THRESHOLD = 0.05  # §2.3: NaN policy, stop if excluded traces exceed 5%

# Metadata per AR-set from Table 1 of the SHARP TMC paper (v5.1 errata:
# the Drive copy holds the TMC dataset, sets S1-S7 == AR-1..AR-7).
# Hardware is identical in every set (Asus RT-AC86U monitor, Netgear X4S
# Tx/Rx link): domains differ by room, monitor position (M1-M4), person
# (P1-P3), day, and LOS/NLOS.
AR_SET_METADATA: dict[str, dict[str, str]] = {
    "AR-1": {"persona": "P1", "ambiente": "bedroom", "monitor": "M1", "direct_path": "LOS"},
    "AR-2": {"persona": "P1", "ambiente": "bedroom", "monitor": "M1", "direct_path": "LOS"},
    "AR-3": {"persona": "P2", "ambiente": "bedroom", "monitor": "M1", "direct_path": "LOS"},
    "AR-4": {"persona": "P1", "ambiente": "bedroom", "monitor": "M2", "direct_path": "NLOS"},
    "AR-5": {"persona": "P2", "ambiente": "bedroom", "monitor": "M2", "direct_path": "NLOS"},
    "AR-6": {"persona": "P1", "ambiente": "living_room", "monitor": "M3", "direct_path": "LOS"},
    "AR-7": {"persona": "P3", "ambiente": "laboratory", "monitor": "M4", "direct_path": "LOS"},
}
MONITOR_HARDWARE = "asus-rt-ac86u"  # same monitor device for all sets

# Campaigns expected per set in the Drive copy (v5.1): uneven a-c.
EXPECTED_SET_CAMPAIGNS: dict[str, list[str]] = {
    "AR-1": ["a", "b", "c"],
    "AR-2": ["a", "b"],
    "AR-3": ["a"],
    "AR-4": ["a", "b"],
    "AR-5": ["a"],
    "AR-6": ["a", "b"],
    "AR-7": ["a"],
}


def list_files(stage_dir: str | Path, pattern: str = "**/*.txt") -> list[Path]:
    """Recursively lists the files staged locally. First cell of the
    notebook: a human inspects the output before confirming the regex
    (§2.1 point 1)."""
    return sorted(Path(stage_dir).glob(pattern))


def print_naming_patterns(files: list[Path], n_examples: int = 30) -> None:
    """Prints n_examples sample names and the distinct patterns (digits
    replaced by '#') to help with human inspection of the regex
    (§2.1 point 1)."""
    for f in files[:n_examples]:
        print(f.name)
    stems = sorted({re.sub(r"\d+", "#", f.name) for f in files})
    print(f"\n{len(stems)} distinct patterns (digits replaced by '#'):")
    for s in stems:
        print(" ", s)


def parse_filename(name: str) -> dict[str, Any]:
    """Extracts set_raw, campagna, attivita, repetition, stream from a
    file name according to FILENAME_PATTERN. Raises ValueError if the
    name doesn't match — the caller must log and exclude, not guess
    (§2.1 point 2)."""
    m = FILENAME_PATTERN.match(name)
    if m is None:
        raise ValueError(f"file name not recognized by the expected pattern: {name!r}")
    d = m.groupdict()
    campagna = d["campagna"] or ""
    return {
        "set_raw": f"{d['set_num']}{campagna}",
        "set_num": d["set_num"],
        "campagna": campagna,
        "attivita": d["attivita"],
        "repetition": d["repetition"] or "",
        "stream": int(d["stream"]),
    }


def build_ar_map(set_raw_values: list[str], out_path: str | Path) -> dict[str, str]:
    """Builds the set_raw -> AR-set (AR-1…AR-7) map and saves it as an
    artifact in reports/name_to_arset.json (§2.1 point 3). Do not
    hardcode this map anywhere else in the code."""
    ar_map: dict[str, str] = {}
    for set_raw in sorted(set(set_raw_values)):
        m = re.match(r"^[A-Za-z]+(\d+)", set_raw)
        if m is None:
            logger.warning("set_raw not mappable to an AR-set: %r", set_raw)
            continue
        ar_map[set_raw] = f"AR-{int(m.group(1))}"
    write_json(out_path, ar_map)
    return ar_map


def load_trace(filepath: str | Path) -> np.ndarray:
    """Loads a single Doppler file-stream (numpy pickle, shape
    (N_frame, 100)). Raises if the file isn't readable — no silent
    fallback on corrupted data."""
    with open(filepath, "rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr)


def build_inventory(
    stage_dir: str | Path,
    out_dir: str | Path = "reports",
    stft_hop_s: float | None = 0.006,
) -> pd.DataFrame:
    """Scans the staged Doppler files and produces reports/inventory.csv,
    one row per file-stream (§2.2). Also writes name_to_arset.json.

    `stft_hop_s`: if not verified against the real metadata, pass None —
    it will be recorded as NaN and must be flagged as a blocker (§2.1).
    """
    files = list_files(stage_dir)
    rows: list[dict[str, Any]] = []
    parsed_ok: list[dict[str, Any]] = []
    for fp in files:
        try:
            parsed = parse_filename(fp.name)
        except ValueError as exc:
            logger.warning("dropping unparsable file: %s", exc)
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
        except Exception as exc:  # unreadable data: log, don't invent values
            logger.error("could not read %s: %s", fp, exc)
            shape_0 = shape_1 = n_frame = None
            has_nan = None
            dtype = "unreadable"

        ar_set = ar_map.get(p["set_raw"], "unknown")
        meta = AR_SET_METADATA.get(ar_set, {})
        rows.append(
            {
                "filepath": p["filepath"],
                "trace_id": f"{p['set_raw']}_{p['attivita']}{p['repetition']}",
                "ar_set": ar_set,
                "campagna": p["campagna"],
                "attivita": p["attivita"],
                "repetition": p["repetition"],
                "persona": meta.get("persona", "unknown"),
                "ambiente": meta.get("ambiente", "unknown"),
                "monitor": meta.get("monitor", "unknown"),
                "direct_path": meta.get("direct_path", "unknown"),
                "hardware": MONITOR_HARDWARE,
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
    logger.info("inventory.csv written: %d rows (%s)", len(df), out_path)
    return df


def trace_table(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Collapses the per-stream inventory to one row per trace_id (the
    split unit, §0.2). Assumes ar_set/campagna/attivita are consistent
    across the 4 streams of the same trace."""
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
    """Verifies the axes (§2.3): shape_1 == 100 (velocity bins) for
    every file. Raises AssertionError with the list of non-conforming
    files — an explicit blocker, transposed axes or a different ND to
    investigate."""
    bad = inventory_df[inventory_df["shape_1"] != EXPECTED_VELOCITY_BINS]
    assert bad.empty, (
        f"{len(bad)} files with shape_1 != {EXPECTED_VELOCITY_BINS}: "
        f"{bad['filepath'].tolist()[:10]}"
    )


def assert_coverage(inventory_df: pd.DataFrame, expected_ar_sets: list[str] | None = None) -> set[str]:
    """Verifies AR-1…AR-7 (= paper sets S1–S7) coverage after merging
    the two zips, including the expected campaigns per set (v5.1, §1.1).
    Returns the missing expected (set, campaign) pairs as strings; the
    caller decides whether it's a blocker (usually yes)."""
    if expected_ar_sets is None:
        expected_ar_sets = sorted(EXPECTED_SET_CAMPAIGNS)
    present_sets = set(inventory_df["ar_set"].unique())
    missing: set[str] = set(expected_ar_sets) - present_sets

    present_campaigns = set(
        map(tuple, inventory_df[["ar_set", "campagna"]].drop_duplicates().values)
    )
    for ar_set in expected_ar_sets:
        for campagna in EXPECTED_SET_CAMPAIGNS.get(ar_set, []):
            if (ar_set, campagna) not in present_campaigns:
                missing.add(f"{ar_set}{campagna}")

    if missing:
        logger.error("missing AR-sets/campaigns after merging the zips: %s", sorted(missing))
    return missing


def build_contingency_table(inventory_df: pd.DataFrame, out_path: str | Path) -> pd.DataFrame:
    """Activity × AR-set contingency table, counting *traces* (not
    streams) — reports/contingency.csv (§2.3)."""
    traces = trace_table(inventory_df)
    table = pd.crosstab(traces["ar_set"], traces["attivita"])
    table.to_csv(out_path)
    return table


def apply_nan_policy(
    inventory_df: pd.DataFrame, out_dir: str | Path = "reports", threshold: float = NAN_EXCLUSION_THRESHOLD
) -> pd.DataFrame:
    """Excludes traces with at least one NaN stream, logging them to
    reports/excluded_traces.csv with the reason. Raises if the fraction
    of excluded traces exceeds `threshold` (§2.3: stop, don't proceed)."""
    traces = trace_table(inventory_df)
    nan_trace_ids = set(inventory_df.loc[inventory_df["has_nan"] == True, "trace_id"])  # noqa: E712
    excluded = traces[traces["trace_id"].isin(nan_trace_ids)].copy()
    excluded["reason"] = "nan_in_stream"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    excluded.to_csv(Path(out_dir) / "excluded_traces.csv", index=False)

    frac = len(excluded) / len(traces) if len(traces) else 0.0
    assert frac <= threshold, (
        f"{frac:.1%} of traces excluded for NaN, above the {threshold:.0%} threshold: "
        "stop, decide on imputation before proceeding."
    )
    logger.info("traces excluded for NaN: %d/%d (%.1f%%)", len(excluded), len(traces), 100 * frac)
    clean_trace_ids = set(traces["trace_id"]) - nan_trace_ids
    return inventory_df[inventory_df["trace_id"].isin(clean_trace_ids)].copy()


# Activity codes that are not HAR activities but slipped through the naming
# regex (day-1 finding, confirmed on real data 2026-07-15): "LOS" is a
# single-occurrence (AR-5 only) near-static recording — mean/min/max
# essentially identical to "E" (empty) traces in the same set, and
# std-over-time (~0.004) an order of magnitude below any real activity
# (~0.03+) and close to "E" (~0.001). Consistent with a line-of-sight
# calibration/reference recording, not a subject performing an activity.
NON_ACTIVITY_LABELS = {"LOS"}


def exclude_non_activities(
    inventory_df: pd.DataFrame, out_dir: str | Path = "reports"
) -> pd.DataFrame:
    """Excludes traces whose activity code is in NON_ACTIVITY_LABELS
    (calibration/reference recordings that match the file-naming regex
    but are not HAR activities). Logs them to reports/excluded_traces.csv
    with reason 'non_activity_reference', appended to whatever
    apply_nan_policy already wrote there — call this after
    apply_nan_policy in the notebook."""
    traces = trace_table(inventory_df)
    excluded = traces[traces["attivita"].isin(NON_ACTIVITY_LABELS)].copy()
    if excluded.empty:
        return inventory_df.copy()

    excluded["reason"] = "non_activity_reference"
    out_path = Path(out_dir) / "excluded_traces.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    logger.info(
        "excluded %d non-activity trace(s) %s: %s",
        len(excluded), sorted(NON_ACTIVITY_LABELS), sorted(excluded["trace_id"]),
    )

    clean_trace_ids = set(traces["trace_id"]) - set(excluded["trace_id"])
    return inventory_df[inventory_df["trace_id"].isin(clean_trace_ids)].copy()


# C0 compares against the TMC paper's core tables: 5 classes (4 activities
# + empty), see v5.1-5. The exact label letters used in the file names are
# confirmed on day 1 against the inventory.
C0_PAPER_CLASSES = ["walking", "running", "jumping", "sitting", "empty"]


def decide_classes(inventory_df: pd.DataFrame) -> dict[str, Any]:
    """Records the observed activities (7 activities + empty expected =>
    n_att 8) and the paper's 5-class set for C0 (v5.1-5). The mapping
    from observed label letters to the C0 class names is confirmed by
    hand on day 1."""
    labels = sorted(inventory_df["attivita"].unique())
    return {
        "n_att": len(labels),
        "labels": labels,
        "c0_paper_set": C0_PAPER_CLASSES,
    }
