"""§0.7 single final-test-session orchestration: the frozen pre-registered
row list turned into shared-harness checkpoint -> CSV calls. Ref. §0.7, §9.

Thin-notebook rule (CLAUDE.md architecture): notebook 05 declares the frozen
ROWS table and the run mode and displays the results; this module owns the
logic that reads the frozen artifacts and drives the four evaluation kinds,
the same reason harness/probe/diagnostics logic lives in the package and not
inline in a notebook (the once-only test access is exactly the code that must
be diffable and unit-testable, not retyped per session).

Nothing here selects a checkpoint or edits a split — selection happened on
val, upstream. Every path goes through the audited harness entry points
(evaluate / evaluate_c0 / cache_features / evaluate_features), so each
set_name="test" call appends to test_invocations.jsonl BEFORE the data is
touched (§0.7). Four row kinds:

- ``e2e``   -> evaluate (optional adapt_bn = the C1+AdaBN row, §9).
- ``c0``    -> evaluate_c0 (SHARP-repo decision fusion, P1, 5-class, §2.1).
- ``probe`` -> cache_features then evaluate_features with a persisted linear
  head; the C3 row reads its checkpoint+head from phase_b_selection.json.
- ``t3a``   -> cache raw / post-AdaBN features, build the §9 T3A prototype
  head from C1's classifier weights, then evaluate_features.

AdaBN rows pin batch_size to harness.ADAPT_BN_BATCH explicitly (§9: the
adaptation batch is fixed a priori and the harness asserts it — pinning it
here keeps the row correct independently of the harness default).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .harness import (
    ADAPT_BN_BATCH,
    cache_features,
    evaluate,
    evaluate_c0,
    evaluate_features,
)
from .transductive import head_weight_from_checkpoint, t3a_head


def required_paths(row: dict[str, Any], ckpt_root: Path) -> list[Path]:
    """Every Drive artifact a row needs, for the readiness assert (§0.7): a
    missing one must stop the session before any evaluation runs.

    The C3 ``from_phaseb`` row rebuilds its ``epoch{e}.ckpt`` /
    ``probe_head_epoch{e}.npz`` from phase_b_selection.json's
    ``selected_epoch``: the stored ``selected_checkpoint``/``selected_head``
    are ABSOLUTE paths from the selecting session and may not resolve under a
    different mount/shortcut, while the filenames are deterministic."""
    d = Path(ckpt_root) / row["folder"]
    if row["kind"] == "probe" and row.get("from_phaseb"):
        sel = d / "phase_b_selection.json"
        if not sel.exists():
            return [sel]
        e = json.loads(sel.read_text())["selected_epoch"]
        return [d / f"epoch{e}.ckpt", d / f"probe_head_epoch{e}.npz"]
    if row["kind"] == "probe":
        return [d / row["ckpt"], d / row["head"]]
    return [d / "best.ckpt"]  # e2e, c0, t3a all need best.ckpt


def readiness_missing(rows: list[dict[str, Any]], ckpt_root: Path) -> list[tuple[str, str]]:
    """(key, missing_path) for every artifact absent on Drive — empty means
    ready. The assert adapts to whatever the caller declares in ``rows``."""
    missing: list[tuple[str, str]] = []
    for r in rows:
        for p in required_paths(r, Path(ckpt_root)):
            if not Path(p).exists():
                missing.append((r["key"], str(p)))
    return missing


def finalize_csvs(
    key: str, out_dir: Path, set_name: str, fusion: str, final_dir: Path
) -> None:
    """Copy the harness CSVs to ``final_dir`` under notebook 06's naming
    contract ``<key>_<set>_<fusion>_<kind>.csv``. The row key replaces the
    checkpoint stem so streams never collide (every e2e checkpoint is
    ``best.ckpt``) — the exact miskeying notebook 06 warns against."""
    final = Path(final_dir)
    final.mkdir(parents=True, exist_ok=True)
    for csv in Path(out_dir).glob("*.csv"):
        kind = csv.stem.rsplit("_", 1)[1]  # windows | metrics | confusion
        (final / f"{key}_{set_name}_{fusion}_{kind}.csv").write_bytes(csv.read_bytes())


def row_accuracy(out_dir: Path) -> tuple[float, int, int]:
    """Fused accuracy + trace/window counts from the row's windows CSV — the
    dry-run self-verification (near the recorded val macro-F1s, no crash =
    the path is sound; accuracy, not macro-F1, so not identical)."""
    w = next(Path(out_dir).glob("*_windows.csv"))
    df = pd.read_csv(w)
    return float((df.y_pred == df.y_true).mean()), int(df.trace_id.nunique()), len(df)


def run_row(
    row: dict[str, Any],
    set_name: str,
    *,
    session_dir: Path,
    ckpt_root: Path,
    stage_dir: Path,
    repo_dir: Path,
    final_dir: Path,
) -> dict[str, Any]:
    """Evaluate one §0.7 row through the shared harness and copy its CSVs to
    ``final_dir``. Returns ``out_dir``, ``fusion`` and ``t3a_audit`` — the
    §9 T3A audit arrays (``n_supports`` / ``pseudo_label_counts``, per class)
    for a ``t3a`` row, ``None`` otherwise; the caller prints them (the harness
    ignores them). AdaBN rows pin ``batch_size=ADAPT_BN_BATCH`` (§9)."""
    d = Path(ckpt_root) / row["folder"]
    out = Path(session_dir) / row["key"]
    out.mkdir(parents=True, exist_ok=True)
    kind = row["kind"]
    fusion = "sharp_c0" if kind == "c0" else "softmax_avg"
    adapt = row.get("adapt_bn", False)
    bn_kw = {"batch_size": ADAPT_BN_BATCH} if adapt else {}  # §9: AdaBN batch fixed a priori
    t3a_audit: dict[str, Any] | None = None

    if kind == "c0":
        evaluate_c0(d / "best.ckpt", row["split"], set_name,
                    stage_dir=stage_dir, out_dir=out, repo_dir=repo_dir)

    elif kind == "e2e":
        evaluate(d / "best.ckpt", row["split"], set_name, stage_dir=stage_dir,
                 out_dir=out, repo_dir=repo_dir, adapt_bn=adapt, **bn_kw)

    elif kind == "probe":
        if row.get("from_phaseb"):
            e = json.loads((d / "phase_b_selection.json").read_text())["selected_epoch"]
            ckpt, head_path = d / f"epoch{e}.ckpt", d / f"probe_head_epoch{e}.npz"
        else:
            ckpt, head_path = d / row["ckpt"], d / row["head"]
        feat = cache_features(ckpt, row["split"], set_name, stage_dir=stage_dir,
                              repo_dir=repo_dir, out_path=out / f"features_{set_name}.npz")
        hz = np.load(head_path, allow_pickle=False)
        head = {"weight": hz["weight"], "bias": hz["bias"]}
        evaluate_features(feat, head, set_name, out_dir=out, run_name=row["key"], repo_dir=repo_dir)

    elif kind == "t3a":
        ckpt = d / "best.ckpt"
        feat = cache_features(ckpt, row["split"], set_name, stage_dir=stage_dir,
                              repo_dir=repo_dir, adapt_bn=adapt,
                              out_path=out / f"features_{set_name}.npz", **bn_kw)
        raw = np.load(feat, allow_pickle=False)["features"]
        full = t3a_head(head_weight_from_checkpoint(ckpt), raw)  # prototypes from C1's head
        t3a_audit = {"n_supports": full["n_supports"],
                     "pseudo_label_counts": full["pseudo_label_counts"]}
        head = {"weight": full["weight"], "bias": full["bias"]}
        evaluate_features(feat, head, set_name, out_dir=out, run_name=row["key"], repo_dir=repo_dir)

    else:
        raise ValueError(f"unknown row kind {kind!r}")

    finalize_csvs(row["key"], out, set_name, fusion, final_dir)
    return {"out_dir": out, "fusion": fusion, "t3a_audit": t3a_audit}
