"""Day 3 — shared evaluation harness: the single checkpoint -> CSV
interface for every evaluation stream. Ref. §0.4, §0.7, §2.1, §9.

One code path for all streams: forward -> per-sample softmax -> antenna
fusion per (trace, window) -> accuracy + macro-F1 per AR-set (never
aggregate-only, §9) -> CSVs. Two fusion methods:

- "softmax_avg" (§1.3, §9): mean of the antenna softmax, then argmax —
  C1-C4, val and test, and the in-loop selection metric of train.py
  (same functions, so selection and reporting cannot drift).
- "sharp_c0" (§2.1, TMC §4.2): the SHARP repo's decision fusion,
  verified against `francescamen/SHARP` `Python_code/CSI_network.py`:
  majority vote over the 4 per-antenna argmax labels; ties (equal top-2
  counts, or more than 2 distinct labels) fall back to the argmax of the
  summed softmax. C0 only, via evaluate_c0().

Rule §0.7: the test set is evaluated once, at the end, with the
val-selected checkpoint only — and EVERY access to the test set (eval,
C0 wrapper, feature caching) appends to `test_invocations.jsonl` in the
output directory BEFORE touching the data. No access path outside the
audit trail.

Also here (§5.3): cache_features — frozen encoder, features extracted
once WITHOUT augmentation (declared, §5.3 note i) and cached as .npz for
the linear probes — and evaluate_features, the probe-side twin of
evaluate() for phase B / C1-lin / C2-lin test runs.

AdaBN (§9 v5.2 transductive rows, spec pinned 2026-07-18) is a flag on
the two checkpoint-loading entry points, not a separate path:
evaluate(..., adapt_bn=True) is the C1+AdaBN row, cache_features(...,
adapt_bn=True) is the post-AdaBN cache the C1+AdaBN+T3A row consumes
(composition order AdaBN -> T3A). The flag re-estimates BN statistics
on the SAME set it then evaluates/caches (unlabeled inputs only),
enters the §0.7 audit log and the CSV meta, and switches the output
stems ("<ckpt>_adabn_...") so the plain C1 artifacts are never
overwritten. T3A itself lives in transductive.py (numpy on caches).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DopplerDataset
from .models import build_backbone
from .models.heads import ActivityHead
from .utils import get_git_hash, get_logger

logger = get_logger(__name__)

TEST_LOG_FILENAME = "test_invocations.jsonl"
FUSION_METHODS = ("softmax_avg", "sharp_c0")
ADAPT_BN_BATCH = 256  # §9 AdaBN spec: adaptation batch size fixed a priori

Fusion = Literal["softmax_avg", "sharp_c0"]


# ---------------------------------------------------------------- metrics

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 averaged ONLY over classes present in the ground truth
    of the evaluated set (§9) — same definition in val and test; the
    absent classes are listed in the metrics CSV."""
    scores = []
    for c in np.unique(y_true):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class (§9: confusion
    matrix for the primary rotation)."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


# ---------------------------------------------------------------- forward

@torch.no_grad()
def predict(
    backbone: nn.Module, head: nn.Module, loader: DataLoader,
    device: torch.device, amp: bool,
) -> dict[str, np.ndarray]:
    """Single forward pass over a loader. Returns per-sample arrays:
    `probs` (N, C) float32 softmax, `y`, `trace_id` (object), `antenna`,
    `window_start`. Every aggregation (fusion or per-antenna appendix,
    §6-ablation 1) reuses this one forward."""
    backbone.eval()
    head.eval()
    probs, ys, tids, ants, starts = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            p = torch.softmax(head(backbone(x)), dim=1)
        probs.append(p.float().cpu().numpy())
        ys.append(batch["y"].numpy())
        tids.extend(batch["trace_id"])
        ants.append(batch["antenna"].numpy())
        starts.append(batch["window_start"].numpy())
    return {
        "probs": np.concatenate(probs),
        "y": np.concatenate(ys),
        "trace_id": np.array(tids, dtype=object),
        "antenna": np.concatenate(ants),
        "window_start": np.concatenate(starts),
    }


def _sharp_c0_vote(win_probs: np.ndarray) -> int:
    """SHARP-repo antenna fusion for ONE window (TMC §4.2; CSI_network.py):
    majority vote over the per-antenna argmax; equal top-2 counts or >2
    distinct labels fall back to argmax of the summed softmax."""
    votes = win_probs.argmax(axis=1)
    lab_unique, count = np.unique(votes, return_counts=True)
    if lab_unique.shape[0] == 1:
        return int(lab_unique[0])
    lab_merge_max = int(win_probs.sum(axis=0).argmax())
    order = np.flip(np.argsort(count))
    if count[order[0]] == count[order[1]] or lab_unique.shape[0] > 2:
        return lab_merge_max
    return int(lab_unique[order[0]])


def fuse_windows(
    probs: np.ndarray, y: np.ndarray, trace_ids: np.ndarray, window_starts: np.ndarray,
    method: Fusion = "softmax_avg",
) -> dict[str, np.ndarray]:
    """Antenna fusion: groups per-sample predictions by (trace_id,
    window_start) — the evaluation unit is the disjoint window, §1.2/§9 —
    and fuses over the antennas with `method`. Returns per-window arrays
    `trace_id`, `window_start`, `y_true`, `y_pred`, and `probs` (mean
    softmax, kept for the windows CSV under both methods)."""
    assert method in FUSION_METHODS, f"unknown fusion {method!r}, expected {FUSION_METHODS}"
    groups: dict[tuple[Any, int], list[int]] = {}
    for i, key in enumerate(zip(trace_ids, window_starts)):
        groups.setdefault((key[0], int(key[1])), []).append(i)

    keys = sorted(groups)
    tids, starts, y_true, y_pred, fused = [], [], [], [], []
    for tid, ws in keys:
        idx = groups[(tid, ws)]
        assert len(set(y[idx])) == 1, f"inconsistent labels within window ({tid}, {ws})"
        win_probs = probs[idx]
        mean_probs = win_probs.mean(axis=0)
        pred = int(mean_probs.argmax()) if method == "softmax_avg" else _sharp_c0_vote(win_probs)
        tids.append(tid)
        starts.append(ws)
        y_true.append(int(y[idx[0]]))
        y_pred.append(pred)
        fused.append(mean_probs)
    return {
        "trace_id": np.array(tids, dtype=object),
        "window_start": np.array(starts),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "probs": np.stack(fused),
    }


def _adapt_bn(backbone: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> int:
    """AdaBN re-estimation (§9, spec pinned 2026-07-18): reset every
    BatchNorm running statistic, then ONE full no-grad pass over the
    loader with only the _BatchNorm modules in train mode and
    momentum=None — the cumulative-mean AdaBN estimator; the default
    exponential momentum would weight recent batches arbitrarily by
    order. Weights are untouched; every non-BN module stays in eval
    (deterministic — no dropout-style stochasticity can leak in). The
    loader must be the deterministic eval loader (shuffle=False -> the
    dataset's sorted-trace, ascending antenna/temporal-window order, the
    declared order) at ADAPT_BN_BATCH. Returns the number of adapted BN
    modules; the caller re-uses the same loader for the actual
    eval/caching pass, statistics frozen (backbone back in full eval).
    """
    bns = [m for m in backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert bns, "AdaBN on a backbone without BatchNorm modules — nothing to adapt (wrong model?)"
    backbone.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over the full pass
        m.train()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                backbone(x)
    backbone.eval()
    logger.info(
        "AdaBN: re-estimated running stats of %d BatchNorm modules over %d samples",
        len(bns), len(loader.dataset),
    )
    return len(bns)


# ------------------------------------------------------------- audit trail

def _log_test_invocation(out_dir: Path, entry: dict[str, Any]) -> None:
    """Appends one JSONL line per test-set access (§0.7), BEFORE the data
    is touched — the audit trail exists even if the eval then crashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with open(out_dir / TEST_LOG_FILENAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.warning("TEST-SET ACCESS logged to %s: %s", out_dir / TEST_LOG_FILENAME, entry)


# ---------------------------------------------------------------- reports

def _write_reports(
    res: dict[str, np.ndarray], arset_of_trace: dict[str, str], labels: list[str],
    out_dir: Path, stem: str, set_name: str, fusion: Fusion, meta: dict[str, Any],
) -> Path:
    """Windows CSV + metrics CSV (+ confusion CSV) from per-sample
    predictions. Metrics per AR-set AND aggregate (§9: never
    aggregate-only), fused rows plus the free per-antenna appendix rows
    (§6-ablation 1: same forward, two aggregations). Returns the metrics
    CSV path — the `checkpoint -> CSV` contract of §0.4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fused = fuse_windows(res["probs"], res["y"], res["trace_id"], res["window_start"], fusion)
    arsets = np.array([arset_of_trace[t] for t in fused["trace_id"]], dtype=object)

    win_df = pd.DataFrame({
        "trace_id": fused["trace_id"], "window_start": fused["window_start"],
        "ar_set": arsets, "y_true": fused["y_true"], "y_pred": fused["y_pred"],
        **{f"p_{lab}": fused["probs"][:, i] for i, lab in enumerate(labels)},
    })
    win_path = out_dir / f"{stem}_windows.csv"
    win_df.to_csv(win_path, index=False)

    def rows_for(y_true: np.ndarray, y_pred: np.ndarray, tids: np.ndarray, aggregation: str) -> list[dict[str, Any]]:
        sample_arsets = np.array([arset_of_trace[t] for t in tids], dtype=object)
        out = []
        for scope in ["ALL", *sorted(set(sample_arsets))]:
            m = np.ones(len(y_true), bool) if scope == "ALL" else sample_arsets == scope
            absent = [labels[c] for c in range(len(labels)) if c not in set(y_true[m].tolist())]
            out.append({
                "set_name": set_name, "aggregation": aggregation, "ar_set": scope,
                "n_windows": int(m.sum()), "n_traces": len(set(tids[m].tolist())),
                "accuracy": float((y_pred[m] == y_true[m]).mean()),
                "macro_f1": macro_f1(y_true[m], y_pred[m]),
                "absent_classes": ";".join(absent), **meta,
            })
        return out

    rows = rows_for(fused["y_true"], fused["y_pred"], fused["trace_id"], f"fused_{fusion}")
    for antenna in sorted(set(res["antenna"].tolist())):  # per-antenna appendix (§6-ablation 1)
        m = res["antenna"] == antenna
        rows += rows_for(res["y"][m], res["probs"][m].argmax(axis=1), res["trace_id"][m], f"antenna_{antenna}")

    metrics_path = out_dir / f"{stem}_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    cm = confusion_matrix(fused["y_true"], fused["y_pred"], len(labels))
    pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=labels).to_csv(
        out_dir / f"{stem}_confusion.csv"
    )

    head = rows[0]
    logger.info(
        "%s [%s, %s]: accuracy %.4f, macro-F1 %.4f (%d windows) -> %s",
        stem, set_name, rows[0]["aggregation"], head["accuracy"], head["macro_f1"],
        head["n_windows"], metrics_path,
    )
    return metrics_path


# ------------------------------------------------------------ entry points

def _load_end_to_end(checkpoint: str | Path) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg["loss"]["type"] == "ce", (
        f"checkpoint {checkpoint} was trained with loss {cfg['loss']['type']!r}: SupCon "
        "encoders are evaluated via linear probe (§6-C3/C4 phase B) — use "
        "cache_features + probe.linear_probe + evaluate_features, not evaluate()."
    )
    backbone = build_backbone(cfg)
    backbone.load_state_dict(ckpt["backbone"])
    # Same head sizing as train.py: sharp_like fixes the feature size by
    # geometry (backbone.feature_dim = 25500), d_enc only applies to V-B.
    head = ActivityHead(getattr(backbone, "feature_dim", cfg["d_enc"]), cfg["n_att"])
    head.load_state_dict(ckpt["head"])
    return backbone, head, cfg


def _make_loader(
    split_file: str | Path, set_name: str, stage_dir: str | Path, repo_dir: str | Path,
    labels: list[str] | None, batch_size: int, num_workers: int, pin: bool,
) -> tuple[DopplerDataset, DataLoader]:
    repo_dir = Path(repo_dir)
    dataset = DopplerDataset(
        split_file, set_name, stage_dir,
        inventory_csv=repo_dir / "reports" / "inventory.csv",
        arset_map=repo_dir / "reports" / "name_to_arset.json",
        labels=labels,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return dataset, loader


def evaluate(
    checkpoint: str | Path, split_file: str | Path, set_name: Literal["val", "test"],
    *,
    stage_dir: str | Path,
    out_dir: str | Path | None = None,
    repo_dir: str | Path = ".",
    fusion: Fusion = "softmax_avg",
    labels: list[str] | None = None,
    batch_size: int = 256,
    num_workers: int = 2,
    device: str | None = None,
    amp: bool = True,
    adapt_bn: bool = False,
) -> Path:
    """Evaluates an end-to-end (CE) checkpoint on one set of a frozen
    split and writes windows/metrics/confusion CSVs next to the
    checkpoint (or in `out_dir`). Antenna fusion per §1.3/§9 —
    "sharp_c0" only via evaluate_c0(). Returns the metrics CSV path.

    adapt_bn=True is the AdaBN row (§9 transductive spec): BN running
    statistics re-estimated on this same set's unlabeled inputs via
    _adapt_bn BEFORE the prediction pass — weights untouched, output
    stem "<ckpt>_adabn_<set>_<fusion>" so the plain row's CSVs are
    never overwritten. Pre-registered use: C1 on test, in the single
    §0.7 session (a val run would only be a code smoke check).

    Every set_name="test" invocation is logged to test_invocations.jsonl
    BEFORE the data is read (§0.7): one final test eval per stream, with
    the val-selected checkpoint only — the log is the proof.
    Ref. §0.4, §0.7, §1.3, §9.
    """
    checkpoint = Path(checkpoint)
    out = Path(out_dir) if out_dir is not None else checkpoint.parent
    backbone, head, cfg = _load_end_to_end(checkpoint)
    labels = labels if labels is not None else cfg.get("labels")
    if adapt_bn:
        assert batch_size == ADAPT_BN_BATCH, (
            f"AdaBN batch size is fixed a priori at {ADAPT_BN_BATCH} (§9), got {batch_size}"
        )

    if set_name == "test":
        _log_test_invocation(out, {
            "kind": "evaluate", "checkpoint": str(checkpoint), "split_file": str(split_file),
            "set_name": set_name, "fusion": fusion, "adapt_bn": adapt_bn,
            "config_name": cfg.get("name"), "git_hash": get_git_hash(repo_dir),
        })

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone.to(dev)
    head.to(dev)
    dataset, loader = _make_loader(
        split_file, set_name, stage_dir, repo_dir, labels, batch_size, num_workers, dev.type == "cuda"
    )
    if adapt_bn:
        _adapt_bn(backbone, loader, dev, amp)
    res = predict(backbone, head, loader, dev, amp)
    arset_of_trace = {t: info["ar_set"] for t, info in dataset.trace_info.items()}
    stem = f"{checkpoint.stem}{'_adabn' if adapt_bn else ''}_{set_name}_{fusion}"
    meta = {
        "checkpoint": str(checkpoint), "git_hash": get_git_hash(repo_dir),
        "fusion": fusion, "adapt_bn": adapt_bn,
    }
    return _write_reports(res, arset_of_trace, dataset.labels, out, stem, set_name, fusion, meta)


def evaluate_c0(
    checkpoint: str | Path, split_file: str | Path, set_name: Literal["val", "test"], **kwargs: Any
) -> Path:
    """C0's SHARP-repo-style evaluation (§2.1): same forward and audit
    trail as every other stream, antenna aggregation = the paper's
    decision fusion (TMC §4.2, ratified 2026-07-16) instead of softmax
    averaging. This wrapper IS the §0.7 requirement that C0's eval has
    no test access path outside the common logger."""
    assert "fusion" not in kwargs, "evaluate_c0 fixes fusion='sharp_c0' (§2.1) — don't override"
    assert not kwargs.get("adapt_bn"), (
        "AdaBN is pre-registered on C1 only (§9 frozen row list) — no C0+AdaBN row exists"
    )
    return evaluate(checkpoint, split_file, set_name, fusion="sharp_c0", **kwargs)


def cache_features(
    checkpoint: str | Path, split_file: str | Path, set_name: Literal["train", "val", "test"],
    *,
    stage_dir: str | Path,
    out_path: str | Path | None = None,
    repo_dir: str | Path = ".",
    labels: list[str] | None = None,
    batch_size: int = 256,
    num_workers: int = 2,
    device: str | None = None,
    amp: bool = True,
    adapt_bn: bool = False,
) -> Path:
    """Extracts d_enc features from a checkpoint's FROZEN encoder over
    one set, WITHOUT augmentation (§5.3, declared note i), and caches
    them to a compressed .npz — computed once, every probe (C1-lin,
    C2-lin, C3/C4 phase B, §7 diagnostics) reuses the file. Test-set
    caching is a test access: logged (§0.7).

    adapt_bn=True is the post-AdaBN cache (§9: input of the
    C1+AdaBN+T3A row, composition AdaBN -> T3A): BN statistics
    re-estimated on this same set via _adapt_bn before extraction;
    default filename gains an "_adabn" infix so the plain cache is
    never overwritten. The adaptation is deterministic (same loader
    order, cumulative estimator), so the encoder here is identical to
    the one evaluate(adapt_bn=True) predicts with.

    The .npz holds: features (N, d_enc) float32, y, ar_set (train-domain
    index, -1 for held-out), antenna, window_start, trace_id, arset_name
    (per sample), plus labels/arset_labels/checkpoint/git_hash/adapt_bn
    metadata.
    """
    checkpoint = Path(checkpoint)
    infix = "_adabn" if adapt_bn else ""
    out_path = (
        Path(out_path) if out_path is not None
        else checkpoint.parent / f"features_{checkpoint.stem}{infix}_{set_name}.npz"
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    labels = labels if labels is not None else cfg.get("labels")
    if adapt_bn:
        assert batch_size == ADAPT_BN_BATCH, (
            f"AdaBN batch size is fixed a priori at {ADAPT_BN_BATCH} (§9), got {batch_size}"
        )

    if set_name == "test":
        _log_test_invocation(out_path.parent, {
            "kind": "cache_features", "checkpoint": str(checkpoint),
            "split_file": str(split_file), "set_name": set_name, "adapt_bn": adapt_bn,
            "config_name": cfg.get("name"), "git_hash": get_git_hash(repo_dir),
        })

    backbone = build_backbone(cfg)
    backbone.load_state_dict(ckpt["backbone"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone.to(dev)
    backbone.eval()

    dataset, loader = _make_loader(
        split_file, set_name, stage_dir, repo_dir, labels, batch_size, num_workers, dev.type == "cuda"
    )
    if adapt_bn:
        _adapt_bn(backbone, loader, dev, amp)
    feats, ys, arsets, ants, starts, tids = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
                f = backbone(x)
            feats.append(f.float().cpu().numpy())
            ys.append(batch["y"].numpy())
            arsets.append(batch["ar_set"].numpy())
            ants.append(batch["antenna"].numpy())
            starts.append(batch["window_start"].numpy())
            tids.extend(batch["trace_id"])

    arset_of_trace = {t: info["ar_set"] for t, info in dataset.trace_info.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=np.concatenate(feats),
        y=np.concatenate(ys),
        ar_set=np.concatenate(arsets),
        antenna=np.concatenate(ants),
        window_start=np.concatenate(starts),
        trace_id=np.array(tids, dtype=str),
        arset_name=np.array([arset_of_trace[t] for t in tids], dtype=str),
        labels=np.array(dataset.labels, dtype=str),
        arset_labels=np.array(dataset.arset_labels, dtype=str),
        checkpoint=np.array(str(checkpoint)),
        git_hash=np.array(get_git_hash(repo_dir)),
        set_name=np.array(set_name),
        adapt_bn=np.array(adapt_bn),
    )
    logger.info("cached %d features (d=%d) -> %s", len(np.concatenate(ys)), feats[0].shape[1], out_path)
    return out_path


def evaluate_features(
    features_npz: str | Path, head_state: dict[str, np.ndarray], set_name: Literal["val", "test"],
    *,
    out_dir: str | Path | None = None,
    run_name: str = "probe",
    repo_dir: str | Path = ".",
) -> Path:
    """Probe-side twin of evaluate() for phase B / C1-lin / C2-lin: a
    linear head (probe.linear_probe output: "weight" (C, d_enc), "bias"
    (C,)) applied to cached features, then the SAME fusion, metrics and
    test-invocation logging as every other stream (§0.7, §5.3, §9)."""
    features_npz = Path(features_npz)
    data = np.load(features_npz, allow_pickle=False)
    assert str(data["set_name"]) == set_name, (
        f"{features_npz} caches the {data['set_name']!r} set, not {set_name!r} — wrong file."
    )
    out = Path(out_dir) if out_dir is not None else features_npz.parent

    if set_name == "test":
        _log_test_invocation(out, {
            "kind": "evaluate_features", "features": str(features_npz),
            "encoder_checkpoint": str(data["checkpoint"]), "set_name": set_name,
            "run_name": run_name, "git_hash": get_git_hash(repo_dir),
        })

    logits = data["features"] @ head_state["weight"].T + head_state["bias"]
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    res = {
        "probs": probs.astype(np.float32),
        "y": data["y"],
        "trace_id": data["trace_id"].astype(object),
        "antenna": data["antenna"],
        "window_start": data["window_start"],
    }
    arset_of_trace = dict(zip(data["trace_id"].tolist(), data["arset_name"].tolist()))
    labels = [str(x) for x in data["labels"]]
    stem = f"{run_name}_{set_name}_softmax_avg"
    meta = {"checkpoint": str(data["checkpoint"]), "git_hash": get_git_hash(repo_dir), "fusion": "softmax_avg"}
    return _write_reports(res, arset_of_trace, labels, out, stem, set_name, "softmax_avg", meta)
