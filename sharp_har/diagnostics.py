"""Fixed-encoder diagnostics on cached features. Ref. §7 (v5.2), §9.

Three instruments live here, all val/train-only (§0.7: a diagnostic
never touches the test set):

1. NCM + kNN readout scores (§7 v5.2 "diagnostiche aggiuntive
   val-only") — promoted from the diagnostics notebook 2026-07-18,
   math verified byte-identical;
2. the train-feature DOMAIN PROBE (§7 esito v5.2 / §9 key figure) —
   promoted 2026-07-18: the instrument was replicated verbatim across
   the C1/C2/C3 executed sessions and §10.3 pre-registers a fourth run
   (E2′ S6-out train), which makes it re-run pipeline code, not a
   one-off (same criterion as the NCM/kNN promotion; equivalence to
   the notebook-local original verified on synthetic data — the
   recorded C1/C2/C3 rows stand);
3. concat_caches, the alignment-checked feature concatenation for the
   §7 C1⊕C3 diagnostic and its C1⊕C1′ ensemble control (the probe
   recipe itself stays probe.linear_probe, unchanged).

Declared hyperparameters (§7 v5.2, fixed a priori, identical recipe
for every encoder):
- features L2-normalized, cosine similarity (on normalized vectors,
  equivalent to Euclidean — standard metric for contrastive features);
- NCM: one centroid per class = mean of L2-normalized TRAIN features,
  re-normalized L2 (without this, centroids of different norms would
  make cosine-argmax and Euclidean-argmin diverge); class score =
  cosine similarity to the centroid;
- kNN: k=20 (team call); class score = fraction of votes among the k
  nearest TRAIN neighbors, tie broken by the mean similarity of the
  voting neighbors;
- antenna fusion = mean of class scores over the 4 antennas, then
  argmax — the query side goes through `harness.fuse_windows`
  ("softmax_avg" only averages a per-sample class-score array and
  takes the argmax, so it does not require true probabilities);
- reference pool (declared, §7 refinement 2026-07-18): centroids and
  neighbors are computed on the single pool of TRAIN (window, antenna)
  samples across all 4 antennas — the declared fusion operates on the
  query side only, the §1.3 analogue.

Both scorers return an (n_query, n_classes) float64 class-score array,
the shape `harness.fuse_windows` expects from a softmax, so fusion,
metrics (`harness.macro_f1`) and the reference (`probe.majority_baseline`)
reuse the frozen harness/probe code unchanged. The frozen §5.3 linear
probe recipe in `probe.py` is untouched by design.

Val-only diagnostics: callers feed train (reference) and val (query)
caches — never test (§0.7; promotion to a test row would need a team
decision, none planned).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .harness import fuse_windows, macro_f1
from .inventory import AR_SET_METADATA
from .probe import linear_probe, majority_baseline
from .utils import get_logger

logger = get_logger(__name__)

K_NEIGHBORS = 20  # team call, §7 v5.2 — fixed a priori, never tuned
TIE_EPS = 1e-6  # << 1/K_NEIGHBORS: only ever breaks exact vote-count ties

# Domain probe (§7 esito v5.2): targets and inner-split fraction exactly
# as in the three executed sessions — frozen, never extended per-run.
DOMAIN_TARGETS = ("y", "ar_set", "ambiente", "direct_path", "persona", "monitor")
DOMAIN_EVAL_FRAC = 1 / 3


def l2norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (declared cosine metric, §7 v5.2)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0.0).all(), (
        f"{int((norms == 0.0).sum())} zero-norm feature rows — corrupt cache? "
        "(would silently become NaN scores downstream)"
    )
    return x / norms


def ncm_scores(
    train_x: np.ndarray, train_y: np.ndarray, n_classes: int, query_x: np.ndarray
) -> np.ndarray:
    """Cosine similarity to per-class centroids (mean of L2-normed train
    features, RE-normalized L2 — declared, §7 v5.2). Inputs must already
    be L2-normalized (`l2norm`)."""
    present = np.unique(train_y)
    assert len(present) == n_classes, (
        f"train reference covers {len(present)}/{n_classes} classes — a missing "
        "class would silently yield a NaN centroid (wrong cache or class subset?)"
    )
    centroids = np.stack([train_x[train_y == c].mean(axis=0) for c in range(n_classes)])
    centroids = l2norm(centroids)
    return query_x @ centroids.T


def knn_scores(
    train_x: np.ndarray, train_y: np.ndarray, n_classes: int, query_x: np.ndarray,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """Vote-fraction among the k nearest train neighbors (cosine
    similarity), tie-broken by mean similarity of the voting neighbors
    (declared, §7 v5.2). Inputs must already be L2-normalized; the
    TIE_EPS term is bounded by 1e-6, below the 1/(4k) granularity the
    fused vote fractions can differ by, so it never overrides a real
    vote-count difference."""
    assert train_x.shape[0] > k, f"need more than k={k} train samples, got {train_x.shape[0]}"
    sims = query_x @ train_x.T  # (n_query, n_train)
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    scores = np.zeros((query_x.shape[0], n_classes), dtype=np.float64)
    for i in range(query_x.shape[0]):
        idx = top_idx[i]
        labels_i, sims_i = train_y[idx], sims[i, idx]
        for c in range(n_classes):
            m = labels_i == c
            if m.any():
                scores[i, c] = m.sum() / k + TIE_EPS * sims_i[m].mean()
    return scores


# --------------------------------------------------- domain probe (§7/§9)

def _as_arrays(features: str | Path | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    if isinstance(features, (str, Path)):
        return np.load(features, allow_pickle=False)
    return features


def build_domain_targets(
    arset_name: np.ndarray, arset_int: np.ndarray, arset_labels: np.ndarray
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """ar_set (the adversary's actual target) + the metadata attributes
    it conflates (ambiente, direct_path, persona, monitor). The ar_set
    label space comes from the cache's own arset_labels (the
    train-domain index order the integers refer to); the derived spaces
    are built from the AR-sets OBSERVED in the cache, so no class is
    allocated for held-out domains. Deterministic given the frozen
    train trace list. Verbatim from the executed C1/C2/C3 sessions."""
    names = np.array([str(a) for a in arset_name])
    ar_labels = [str(a) for a in arset_labels]
    assert int(arset_int.max()) < len(ar_labels), "ar_set index outside arset_labels"
    assert set(names) <= set(ar_labels), "arset_name outside the train-domain label set"
    out: dict[str, tuple[np.ndarray, list[str]]] = {"ar_set": (arset_int.astype(np.int64), ar_labels)}
    for attr in ("ambiente", "direct_path", "persona", "monitor"):
        vals = sorted({AR_SET_METADATA[n][attr] for n in set(names)})
        out[attr] = (
            np.array([vals.index(AR_SET_METADATA[n][attr]) for n in names], dtype=np.int64),
            vals,
        )
    return out


def inner_trace_split(
    trace_id: np.ndarray, arset_int: np.ndarray, seed: int = 42,
    eval_frac: float = DOMAIN_EVAL_FRAC,
) -> tuple[set[str], set[str]]:
    """Trace-disjoint fit/eval split of a TRAIN cache, stratified by
    AR-set (§0 rule 2: never by window). Deterministic given trace ids,
    AR-set assignment and seed: sorted trace order, one shared
    default_rng(seed) consumed over AR-sets in ascending order.
    Verbatim from the executed C1/C2/C3 sessions — changing any detail
    would break comparability with the recorded rows."""
    trace_to_ar: dict[str, int] = {}
    for t, a in zip(trace_id, arset_int):
        trace_to_ar.setdefault(str(t), int(a))

    by_ar: dict[int, list[str]] = {}
    for t, a in trace_to_ar.items():
        by_ar.setdefault(a, []).append(t)

    rng = np.random.default_rng(seed)
    fit, evl = [], []
    for a in sorted(by_ar):
        ts = np.array(sorted(by_ar[a]), dtype=object)
        ts = ts[rng.permutation(len(ts))]
        n_ev = max(1, int(round(len(ts) * eval_frac)))
        evl += list(ts[:n_ev])
        fit += list(ts[n_ev:])

    fit_s, ev_s = set(fit), set(evl)
    assert not (fit_s & ev_s), "inner split is not trace-disjoint (§0 rule 2)"
    assert fit_s and ev_s
    return fit_s, ev_s


def fused_head_scores(
    x: np.ndarray, y: np.ndarray, trace_id: np.ndarray, window_start: np.ndarray,
    weight: np.ndarray, bias: np.ndarray,
) -> dict[str, float]:
    """Antenna-fused (§1.3) accuracy, macro-F1 and majority baseline of
    a linear head on cached features — the shared readout of the domain
    probe and the §7 concat diagnostic (identical math to the executed
    sessions' notebook-local `fused_accuracy` and to
    probe.probe_encoder's summary block)."""
    logits = x @ weight.T + bias
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)).astype(np.float32)
    fused = fuse_windows(probs, y, trace_id.astype(object), window_start)
    return {
        "accuracy": float((fused["y_pred"] == fused["y_true"]).mean()),
        "macro_f1": macro_f1(fused["y_true"], fused["y_pred"]),
        "majority_baseline": majority_baseline(fused["y_true"]),
    }


def domain_probe(
    features: str | Path | Mapping[str, np.ndarray], label: str,
    *,
    seed: int = 42,
    targets: tuple[str, ...] = DOMAIN_TARGETS,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Is the domain linearly readable from a frozen encoder's TRAIN
    features? The §9 invariance evidence (esito §7 v5.2): the frozen
    §5.3 recipe (probe.linear_probe, untouched) on an inner
    trace-disjoint stratified split of the train traces, one row per
    target, each read against its own majority baseline; "y" is the
    positive control (validates the plumbing; saturated — the encoder
    trained on these traces, the declared memorization confound).

    `features` = a harness.cache_features .npz of the TRAIN set. Rows
    carry the exact keys the executed C1/C2/C3 sessions recorded.
    Promoted 2026-07-18 from the notebook-local cell (verbatim; §10.3's
    pre-registered E2′ S6-out replication makes this a re-run
    instrument -> pipeline code per the thin-notebook rule).
    Train/val only — never a test cache (§0.7)."""
    d = _as_arrays(features)
    if "set_name" in d:
        assert str(d["set_name"]) == "train", (
            f"domain_probe is defined on TRAIN caches (§7 esito v5.2), got {d['set_name']!r}"
        )
    x, tid, ws = d["features"], d["trace_id"], d["window_start"]
    target_map = build_domain_targets(d["arset_name"], d["ar_set"], d["arset_labels"])
    target_map["y"] = (d["y"].astype(np.int64), [str(lab) for lab in d["labels"]])
    fit_s, ev_s = inner_trace_split(tid, d["ar_set"], seed=seed)

    m_fit = np.array([str(t) in fit_s for t in tid])
    m_ev = np.array([str(t) in ev_s for t in tid])
    logger.info(
        "[%s] domain probe: %d samples (d=%d), inner split %d fit / %d eval traces "
        "(%d / %d samples)",
        label, x.shape[0], x.shape[1], len(fit_s), len(ev_s), m_fit.sum(), m_ev.sum(),
    )

    rows: list[dict[str, Any]] = []
    for target in targets:
        y, vals = target_map[target]
        n_cls = len(vals)
        fit_d = {"features": x[m_fit], target: y[m_fit], "trace_id": tid[m_fit], "window_start": ws[m_fit]}
        ev_d = {"features": x[m_ev], target: y[m_ev], "trace_id": tid[m_ev], "window_start": ws[m_ev]}

        res = linear_probe(fit_d, ev_d, target=target, n_classes=n_cls, seed=seed, device=device)
        scores = fused_head_scores(x[m_ev], y[m_ev], tid[m_ev], ws[m_ev], res["weight"], res["bias"])
        rows.append({
            "run": label, "target": target, "n_classes": n_cls,
            "eval_accuracy": scores["accuracy"],
            "majority_baseline": scores["majority_baseline"],
            "delta": scores["accuracy"] - scores["majority_baseline"],
            "macro_f1": res["best_val_macro_f1"], "best_epoch": res["best_epoch"],
            "classes_present_in_eval": len(sorted(set(y[m_ev].tolist()))),
            "labels": vals,
        })
        logger.info(
            "[%s] %-12s (%d cls): acc %.3f vs baseline %.3f (delta %+.3f), macro-F1 %.3f%s",
            label, target, n_cls, scores["accuracy"], scores["majority_baseline"],
            scores["accuracy"] - scores["majority_baseline"], res["best_val_macro_f1"],
            "  <- CONTROL" if target == "y" else "",
        )
    return rows


# -------------------------------------------------- concat diagnostics (§7)

def concat_caches(
    *caches: str | Path | Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Row-wise feature concatenation of two harness.cache_features
    .npz over the SAME split set — the §7 C1⊕C3 diagnostic and its
    C1⊕C1′ ensemble control. Blocking asserts on per-row alignment
    (trace_id, window_start, antenna) and on identical y/labels/
    set_name: the caches are extracted with shuffle=False over one
    frozen split, so any mismatch means wrong files, never a reorder to
    fix silently. Returns a mapping ready for probe.linear_probe
    (recipe unchanged, §7: raw features, no per-block rescaling — the
    linear head absorbs scale)."""
    assert len(caches) >= 2, "concat needs at least two caches"
    ds = [_as_arrays(c) for c in caches]
    ref = ds[0]
    for i, d in enumerate(ds[1:], start=2):
        for key in ("trace_id", "window_start", "antenna", "y"):
            assert np.array_equal(np.asarray(ref[key]), np.asarray(d[key])), (
                f"cache #{i} misaligned on {key!r} — different split set, encoder run "
                "over a different trace list, or corrupted cache"
            )
        for key in ("labels", "set_name"):
            if key in ref and key in d:
                assert np.array_equal(np.asarray(ref[key]), np.asarray(d[key])), (
                    f"cache #{i} disagrees on {key!r} — not the same evaluation contract"
                )
    features = np.concatenate(
        [np.asarray(d["features"], dtype=np.float32) for d in ds], axis=1
    )
    logger.info(
        "concat_caches: %d caches -> features %s (blocks %s)",
        len(ds), features.shape, [int(np.asarray(d["features"]).shape[1]) for d in ds],
    )
    return {
        "features": features,
        "y": np.asarray(ref["y"]),
        "trace_id": np.asarray(ref["trace_id"]),
        "window_start": np.asarray(ref["window_start"]),
        "antenna": np.asarray(ref["antenna"]),
    }
