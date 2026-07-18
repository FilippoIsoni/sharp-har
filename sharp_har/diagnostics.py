"""Fixed-encoder readout diagnostics: NCM and kNN class scores on
cached features. Ref. §7 (v5.2 "diagnostiche aggiuntive val-only").

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

import numpy as np

K_NEIGHBORS = 20  # team call, §7 v5.2 — fixed a priori, never tuned
TIE_EPS = 1e-6  # << 1/K_NEIGHBORS: only ever breaks exact vote-count ties


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
