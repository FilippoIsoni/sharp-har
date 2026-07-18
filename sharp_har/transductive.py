"""T3A for the pre-registered transductive test rows (C1+T3A and
C1+AdaBN+T3A). Ref. §9 (v5.2 transductive rows; operational spec pinned
2026-07-18), §0.7. AdaBN, the other §9 technique, lives in harness.py
(`adapt_bn=True` on evaluate/cache_features): it re-estimates BN
statistics inside the model, so it belongs to the checkpoint->CSV path,
while T3A is pure numpy on cached features and only replaces the head.

Scope (§9, fixed a priori — no selection on val or test, ever): C1
only, unlabeled test features, inside the single logged §0.7 session.

T3A (Iwasawa & Matsuo, NeurIPS 2021) — declared BATCH variant:

1. initial prototypes = L2-normalized rows of C1's end-to-end head
   weight matrix; the bias is dropped (the paper's initial templates
   are the weight rows alone);
2. pseudo-label + softmax entropy of every (window, antenna) test
   sample from the dot product with the INITIAL prototypes. Declared
   deviation: the official implementation scores with the biased
   original classifier — our prototype path drops the bias everywhere
   for one consistent scoring rule;
3. per class, keep the M = T3A_M lowest-entropy samples among those
   pseudo-labeled to that class (fewer than M -> all; none -> the
   prototype stays initial). M = 20 fixed a priori = the central value
   of the paper's filter grid {1, 5, 20, 50, 100, inf} (grid confirmed
   against the official repo, where inf is filter_K = -1). Declared
   deviation: the official code subjects the warmup supports (= weight
   rows) to the same filter; the §9 formula keeps the initial
   prototype unconditionally;
4. adjusted prototype = L2-renormalized mean of {initial prototype}
   union {L2-normalized kept supports}. Supports are L2-normalized
   BEFORE averaging, as in the official implementation (sum of
   normalized supports, renormalized — mean and sum coincide after
   renormalization);
5. prediction = harness.evaluate_features with the returned head
   (weight = adjusted prototypes, bias = 0): softmax over prototype
   logits on the RAW cached features, antenna fusion, argmax — the
   §1.3 path unchanged, so the C1+T3A row differs from the C1 row in
   the head ONLY. For C1+AdaBN+T3A the input is the post-AdaBN cache
   (composition order AdaBN -> T3A, §9).

Why batch instead of the paper's online pass (declared in the report):
the online protocol is order-dependent — an arbitrary processing order
would have to be declared and the result would depend on it, a free
parameter with no benefit under the declared batch-deployment
assumption. Single assignment: pseudo-labels and entropies come from
the initial prototypes only; there is no second pass with the adjusted
prototypes (that would be an undeclared iterative variant).

Determinism: everything is closed-form given the cache; entropy ties
in the top-M filter break by cache row order (stable sort — the
dataset's sorted-trace, ascending antenna/window order), pseudo-label
ties by lowest class index (argmax first occurrence).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .diagnostics import l2norm
from .utils import get_logger

logger = get_logger(__name__)

T3A_M = 20  # §9: fixed a priori, central value of the paper's grid — never tuned


def head_weight_from_checkpoint(checkpoint: str | Path) -> np.ndarray:
    """C1's end-to-end head weight rows (n_att, d_enc) for t3a_head,
    straight from a train_run checkpoint (ActivityHead state dict, key
    "fc.weight"); the bias is deliberately not read (§9 step 1). Local
    torch import: T3A itself stays numpy-only on cached features."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["config"]["loss"]["type"] == "ce", (
        f"checkpoint {checkpoint} was trained with loss {ckpt['config']['loss']['type']!r}: "
        "T3A adjusts an end-to-end CE head — §9 pre-registers it on C1 only."
    )
    return ckpt["head"]["fc.weight"].numpy().astype(np.float64)


def t3a_head(
    head_weight: np.ndarray, features: np.ndarray, *, m: int = T3A_M
) -> dict[str, np.ndarray]:
    """The §9 batch-variant T3A adjustment: unlabeled cached features ->
    adjusted-prototype head. Returns a head state consumable by
    harness.evaluate_features — "weight" (n_classes, d) float32 = the
    adjusted L2-normalized prototypes, "bias" = zeros — plus the audit
    arrays "n_supports" and "pseudo_label_counts" (per class; the §0.7
    session prints them, evaluate_features ignores them).

    `features` are the RAW rows of a harness.cache_features .npz (no
    pre-normalization: the official prediction is raw feature · unit
    prototype); `head_weight` comes from head_weight_from_checkpoint.
    """
    head_weight = np.asarray(head_weight, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    assert head_weight.ndim == 2, f"head_weight must be (n_classes, d), got {head_weight.shape}"
    n_classes, d = head_weight.shape
    assert features.ndim == 2 and features.shape[1] == d, (
        f"features {features.shape} do not match head width d={d} — wrong cache or head?"
    )
    assert features.shape[0] > 0, "empty feature cache"
    assert m >= 1, f"filter size m must be >= 1, got {m}"

    protos0 = l2norm(head_weight)  # step 1: initial prototypes, bias dropped
    logits = features @ protos0.T
    shifted = logits - logits.max(axis=1, keepdims=True)  # stable log-softmax
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    entropy = -(np.exp(logp) * logp).sum(axis=1)
    pseudo = logits.argmax(axis=1)  # step 2

    weight = np.empty_like(protos0)
    n_supports = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        idx = np.flatnonzero(pseudo == c)
        members = protos0[c : c + 1]
        if idx.size:
            keep = idx[np.argsort(entropy[idx], kind="stable")[:m]]  # step 3
            members = np.concatenate([members, l2norm(features[keep])], axis=0)
            n_supports[c] = keep.size
        weight[c] = members.mean(axis=0)  # step 4
    weight = l2norm(weight)

    logger.info(
        "t3a_head: m=%d, supports per class %s (pseudo-label counts %s)",
        m, n_supports.tolist(), np.bincount(pseudo, minlength=n_classes).tolist(),
    )
    return {
        "weight": weight.astype(np.float32),
        "bias": np.zeros(n_classes, dtype=np.float32),
        "n_supports": n_supports,
        "pseudo_label_counts": np.bincount(pseudo, minlength=n_classes),
    }
