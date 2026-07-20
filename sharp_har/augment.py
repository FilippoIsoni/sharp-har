"""Day 3 — Doppler CSI augmentation, fixed order. Ref. §3.

On-the-fly in training only, applied AFTER standardization, so masked
regions are filled with 0 (= the post-standardization mean). Two
probability profiles (§3 table): "ce" for C1/C2 batches and
"supcon_view" for the SupCon views of C3/C4, where each sample yields
2 views = 2 independent augmentations of the same (window, antenna).

Additive third profile "ce_amp" (§3 amendment, team-ratified
2026-07-20, pre-registered in splits/CHANGELOG.md): the C1-aug arm.
Identical to "ce" except the SINGLE amplitude channel — scaling
s ~ U(0.6, 1.5) at p=0.8 (vs U(0.8, 1.2) at p=0.5) — the one §3 lever
that is physically coherent on μ-Doppler (global attenuation ≈ the
room/distance component of a cross-environment shift), label-safe
(a global scalar preserves the class pattern; contrast the forbidden
velocity manipulations) and in-scope (no raw CSI needed). The frozen
"ce"/"supcon_view" profiles and the §3 width table stay byte-identical:
the profile is selected per run via the config key `train.augment_profile`
(train.py, default "ce"), so every existing run is untouched. The name
describes the transform, not a rotation: the arm runs on both the
S7-out and S6-out rotations.

Forbidden by design and deliberately not implemented here: velocity-axis
flip (inverts the Doppler sign) and time flip (sit-down <-> stand-up).

Widths in the §3 table assume the reference 340x100 geometry; for a
different verified geometry they rescale proportionally (§3), which
augment_cfg does from the actual (time_steps, velocity_bins).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

# Fixed order, applied after standardization (x - mu) / sigma. Masked
# regions are filled with 0. Forbidden: velocity flip, time flip.
AUGMENTATION_ORDER = (
    "time_shift",
    "time_masking",
    "velocity_masking",
    "amplitude_scaling",
    "gaussian_noise",
)

REFERENCE_TIME_STEPS = 340
REFERENCE_VELOCITY_BINS = 100

# Application probabilities per profile (§3 table). Everything else
# (widths, ranges, sigma) is shared between the profiles, except the
# per-profile overrides below. "ce" and "supcon_view" are the frozen §3
# table and MUST stay byte-identical (every archived run reproduces
# through them); "ce_amp" is the additive 2026-07-20 amendment, written
# out literally (not spread from "ce") so each frozen value is auditable
# in place.
_PROFILE_PROBS: dict[str, dict[str, float]] = {
    "ce": {
        "p_time_shift": 0.5,
        "p_time_masking": 0.5,
        "p_velocity_masking": 0.5,
        "p_amplitude_scaling": 0.5,
        "p_gaussian_noise": 0.0,  # "—" in the table: never applied on the CE path
    },
    "supcon_view": {
        "p_time_shift": 0.5,
        "p_time_masking": 0.8,
        "p_velocity_masking": 0.8,
        "p_amplitude_scaling": 0.8,
        "p_gaussian_noise": 0.5,
    },
    "ce_amp": {  # C1-aug arm: = "ce" except the amplitude channel (§3 amendment 2026-07-20)
        "p_time_shift": 0.5,
        "p_time_masking": 0.5,
        "p_velocity_masking": 0.5,
        "p_amplitude_scaling": 0.8,
        "p_gaussian_noise": 0.0,
    },
}

# Per-profile parameter overrides, applied on top of the shared §3 width
# table by augment_cfg. One conceptual lever per arm: "ce_amp" touches
# ONLY the amplitude range (clean attribution — bundling other channels
# would be variant (a) "more of everything", uninterpretable). The
# amplitude scalar is geometry-free, so overriding after the axis
# rescaling is exact.
_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "ce_amp": {"amplitude_range": (0.6, 1.5)},
}


def _rescale(value: int, actual: int, reference: int) -> int:
    """Proportional rescaling of a table width/shift to a non-reference
    axis size (§3), never below 1."""
    return max(1, round(value * actual / reference))


def augment_cfg(
    profile: Literal["ce", "supcon_view", "ce_amp"],
    time_steps: int = REFERENCE_TIME_STEPS,
    velocity_bins: int = REFERENCE_VELOCITY_BINS,
) -> dict[str, Any]:
    """Builds the parameter dict `apply` consumes for one of the §3
    profiles, with widths rescaled from the reference 340x100 geometry
    to the actual one recorded in the frozen split (day-1 axes check
    confirmed 340x100, so the rescaling is the identity for this
    project's rotations). Per-profile overrides (_PROFILE_OVERRIDES,
    §3 amendment 2026-07-20) apply last."""
    assert profile in _PROFILE_PROBS, f"unknown profile {profile!r}, expected {sorted(_PROFILE_PROBS)}"
    t = lambda v: _rescale(v, time_steps, REFERENCE_TIME_STEPS)  # noqa: E731
    d = lambda v: _rescale(v, velocity_bins, REFERENCE_VELOCITY_BINS)  # noqa: E731
    return {
        **_PROFILE_PROBS[profile],
        "time_shift_max": t(10),                     # shift ~ U{-10, ..., +10}
        "n_masks": (1, 2),                           # 1-2 masks, equiprobable (both axes)
        "time_mask_width": (t(5), t(20)),            # width ~ U{5, ..., 20}
        "velocity_mask_width": (d(2), d(10)),        # width ~ U{2, ..., 10}
        "amplitude_range": (0.8, 1.2),               # s ~ U(0.8, 1.2)
        "noise_sigma": 0.05,
        **_PROFILE_OVERRIDES.get(profile, {}),
    }


def _mask_axis(
    x: torch.Tensor, axis: int, width_range: tuple[int, int], n_masks_range: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """Zeroes n ~ U{n_masks_range} contiguous spans of width
    ~ U{width_range} along `axis` (1 = time, 2 = velocity), in place.
    0 is the post-standardization mean (§3)."""
    size = x.shape[axis]
    n_masks = int(rng.integers(n_masks_range[0], n_masks_range[1] + 1))
    for _ in range(n_masks):
        width = min(int(rng.integers(width_range[0], width_range[1] + 1)), size)
        start = int(rng.integers(0, size - width + 1))
        if axis == 1:
            x[:, start : start + width, :] = 0.0
        else:
            x[:, :, start : start + width] = 0.0


def apply(x: torch.Tensor, cfg: dict[str, Any], rng: np.random.Generator) -> torch.Tensor:
    """Applies the augmentation pipeline in AUGMENTATION_ORDER to a
    single already-standardized (1, time, velocity) sample and returns a
    new tensor of the same shape (the input is never modified). Called
    twice with independent draws to generate SupCon's positive pairs
    (§3). All randomness comes from `rng`, so the caller controls
    determinism (seed 42 + per-epoch reseed, §0.5).
    """
    assert x.ndim == 3, f"expected (1, time, velocity), got shape {tuple(x.shape)}"
    x = x.clone()  # masking/shift below operate in place on the copy

    if rng.random() < cfg["p_time_shift"]:
        m = cfg["time_shift_max"]
        shift = int(rng.integers(-m, m + 1))
        if shift:
            x = torch.roll(x, shifts=shift, dims=1)  # circular (§3)

    if rng.random() < cfg["p_time_masking"]:
        _mask_axis(x, 1, cfg["time_mask_width"], cfg["n_masks"], rng)

    if rng.random() < cfg["p_velocity_masking"]:
        _mask_axis(x, 2, cfg["velocity_mask_width"], cfg["n_masks"], rng)

    if rng.random() < cfg["p_amplitude_scaling"]:
        lo, hi = cfg["amplitude_range"]
        x = x * float(rng.uniform(lo, hi))

    if rng.random() < cfg["p_gaussian_noise"]:
        noise = rng.standard_normal(tuple(x.shape), dtype=np.float32) * cfg["noise_sigma"]
        x = x + torch.from_numpy(noise)

    return x


class Augmenter:
    """Picklable callable wrapper around `apply`, usable as
    DopplerDataset's `transform` (CE path) or invoked twice per sample
    for SupCon views. Owns a np.random.Generator; `reseed` it per epoch
    with utils.epoch_seed so the augmentation stream is deterministic
    and resume-reproducible (§0.5, §8.2). With multi-worker DataLoaders
    reseed per worker too (worker_init_fn), or the workers' forked
    generators draw identical streams."""

    def __init__(
        self,
        profile: Literal["ce", "supcon_view", "ce_amp"],
        seed: int,
        time_steps: int = REFERENCE_TIME_STEPS,
        velocity_bins: int = REFERENCE_VELOCITY_BINS,
    ) -> None:
        self.cfg = augment_cfg(profile, time_steps, velocity_bins)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return apply(x, self.cfg, self._rng)


class TwoViewAugmenter(Augmenter):
    """SupCon-view variant (§3): every sample yields 2 views = 2
    independent augmentations of the same (window, antenna), stacked to
    (2, 1, time, velocity) — the ONE transform whose output shape
    differs from its input, by contract with the phase-A training loop
    (which reshapes the collated (B, 2, 1, T, V) batch to 2B views and
    duplicates the labels). Built on the "supcon_view" profile."""

    def __init__(
        self,
        seed: int,
        time_steps: int = REFERENCE_TIME_STEPS,
        velocity_bins: int = REFERENCE_VELOCITY_BINS,
    ) -> None:
        super().__init__("supcon_view", seed, time_steps, velocity_bins)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([apply(x, self.cfg, self._rng), apply(x, self.cfg, self._rng)])
