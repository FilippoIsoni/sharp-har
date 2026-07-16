# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-16 · **Phase: day 2 done (gate GO) → day 3**

## Done

- Repo scaffold: package `sharp_har/`, configs C0–C4, thin notebooks, split/report dirs.
- **Day 1 complete, run on real Drive data (`DATASET_SHARP`).** 408 file-streams inventoried
  (`reports/inventory.csv`), axes + AR-set/campaign coverage verified, 0% NaN.
- Two real-data findings fixed in `inventory.py` and frozen into the split JSONs:
  - `FILENAME_PATTERN` extended for activity repetition numbers (`S4a_C1`, `S6b_J_0`, …) —
    the plain-letters-only regex was silently dropping every repeated recording.
  - `S5a_LOS` excluded as a non-activity calibration/reference trace (near-static signal,
    single occurrence) — see `NON_ACTIVITY_LABELS`, logged to `reports/excluded_traces.csv`.
  - `S4a_L`/`S5a_L` staged from **both** `doppler_traces.zip` and `doppler_traces_S4_S5.zip`
    with different `n_frame` (two distinct physical recordings, not copies) — split into
    `S4a_Lalt`/`S5a_Lalt` instead of arbitrarily discarding one; see
    `resolve_duplicate_streams()` + `assert_no_duplicate_files()` safety net, logged to
    `reports/duplicate_traces_split.csv`.
- **`splits/p2_lab.json` and `splits/p1_sharp.json` are frozen** (§0.1 — not to be
  regenerated or edited). p2_lab: leave-S7-out, train=81 val=9 test=11 (40 rare-cell pins,
  seed 42). p1_sharp: train=22 val=5 test=74 (train S1a/b/c, test S2–S7). n_att=8
  (C, E, H, J, L, R, S, W); window-count sanity check within ~5% of the assumed hop (186/197
  train-stride, 55/58 eval-stride) — accepted, not a blocker.
- **Day 2 implementation:** `data.py` (windowing/normalization/antennas, class-subset
  filtering), `resnet_vb.py` (V-B, 4.66 GFLOPs / 2.93 escalated, 2.79 M params) +
  `ActivityHead`, `ce_with_label_smoothing`, `train_run` (atomic checkpoints, auto-resume,
  per-epoch reseed, bit-exact resume verified on synthetic data), notebook `02_smoke_gate`.
- Day-2 hardening after the first Colab run: **resume on GPU fixed** — RNG states must be
  restored as CPU ByteTensors, so `last.ckpt` now loads with `map_location="cpu"` and
  `_restore_rng` coerces (the dtype-only attempt ed42717 was insufficient); `grad_clip`
  read from the config; notebooks install requirements AFTER the clone (was a silent no-op
  on fresh runtimes); numpy pin relaxed to `<3.0` (stock Colab env, proven by day 1).
- **Day 2 closed — smoke gate GO** (2026-07-16, Colab T4, torch 2.11, commit `1e65c12`,
  `reports/gate_day2.json`): staging 57 s (≤900), warm 0.526 s/step → projected C1 2.34 h
  (≤4 h), phase A ~7.01 h (≤8 h, declared 2×-per-step approximation). End-to-end C1_smoke
  run + real resume from the Drive checkpoint (`resumed ... at epoch 2`) verified.
  No escalation needed; these measurements recalibrate the §8.4 budget.

## In progress

- Nothing in flight — day 3 starts next.

## Next steps (in order)

1. **Day 3, first GPU task:** measure the real full-batch phase-A step (512 SupCon views)
   to replace the 2× approximation — the margin to the 8 h rule is only ~1 h; if the
   measured projection exceeds it, escalation §5.2 before any phase-A run.
2. **Day 3:** `harness.py` (checkpoint → CSV, test-invocation logging, C0 wrapper),
   `sampler.py` P×K, `augment.py`, `probe.py`, feature caching.
3. Housekeeping: delete the scratch `C1_smoke` folder on Drive; the real C1 starts fresh
   from the unmodified `c1_ce.yaml`.
4. **Days 4–9 (vertical ownership, §10.2):** A → C0 + C1 · B → C3, then C4 · C → C2, then
   C4 + probes + C1-lin/C2-lin.

## Blockers / open decisions

- Team to ratify: leave-S7-out as primary rotation (small test set, ~1 campaign; person P3
  unseen — declared in §2.2).
- Team to ratify: C0 class set — proposal on the table: `n_att: 5`,
  `labels: ["E","J","L","R","W"]` (paper's core 5 classes; L = sitting still per the
  dataset paper's letter map, C is sit-down/stand-up). `c0_sharp.yaml` gets updated only
  after ratification.
- Team to decide before `harness.py`: C0 evaluation fusion rule — the paper's decision
  fusion (TMC §4.2) in the C0 wrapper vs the pipeline's §1.3 softmax averaging
  (C1–C4 stay on softmax averaging either way).
