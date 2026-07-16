# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-16 · **Phase: §10.2 runs in flight (C0 rerun, C1 finishing)**

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
- **Team decisions ratified (2026-07-16):**
  - Leave-S7-out confirmed as the primary P2 rotation (small test set and unseen person P3
    stay declared limitations, §2.2).
  - C0 = paper's 5-class core: `c0_sharp.yaml` updated to `n_att: 5`,
    `labels: ["E","J","L","R","W"]` (letter map from the dataset paper; L = sitting still,
    C = sit-down/stand-up stays out).
  - C0 test evaluation uses the paper's decision fusion (TMC §4.2) inside the harness C0
    wrapper; C1–C4 stay on softmax averaging (§1.3). Both paths log through the same
    test-invocation harness (§0 rule 7).

- **Day 3 implementation complete** (verified end-to-end on synthetic data locally):
  - `harness.py`: single checkpoint → CSV path (windows/metrics/confusion per AR-set +
    aggregate, per-antenna appendix rows), softmax-averaging fusion + C0 wrapper with the
    SHARP-repo decision fusion (majority vote, softmax-sum tie-break — verified against
    `francescamen/SHARP` `CSI_network.py`), test-invocation JSONL logging on EVERY test
    access (eval, C0, feature caching, probe eval), `cache_features` → npz.
  - `sampler.py` `PKSampler`: round-robin AR-sets, without-replacement trace queues,
    ≥340-offset reuse constraint, per-epoch deterministic reseed, §4.2 composition logging.
  - `augment.py` (CE/SupCon-view profiles, widths rescale with axes), `supcon_loss`
    (float32 under AMP), `ProjectionHead`, `probe.py` (frozen recipe, fused val macro-F1
    early stopping, majority baseline), `models/build_backbone` shared factory.
  - `train.py` in-loop val metric now computed by the harness functions (selection ≡
    reporting); train still gates SupCon/GRL/P×K wiring to day 4.
  - `bench.py` + notebook `02b_phase_a_gate`: real 512-view forward+backward with the real
    sampler and augmentation → `reports/gate_day3_phase_a.json`, rule ≤ 8 h (§10.1).

- **Phase-A gate run (2026-07-16, Colab T4, commit `faa0cdb`):** measured 1.373 s/step at
  full batch (512 views, real P×K + in-step augmentation) → projected 9.16 h > 8 h →
  **NO-GO** (`reports/gate_day3_phase_a.json`, committed verbatim). Internal consistency
  verified: per-class trace counts sum to 81 (= frozen p2_lab train), distinct-AR-sets per
  class match the frozen contingency table, min reuse offset 400 ≥ 340. Peak memory
  8.57/11.37 GiB → §5.2 verified: no activation checkpointing needed.
- **Escalation (a) applied** (first rung of the pre-committed §5.2 ladder, recorded in
  `splits/CHANGELOG.md`): phase-A epoch_steps 400 → 300 in `c3_supcon.yaml` +
  `c4_supcon_grl.yaml` → projected 6.87 h ≤ 8 h. C1/C2 stay at 400 (their gate passed);
  backbone untouched (escalation (b) would break shared-encoder comparability); grid
  40/50/60 unchanged. Declared: phase-A epochs are 300 steps vs 400 in CE runs.

- **Day 4 wiring complete** (verified end-to-end on synthetic data: C0/C1/C2/C3/C4-style
  runs all train, deterministic, bit-exact resume on the SupCon path too):
  - `train_run` now drives every config: CE path gains the §3 "ce" augmentation profile
    (in DataLoader workers, per-(epoch, worker) reseed — resume-reproducible; changing
    num_workers changes the augmentation stream, declared); SupCon phase A wired with
    P×K batches + `TwoViewAugmenter` (2 views per sample built in the workers, as the
    day-3 gate note hoped) + `supcon_loss`, no in-loop selection/best.ckpt (§6-C3), grid
    checkpoints; GRL path (C2/C4): `ARSetHead` + literal §6-C2 loss
    `L = L_task + β·λ(p)·L_env`, fixed ramp `losses.grl_lambda` (λ_max/β/ramp_epochs from
    the config), mandatory monitoring in history.csv (arset_train_acc, grl_lambda).
  - `losses.GRL` (reversal autograd.Function), `models/sharp_like.py` implemented
    faithful to the TMC figure with Keras-style same padding (feature_dim 25500; heads
    consume `backbone.feature_dim`, d_enc does not apply to C0).
  - `c0_sharp.yaml`: `train.augment: false` — the SHARP repo does not use the §3 set,
    faithfulness wins (declared).
  - Notebooks `00_setup_smoke` (env + staging + frozen-artifact asserts) and `03_train`
    (config-driven runner, val-only evaluation, §0.7 test warning) implemented.

- **Day-4 review pass:** fixed harness `_load_end_to_end` rebuilding the head with
  `d_enc` instead of `backbone.feature_dim` (C0/sharp_like eval would have crashed on
  size mismatch); added `viz.py` (plot_history / plot_confusion / compare_runs, panels
  driven by history.csv columns) with a thin "Training curves" section in notebook 03;
  added `notebooks/runs/` — executed run notebooks committed verbatim as
  `YYYY-MM-DD_<config>.ipynb` (see its README), 03 stays an output-free template.

## In progress

- **C0 rerun on GPU** (owner A): the first attempt is archived in
  `notebooks/runs/2026-07-16_c0_sharp.ipynb` — interrupted at epoch 11/60 on a **CPU
  runtime**, best val macro-F1 0.667 @ epoch 6 (val = 3 traces only → noisy selection
  metric, declared). Decision: restart fresh on GPU (delete or rename the Drive `C0`
  folder first) rather than resuming a mixed CPU/GPU run.
- **C1 training** (owner A2): running without issues, near completion. The harness
  head-sizing fix does NOT affect it: training never goes through `_load_end_to_end`,
  and for `resnet_vb` feature_dim = d_enc, so old and new code build the same head.
  On finish: archive the executed notebook under `notebooks/runs/`, then C1-lin probe.

## Next steps (in order)

1. **Launch C3 (owner B) and C2 (owner C) in parallel** via notebook `03` — independent
   of C0/C1. C3 ≈ 6.9 h (phase A: no val metric, no best.ckpt; deliverables are
   epoch40/50/60.ckpt). C2: watch the §6-C2 monitoring panel — arset_train_acc must fall
   as λ ramps without collapsing val macro-F1; if not, DISCUSS λ_max → 0.5, don't tune solo.
2. Every finished run: executed notebook committed verbatim to `notebooks/runs/`
   (`YYYY-MM-DD_<config>.ipynb`) + STATUS line, same commit. Val only, never test.
3. After C1/C2 best.ckpt: **C1-lin / C2-lin probes** (cache_features train+val →
   linear_probe → fused val F1).
4. After C3 (and C2's outcome): **phase B grid** on C3's epoch40/50/60 → val-selected
   checkpoint; only then **launch C4** (inherits any GRL contingency), then its phase B.
5. **Single final test session** (§0.7) once ALL streams have a val-selected checkpoint:
   evaluate_c0 (C0), evaluate (C1/C2), evaluate_features (C3/C4 + probes); commit per-AR-set
   CSVs + `test_invocations.jsonl`. Then `viz.compare_runs` + per-AR-set table (§0.5:
   <~2 points = "comparable").
6. Optional prep while runs are in flight: thin runner notebooks `04_probe` (phase B /
   C1-lin / C2-lin) and `05_test_final` — logic already lives in the package.
7. Housekeeping: delete the scratch `C1_smoke` folder on Drive (still pending).

## Blockers / open decisions

- None.
