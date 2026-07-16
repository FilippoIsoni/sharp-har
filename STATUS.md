# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-16 · **Phase: day 3 code done → phase-A gate run on Colab**

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

## In progress

- **Phase-A gate run on Colab (Melissa): notebook `02b_phase_a_gate`** — margin to the
  8 h rule was only ~1 h under the day-2 2× approximation; if the measured projection
  exceeds it, escalation §5.2 BEFORE any phase-A run. Commit the JSON verbatim.

## Next steps (in order)

1. Commit `reports/gate_day3_phase_a.json` from the Colab run + STATUS update (GO/NO-GO).
2. Housekeeping: delete the scratch `C1_smoke` folder on Drive; the real C1 starts fresh
   from the unmodified `c1_ce.yaml`.
3. **Day 4: wiring** — SupCon phase-A path in `train_run` (P×K + 2 views, checkpoint grid
   40/50/60), GRL/`ARSetHead` (C2/C4), `sharp_like` backbone (C0), notebooks `00`/`03`.
4. **Days 4–9 (vertical ownership, §10.2):** A → C0 + C1 · B → C3, then C4 · C → C2, then
   C4 + probes + C1-lin/C2-lin.

## Blockers / open decisions

- None.
