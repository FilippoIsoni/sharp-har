# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-15 · **Phase: day 1 done → day 2**

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

## In progress

- **Day 2:** `data.py` implemented and unit-tested on synthetic data reproducing every
  real-data quirk (repetitions, dual-archive alt traces, LOS file, shape drift, held-out-domain
  ar_set sentinel −1). Window volumes from frozen artifacts: p2_lab train 53,400 / val 1,396 /
  test 1,700 (window, antenna) samples; p1_sharp 14,384 / 980 / 13,672. Next: `resnet_vb.py`,
  CE loss, `train.py` skeleton, notebook 02 smoke + throughput gate. **Dataloader needs the
  §11 cross-review before the gate run.**

## Next steps (in order)

1. **Day 2 (gate, next up):** implement `data.py`, `resnet_vb.py`, CE loss, `train.py` skeleton
   (checkpoint/resume); end-to-end smoke test + throughput gate (s/step, staging time) →
   written go/no-go, recalibrate §8.4 budget.
2. **Day 3:** `harness.py` (test-invocation logging), `sampler.py` P×K, `augment.py`, `probe.py`,
   feature caching; phase-A full-batch memory test (512 views).
3. **Days 4–9 (vertical ownership, §10.2):** A → C0 + C1 · B → C3, then C4 · C → C2, then C4 +
   probes + C1-lin/C2-lin.

## Blockers / open decisions

- Team to ratify: leave-S7-out as primary rotation (small test set, ~1 campaign; person P3
  unseen — declared in §2.2).
- `configs/c0_sharp.yaml` still has a placeholder `n_att` — the real C0 class count (paper's
  5-class core vs the full inventory's 8) needs a deliberate decision before day 2 touches it,
  not an assumption carried over from the pipeline doc.
