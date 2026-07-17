# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-17 · **Phase: §10.2 runs (C0–C3 done incl. C3 phase B; C4 gate diagnostic in flight)**

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

- **§10.2 tail prep while runs are in flight (code-only, verified on synthetic data):**
  `probe.probe_encoder` (frozen encoder → cached train/val features → §5.3 probe →
  persisted `probe_head_*.npz` + JSON summary; targets `y`/`ar_set`/`persona`, person
  derived per sample from `AR_SET_METADATA` — no cache change) and `probe.select_phase_b`
  (§6-C3/C4 grid → `phase_b_selection.json`, ties → earliest epoch, declared);
  `viz.metrics_table` (run × AR-set pivot from harness metrics CSVs, fused rows only);
  thin runner notebooks `04_probe` (phase B / C1-lin / C2-lin / §7 diagnostics — val
  only) and `05_test_final` (readiness assert → single §0.7 test session → comparison
  table/confusions → artifact copy to `reports/final/` for the final commit).

- **C1 run complete** (2026-07-16, Colab GPU, full 40 epochs): best val macro-F1
  **0.8871 @ epoch 37** (late best, on the cosine-annealed tail). Executed notebook
  archived as `notebooks/runs/2026-07-16_c1_ce.ipynb` (commit 761c6c1 — STATUS line
  landed one commit late, with C2's).
- **C2 run complete** (2026-07-16, Colab GPU): best val macro-F1 **0.8415 @ epoch 13**,
  early stop at 23/40 (patience 10) — stopped mid-schedule (lr ≈ 4.8e-4), unlike C1
  whose best came at the annealed tail; protocol-consistent, declared. Val gap to C1 is
  ~4.6 pts (> §0.5's ~2-pt comparability band). Executed notebook archived as
  `notebooks/runs/2026-07-16_c2_grl.ipynb`.
  **§6-C2 monitoring — reading CORRECTED 2026-07-17:** arset_train_acc sat at ~0.30 for
  the whole run. The original line read this against 1/6 ≈ 0.167 ("held down but not at
  chance"); the reference is the **majority baseline = 0.2969** (AR-1's share of train
  windows), i.e. 0.30 IS the floor — the adversary only ever learned the class prior.
  It was already there at epoch 1-2 with λ still at 0.25-0.46, so it never became a
  discriminator at all. Same 1/n-vs-majority error the `probe.py` docstring warns against.

- **C0 rerun complete on GPU** (2026-07-16, owner A): fresh start from epoch 1
  (Drive `C0` folder cleared — epoch-1 losses differ from the CPU attempt, no mixed
  resume), best val macro-F1 **0.8916 @ epoch 20**, early stop at **31/60** (patience
  10; an earlier STATUS line said 30/60 — the resumed tail ran epoch 31).
  Verified against config/split: 5-class filtering logged (train 15 traces / 9180
  samples, val 3 / 540), constant lr 1e-4, code era `ae1746e` (same training code as
  C1/C2). Caveat: val = 3 traces → very noisy selection (declared). **Archive
  regularized 2026-07-17** per the multi-session rule: `_part1` =
  `2026-07-16_c0_sharp_part1.htm` + `_part1_history.csv` (main session, epochs 1-30;
  HTML export instead of an executed `.ipynb`, declared exception), `_part2` =
  `2026-07-16_c0_sharp_part2.ipynb` (resumed tail, epoch 31; was `c0_sharp_train.ipynb`
  from `fbb8fa0`). Consistency verified before renaming: part-2's full 31-epoch history
  dict ≡ CSV at full float precision (30/30 rows), `.htm` log ≡ CSV line-by-line.
  Note: commit `ac3217d` ("cleaning") had also **deleted the archived CPU-attempt
  notebook** (`2026-07-16_c0_sharp.ipynb`, interrupted @11/60, best 0.667 @6) — against
  the never-remove convention; recoverable from git history (`562c145`) if the team
  wants it restored.

- **C2-lin probe + C2 §7 diagnostics complete** (2026-07-16, archived as
  `notebooks/runs/2026-07-16_c2_grl_probe.ipynb`, heads/caches on Drive under `C2/`):
  - C2-lin (frozen encoder, §5.3 recipe on `best.ckpt`): **val macro-F1 0.8410**,
    val accuracy 0.8023 — ≈ the end-to-end 0.8415, features linearly separable.
  - **ar_set probe: val accuracy 0.352 vs majority baseline 0.390** (macro-F1 0.116).
    Originally recorded as "first half of the §9 invariance evidence" — **that reading
    is withdrawn (2026-07-17)**: C1, with no adversary at all, scores 0.287 on the same
    probe, i.e. lower. The contrast the §9 claim needs does not exist.
  - persona probe: 0.928 = majority baseline exactly (val is 92.8% one person →
    uninformative here; qualitative as declared, §7).

- **C1-lin probe + C1 §7 diagnostics complete** (2026-07-16, heads/caches on Drive
  under `C1/`): C1-lin **val macro-F1 0.8835** (≈ the end-to-end 0.8871 → features
  linearly separable, encoder/cache confirmed correct); **ar_set 0.287 vs majority
  baseline 0.390**; persona 0.928 = baseline exactly. Executed notebook archived as
  `notebooks/runs/2026-07-17_c1_ce_probe.ipynb` (commit `79e8034`).

- **§7 ar_set probe is structurally unfit on the p2_lab val split** (found 2026-07-17,
  inspectable from the frozen artifacts, no run needed):
  - val = 9 traces over 5 AR-sets (**AR-3 absent**, AR-5 has 1 trace) → effective n is
    9, not 349 windows: 0.287 / 0.352 / 0.390 are the same number statistically;
  - **AR-1 and AR-2 are identical in every `AR_SET_METADATA` attribute** (P1, bedroom,
    M1, LOS — they differ only by campaign), and together they are 55% of val windows.
    The probe is asked to separate two physically indistinguishable sessions, so the
    majority baseline is unreachable by construction — hence the below-baseline scores.
  - Consequence: the §7 "grafico chiave" cannot be produced as designed, in either
    direction. Not a bug; the doc's expectation ("alto in C1/C3, verso la baseline in
    C2/C4") is contradicted by the data.

- **Domain-readability diagnostic on C1 train features** (2026-07-17, archived as
  `notebooks/diagnostics/2026-07-17_c1_ce_domain_probe.ipynb` — a new folder for
  investigation sessions, see its README; imports the frozen §5.3 recipe, does NOT
  modify `probe.py`;
  train features only, no val selection, no test contact §0.7). Inner **trace-disjoint**
  stratified split of the 81 train traces (55 fit / 26 eval, all 6 AR-sets present),
  probed for several targets against each target's own majority baseline:

  | target | acc | baseline | delta |
  |---|---|---|---|
  | **y (positive control)** | **1.000** | 0.197 | **+0.803** |
  | ar_set | 0.196 | 0.286 | −0.090 |
  | ambiente | 0.854 | 0.854 | +0.000 |
  | direct_path | 0.633 | 0.731 | −0.098 |
  | persona | 0.818 | 0.818 | +0.000 |
  | monitor | 0.499 | 0.584 | −0.086 |

  - `ambiente`/`persona` are **exactly** constant predictors (macro-F1 0.4606 and 0.4500
    match the constant-predictor arithmetic to 4 digits) and select epoch 1: nothing
    beats predicting the majority.
  - The `y` control validates the plumbing but **saturates**: the eval traces are still
    *train* for the encoder, so 1.000 is C1's train accuracy, not generalization. The
    val probe (unseen traces) gives 0.8835 — that gap is the declared memorization
    confound, quantified.
  - **Reading:** on the same traces, with the same features, activity reads at 100% and
    domain reads at nothing. C1's features are a near-pure activity code — the CE
    objective never needed the environment, so the encoder never kept it.
  - **Verdict: the GRL had nothing to remove.** C2 paid 4.6 val pts + early stop for a
    regulariser with no target. Honest scope: "not *linearly* readable, on traces the
    encoder has seen" — though the adversary's own MLP (256→128→6) failed too.
  - Root cause is structural: **train has 2 environments, one of which is a single
    AR-set** (AR-6 living room; AR-1…AR-5 all bedroom). There is no second living-room
    set to generalise to, so environment-invariance is barely definable on this train.

- **Confirmatory domain diagnostic on C2 train features complete** (2026-07-17, archived
  as `notebooks/diagnostics/2026-07-17_c2_grl_domain_probe.ipynb` — identical code to the
  C1 session, run where only C2's cache was staged):
  - Every domain target sits at its majority baseline on C2 too (ar_set −0.030,
    direct_path +0.008, monitor +0.015; `ambiente`/`persona` are the exact same constant
    predictors as C1) → the adversary removed nothing; on macro-F1 the domain is even
    slightly *more* readable than C1 (ar_set 0.144 vs 0.066) — noise, but in the
    opposite direction to the GRL's purpose.
  - **`y` control: 0.893 on traces the encoder trained on, vs C1's 1.000** → the GRL
    cost train fit itself, not just transfer (memorization gap: C2 0.893→0.8415 val,
    C1 1.000→0.8835 val). Answers the fit-vs-transfer question left open above.
  - Evidence for the GRL team call is now complete and symmetric: no readable domain in
    CE features, no extra invariance in GRL features, measured fit + val cost.

- **Split audit complete** (2026-07-17, code-only reread of frozen artifacts + split/data/
  harness code — no run needed): `p2_lab.json` is **correct**. Verified programmatically:
  train/val/test trace-disjoint (81/9/11; 101 traces × 4 antennas + excluded LOS = the 408
  inventoried streams), windows never cross traces, all 4 antennas of a trace on one side,
  μ/σ train-only, test = all of AR-7, and the dual-archive duplicates (`S4a_L`/`S4a_Lalt`,
  `S5a_L`/`S5a_Lalt`) all sit in train — no quasi-leakage. Two criticalities found,
  neither invalidating C1/C2 (identical split, metric and protocol → the contrast stands):
  - **val contains no C, E, S traces** (5 of 8 classes present; 4/9 traces are J):
    checkpoint selection is blind to 3 classes, and val macro-F1 is a **5-class** number
    (harness `macro_f1` averages over classes present in y_true, declared via the
    `absent_classes` CSV column) — NOT scale-comparable to the 8-class test macro-F1;
    expect test to read lower for this reason alone.
  - val is small and skewed by construction: 9 traces / 349 fused windows (H = one trace,
    25 windows, yet 1/5 of the macro-F1); AR-3's absence is deterministic, not bad luck
    (1 leftover trace after rare-cell pinning → round(0.15·1) = 0). Selection noise on
    top of the declared in-domain-only limitation.

- **C3 (SupCon phase A) run complete** (2026-07-17, Colab T4, 60/60 epochs over three
  sessions — epochs 1-25, 26-42, 43-60 — with two clean auto-resumes): train loss
  5.914 → 4.430, smooth decrease, plateaued on the annealed tail (last ~6 epochs within
  ±0.004). Throughput 1.03-1.05 s/step vs the gate's 1.373 projection → ≈5.5 h GPU total,
  well inside the ≤8 h budget. §4.2 sampler invariants hold in every logged epoch
  (300 batches, min reuse offset 400 ≥ 340). No in-loop selection by design (§6-C3,
  `best_val_macro_f1` = -1): deliverables are **epoch40/50/60.ckpt** on Drive under `C3/`
  (grid epochs all completed; confirm the three files exist before phase B). Archived as
  `notebooks/runs/2026-07-17_c3_supcon_part1.ipynb` (epochs 26-42), `_part2.ipynb`
  (epochs 43-60; full 60-epoch history in its final dict) + `_part1_history.csv`
  (epochs 1-42 snapshot — the first session's notebook was not saved, declared C0-style
  exception; CSV vs part-2 dict verified to ≤3.3e-13 relative, float truncation only).

- **C3 phase B complete** (2026-07-17, owner A via a Drive shortcut to owner B's `C3/`
  — write access verified in-session; archived as `notebooks/runs/2026-07-17_phase_b_c3.ipynb`):
  grid probed with the frozen §5.3 recipe → **selected epoch 40, val macro-F1 0.8190**
  (epoch50 0.8150, epoch60 0.8120 — the whole grid spans 0.7 pts and one fused window
  of val accuracy: plateau confirmed, the last 20 phase-A epochs bought nothing).
  C3-lin sits ~6.5 pts under C1-lin (0.8835) and under C2-lin (0.8410) on the 5-class
  val — declared-noisy, the final test decides. §7 record for C3 collected in the same
  session with the known underpowered-val caveats: ar_set 0.289 vs baseline 0.390
  (≈ C1's 0.287), persona 0.928 = baseline exactly — third encoder, third loss family,
  same instrument-limited result. Only `epoch40.ckpt` + `probe_head_epoch40.npz` go to
  the final test session.

## In progress

- **C3 domain diagnostic (the C4 gate)** running (owner A): template
  `notebooks/diagnostics/2026-07-17_c3_supcon_domain_probe.ipynb` on the phase-B
  feature caches — do SupCon features retain the domain CE discards?

## Next steps (in order)

1. Every finished run: executed notebook committed verbatim to `notebooks/runs/`
   (`YYYY-MM-DD_<config>.ipynb`) + STATUS line, same commit. Val only, never test.
2. **Team discussion — the GRL premise, not λ_max.** The old item here was "pick λ_max
   for C4"; it is moot — no λ removes information that is already absent. Evidence is
   now complete (C1 + C2 domain diagnostics in Done). What needs deciding: (a) does the
   GRL branch survive at all, (b) what §7 reports now that its key figure cannot exist,
   (c) whether C4 (~7 h) is still worth launching. Proposed framing: not "the GRL
   failed" but "on this dataset a plain CE encoder already yields features with no
   readable environment — the GRL is redundant, and we showed it".
3. **C4 only if the branch survives step 2** (the in-flight C3 domain diagnostic is
   that call's last evidence: SupCon compresses less than CE, so the GRL premise may be
   alive for C4 even though it was dead for C2), then its phase A + phase B.
4. **Single final test session** via notebook `05` (§0.7) once ALL streams have a
   val-selected checkpoint: readiness assert, then evaluate_c0 (C0), evaluate (C1/C2),
   evaluate_features (C1-lin/C2-lin/C3/C4), `viz.metrics_table` + confusions; commit
   `reports/final/` (per-AR-set CSVs + `test_invocations.jsonl`) in the same commit as
   the archived notebook (§0.5: <~2 points = "comparable"). Whoever runs it needs
   Editor-shortcut access to EVERY run folder (C0…C4) from one account, as done for C3.
5. Housekeeping: delete the scratch `C1_smoke` folder on Drive (still pending).

## Blockers / open decisions

- **Does the GRL branch (C2/C4) survive?** — BLOCKS C4, does not block C3 or phase B.
  Evidence is now complete (Done above): the domain is not readable from a plain CE
  encoder (no target for the adversary), the C2 encoder is no more invariant than C1's,
  and the GRL measurably cost both train fit (`y` control 0.893 vs 1.000) and val
  macro-F1 (−4.6 pts). Team call, not a solo tune (§7 is the doc: a discrepancy gets
  discussed, never silently resolved).
  Scope note for the call (2026-07-17 split audit): on this protocol the ar_set target
  is dominated by within-environment session distinctions (5 of 6 train domains are the
  same bedroom; AR-1/AR-2 metadata-identical), `ambiente` ≡ AR-6 membership (one
  living-room set, 85/15 windows), and the test environment is outside the adversary's
  label space by LOEO construction — so retargeting the adversary to `ambiente` would
  not rescue C4 either. The negative result is dataset/protocol-scoped, not a general
  claim about GRL.
- **What does §7 report?** — its ar_set probe cannot support the committed claim in
  either direction on this split. Options: (a) declare it underpowered and rest §9 on
  the diagnostic above, (b) re-target the probe (`ambiente`, or AR-1/AR-2 merged) — a
  change of target vs the doc, so team call. Note the deeper limit no probe fixes: val
  is in-domain by construction, and the unseen environment (S7) is test, closed until
  §0.7's single session.