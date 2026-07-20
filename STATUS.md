# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-19 · **Phase: v5.2 tail — E1′ closed at n=2 (C1 seed-stable, GRL-specific instability), C1_s43 cache landed; E2′ S6-out domain diagnostic DONE (structural verdict replicates with the lab as 2nd env); NCM/kNN §7 complete for C1/C2/C3/C1_s43; ALL code deliverables implemented (T3A/AdaBN/domain-probe/concat — cross-review pending); §7 concat DONE (no CE↔SupCon complementarity); SupCon fair-shot DECIDED (C3-ft runs, seed-44 does not); C3-ft DONE + epilogue diagnostics DONE (hypothesis falsified 0.8183 ≈ C3-lin; SEVEN instruments agree on the SupCon ceiling, fine-tune visibly forgetting the init toward C1); cross-review + notebook-05 = the ONLY prep left before the single test session. All 13 row checkpoints exist.** · **Deadline: 2026-07-30 (code freeze 2026-07-28, §10.4)**

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

- **C3 domain diagnostic complete — the C4 gate closes** (2026-07-17, owner A,
  executed copy committed over the template in `notebooks/diagnostics/`, commit
  `1124a12` — archived without this STATUS/README line, added right after):
  - Control `y` = 0.995/0.993/0.996 over epoch40/50/60 (SupCon never trained a
    classifier, yet activities are ~99.5% linearly separable on seen traces).
  - **Every domain target at or below its majority baseline on all three grid
    checkpoints**; `ambiente`/`persona` are the exact constant predictors (0.4606 /
    0.4500 — third replication of the instrument across machines and encoders).
  - No maturity trend 40→60: the compression hypothesis does not hold for domain.
    Structural reading: P×K deliberately mixes AR-sets within each class and SupCon
    pulls same-class views together, so domain suppression is built into the phase-A
    objective itself — no adversary needed.
  - **Consequence: the GRL has no target under either loss family (CE or SupCon).**
    C4's expected outcome is "C3 plus noise" (C2 precedent: or worse). The remaining
    team call is between skipping C4 (~7 h saved, negative result reported with
    complete evidence) and running it as a pre-registered negative control — there is
    no remaining scenario in which C4 is promising.

- **E1 setup committed (2026-07-17):** rule-5 amendment recorded in
  `splits/CHANGELOG.md`; configs `c1_ce_s43`/`c2_grl_s43` (byte-identical to the
  seed-42 originals except name/seed) + pre-configured runner notebooks in
  `notebooks/e1_seed_replicates/` (RUN pinned, GPU sanity cell). Measures the §0.5
  seed noise floor + C2-findings robustness; C1_s43 features double as the E2
  concat control. C3 deliberately not replicated (declared). One pre-registered
  test row each; probes/techniques stay on seed 42.

- **Team decisions ratified (2026-07-17) — recorded as pipeline doc v5.2** (header
  block + amended §0.7, §2.2, §6, §7, §9, §10.3, §10.4):
  - **C4 closed without running**, on evidence: the GRL has no readable domain target
    under either loss family; expected outcome "C3 plus noise". Final table has one
    fewer stream; no extension involves C4.
  - **§7 ar_set probe declared underpowered** on p2_lab (structural — split-audit
    findings); §9 invariance evidence rests on the train-feature domain diagnostics
    (replicated on C1/C2/C3).
  - **Transductive rows pre-registered** for the single §0.7 test session: C1+AdaBN,
    C1+T3A, C1+both (ratified as "optional"; pinned **unconditional** in the same-day
    refinement pass — a conditional row inside a frozen list would reopen an in-session
    choice, §9) — C1 only, unlabeled test data, hyperparameters fixed a priori (T3A
    paper defaults), same logging harness; **row list frozen, no additions once the
    session is open** (§0.7 v5.2 note).
  - **Extensions redefined (§10.3):** E1′ = seed-43 replicates of C1/C2 (as committed
    above; seed 44 only on an explicit further call); E2′ = living-out rotation for C1
    only + domain-diagnostic replication on its train; C3 declared single-seed.
    Val-only NCM/kNN/concat diagnostics (seed 42, fixed hyperparameters, incl. the
    C1⊕C1′ ensemble control) added to §7.
- **Doc refinement pass (2026-07-17, doc/CHANGELOG-only, no code):** frozen §0.7
  test-row list enumerated (12 rows; notebook-05 readiness assert will check it
  hard-coded); C1+AdaBN+T3A pinned unconditional; NCM/kNN hyperparameters declared
  (cosine on L2-normed features, kNN k=20 majority vote + mean-similarity tie-break,
  score-mean antenna fusion); T3A pinned M=20 a priori (the paper has no single
  default — it selects filter_K per dataset via validation, which would be tuning);
  t-SNE subsampling fixed at 8 windows/trace; §0.5 rule-5 pointer to the E1′
  amendment; §8.4/§11 C4 rows annotated closed (budget note: E1′+E2′ ≈ 7.1 h, same
  order as the ~6.9 h freed by C4, well under the pre-v5.2 extension envelope);
  stale `E2` refs → `E2′` (§0.3, §1.4); deadline recorded 2026-07-30 →
  freeze 2026-07-28; E2′ artifacts pre-declared (`splits/p2_living.json`,
  `configs/c1_ce_s6out.yaml`, Drive `C1_s6out`); CHANGELOG E-label collision fixed.
- **Transductive + viz specs pinned (2026-07-18, doc-only, §9):** T3A fixed as a
  declared **batch variant** on cached features (single assignment with the initial
  prototypes = L2-normed head weight rows, bias dropped as in the paper; per-class
  top-M=20 entropy filter; adjusted prototype = renormalized mean of initial ∪
  supports; prediction path = §1.3 unchanged → the row differs from C1 only in the
  head). Chosen over the paper's online pass because that is order-dependent: an
  arbitrary processing order would have to be declared and results would depend on
  it — a free parameter with no benefit under the declared batch-deployment
  assumption; deviation declared in the report. AdaBN fixed as reset + cumulative
  full-pass BN re-estimation (`momentum=None`, batch 256, inventory order, weights
  untouched) — momentum updates weight recent batches exponentially (order-
  sensitive). t-SNE figure pinned to **train features** — the only split with all
  6 AR-sets and both environments (val: 9 traces, 5 AR-sets, AR-3 absent; test =
  single domain S7 → no cross-domain structure to show); same declared scope as
  the §7 train-feature diagnostics, zero test contact.
- **E2′ setup committed (2026-07-18, code-only — the freeze itself happens in the
  Colab session):** `build_p2_rotation` gains a blocking `reference` consistency
  check against the frozen primary rotation (identical trace universe — checked
  before μ/σ so a diverged inventory fails fast — identical axes/window/classes,
  and own non-copied μ/σ; any failure aborts before the JSON is written; negative
  paths verified locally); config `c1_ce_s6out.yaml` (byte-identical to C1 except
  name/protocol/split_file, seed 42, Drive folder `C1_s6out`) + pinned runners in
  `notebooks/e2_living_out/` (one-shot split session with day-1 gates, session
  scratch reports, contingency inspection and a §0.1 freeze guard; training
  runner gated on the frozen split being in the clone). **Local dry-run on the
  frozen universe predicts the exact partition** (assignment depends only on
  trace ids + seed 42, data values only enter μ/σ): train=79 val=7 test=15
  pinned=41. Declared: val = AR-1/2/4/5 only (no AR-3/6/7); val classes miss C
  and W → this rotation's val macro-F1 is a **6-class** number; the dual-archive
  twin pair separates — **S4a_L in train, S4a_Lalt in val** — a selection-side
  quasi-leakage (S6 test untouched); AR-7 (11 traces) entirely in train.
  *(Partition numbers and both caveats superseded by the review pass below.)*
- **E2′ review pass — declared caveats resolved (2026-07-18, before the freeze):**
  - **Twin quasi-leakage AMENDED away** (`splits/CHANGELOG.md` 2026-07-18,
    pre-registered): dual-archive `*alt` twins are two recordings of the same
    physical session → not independent split units; `build_p2_rotation` now binds
    them to their base's side (`bind_alt_twins=True` default). No retroactive
    inconsistency: p2_lab already satisfies the invariant by draw (all four twins
    in train) and is never regenerated (byte-identical regeneration would need
    `bind_alt_twins=False`, declared in the docstring).
  - **Validation:** the dry-run reproduces the frozen p2_lab partition EXACTLY
    (train/val/test/pinned) under pre-amendment mechanics — the methodology
    predicts the real session. Amended AR-6 partition: **train=80 val=6 test=15
    pinned=41**, both twin pairs in train, val = S1b_E, S1b_J2, S1c_S, S2a_R,
    S4a_C2, S4b_J1 (AR-1/2/4).
  - **Val class absence ACCEPTED per doc, not amended:** the amended draw gives
    val classes {C, E, J, R, S} (**H, L, W absent** → 5-class val macro-F1, same
    caveat family as p2_lab's 5-class val). §2.2 explicitly pre-accepts rare
    cells missing from val, and selection is within-run, where a k-class metric
    stays valid; forcing class coverage would change the §2.2 stratification
    itself (doc wins). Runner notebooks + README updated to the amended numbers.
- **Housekeeping: scratch `C1_smoke` folder deleted from Drive** (2026-07-17). Note:
  so far each collaborator manages run folders on their own Drive — the final test
  session needs Editor shortcuts to EVERY run folder (incl. new `C1_s43`/`C2_s43` and
  the S6-out folder) from one account; verify before the session day.

- **C2_s43 run complete** (2026-07-18, executed notebook committed in place at
  `notebooks/e1_seed_replicates/03_train_c2_grl_s43.ipynb` — kept there, not moved to
  `notebooks/runs/`, declared deviation from that folder's stated convention): best val
  macro-F1 **0.7870 @ epoch 6**, early stop at 16/40 (patience 10, clean single session,
  no resume). Two readings:
  - **§6-C2 finding CONFIRMED, robust to init:** `arset_train_acc` sits at 0.30–0.32
    across all 16 epochs (epoch 1: 0.257, epoch 16: 0.303) — the same majority-floor
    plateau as the seed-42 run, reproduced with an independent initialization. The
    adversary-learns-nothing finding does not depend on which seed drew it.
  - **§0.5 seed-noise floor MEASURED, and it changes what can be claimed:**
    |0.7870 − 0.8415| = **5.45 points** between C2 and C2_s43 — bigger than the
    "GRL costs −4.6 pts vs C1" gap this whole extension was built to calibrate. A
    same-config seed swing (5.45 pts) exceeding the cross-config gap (4.6 pts) means
    that gap is **not currently distinguishable from seed noise** on val macro-F1
    alone (the domain-diagnostic verdict on C4 is unaffected — it never relied on this
    gap, see the C1/C2/C3 train-feature diagnostics above).
  - **Pre-registered seed-44 trigger CONDITION MET** (`splits/CHANGELOG.md`
    2026-07-18 addendum: "a seed twin lands outside the §0.5 ~2-point band from its
    seed-42 sibling"): 5.45 pts is that case. Per the rule this makes `C2_s44` (and by
    the pair clause, `C1_s44`) launch-eligible — **decision deliberately held** pending
    `C1_s43` (does C1 show the same seed sensitivity, or is it C2/GRL-specific?),
    due by table-freeze day regardless of outcome.

- **C1_s43 run complete — E1′ measurements done** (2026-07-18, three sessions:
  epochs 1-8 launched by owner B (notebook not saved — declared C0-style exception),
  9-32 (snapshot archived as `notebooks/runs/2026-07-18_c1_ce_s43_part1.ipynb`, log
  through 31), 33-40 finished by owner A (`_part2.ipynb`, full 40-epoch history in its
  final dict — verified against part-1's log lines to print precision, ≤4.9e-05)):
  **best val macro-F1 0.8784 @ epoch 37**, full 40/40, no early stop — the same
  late-cosine-tail best epoch as C1 seed-42 (0.8871 @ 37).
  - **Seed swing on C1: 0.87 pts — inside the §0.5 band. The E1′ instability is
    C2/GRL-specific, not pipeline-wide**: C1 reproduces both the score (±0.9) and
    the training shape (best @ 37) across seeds; C2 swung 5.45 pts with a different
    best epoch (13 vs 6). The GRL doesn't just cost performance — it destabilizes
    training run-to-run, coherent with the noisy val trajectory and early stops
    already on record.
  - **The C1–C2 val gap direction now holds across seeds**: every C1 value
    {0.8871, 0.8784} beats every C2 value {0.8415, 0.7870}; the smallest
    cross-pair gap is 3.69 pts > the 2-pt band. n=2 per config — direction robust,
    magnitude uncertain (3.7 to 10 pts); report it as a range, not a number.
  - Archive regularized per the recorded plan: both parts + `2026-07-18_c2_grl_s43.ipynb`
    moved to `notebooks/runs/` (indexed), clean C2_s43 template restored in
    `notebooks/e1_seed_replicates/` from `8aa804b`.

- **§9 key figure produced** (2026-07-18, `notebooks/diagnostics/2026-07-18_embeddings_c1_vs_c3.ipynb`,
  asset `reports/embeddings_c1_vs_c3.png`): PCA-50→t-SNE on train features, C1
  (CE) vs C3 (SupCon), colored by activity and by AR-set. **By-activity:** 8
  tight clusters in both encoders; C3's L/S/E clusters visibly chain into one
  continuous shape rather than 3 discrete blobs — a plausible geometric
  reading of why the *linear* probe scores C3 lower (0.8190 vs C1-lin 0.8835)
  even though the structure may still be there for a non-linear readout.
  **By-AR-set (the figure's actual job):** colors uniformly mixed within every
  cluster in BOTH encoders, no visible domain sub-structure — a visual echo
  of the domain-diagnostic delta≈0 verdict, not an independent proof (train,
  seen traces — same declared scope as the §7 diagnostics). Fixed a real bug
  along the way: the first render appeared "cut off" — actually a duplicate
  Jupyter auto-display (bare `fig` as the last expression re-triggers Out[]
  on top of pyplot's own draw); fixed with the trailing-`;` convention already
  used in `03_train.ipynb`/`05_test_final.ipynb`. `sharp_har/viz.plot_embeddings`
  is package code — cross-review CLOSED 2026-07-18 (see the review pass below);
  T3A/AdaBN still pending theirs (CLAUDE.md: no logic in notebooks).

- **NCM/kNN + embeddings cross-review pass (2026-07-18, code+doc, local):**
  - **NCM/kNN scorers promoted** from the diagnostics notebook to
    `sharp_har/diagnostics.py` (declared deviation from the "diagnostics-style
    notebook" plan line): they are pre-registered §7 metrics with doc-declared
    hyperparameters, re-run across sessions (C3 rerun queued, C2 when unblocked,
    C1_s43 footnote) → pipeline code per the thin-notebook rule, not a one-off;
    the diagnostics README's dividing line is sharpened accordingly. Math
    verified **byte-identical** to the notebook-local cell that produced the C1
    numbers (synthetic equivalence check: NCM and kNN score arrays exactly
    equal; tie-break epsilon bound verified < 1/(4k) granularity) — **the
    recorded C1 rows stand**; owner A's executed C1 session still archives
    verbatim with the pre-promotion cell (declared). Blocking asserts added
    per project philosophy (zero-norm feature rows, train class coverage,
    n_train > k). `probe.py` untouched, as declared.
  - **kNN/NCM reference pool declared in §7** (refinement 2026-07-18): single
    train pool of (window, antenna) samples across the 4 antennas; the declared
    fusion operates query-side only. Was implicit in the first execution; kept
    (switching to per-antenna neighbors would invalidate the recorded C1 rows
    for no statistical benefit), now declared.
  - **§9 t-SNE recipe wording fixed:** "8 campioni (finestra, antenna) per
    trace" — the cache row unit, the semantics the committed figure was
    produced with — not "8 finestre" (ambiguous vs windows ×4 antennas).
    Sample-counting serves the declared anti-near-duplicate motivation better
    than 8 windows × 4 correlated antenna views; regenerating the figure under
    the other reading would cost a session for zero qualitative gain.
    `viz.plot_embeddings`: param renamed `samples_per_trace`, suptitle updated,
    `learning_rate="auto"` pinned explicitly (= sklearn ≥1.2 default; recipe
    can't drift with library defaults). **plot_embeddings cross-review closed**
    (recipe conformance §9, per-encoder determinism of the subsample, PCA guard,
    train-only assert all verified). Committed PNG untouched (measured artifact;
    its title says "windows/trace" — superseded wording, declared here).
  - Choices reviewed and deliberately KEPT (would-be changes = post-hoc tuning
    or churn): k=20, cosine/L2 metric, vote-fraction scores, perplexity 30
    (n≈650 points/encoder), PCA-50, init="pca", hardcoded declared linear-probe
    reference prints, majors-only dependency pins (stock-Colab philosophy).
    `umap-learn` dropped from requirements (never imported; §9's "t-SNE/UMAP"
    resolved to t-SNE, declared in the figure title).

- **E2′ P2-living split frozen** (2026-07-18, Colab session, executed copy
  `notebooks/runs/2026-07-18_s6out_split.ipynb`): `splits/p2_living.json`
  committed (§0.1, own train-only μ/σ). `build_p2_rotation(AR-6,
  reference=p2_lab.json)` passed the frozen-reference universe/metadata check;
  the printed partition matched the dry-run **exactly** — train=80 val=6
  test=15 pinned=41, both twin pairs (S4a_L/Lalt, S5a_L/Lalt) bound to train,
  AR-7 wholly in train, val = {S1b_E, S1b_J2, S1c_S, S2a_R, S4a_C2, S4b_J1}
  (AR-1×3/2×1/4×2), **val classes {C,E,J,R,S} — H/L/W absent → 5-class val
  macro-F1** (declared, §2.2), test = 15 AR-6 traces (S6a×9, S6b×6), single
  living-room domain.

- **C1 S6-out run complete** (2026-07-18, Colab GPU, `configs/c1_ce_s6out.yaml`,
  seed 42; archived `notebooks/runs/2026-07-18_c1_ce_s6out_part1.ipynb` (training,
  clean single session) + `_part2.ipynb` (CPU curves session, no GPU)): **best val
  macro-F1 0.7761 @ epoch 12**, early stop 22/40 (patience 10). `best.ckpt` on Drive
  `C1_s6out`. Two readings:
  - **Val trajectory is very noisy — best came mid-schedule, not on the tail.** The
    6-trace / 956-sample **5-class** val (H/L/W absent) swings 0.14 to 0.78
    epoch-to-epoch; best @12 with lr still high on cosine, epochs 15/19 (0.759/0.767)
    neared but never beat it, early stop @22. This is the C2-like "early best" shape,
    opposite to C1-S7's tail best @37 — driven by the tiny skewed val, not by the loss.
    Selection is fragile here (declared, 2.2 / split-audit family), but a within-run
    k-class metric stays valid for checkpoint choice.
  - **NOT scale-comparable to C1-S7's 0.8871:** different rotation, different 5-class
    set, noisier val. Do NOT read 0.7761 as "S6 is harder" — the S6-out number that
    enters the story is the **8-class TEST row** (once, 0.7), not this val. This run's
    only deliverable is the val-selected `best.ckpt` for that one pre-registered row.
  - Remaining on this rotation: the domain-diagnostic replication on its **train**
    features (9, the other reason E2' exists — 2nd environment is the lab here).

- **Pre-freeze implementation pass — every remaining code deliverable implemented**
  (2026-07-18, code+templates, local; **cross-review PENDING**, required §10.4):
  - **T3A** in new `sharp_har/transductive.py` (`t3a_head` + `head_weight_from_checkpoint`):
    §9 pinned batch variant, pure numpy on cached features; returns an
    `evaluate_features`-compatible head (adjusted prototypes, bias 0) → the C1+T3A row
    differs from C1 only in the head, as declared. Official-repo pre-check done (the
    cross-review item): filter grid {1,5,20,50,100,∞(-1)} confirmed; supports
    L2-normalized before averaging (official math, now explicit in §9); two declared
    deviations made explicit in doc+docstring — bias-free pseudo-labeling (official
    scores with the biased classifier) and initial prototype never filtered (official
    filters warmup supports too). Placement: test-row technique ≠ val-only diagnostic,
    so NOT in `diagnostics.py`.
  - **AdaBN** as `adapt_bn=True` on `harness.evaluate`/`cache_features` (the doc's
    "harness addition"): deterministic full-pass cumulative BN re-estimation
    (`momentum=None`, only BatchNorm modules in train mode, weights untouched),
    `_adabn`-infixed stems/filenames so the plain C1 artifacts can never be overwritten
    inside the single session, flag recorded in the §0.7 JSONL log + CSV/npz metadata,
    `ADAPT_BN_BATCH=256` enforced by assert; `evaluate_c0` refuses the flag (no C0+AdaBN
    row exists). §9 "ordine di inventario" wording refined: operative order = the eval
    loader's deterministic dataset order (declared, not silently resolved).
  - **Domain-probe instrument promoted** to `diagnostics.domain_probe`
    (+ `inner_trace_split`, `build_domain_targets`, `fused_head_scores`): the
    pre-registered S6-out replication makes it re-run pipeline code — same criterion as
    the NCM/kNN promotion. Math verified **byte-identical** to the executed C1/C2/C3
    notebook cell (full-row equality on synthetic data): the recorded rows stand; the
    three archived sessions predate the promotion (declared in the diagnostics README).
  - **`diagnostics.concat_caches`**: alignment-asserted concatenation for the §7 concat
    rows (per-row trace/window/antenna/y + labels/set_name contract asserts — a
    mismatch is a wrong-file error, never a reorder to repair).
  - **Templates committed:** `notebooks/e2_living_out/04_domain_probe_c1_s6out.ipynb`
    (one session: cache `C1_s6out` TRAIN features over p2_living — 50948 samples pinned
    by assert — then `domain_probe`; executed copy → `notebooks/diagnostics/`) and
    `notebooks/diagnostics/2026-07-18_concat_c1_c3.ipynb` (C1⊕C3 + C1⊕C1′ control,
    frozen probe recipe unchanged, SKIP-with-note while the C1_s43 cache is missing)
    — this template was consumed by its executed run, archived as
    `2026-07-19_concat_c1_c3.ipynb` (2026-07-19 repo cosmesis).
  - **Synthetic verification 25/25 PASS** (local suite): T3A ≡ an independent naive
    transcription of the §9 formula, bitwise determinism, empty-class/under-M edges,
    blocking asserts; AdaBN weights bitwise untouched, first-BN running_mean ≡ global
    channel mean (cumulative estimator), bitwise deterministic, eval mode restored;
    domain-probe full-row byte-equivalence; concat hstack + assert firing.
  - Deliberately deferred (not forgotten): the notebook-05 template extension
    (transductive rows + post-AdaBN caching + hard-coded row-list readiness assert) —
    the OPEN SupCon fair-shot call can still amend the frozen row list (a C3-ft row),
    and the assert must be written against the FINAL list; extending now would bake in
    a list that may legitimately change within days. Still on the §10.4 freeze
    checklist, with the report.

- **C1_s43 feature cache landed** (2026-07-18, owner B/Melissa on Colab, executed
  copy `notebooks/runs/2026-07-18_c1_s43_feature_cache.ipynb`): `cache_features` on
  `C1_s43/best.ckpt` → train **53400** + val **1396** (d=256) npz on Drive `C1_s43`,
  sample-count asserts passed. Unblocks the C1⊕C1′ concat control and the C1_s43
  NCM/kNN footnote (both now runnable).

- **E2′ S6-out domain diagnostic complete — structural verdict replicates on a
  second rotation** (2026-07-18, executed `notebooks/diagnostics/2026-07-18_probe_c1_s6out.ipynb`;
  caches `C1_s6out` TRAIN features over p2_living, 50948 samples pinned by assert,
  then `diagnostics.domain_probe`, inner trace-disjoint split 53 fit / 27 eval):
  - **Control `y` (8 cls) passes: acc 0.870, baseline 0.210, delta +0.660**, macro-F1
    0.852 — lower than the S7-rotation controls (1.000 / 0.893 / 0.995) because this
    inner split's eval traces are held-out *within train* (trace-disjoint), so 0.870
    is generalization, not memorization; large positive delta, plumbing sound.
  - **Every domain target at or below its majority baseline:** ar_set +0.011,
    ambiente +0.000, direct_path -0.029, persona +0.000, monitor +0.002 -
    `ambiente`/`persona` are again exact constant predictors. The "no readable domain
    in CE features" finding now holds on a rotation whose train second environment is
    the **laboratory** (S7), not living-room S6 - the fourth replication and the one
    E2′ existed to produce. No counterexample: the GRL-has-no-target verdict is not an
    artifact of the primary rotation's composition.

- **NCM/kNN §7 complete for C1/C2/C3/C1_s43** (2026-07-18, executed
  `notebooks/diagnostics/2026-07-18_ncm_knn_c1_c2_c3_full.ipynb` - Melissa now has the
  C2/C3 shortcuts that blocked the earlier owner-A run; renamed from the upload's
  `(1)` suffix during the 2026-07-19 repo cosmesis, the owner-A C1-only run stays
  archived as `...c1_c2_c3.ipynb`). Val, cached features,
  frozen §7 hyperparameters. Majority baseline 0.3209 throughout:
  | run | NCM acc / F1 | kNN acc / F1 | linear ref (F1) |
  |---|---|---|---|
  | C1 | 0.8653 / 0.8888 | 0.8453 / 0.8563 | 0.8835 |
  | C2 | 0.7765 / 0.8176 | 0.8424 / 0.8663 | 0.8410 |
  | C3 | 0.6963 / 0.7178 | 0.7937 / 0.8047 | 0.8190 |
  | C1_s43 | 0.8567 / 0.8707 | 0.8281 / 0.8497 | (0.8784 e2e) |
  - **C3 readout question resolved:** kNN (0.7937 acc) sharply beats NCM (0.6963) -
    the t-SNE chaining showed up as a non-linear-readout gain - but kNN macro-F1 0.8047
    still sits **under** C3's linear probe 0.8190. So the linear recipe did NOT
    understate SupCon: C3 is lowest under every readout tried. Report line stands.
  - **C1 robust to readout** (NCM/kNN within ~2 pts of linear); **C1_s43 footnote
    confirms seed robustness on NCM/kNN too** (within ~1 pt of C1 seed-42). C2 is the
    noisy one across readouts (NCM 0.7765 / kNN 0.8424).

- **§7 concat diagnostic complete — no CE↔SupCon complementarity** (2026-07-19,
  executed `notebooks/diagnostics/2026-07-19_concat_c1_c3.ipynb`; frozen §5.3 probe
  recipe unchanged, `diagnostics.concat_caches` alignment-asserted, val-only, no test
  contact). Both pairs ran (the control un-SKIPped once the C1_s43 cache landed):
  - **C1⊕C3 val macro-F1 0.8684** (−0.0151 vs C1-lin 0.8835); **control C1⊕C1′
    0.8882** (+0.0047 vs C1-lin); **candidate − control = −0.0197**, needed > +0.02
    (§0.5 band). A second CE encoder (the seed-43 twin) gives the small generic
    ensemble bump; **C3 does WORSE than that** — SupCon is not merely redundant with
    CE, it's a worse concat partner than a same-loss twin. Coherent with C3 losing on
    every readout (linear 0.8190, kNN 0.8047, NCM 0.7178) and the 16/349 error-overlap.
  - Caveats: both probes select epoch 1/6 on the fragile 9-trace/5-class val → the
    −0.0197 magnitude is noisy, but the direction is robust across all C3 evidence.
    The per-block feature-norm print (review caveat) was not added — moot here: the
    result is a null/negative and the norm-matched control took its expected bump, so
    the scale-asymmetry "spurious positive" risk demonstrably did not occur.
  - **Consequence:** Candidate B (joint CE+SupCon) stays future-work with a negative
    concat number, exactly as pre-declared; third convergent answer to "was the probe
    unfair to SupCon?" (no). Closes the last val-only §7 diagnostic.

## In progress

- **Seed-44 decision — FINAL (team-confirmed 2026-07-19): no s44 runs, E1′ closed
  at n=2.** Rationale: the open question the trigger was held for ("pipeline-wide or
  GRL-specific?") was answered by `C1_s43` (GRL-specific); every remaining claim
  is directional and already closed (no GRL target ×3 diagnostics; C1 stable
  0.87 pts; every C1 value > every C2 value, min gap 3.69 > band) — an s44 would
  only sharpen the magnitude interval, which no report claim uses. This is the
  rule's own default branch ("otherwise E1 stays at n=2, declared" — CHANGELOG
  2026-07-18), so no pair-clause deviation is needed. **Standing constraint for
  the report: the GRL val cost is stated as a RANGE (≈3.7–10 pts, n=2), never as
  the single seed-42 number (−4.6).**
- **C3-ft RUN COMPLETE (2026-07-19, owner A) — pre-registered hypothesis
  FALSIFIED, cleanly.** Archived as `notebooks/runs/2026-07-19_c3_ft.ipynb`
  (clean single session; the init log line verified: backbone from
  `C3/epoch40.ckpt`, head fresh). **Best val macro-F1 0.8183 @ epoch 4**, early
  stop 14/40. Readings:
  - 0.8183 ≈ C3-lin (0.8190): the best landed during warmup (LR still 8e-5,
    encoder ≈ init + aligned fresh head); the epoch peak LR hit (5), val dropped
    to 0.685 and never re-passed the best. Full-network fine-tuning does not
    lift the SupCon representation above its own linear-probe ceiling.
  - Gap to C1: **~6.9 pts** — outside the band, outside seed noise (0.87), same
    direction as every other readout. The hypothesis "comparable to C1 ±1"
    is falsified; with this, SIX independent instruments (linear probe, NCM,
    kNN, t-SNE, concat, full fine-tune) agree on the ~0.82 SupCon ceiling.
  - Declared post-hoc observation (labeled as such, NOT a rerun license): the
    early-best × patience-10 interaction stopped the run at 14, before the
    cosine tail where C1/C1_s43 found their bests — the "protected floor"
    (converging to a CE-like solution) never got the schedule time to
    materialize. Future-work sentence material.
  - **Process note (declared):** the run launched ahead of the wiring
    cross-review (owner A's call, after the launch-check line was designed in);
    the init line + the warmup trajectory behaviorally confirm the wiring, and
    the formal code review still applies pre-freeze. `best.ckpt` (epoch 4) on
    Drive `C3_ft` is the 13th-row checkpoint for the single test session.
  - **Epilogue diagnostics DONE (2026-07-20, declared no-decision-feeds session,
    `notebooks/diagnostics/2026-07-20_c3_ft_diagnostics.ipynb`) — all as
    pre-written, chapter fully closed:** domain probe = 5th replication, every
    target ≤ baseline (`y` control 0.981) → invariance holds across BOTH loss
    families composed in sequence; NCM 0.7736 / kNN 0.7984 confirm the ~0.82
    SupCon ceiling on the fine-tuned encoder too; t-SNE (asset
    `reports/embeddings_c1_c3_c3ft_diagnostic.png`, diagnostic — NOT the §9
    figure) shows the CE fine-tune *partially un-chaining* C3's L/S/E geometry
    (J, W/R became discrete C1-like blobs; L/S/E a residual filament, best
    stopped at epoch 4) — the fine-tune was turning the SupCon encoder INTO C1
    (forgetting the init), not adding to it. "SupCon buys nothing" now visible,
    not just numeric. Seven instruments agree.
- **Cross-review of the pre-freeze implementation pass** (T3A `transductive.py`,
  harness `adapt_bn`, `diagnostics.domain_probe`/`concat_caches`/`fused_head_scores`,
  the two new templates) — required before the single test session (§10.4). The
  official-repo grid/math pre-check and the 25/25 synthetic suite are recorded in
  Done; the reviewer re-confirms rather than starts from zero. All v5.2-tail local
  prep is now implemented — what remains on the rotations is running sessions, not
  writing code.

## Next steps (in order)

1. Every finished run: executed notebook committed verbatim to `notebooks/runs/`
   (`YYYY-MM-DD_<config>.ipynb`) + STATUS line, same commit. Val only, never test.
2. **E1 tail — DONE** (C1_s43 cache landed, see Done); only routine team
   confirmation of the semi-final no-s44 decision remains to close E1′ on paper.
3. **E2′ living-out — DONE** (split frozen + C1 S6-out run + S6-out domain
   diagnostic all complete, see Done; verdict replicates with the lab as 2nd env).
   The `best.ckpt` on Drive `C1_s6out` is ready for the single §0.7 test row.
4. **Val-only diagnostics — COMPLETE:** NCM/kNN (C1/C2/C3/C1_s43) and the §7 concat
   (C1⊕C3 vs C1⊕C1′ → no complementarity) all done and archived, see Done. Nothing
   left on this line before the single test session.
5. **C3-ft (approved 2026-07-19):** specify the training recipe (In progress), wire
   init-from-checkpoint in `train.py` + `c3_ft.yaml` + §8.4 budget + pre-registered
   hypothesis, run (~2 h), archive to `notebooks/runs/`, §0.7 row-list amendment to
   13 rows. Cross-review the wiring with the rest of the pre-freeze pass (step 6).
6. **Cross-review before code freeze** (implementation DONE 2026-07-18, see Done):
   T3A (`transductive.py`), AdaBN (harness `adapt_bn`), `diagnostics.domain_probe`/
   `concat_caches` + the new C3-ft init wiring. Then, with the FINAL row list fixed
   (now includes C3-ft): extend the notebook 05 template with the pre-registered
   transductive rows + post-AdaBN feature caching + the hard-coded frozen row-list
   readiness assert (§0.7) — deliberately deferred, see Done.
7. **Single final test session** via notebook `05` (§0.7) once ALL streams have a
   val-selected checkpoint: readiness assert; rows = the frozen v5.2 list ONLY
   (C0, C1 ± s43, C2 ± s43, C1-lin/C2-lin, C3, C1+AdaBN, C1+T3A, C1+both
   (unconditional, §9), the S6-out rotation's C1, **+ C3-ft (13th)**) — evaluate_c0,
   evaluate, evaluate_features, `viz.metrics_table` + confusions; commit
   `reports/final/` (per-AR-set CSVs + `test_invocations.jsonl`) in the same commit as
   the archived notebook. Editor shortcuts to EVERY run folder from one account,
   verified beforehand.
8. **Report + presentation** with the §10.4 v5.2 declaration list; code freeze
   2026-07-28 (deadline 2026-07-30); PCA+t-SNE figure C1 vs C3, domain-diagnostics
   table as the §9 key figure.

## Blockers / open decisions

- **DECIDED (team, 2026-07-19): the "fair-shot" SupCon extension = C3-ft (Candidate
  A) IS RUN; implementation to be specified (see "In progress"). Seed 44 is NOT run
  (E1′ final at n=2, above).** The C3-ft decision opens the §0.7 row-list amendment
  window (13th row) and now gates the notebook-05 readiness assert — the assert must
  be written against the list *including* C3-ft. Record of the call that led here
  (candidate survey after the C3 verdict — loses to C1 on all three readouts):
  - **Candidate A — CE fine-tuning of C3's encoder ("C3-ft"):** init the CE run
    from `C3/epoch40.ckpt`, full-network fine-tune, ~2 h. Answers the reviewer
    question the report will face anyway ("was the linear probe unfair to
    SupCon?") with a protected floor (worst case ≈ a CE from a different init).
    Expected outcome: comparable to C1 (±1 pt) — success is pre-defined as
    "comparable", not "beats". Needs: init-from-checkpoint wiring in `train.py`
    (small, cross-review), `c3_ft.yaml`, §8.4 budget line, **§0.7 frozen
    row-list amendment** (13th row — the list can be amended only by a team
    call BEFORE the single test session opens), hypothesis pre-registered.
  - **Candidate B — joint CE+SupCon (single encoder) — downgraded to future
    work, not recommended as a run:** of its two payoff channels, the
    information-capture channel is empirically weakened (error-overlap C1 vs
    C3: 16/349 rescue windows, unstructured — the encoders are largely
    redundant), leaving only the modest regularization channel; plus extra
    wiring, an α fixed blind, and the C2 precedent that a second gradient on
    the CE encoder is never free. Propose as a future-work sentence with the
    concat number in it.
  - **Excluded on evidence (not preference):** longer training (grid 40/50/60
    flat-to-declining; C3 already had more gradient steps than C1); bigger
    batch in any form (phase-A optimization shows zero starvation symptoms —
    glass-smooth loss, ±0.004 tail, one-fused-window grid spread, y-control
    99.5%: the objective's optimum is the problem, not the optimization;
    additionally the SupCon large-batch lever targets many-class regimes —
    with 8 classes P×K already yields ~64 views/class/batch); GradCache
    (mathematically invalid with BatchNorm); memory banks/queues (same
    saturation + review burden); τ/augmentation tuning (no selection budget on
    a 9-trace val; §3 frozen).
  - Timing: DECIDED 2026-07-19. The C3-ft run + init-from-checkpoint cross-review
    + `c3_ft.yaml` + §8.4 budget line + §0.7 row-list amendment (13th row) +
    pre-registered hypothesis must all land before the 2026-07-28 freeze and before
    the single test session opens. Open sub-decision: the C3-ft *training recipe*
    (see "In progress").
- None else blocking. Both former open calls closed 2026-07-17 by the v5.2 team
  decisions (GRL branch: C4 never runs; §7: underpowered, §9 rests on the
  domain diagnostics).
- Resolved 2026-07-18 (was: S6-out twin quasi-leakage): closed by the
  pre-registered twin-binding amendment (`splits/CHANGELOG.md` 2026-07-18) —
  see the E2′ review pass in Done. Nothing left to ratify beyond confirming the
  expected partition printout at the freeze session.
- Minor, non-blocking: seed 44 was floated in planning but the committed E1
  amendment deliberately stops at one replicate per config (n=2 → observed range,
  no significance claims) — adding s44 would be a new explicit call. (The former
  note (ii), the CHANGELOG E-label collision, was fixed in the 2026-07-17 doc
  refinement pass.)