# PROJECT — WiFi CSI Human Activity Recognition on SHARP, under Leave-One-Environment-Out

> **Single source of truth for this project.** It supersedes and replaces
> `pipeline_wifi_har_v5.md`, `STATUS.md`, `CONSOLIDATION_REVIEW.md`,
> `CONCEPTUAL_STRESS_TEST.md` and `NOTEBOOK_06_REVIEW.md`, which are archived
> under `docs/archive/` — each with a supersession banner, kept verbatim below
> it — for the record only. Nothing should be read from them any more.
>
> **Part I keeps the pipeline document's section numbering (§0–§9)** because
> ~380 docstring citations in `sharp_har/` point at it (`Ref. §4.2`, `Ref. §9`, …).
> Those references resolve here. Parts II–VI are new and unnumbered.
>
> Every number in this document traces to a committed artifact:
> `splits/*.json`, `reports/gate_*.json`, `reports/final/`, `report/tables/*.csv`.
> Nothing is quoted from a chat log or from notebook stdout.

**Status — 2026-07-25.** Experiments are **closed**. The single test session ran
2026-07-22 (16 pre-registered rows, one shot, full audit trail); the analysis
closed 2026-07-23 (8 measured tables); the IEEE report is drafted, refined and
builds to exactly 6 pages. Remaining work: report polish and the presentation.
Deadline 2026-07-30, code freeze 2026-07-28. **No new run, no new test contact,
no split regeneration.**

**The result in five lines.** A plain cross-entropy ResNet baseline (C1) is the
best model on the held-out environment (test macro-F1 **0.804**, accuracy
0.805, 11 traces). Supervised contrastive pre-training (C3) is **worse**
(−0.078 accuracy, CI excludes zero) under every readout tried, and a full
fine-tune from it does not recover. A domain-adversarial GRL (C2) is worse still
(−0.158, CI crosses zero) **because it has no target**: no domain attribute is
linearly readable from the encoder's train features, replicated on five
encoders and two rotations. This is structural — the dataset's environment
incidence (bedroom = 5 capture sets, living room = 1, laboratory = 1) means
*no* leave-one-environment-out rotation of it poses environment-invariance
non-degenerately. The deep backbone, not the objective, is what matters: the
paper's own shallow network loses 23.5 points in the identical recipe.

---

## Contents

- **Part I — Specification** (§0 principles · §1 data · §2 splits · §3 augmentation · §4 batching · §5 architectures · §6 configurations · §7 diagnostics · §8 optimization & budget · §9 evaluation)
- **Part II — Execution record** (gates, runs, the single test session)
- **Part III — Measured results**
- **Part IV — Findings, declared limitations, allowed wording**
- **Part V — Decision log** (pre-registrations and amendments)
- **Part VI — Repository, reproduction, remaining work, sources**
- **Appendix A — Supersession map** (where each archived document went)

---

# PART I — SPECIFICATION

## §0. Non-negotiable principles

Violating any of these invalidates results. They held for the whole project.

1. **Splits are frozen** as JSON on Git on day 1 and never regenerated or edited.
2. **Split by trace, never by window.** Windows of one trace on two sides of a
   split is leakage. All 4 antennas of a trace stay on the same side.
3. **μ/σ from train only**, per rotation, stored in that rotation's split file,
   reused identically on val and test.
4. **Every run saves** its full YAML config, seed, git hash, checkpoints and
   per-AR-set CSV metrics. **One shared eval harness** (`checkpoint → CSV`) for
   every stream.
5. **Single seed 42** (init, per-epoch sampler reseed, augmentation).
   Differences below ~2 points are *comparable*, not improvements.
   *Amendment (E1′):* two pre-registered seed-43 replicates (`C1_s43`,
   `C2_s43`); seed 42 remains primary; probes, diagnostics and test-time
   techniques stay on seed 42 only.
   *Declared:* a fixed seed is **not** bit-exact reproducibility — cuDNN
   benchmark is on for throughput.
6. **No run outside the budget table (§8.4)** and none before its throughput
   gate passed.
7. **The test set is evaluated once, at the end**, with val-selected checkpoints
   only. Every test access goes through the logging harness
   (`test_invocations.jsonl`), including C0's SHARP-repo-style evaluation via
   wrapper. **The row list is frozen before the session opens and never extended
   with the session open.** Final list = 16 rows (below). Transductive rows
   (AdaBN/T3A) use *unlabeled* test data inside that same single session.

**The frozen 16-row test list** (amended 12→13→16→17→16, always with the session
closed): C0 · C1 · C1_s43 · C2 · C2_s43 · C1-lin · C2-lin · C3 (epoch 40, phase-B
selected) · C3-ft · C1+AdaBN · C1+T3A · C1+AdaBN+T3A · C1 S6-out · C1_aug ·
C1_s6out_aug · C1_sharplike. Per-antenna appendix rows are a different
aggregation of the same invocations, not extra rows.

## §1. Data and preprocessing

### §1.1 Dataset reality

The shared Drive folder `DATASET_SHARP` holds the **SHARP TMC** dataset
(`doppler_traces.zip` ≈ 762 MB + `doppler_traces_S4_S5.zip` ≈ 10 MB), *not* the
extended 7-environment/13-subject dataset of the IEEE Comm. Mag. paper. Training
data comes **exclusively** from this folder; never re-download from IEEE
DataPort. AR-8 / AR-9 and campaigns d/e do not exist.

Capture sets **S1–S7 ≡ AR-1…AR-7** (1:1), **12 campaigns**, identical hardware
everywhere (Asus RT-AC86U monitor, Netgear X4S link) — so there is no
hardware confound; domains differ by room, monitor position, subject, day and
LOS/NLOS.

| Set | Campaigns | Environment | Monitor | Subject | Path |
|---|---|---|---|---|---|
| S1 | a, b, c | bedroom | M1 | P1 | LOS |
| S2 | a, b | bedroom (different day) | M1 | P1 | LOS |
| S3 | a | bedroom | M1 | P2 | LOS |
| S4 | a, b | bedroom | M2 | P1 | **NLOS** |
| S5 | a | bedroom | M2 | P2 | **NLOS** |
| S6 | a, b | living room | M3 | P1 | LOS |
| S7 | a | **laboratory** | M4 | **P3** | LOS |

**Environment incidence — the load-bearing structural fact:**
bedroom = {AR-1…AR-5} (5 sets), living room = {AR-6} (1 set), laboratory =
{AR-7} (1 set). No environment other than the bedroom is represented by more
than one capture set. See §9 / Part IV for the consequence.

**Classes.** `n_att = 8`, letters from the file names: **C, E, H, J, L, R, S, W**.
Five form the reference paper's core set — E (empty room), W (walking),
R (running), J (jumping), L (sitting still); C (sit-down/stand-up), H (arm
exercise) and S (standing still) come from the extended set. The H and S glosses
are **not** documented in the released package: they were supplied by the team
on 2026-07-25 for the report and are descriptive only — no code, split or metric
depends on them. **C0 uses the 5-class paper set** `["E","J","L","R","W"]`;
C1–C4 use all 8.

**Day-1 inventory (`reports/inventory.csv`, `reports/contingency.csv`).**
408 file-streams = 101 traces × 4 antennas + the excluded LOS trace. Axes
verified **340 time steps × 100 velocity bins**, STFT hop = Tc ≈ 6 ms, **0 % NaN**.
Three real-data findings, fixed in `inventory.py` and frozen into the splits:

- `FILENAME_PATTERN` extended for activity repetition numbers (`S4a_C1`,
  `S6b_J_0`, …) — the letters-only regex was silently dropping every repeated
  recording.
- `S5a_LOS` excluded as a non-activity calibration trace
  (`NON_ACTIVITY_LABELS`, logged to `reports/excluded_traces.csv`).
- `S4a_L` / `S5a_L` appear in **both** archives with different `n_frame` — two
  distinct physical recordings, not copies. Split into `S4a_Lalt` / `S5a_Lalt`
  rather than discarding one (`resolve_duplicate_streams()` +
  `assert_no_duplicate_files()`, logged to `reports/duplicate_traces_split.csv`).
  These twin pairs are **bound split units** (§2.2).

### §1.2 Windowing

- Unit: per (trace, antenna) a Doppler time × velocity matrix.
- **Window = 340 time steps × all 100 velocity bins.** Incomplete tail windows
  are dropped.
- **Train stride 100** (~71 % overlap) — a redundancy reduction motivated by
  compute budget alone. **Val/test stride 340** (disjoint windows), so
  evaluation units are uncorrelated; ~3.4× fewer eval windows, irrelevant to
  budget.
- **All window counts in this project are in (window, antenna) samples**, i.e.
  already ×4 over antennas. ~158 train-stride / 47 eval-stride windows per
  antenna per trace over the 101 traces (~165 / 49 on the p2_lab train side,
  whose traces run longer); primary rotation = **53 400** train samples,
  and 349 / 425 fused eval windows on val / test.
  *Corrected 2026-07-26:* this bullet previously read "~197 / 58 … ≈ 69 k",
  which no artifact supports; the figures above are recomputed from
  `reports/inventory.csv` + `splits/p2_lab.json`. The report always carried
  the right number (§IV, "≈53k"), so nothing measured depended on the error.
- Windows inherit activity, subject, environment, AR-set, trace-id and antenna
  from the parent trace.

### §1.3 Antennas

4 antennas → 4 Doppler streams per trace. **Training:** each (window, antenna)
is an independent sample. **Val and test:** fusion per window = **mean of the
softmax over available antennas, then argmax** — identical in val and test.
Per-antenna rows go in the appendix (same forward pass, different aggregation).
*Exception:* C0 uses the SHARP repo's decision fusion (§2.1).

### §1.4 Normalization

Single-channel input → μ and σ are **two global scalars** computed over all
train windows of all 4 antennas, after windowing and before augmentation:
`(x − μ_train)/σ_train`, identical on val/test, stored in the rotation's split
file. Declared and accepted: computing μ/σ on overlapping windows weights
central frames ~3.4× more than edges — numerically negligible for two scalars,
and it keeps a single code path.

Frozen values: p2_lab μ 0.118294 σ 0.180123 · p2_living μ 0.119122 σ 0.180886 ·
p1_sharp μ 0.117743 σ 0.180229.

## §2. Data splitting

### §2.1 Protocol P1 — SHARP reproduction (C0 only)

Train = S1 (all campaigns a/b/c) as in the paper/repo; **val = 20 % of S1
traces** (by trace, seed 42) for early stopping — a **declared deviation**: the
SHARP repo trains on all of S1. Test = S2–S7. Frozen partition
(`splits/p1_sharp.json`): **train 22 / val 5 / test 74 traces**.
Evaluation uses the **repo's decision fusion** (majority vote over antennas,
softmax-sum tie-break — verified against `francescamen/SHARP` `CSI_network.py`),
invoked through the harness wrapper so it is logged like everything else (§0.7).

### §2.2 Protocol P2 — cross-domain (main protocol, C1–C4)

**Primary rotation: leave-S7-out (laboratory)** — the hardest set: environment,
day *and* subject (P3) never seen in train. Train = S1–S6 (6 domains for the
adversarial head), val = 15 % of train traces stratified by (AR-set, activity),
seed 42.

**Rare-cell guarantee (mechanical).** For every (AR-set, activity) cell with
< 4 traces the split script **pins one trace (seed 42) into train**, then
stratifies the rest (degrading to AR-set-only for that cell). A **blocking
assert** then requires every inventory cell to have ≥ 1 train trace. Accepted
consequence: rare cells may be absent from val.

**Twin binding.** `*alt` dual-archive twins are two recordings of the same
physical session → not independent split units; they are removed from the
stratification pool and assigned to their base trace's side
(`build_p2_rotation(bind_alt_twins=True)`, default from 2026-07-18). `p2_lab`
already satisfies this by draw and is never regenerated.

**Adversarial label = AR-set identity** (campaigns aggregated; 6 train domains
in the primary rotation). The report says **"AR-set invariance"**, never
"environment invariance" — room, monitor, subject, day and LOS/NLOS are
confounded inside a set. Subject is measured diagnostically only (§7).

**Second rotation (E2′): leave-S6-out (living room), C1 only.**
Leave-bedroom-out is **declared infeasible** (26 remaining train traces) and was
formally rejected (Part V). Each rotation has its own split file and its own μ/σ.

**Frozen partitions:**

| Split file | Protocol | Train | Val | Test | Pinned | Val class coverage |
|---|---|---|---|---|---|---|
| `splits/p2_lab.json` | P2-lab (S7 out) | 81 | 9 | 11 | 40 | {H,J,L,R,W} — **C, E, S absent** |
| `splits/p2_living.json` | P2-living (S6 out) | 80 | 6 | 15 | 41 | {C,E,J,R,S} — **H, L, W absent** |
| `splits/p1_sharp.json` | P1 (5-class) | 22 | 5 | 74 | 0 | — |

**Known val weaknesses (declared, not fixable without unfreezing).**
p2_lab val = 9 traces / 349 fused windows over 5 AR-sets (AR-3 absent
deterministically, AR-5 = 1 trace; 4 of 9 traces are J; H = one trace = 1/5 of
macro-F1). **Val macro-F1 is therefore a 5-class number and is NOT
scale-comparable to the 8-class test macro-F1.** Blindness is *common-mode*
across all configs, so it cancels to first order in comparisons; it is a real
confound for absolute scale and per-class reading only. Same caveat family on
p2_living. Selection remains valid *within* a run.

**Split audit (2026-07-17, code-only re-read of the frozen artifacts):**
`p2_lab.json` is correct — train/val/test trace-disjoint, windows never cross
traces, all 4 antennas on one side, μ/σ train-only, test = all of AR-7, and all
four `*alt` twins in train (no quasi-leakage).

### §2.3 Split file format

```json
{"protocol": "P2-lab",
 "axes": {"time": 340, "velocity": 100, "layout": "time_x_velocity", "stft_hop_s": 0.006},
 "window": {"train_stride": 100, "eval_stride": 340},
 "classes": {"n_att": 8, "labels": ["C","E","H","J","L","R","S","W"],
             "c0_paper_set": ["walking","running","jumping","sitting","empty"]},
 "split_seed": 42,
 "pinned_train_traces": ["..."],
 "train": ["S1a_W", "..."], "val": ["..."], "test": ["S7a_W", "..."],
 "norm": {"mu": 0.118..., "sigma": 0.180...}}
```

Lists hold trace-ids, never window indices.

## §3. Data augmentation

On-the-fly in training, **after standardization**, fixed order: time shift →
time masking → velocity masking → amplitude scaling → gaussian noise. Masks
filled with **0** (= post-standardization mean). Fixed set, no ablation grid.

| Augmentation | Parameters (time 340 / velocity 100) | CE (`ce`) | SupCon views (`supcon_view`) |
|---|---|---|---|
| Circular time shift | shift ~ U{−10,…,+10} frames | p = 0.5 | p = 0.5 |
| Time masking | 1–2 masks, width ~ U{5,…,20} | p = 0.5 | p = 0.8 |
| Velocity masking | 1–2 masks, width ~ U{2,…,10} bins | p = 0.5 | p = 0.8 |
| Amplitude scaling | s ~ U(0.8, 1.2) | p = 0.5 | p = 0.8 |
| Gaussian noise | additive, σ = 0.05 | — | p = 0.5 |

- Widths assume 340×100 and rescale proportionally if the axes differ. Velocity
  is masked sparingly (≤10 %): it is the feature separating walking/running.
- **Forbidden:** velocity-axis flip (inverts the Doppler sign) and time flip
  (sit-down ↔ stand-up).
- The ±10 time shift is local robustness, **not** compensation for the offset
  diversity lost at stride 100.
- SupCon: 2 views = 2 independent augmentations of the same (window, antenna).

**Additive profile `ce_amp`** (C1-aug arm, pre-registered 2026-07-20): identical
to the CE column **except the amplitude channel** — s ~ U(0.6, 1.5), p = 0.8.
The table above is **unchanged**: `ce` and `supcon_view` stay byte-identical (so
every archived run still reproduces), the new profile is additive in
`augment.py` and selected per run by `train.augment_profile` (default `"ce"`,
CE path only — SupCon views are pinned to `supcon_view` by assert).
*Lever rationale:* global attenuation is the only manipulation that is
physically coherent on μ-Doppler, label-safe and in scope. Time-warp is rejected
on physics (time and velocity are coupled in μ-Doppler, so stretching time alone
models no real motion); velocity-warp on label safety (it moves walk → run,
the same reason the flip is forbidden); channel simulation needs raw CSI we do
not hold. One conceptual lever → clean attribution. This does **not** reopen the
eliminated augmentation ablation (§6): it is a single pre-registered, paired arm.

## §4. Mini-batch composition

### §4.1 CE (C1, C2)

Batch 256 windows, uniform shuffling over (window, antenna) samples.
**Unweighted CE, declared:** selection and reporting use macro-F1 while training
optimizes uniform CE — standard, kept for comparability.
**Declared C2-vs-C4 asymmetry:** in C2 the AR-set head sees uniform batches, so
the GRL gradient is dominated by data-rich sets; C4 would have balanced domains
via P×K. The majority baseline in the monitoring corrects the *metric*, not the
*gradient*. OOM fallback (CE only): batch 128 + accumulation ×2.

### §4.2 SupCon P×K sampler (C3, C4)

P = n_att = 8, K = ⌊256/8⌋ = 32 → **256 windows = 512 views**.
Per class: (1) round-robin over the train AR-sets where the class exists
(inducing "same activity, different domain" positives); (2) a trace without
replacement inside the current set; (3) uniform window, uniform antenna;
(4) **hard constraint: at most one window per trace per class per batch** while
traces ≥ K; below that, reuse with a different window **and minimum temporal
offset |Δstart| ≥ 340** (disjoint) — without it, stride-100 reuse would
reintroduce exactly the trivial positives the distinct-trace rule exists to
remove.
Declared: the round-robin over-samples trace-poor sets relative to their window
counts — intentional domain balancing.
**Sampler reseeded deterministically per epoch** (`seed_epoch = f(seed, epoch)`),
so resume reproduces the batch sequence without saving sampler state.
**Mandatory logging:** distinct AR-sets and unique traces per class, reuse count
and minimum observed offset. **Gradient accumulation is FORBIDDEN in phase A**
(SupCon is intra-batch: two half-batches give two independent losses). OOM
fallback: reduce K, declared.

## §5. Architectures

### §5.1 SHARP-like network (C0, and the `C1_sharplike` ablation)

Faithful to the TMC figure, Keras-style *same* padding: maxpool 2×2 ∥ conv
5@(2×2) ∥ conv 3@(1×1)→6@(2×2)→9@(4×4), concat, conv 3@(1×1), flatten,
dropout 0.2, dense → n classes. `feature_dim = 25500 = 3·⌈340/2⌉·⌈100/2⌉`;
`d_enc` does not apply (feature size fixed by geometry). Shape: a ~1 k-param
shallow conv front-end plus a ~204 k-param dense head — a **near-linear wide**
model.

### §5.2 Shared backbone for C1–C4: ResNet-18 variant **V-B**

The v3 "small-input" variant would have cost ~73 GFLOPs/window (~28× stock on
our 340×100 input) — out of budget. The stated goal only concerns the
**velocity** axis (do not crush it to ~3 bins), so time is downsampled early and
velocity little:

- **Stem:** conv 3×3 **stride (2,1)** + maxpool 3×3 **stride (2,1)** → time /4
  before layer1, velocity intact.
- **Stages:** layer2 stride (2,2); layer3–4 stride (2,1). Final map ≈ **11×50**.
- **Width ×0.5:** channels 32/64/128/256 → **d_enc = 256** (GAP → 256-d feature).
- **Cost: 4.66 GFLOPs/window, 2.79 M params** (2.93 GFLOPs under escalation (b)).
- Memory: activations ~1/16 of v3 → activation checkpointing was expected to be
  unnecessary and the day-3 gate confirmed it (peak 8.57 GiB allocated /
  11.37 GiB reserved on T4).
- **No architecture ablation** inside the core. Pre-committed escalation ladder
  if the day-2 gate failed: (a) steps/epoch 400 → 300; (b) add stride (2,2) to
  layer3; (c) fewer epochs. Only **(a)** was ever used, and only for phase A.

### §5.3 Heads — all parametrized on `d_enc`, no hardcoded counts

- **Activity classifier** (C1, C2 end-to-end): linear `d_enc → n_att`.
- **SupCon projection head** (C3, C4): MLP `d_enc → d_enc → 128`, ReLU,
  L2-normalized; discarded after pre-training.
- **Adversarial AR-set head** (C2, C4): GRL → MLP `d_enc → d_enc/2 → n_AR-sets`,
  ReLU, dropout 0.3.
- **Linear probe — one frozen recipe** for C1-lin, C2-lin, C3/C4 phase B and the
  §7 probes: encoder **frozen**, `d_enc` features extracted once and cached to
  disk; linear `d_enc → n_classes`; **Adam** lr 1e-3, wd 1e-4, batch 256, max 30
  epochs, early stopping on fused val macro-F1 from the shared harness,
  patience 5, best checkpoint. Minutes of compute.
- **Two declarations:** cached features are extracted **without augmentation**
  (so the apples-to-apples table compares encoders on clean features while the
  end-to-end rows trained and selected with augmentation in the loop); the probe
  uses **Adam** while the core uses AdamW — intentional, the probe recipe is
  frozen and separate, and identical for every row.

## §6. Experimental configurations

C1–C4 share the V-B backbone, the splits, the harness and seed 42, on P2.
C0 runs on P1 with the SHARP-like network.

| Config | Encoder trained with | Adversary | Eval | Protocol | Outcome |
|---|---|---|---|---|---|
| **C0** | CE, SHARP-like net | — | end-to-end, repo decision fusion (via logger) | P1, 5 classes | run |
| **C1** | CE (+ label smoothing 0.1) | — | end-to-end | P2-lab | run — **the denominator of every comparison** |
| **C1-lin** | CE | — | linear probe | P2-lab | run |
| **C2** | CE | CE-GRL | end-to-end | P2-lab | run |
| **C2-lin** | CE | CE-GRL | linear probe | P2-lab | run |
| **C3** | SupCon (2 phases) | — | linear probe | P2-lab | run |
| **C4** | SupCon | CE-GRL | linear probe | P2-lab | **closed without running** (evidence, 2026-07-17) |

Recipes:

- **C0** — CE; Adam lr 1e-4, batch 32, max 60 epochs, early stopping on val
  accuracy, patience 10. `train.augment: false` (the SHARP repo uses no
  augmentation — faithfulness wins, declared). Time-boxed at 3 person-days.
- **C1** — one activity head, CE with label smoothing 0.1; max 40 epochs
  (epoch = 400 steps), early stopping on **fused val macro-F1**, patience 10,
  best checkpoint.
- **C2** — activity head + AR-set head behind a GRL.
  `L = L_att + λ(p)·L_env`; **fixed ramp** `p = min(epoch/20, 1)`,
  `λ(p) = 2/(1+exp(−10p)) − 1` (λ ≈ 1 from epoch 20, independent of early
  stopping, otherwise C2 collapses into C1). **Mandatory monitoring:** AR-set
  head accuracy on **train batches, averaged per epoch**, read against the
  **majority baseline** (never 1/n). Only knob if it misbehaves: λ_max → 0.5
  (never used).
- **C3** — *Phase A:* encoder + projection head, SupCon (Khosla et al.),
  labels = activity, τ = 0.1, 2 views, P×K sampler; **60 fixed epochs, no early
  stopping**; checkpoints at 40/50/60 + `last.ckpt`. Pre-committed grid rule for
  any horizon H: ⌈2H/3⌉, ⌈5H/6⌉, H. *Phase B:* the frozen §5.3 linear probe on
  each grid checkpoint; the best **val macro-F1** is selected and **only that
  goes to test** — this is the operational definition of "plateau".
  Terminology in the report: **SupCon**, not NT-Xent.
- **C4** (design kept for the record, never run) — SupCon + CE-GRL together,
  `L = L_supcon + β·λ(p)·L_env`, β = 0.5, same ramp, same P×K; phase B identical
  to C3.

**Extension arms** (all pre-registered — Part V):

| Arm | What it is | Rotation / seed | Test rows |
|---|---|---|---|
| **E1′** | seed-43 replicates of C1 and C2 | P2-lab, seed 43 | 2 |
| **E2′** | leave-S6-out rotation, C1 only + domain-diagnostic replication on its train | P2-living, seed 42 | 1 |
| **C3-ft** | CE fine-tune of C3's encoder from `C3/epoch40.ckpt`, fresh head, peak lr 1e-4 (10× below C1), otherwise the C1 recipe | P2-lab, seed 42 | 1 |
| **C1-aug** | `ce_amp` profile, paired at fixed seed against the existing baselines, replicated across rotations | P2-lab + P2-living, seed 42 | 2 |
| **C1_sharplike** | the SHARP net inside the **exact** C1 recipe (only the backbone differs) | P2-lab, seed 42 | 1 |
| **AdaBN / T3A** | test-time adaptation on C1, unlabeled test data, hyperparameters fixed a priori | P2-lab | 3 |

**Ablations — only the free ones:** (1) antenna fusion vs per-antenna at test
(same forward pass, two aggregations) → appendix; (2) diagnostic probes (§7) on
cached features. Eliminated as out of budget: augmentation grid, β, architecture,
TCN.

## §7. Diagnostic probes

Frozen encoder, §5.3 recipe, cached features, **val or train only — never test**.

- **AR-set probe** — linear `d_enc → n_AR-sets`, accuracy against the
  **majority baseline**. *Verdict: structurally underpowered on p2_lab and
  abandoned as evidence.* Val has 9 traces over 5 AR-sets (AR-3 absent, AR-5 one
  trace) → effective n = 9, not 349 windows; and **AR-1 and AR-2 are identical in
  every metadata attribute** (P1, bedroom, M1, LOS — they differ only by
  campaign) while being 55 % of val windows, so the majority baseline is
  unreachable by construction. The pipeline's original expectation ("high in
  C1/C3, near baseline in C2/C4") is not testable in either direction. Not a bug
  — an instrument the split cannot support.
- **Subject probe** — qualitative only by design (subject and environment are
  confounded inside a set). Val is 92.8 % one person; it returns the majority
  baseline exactly.
- **Domain-readability probe on TRAIN features** — the instrument that actually
  carries the §9 invariance evidence, promoted to `diagnostics.domain_probe`.
  Inner **trace-disjoint** stratified split of the train traces; probes for
  several targets (`ar_set`, `ambiente`, `direct_path`, `persona`, `monitor`),
  each against **its own majority baseline**, with `y` (activity) as positive
  control. Declared scope: *"not **linearly** decodable on traces the encoder has
  seen"*, with the memorization confound quantified by the `y` gap.
- **NCM and kNN readouts** (val-only, seed 42, hyperparameters fixed a priori):
  features L2-normalized, cosine similarity; NCM = one L2-renormalized centroid
  per class, score = similarity to the centroid; kNN k = 20, score = vote
  fraction among the k neighbours; reference pool = the single train pool of
  (window, antenna) samples over the 4 antennas; antenna fusion = mean of class
  scores then argmax (the analogue of §1.3); tie-break = highest mean similarity.
- **Concatenation** C1⊕C3 (512-d, probe recipe unchanged) with the **ensemble
  control C1⊕C1′** (seed 42 ⊕ seed 43), to separate genuine CE/SupCon
  complementarity from generic ensemble diversity. Alignment between caches is
  asserted row-by-row (`diagnostics.concat_caches`); a mismatch is a wrong-file
  error, never something to repair by reordering.

## §8. Optimization, checkpointing, budget

### §8.1 Hyperparameters

- **Epoch = 400 fixed steps** (with a P×K sampler "one pass over the dataset" is
  ill-defined; step budgets are computable). Horizons: CE ≤ 40 epochs (16 k
  steps), phase A 60 epochs — **300 steps/epoch after escalation (a)**, so
  18 k steps.
- AdamW, wd 1e-4; lr 1e-3, cosine decay over the phase horizon, warmup 5 epochs
  (cosine is scheduled over the full horizon even when early stopping cuts in).
- AMP always on; gradient clipping at norm 1.0 (essential with the GRL);
  cuDNN benchmark on (see the determinism note in §0.5).
- **Declared asymmetry:** a phase-A epoch is 300 steps against 400 in the CE
  runs — epochs are budget units, not comparability units.

### §8.2 Checkpointing and resume

Per run on Drive: `last.ckpt` (overwritten every epoch) + `best.ckpt`
(+ the 3 phase-A grid checkpoints). The checkpoint carries weights, optimizer,
scheduler, **AMP GradScaler state**, epoch, config and **RNG states** (torch,
cuda, numpy, python). The P×K sampler stores no internal state — it is reseeded
deterministically per epoch, so a resume from an epoch boundary reproduces the
batch sequence. Auto-resume from `last.ckpt` is automatic; bit-exact resume was
verified on synthetic data for both the CE and the SupCon path.

### §8.3 Measured compute

Gates, not estimates (`reports/gate_day2.json`, `reports/gate_day3_phase_a.json`,
committed verbatim):

| Gate | Measured | Rule | Verdict |
|---|---|---|---|
| Day-2 staging | 57.3 s | ≤ 900 s | pass |
| Day-2 CE throughput (T4, batch 256) | 0.526 s/step warm (0.561 cold) → **C1 ≈ 2.34 h** | ≤ 4 h | pass |
| Day-3 phase A, real 512-view forward+backward with the real sampler | 1.373 s/step → **9.16 h** | ≤ 8 h | **NO-GO** → escalation (a) |
| Phase-A memory | 8.57 GiB alloc / 11.37 GiB reserved | — | no activation checkpointing needed |

Escalation (a): phase-A steps/epoch 400 → 300 (C3/C4 configs only; C1/C2 stay at
400, their gate passed; backbone untouched, since escalation (b) would break
shared-encoder comparability). Projected phase A **6.87 h ≤ 8 h**; realized
throughput was better still (1.03–1.05 s/step, ≈ 5.5 h) once augmentation moved
into the DataLoader workers.
Sampler invariants verified in the gate itself: per-class trace counts sum to
the frozen train partition, distinct AR-sets per class match the frozen
contingency table, **minimum reuse offset 400 ≥ 340**.

**Declared model-selection limit:** validation is in-domain (15 % of train), so
selected checkpoints may be sub-optimal for OOD; sacrificing one of ~5 train
environments as an OOD val costs too much.

### §8.4 Budget — final consuntivo

| Item | Runs | GPU-h |
|---|---|---|
| Core (C0, C1, C2, C3 phase A; phases B / probes < 1 h total) | 4 | ≈ 10–20 |
| C4 | 0 | 0 (~6.9 h freed) |
| E1′ seed replicates | 2 | ≈ 4.6 |
| E2′ living-out | 1 | ≈ 2.5 |
| C3-ft | 1 | ≈ 2.0 |
| C1-aug (after L6 dropped the third run) | 2 | ≈ 4.6 |
| **Tracked extensions total (the amended §8.4 figure)** | 6 | **≈ 13.7** (inside the pre-v5.2 15–35 h envelope) |
| C1_sharplike | 1 | ≈ 2.3 — approved as a **val-only** run, so it carried no §8.4 amendment; its promotion to a test row added 0 GPU-h (the checkpoint already existed). All-in measured extensions ≈ 16 h |

### §8.5 Data residency, I/O, staging

Package lives in the shared Drive folder `DATASET_SHARP`; each member adds a
**shortcut** (storage counts only against the owner, so every 15 GB free
account stays free for checkpoints). Per session: mount Drive → copy the two
zips to `/content` (sequential reads of large files — the case Drive handles
well) → unzip locally. **During training, read only from `/content`**; mounted
Drive holds checkpoints only (random reads from Drive = idle GPU + rate
limiting: forbidden). Default loading is lazy per-trace from local disk.

## §9. Evaluation protocol

- **Metrics:** accuracy and **macro-F1**, per test set (never only aggregate);
  confusion matrix for the primary rotation.
- **Macro-F1 with missing classes:** averaged only over the classes present in
  the evaluated set's ground truth; absent classes are listed per set
  (`absent_classes` column). Same definition in val and test.
- **Evaluation unit: the stride-340 (disjoint) window.** Prediction = argmax of
  the antenna-averaged softmax. Per-antenna in appendix. C0 excepted (repo
  aggregation, via the logger).
- **Single-seed reporting:** one number per cell, seed 42 declared, no ±std with
  n = 1; where seeds exist, mean ± (min–max), explicitly not a variance estimate.
- **Statistical protocol (the one used in the report).** The independent unit is
  the **trace**, not the window: effective n = 11 (S7) and 15 (S6). Comparisons
  use a **trace-level paired bootstrap**, N = 10⁴, fixed seed: traces are
  resampled with replacement and the *same* resample is applied to both streams,
  so per-trace difficulty cancels. The bootstrap runs on **accuracy**: with six
  of eight classes represented by a single trace, only ~2.8 % of trace resamples
  contain all eight classes, so a macro-F1 bootstrap would average a different
  class set in almost every replicate — macro-F1 is reported descriptively,
  without an interval. **Two variances are kept separate and never merged:**
  test-sampling variance (bootstrap) and training/seed variance (E1′).
  With ~10 comparisons on the same 11 traces, **C1–C2 and C1–C3 are designated
  primary** and the rest exploratory (declared multiplicity caveat).
  *Declared refinements not taken:* a BCa interval would be more honest than the
  percentile interval at n = 11; ECE uses equal-width bins.
- **Key figures (final):** the report references exactly **two** — the t-SNE
  C1-vs-C3 embedding figure and C1's test confusion matrix. Accuracy bars were
  **dropped** (the main table conveys the results; cut for the 6-page budget);
  the domain-diagnostics numbers are **delivered in the prose of §VI-E**, not as
  a table — a 6-row tabular costs ~10 column lines and the paper has under one
  line of slack (measured 2026-07-26: five words appended at the tail push it to
  7 pages). Promoting them to a table is the first thing to do if a page is ever
  freed; the report has exactly **one** table, `tab:test`. Per-class,
  reliability and forest-plot figures are produced by notebook 06 and kept
  unreferenced (the forest plot was the lead add-if-space candidate, excluded
  for lack of page slack).
- **t-SNE recipe (fixed, identical for every encoder):** **train features** —
  the only split with all 6 AR-sets and 2 environments (val has 9 traces / 5
  AR-sets; test is a single domain) — L2-normalized, subsampling of **8
  (window, antenna) samples per trace** (the cache row unit, not 8 windows × 4
  antennas), PCA-50 → t-SNE perplexity 30, `learning_rate="auto"`, `init="pca"`,
  seed 42. Qualitative figure, anchored to the probe numbers; same declared
  scope (and memorization confound) as the §7 train diagnostics; zero test
  contact.
- **Transductive rows (pre-registered, C1 only, unlabeled test data, inside the
  single session, same logger, compared against the plain C1 row).**
  - **AdaBN** = reset the encoder's BN running statistics → one forward pass over
    all test (window, antenna) samples with **only BatchNorm modules in train
    mode and `momentum=None`** (cumulative estimator), batch 256
    (`ADAPT_BN_BATCH`, enforced by assert), deterministic dataset order; weights
    untouched; statistics then frozen for evaluation and for post-AdaBN caching.
    Rationale for the cumulative estimator: the default momentum 0.1 weights
    recent batches exponentially → strong, arbitrary order dependence. Outputs
    carry an `_adabn` infix so plain rows can never be overwritten; the flag is
    recorded in the JSONL log and in CSV/npz metadata. `evaluate_c0` refuses it.
  - **T3A** (Iwasawa & Matsuo, NeurIPS 2021) = prototype adjustment of C1's
    linear head. **Declared batch variant** on cached features (pure numpy):
    (1) pseudo-labels and entropies from the initial prototypes = the L2-normalized
    rows of the head weight matrix, **bias dropped**; (2) per class, keep the
    **M = 20** lowest-entropy samples pseudo-labeled to it; (3) adjusted prototype
    = L2-renormalized mean of {initial} ∪ {L2-normalized supports}; (4) prediction
    = the unchanged §1.3 path. **The row therefore differs from C1 only in the
    head.** M = 20 is the centre of the paper's grid {1,5,20,50,100,∞}, **fixed a
    priori** — the paper selects it per dataset by validation, which for us would
    be tuning. The batch variant replaces the paper's online pass because that is
    order-dependent (an arbitrary declared order the result depends on, a free
    parameter with no benefit under the already-declared batch-deployment
    assumption). Two further declared deviations from the official implementation:
    bias-free pseudo-labeling, and the initial prototype is never subject to the
    filter. Supports are pooled over the 4 antennas. Composition order:
    **AdaBN → T3A**.
- **Positioning (pre-empts the "no SOTA" objection):** "established
  domain-invariant representation-learning methods (SupCon, DANN/GRL) applied to
  a rigorous LOEO protocol with direct diagnostics of invariance — not a new
  architecture." Related work names the newer WiFi-sensing DG lines
  (meta-learning RF-Net, AirFi, self-supervised AutoFi) as **declared out of
  scope** for budget and focus — and, empirically, training-time alignment losses
  share the GRL's lack of a target on this train; test-time adaptation is instead
  **measured** in minimal form (the transductive rows).

---

# PART II — EXECUTION RECORD

Gates and their outcomes are in §8.3. All runs used seed 42 unless stated.
Every completed session is archived verbatim with outputs under
`notebooks/runs/` (`YYYY-MM-DD_<config>.ipynb`); investigation sessions live in
`notebooks/diagnostics/`. Archived notebooks are the human-readable session
record, never a data source — authoritative numbers come from the harness CSVs
and the run directories.

### Training runs (validation numbers — selection only)

| Run | Date | Protocol | Best val macro-F1 | Horizon |
|---|---|---|---|---|
| C0 | 2026-07-16 | P1, 5 classes | **0.8916 @ e20** | early stop 31/60 (val = 3 traces → very noisy selection, declared) |
| C1 | 2026-07-16 | P2-lab | **0.8871 @ e37** | full 40/40 (best on the annealed tail) |
| C2 | 2026-07-16 | P2-lab | **0.8415 @ e13** | early stop 23/40 (mid-schedule best) |
| C3 phase A | 2026-07-17 | P2-lab | — (no in-loop selection by design) | 60/60, loss 5.914 → 4.430, plateau within ±0.004 on the tail |
| C3 phase B | 2026-07-17 | P2-lab | **0.8190 → epoch 40 selected** | grid 40/50/60 = 0.8190 / 0.8150 / 0.8120 (0.7-pt span: plateau confirmed) |
| C2_s43 | 2026-07-18 | P2-lab, s43 | **0.7870 @ e6** | early stop 16/40 |
| C1_s43 | 2026-07-18 | P2-lab, s43 | **0.8784 @ e37** | full 40/40 — same best epoch as C1 |
| C1_s6out | 2026-07-18 | P2-living | **0.7761 @ e12** | early stop 22/40 (6-trace 5-class val, very noisy) |
| C3-ft | 2026-07-19 | P2-lab | **0.8183 @ e4** | early stop 14/40 (best during warmup) |
| C1_aug | 2026-07-21 | P2-lab, `ce_amp` | **0.8830 @ e31** | full 40/40 — paired Δ vs C1 = **−0.004 (flat)** |
| C1_s6out_aug | 2026-07-21 | P2-living, `ce_amp` | **0.9230 @ e37** | full 40/40 — paired Δ vs C1_s6out = **+0.147** (Δmedian +0.186) |
| C1_sharplike | 2026-07-21 | P2-lab, SHARP net | **0.7384 @ e24** | early stop 34/40 — **≈15 pts below C1**; median plateau 0.672 vs 0.759 |

Val-only probe sessions: **C1-lin 0.8835** (≈ the end-to-end 0.8871 → features
linearly separable, cache verified), **C2-lin 0.8410** (≈ C2's 0.8415),
**C3-lin 0.8190**. Throughput note from the ablation: `sharp_like` runs at
0.191 s/step against V-B's ~0.53 — ~2.8× faster, so the real trade-off is
accuracy vs speed, and V-B wins accuracy by a wide margin.

### The single §0.7 test session

Executed **2026-07-22** via notebook `05_test_final.ipynb` with `SET="test"`,
all **16 frozen rows**, one shot, after a full `SET="val"` dry run that
exercised the entire path with zero test contact (the harness logs only when
`set_name == "test"`). Artifacts committed under `reports/final/`: per-row
`*_windows.csv` / `*_metrics.csv` / `*_confusion.csv` plus the merged
`test_invocations.jsonl` — **21 logged test accesses**, the §0.7 audit proof.
Session notebook archived as `notebooks/runs/2026_07_22_test_final.ipynb`.

Post-session analysis ran **2026-07-23** (notebook 06, zero test contact of its
own — it reads only the session CSVs), producing the 8 tables in
`report/tables/` that Part III quotes.

---

# PART III — MEASURED RESULTS

All numbers below are read from `report/tables/*.csv` and `reports/final/`.
`n_traces` is quoted with every row: it is the effective n.

## 1. Main test table

| Stream | Split | n_tr | n_win | Accuracy | Macro-F1 | ECE |
|---|---|---|---|---|---|---|
| **C1** | P2-lab | 11 | 425 | **0.8047** | **0.8038** | 0.243 |
| C1_s43 | P2-lab | 11 | 425 | 0.8000 | 0.7990 | 0.237 |
| C1_lin | P2-lab | 11 | 425 | 0.8071 | 0.8059 | 0.153 |
| C1_T3A | P2-lab | 11 | 425 | 0.8118 | 0.8101 | 0.345 |
| C1_aug | P2-lab | 11 | 425 | 0.7765 | 0.7720 | 0.235 |
| C1_AdaBN_T3A | P2-lab | 11 | 425 | 0.7576 | 0.7525 | 0.306 |
| C1_AdaBN | P2-lab | 11 | 425 | 0.7459 | 0.7221 | 0.214 |
| **C3** | P2-lab | 11 | 425 | 0.7271 | 0.7286 | 0.107 |
| C3_ft | P2-lab | 11 | 425 | 0.7200 | 0.7059 | 0.206 |
| C2_lin | P2-lab | 11 | 425 | 0.7200 | 0.7080 | 0.173 |
| **C2** | P2-lab | 11 | 425 | 0.6471 | 0.6006 | 0.070 |
| C2_s43 | P2-lab | 11 | 425 | 0.6000 | 0.5618 | 0.067 |
| C1_sharplike | P2-lab | 11 | 425 | 0.5694 | 0.5434 | 0.187 |
| C1_s6out | P2-living | 15 | 716 | 0.7430 | 0.6842 | 0.172 |
| C1_s6out_aug | P2-living | 15 | 716 | 0.8156 | 0.8227 | 0.224 |
| C0 | P1, 5 classes | 57 | 2717 | 0.6117 | 0.6053 | 0.103 |

The three blocks are **not** mutually comparable: P2-lab (8 classes, 11 traces),
P2-living (8 classes, 15 traces, its own rotation and μ/σ) and P1 (5 classes,
57 traces, its own split).

## 2. Paired trace-level bootstrap (accuracy, N = 10⁴, seed 42; diff = A − B)

| Comparison | Δ | 95 % CI | P(A>B) | n_tr | Resolved | Primary |
|---|---|---|---|---|---|---|
| C1 vs C2 | +0.1576 | [−0.0336, +0.3233] | 0.946 | 11 | no | **yes** |
| **C1 vs C3** | **+0.0776** | **[+0.0295, +0.1282]** | 1.000 | 11 | **yes** | **yes** |
| **C1 vs C1_sharplike** | **+0.2353** | **[+0.0924, +0.3986]** | 1.000 | 11 | **yes** | no |
| C1 vs C3_ft | +0.0847 | [−0.0582, +0.2212] | 0.864 | 11 | no | no |
| C1 vs C1_aug | +0.0282 | [−0.0233, +0.0924] | 0.779 | 11 | no | no |
| C1 vs C1_AdaBN | +0.0588 | [−0.0307, +0.1929] | 0.791 | 11 | no | no |
| C1 vs C1_T3A | −0.0071 | [−0.0613, +0.0474] | 0.408 | 11 | no | no |
| C1 vs C1_AdaBN_T3A | +0.0471 | [−0.0354, +0.1419] | 0.839 | 11 | no | no |
| C1_s6out_aug vs C1_s6out | +0.0726 | [−0.0289, +0.2025] | 0.894 | 15 | no | no |

**Seed spread on test** (n = 2 — a spread, not a CI): C1 |Δ| = 0.0047 ·
**C2 |Δ| = 0.0471**. The GRL's seed instability seen on validation replicates on
test.

## 3. Class-coverage decomposition (the declared caveat, measured)

Mean per-class F1 on test, split by whether the class was visible to checkpoint
selection on that rotation's val:

| Stream | Rotation | F1 val-seen | F1 val-blind | blind − seen |
|---|---|---|---|---|
| C1 | p2_lab | 0.743 | **0.904** | +0.161 |
| C1_lin | p2_lab | 0.741 | 0.914 | +0.173 |
| C1_s43 | p2_lab | 0.749 | 0.883 | +0.135 |
| C1_aug | p2_lab | 0.736 | 0.832 | +0.096 |
| C1_T3A | p2_lab | 0.746 | 0.917 | +0.171 |
| C1_AdaBN | p2_lab | 0.613 | 0.905 | +0.292 |
| C1_AdaBN_T3A | p2_lab | 0.682 | 0.870 | +0.187 |
| C2 | p2_lab | 0.506 | 0.758 | +0.251 |
| C2_s43 | p2_lab | 0.584 | 0.524 | −0.060 |
| C2_lin | p2_lab | 0.720 | 0.688 | −0.032 |
| C3 | p2_lab | 0.671 | 0.824 | +0.153 |
| C3_ft | p2_lab | 0.662 | 0.778 | +0.116 |
| C1_sharplike | p2_lab | 0.588 | 0.469 | −0.119 |
| C1_s6out | p2_living | 0.670 | 0.709 | +0.039 |
| C1_s6out_aug | p2_living | 0.838 | 0.797 | −0.041 |

On p2_lab, **the val-blind classes are the best**, not the worst — the
pre-declared caveat is empirically false there. On p2_living the picture flips,
driven by a *val-seen* class collapsing (S, F1 0.131). On test **no class is
absent**, so `absent_classes` is empty everywhere and val-blindness shows up as
per-class performance, not as a missing average.

**C1 per-class test F1 (p2_lab):** C 0.933 · E 1.000 · H 0.713 · J 0.812 ·
L 0.871 · R 0.673 · S 0.780 · W 0.649.
**C1_s6out per-class test F1 (p2_living):** seen {C 0.796, E 0.908, J 0.888,
R 0.626, **S 0.131**} · blind {H 0.647, L 0.731, W 0.748}.

## 4. Error structure of the deliverable model

**C1 top confusions (true → pred, windows):** W→R 18 · R→W 18 · H→J 14 ·
S→L 7 · S→H 5 · S→C 5. Walking↔running are adjacent Doppler signatures, H→J
are two whole-body dynamic classes, and the remaining mass is among the
low-motion classes; the empty-room class is perfect. Physically sensible
confusions, not scattered ones.

**Per-trace accuracy on the S7 test (11 traces)** — the informative granularity
(the per-AR-set cut is degenerate on a single-set test; note that per-trace is
nearly per-class here, since 6 classes have one trace):

| Trace | Class | C1 | C2 | C3 | C1_sharplike |
|---|---|---|---|---|---|
| S7a_C | C | 1.000 | 0.839 | 0.911 | 0.804 |
| S7a_E | E | 1.000 | 0.906 | 0.962 | 0.925 |
| S7a_H | H | 0.643 | **0.036** | 0.589 | 0.321 |
| S7a_J1 | J | 0.929 | 1.000 | 0.714 | 0.857 |
| S7a_J2 | J | 1.000 | 1.000 | 0.867 | 1.000 |
| S7a_J3 | J | 0.867 | 1.000 | 0.800 | 0.733 |
| S7a_L1 | L | 0.920 | 0.960 | 0.960 | 0.320 |
| S7a_L2 | L | 0.840 | 0.840 | 0.880 | 0.440 |
| S7a_R1 | R | 0.673 | 0.945 | 0.545 | 0.709 |
| S7a_S | S | 0.696 | 0.446 | 0.482 | **0.018** |
| S7a_W | W | 0.655 | 0.218 | 0.655 | 0.600 |

The C1−C2 gap is **concentrated, not uniform**: it lives on H (+0.607), W
(+0.436) and S (+0.250); C2 matches C1 on the jumping and sitting traces and is
*ahead* on one running trace (−0.273).

**C0 per-AR-set test accuracy (P1, S2–S7 — the non-degenerate per-set cut):**
AR-2 0.662 · AR-3 0.721 · AR-4 0.549 · AR-5 0.499 · AR-6 0.573 · AR-7 0.732.

## 5. Calibration

ECE per stream is in the main table. Reading: **C2 (GRL) is the best calibrated
(0.070) while being the least accurate**, C1 is the worst-calibrated of the
accurate models (0.243) and is systematically **under**confident (accuracy above
confidence in every reliability bin). **T3A worsens calibration** (0.345).
Calibration and accuracy dissociate across the loss families; any deployment
consuming confidence scores needs temperature scaling regardless of
configuration.
*Corrected 2026-07-26 (two readings that were wrong in direction or cause):*
**every one of the 16 streams is under-confident**, not just C1 — accuracy minus
mean fused confidence runs +0.067 (C2_s43) to +0.345 (C1_T3A), C1 at +0.243.
Label smoothing therefore cannot be the driver: C1_lin (+0.153), C3 (+0.101) and
C0 (+0.100) carry none, and C2 has *C1's own* smoothing with the smallest gap.
The common cause is §1.3 fusion — the mean of four softmaxes whose per-antenna
accuracies are 0.694/0.664/0.508/0.720 is far flatter than any of them. And T3A
does **not** "sharpen": it lowers mean confidence 0.561 → 0.467, i.e. it worsens
ECE by deepening the under-confidence. Numbers recomputed from
`reports/final/*_windows.csv`.
*Declared caveat:* C0's ECE is mildly ill-defined — under the repo's majority-vote
fusion, `confidence = max(mean softmax)` is not the probability of the predicted
class. C0 is therefore excluded from the reliability overlay and its ECE carries
a footnote.

## 6. Domain-readability diagnostics (train features, val/train only)

The instrument that carries the invariance evidence. Each target is read against
**its own majority baseline**, with activity as positive control.

**C1 (CE encoder, p2_lab train, inner split 55 fit / 26 eval, all 6 AR-sets):**

| Target | Acc | Baseline | Δ |
|---|---|---|---|
| **y (activity, control)** | **1.000** | 0.197 | **+0.803** |
| ar_set | 0.196 | 0.286 | −0.090 |
| ambiente | 0.854 | 0.854 | +0.000 ✻ |
| direct_path | 0.633 | 0.731 | −0.098 |
| persona | 0.818 | 0.818 | +0.000 ✻ |
| monitor | 0.499 | 0.584 | −0.086 |

✻ = the probe converges to the exact constant majority predictor (macro-F1
0.4606 / 0.4500 match the constant-predictor arithmetic to 4 digits, and it
selects epoch 1).

**Five replications, one verdict — no domain target is meaningfully above its
own majority baseline anywhere (the largest positive deviation across all five
is +0.015):**

| Encoder | Rotation | `y` control | Domain targets |
|---|---|---|---|
| C1 (CE) | p2_lab | 1.000 | all ≤ baseline (largest Δ +0.000, the two constant predictors) |
| C2 (CE+GRL) | p2_lab | **0.893** | all at baseline (ar_set −0.030, direct_path +0.008, monitor +0.015); on macro-F1 the domain is *slightly more* readable than C1 (ar_set 0.144 vs 0.066) |
| C3 (SupCon), epochs 40/50/60 | p2_lab | 0.995 / 0.993 / 0.996 | all ≤ baseline on all three checkpoints; **no maturity trend** |
| C1 (CE) | **p2_living** | 0.870 (baseline 0.210) | all at baseline (ar_set +0.011, direct_path −0.029, monitor +0.002) |
| C3-ft (SupCon → CE) | p2_lab | 0.981 | all ≤ baseline (ar_set −0.111) |

Three readings that matter:

- **The GRL had nothing to remove.** C2 paid a measured cost for a regularizer
  with no target; its adversary head sat at the **majority floor (0.2969, AR-1's
  share of train windows)** for the whole run, already at epochs 1–2 with λ still
  at 0.25–0.46 — it never became a discriminator at all. Reproduced on seed 43
  (0.30–0.32 across all 16 epochs).
- **The GRL cost train *fit*, not only transfer:** C2's `y` control is 0.893
  against C1's 1.000 (memorization gap: C2 0.893 → 0.8415 val, C1 1.000 → 0.8835).
- **Under SupCon the suppression is built into the objective:** P×K deliberately
  mixes AR-sets within each class and SupCon pulls same-class views together, so
  no adversary is needed — which is exactly why C4 had no remaining upside.
  The p2_living replication is the important one: it holds with the **laboratory**
  as the train second environment, so the verdict is not an artifact of the
  primary rotation's composition. The p2_living control (0.870) is *generalization*
  rather than memorization — its inner eval traces are held out within train.

## 7. Readout robustness on validation (NCM / kNN / concat)

Majority baseline 0.3209 throughout.

| Run | NCM acc / macro-F1 | kNN acc / macro-F1 | Linear probe (macro-F1) |
|---|---|---|---|
| C1 | 0.8653 / 0.8888 | 0.8453 / 0.8563 | 0.8835 |
| C2 | 0.7765 / 0.8176 | 0.8424 / 0.8663 | 0.8410 |
| C3 | 0.6963 / 0.7178 | 0.7937 / 0.8047 | 0.8190 |
| C1_s43 | 0.8567 / 0.8707 | 0.8281 / 0.8497 | (0.8784 end-to-end) |
| C3-ft | 0.7736 / — | 0.7984 / — | 0.8183 |

- **The frozen linear recipe did not understate SupCon.** kNN beats NCM sharply
  on C3 (the t-SNE chaining showing up as a non-linear readout gain), but kNN
  macro-F1 (0.8047) still sits **below** C3's own linear probe (0.8190). C3 is
  lowest under every readout tried.
- **C1 is readout-robust** (NCM/kNN within ~2 points of linear) and seed-robust
  on these readouts too (C1_s43 within ~1 point). C2 is the noisy one.
- **Concat: no CE↔SupCon complementarity.** C1⊕C3 = **0.8684** (−0.0151 vs
  C1-lin 0.8835), control **C1⊕C1′ = 0.8882** (+0.0047). Candidate − control =
  **−0.0197**, against a > +0.02 requirement. A second *CE* encoder gives the
  small generic ensemble bump; SupCon does **worse than that** — it is not merely
  redundant with CE, it is a worse concat partner than a same-loss twin.
  (Both probes select epoch 1/6 on the fragile val, so the magnitude is noisy;
  the direction agrees with all other C3 evidence.)

## 8. Geometry (t-SNE, train features, C1 vs C3)

**By activity:** 8 tight clusters in both encoders; C3's L/S/E clusters visibly
**chain into one continuous filament** instead of 3 discrete blobs — a plausible
geometric reading of why a *linear* probe scores C3 lower even if structure is
present. **By AR-set (the figure's actual job):** colours uniformly mixed within
every cluster in **both** encoders, no visible domain sub-structure — a visual
echo of the Δ ≈ 0 diagnostic verdict, not an independent proof (train, seen
traces). A diagnostic 3-row variant (C1 / C3 / C3-ft) shows the CE fine-tune
**partially un-chaining** C3's geometry (J and W/R become discrete, C1-like
blobs; L/S/E remain a residual filament, since the best checkpoint was epoch 4):
the fine-tune was turning the SupCon encoder *into* C1 — forgetting its
initialization rather than building on it.

---

# PART IV — FINDINGS, LIMITATIONS, ALLOWED WORDING

## 1. The findings

> **Sign convention.** Part III §2 tabulates the paired bootstrap as
> Δ = C1 − variant (positive favours C1). The findings below quote each effect
> in its natural orientation, **variant − C1**, and say so where it matters; the
> intervals are the same intervals, mirrored.

**F0 — The CE baseline wins and is stable.** C1 is the best model on the
held-out environment (0.8047 / 0.8038) and no variant resolves an improvement
over it. Its seed-43 replicate lands 0.5 accuracy points away and finds its best
at the same epoch (37).

**F1 — SupCon does not help; the readout is not the reason.** C3 is **7.8
accuracy points below C1** on test (paired Δ = C1 − C3 = +0.078, CI
[+0.030, +0.128] — it **excludes zero**, and it is one of the two *primary*
comparisons: the one that resolves) and below C1 under every readout on frozen
features. The fair-shot fine-tune C3-ft, pre-registered with the
hypothesis "comparable to C1 ± 1 pt", is **falsified**: it lands at 0.8183 val
≈ its own linear-probe ceiling (0.8190) and 0.7200 on test.
*Evidence accounting (important for honesty):* count **two families of
evidence** — (i) multiple readouts on the *same* frozen features (linear, NCM,
kNN, concat, t-SNE), which agree partly by construction and share one
memorization confound, and (ii) the **full fine-tune**, a genuinely different
input path and the load-bearing confirmation. Not "seven independent
instruments".

**F2 — The GRL costs, and its damage sits in the head.** C2 is 15.8 accuracy
points below C1; the direction is stable across seeds and on validation, but the
paired interval (Δ = C1 − C2 = +0.158, CI [−0.034, +0.323]) **crosses zero on 11
traces** — a 16-point gap that the test set cannot resolve. A linear probe on C2's *frozen features*
reaches 0.720 against C2's own trained head at 0.647 (probe − end-to-end:
**+0.073 for C2 against +0.002 for C1**): a substantial part of the damage is in
the classifier head, not in the representation. The damage is also concentrated
(H, W, S traces), not uniform.

**F3 — The adversary has no target, and this is structural.** No domain
attribute is linearly readable from train features, on five encoders and two
rotations (Part III §6). The root cause is the split composition: one
environment dominates train and the second is a single capture set. The
**incidence proof** generalizes it: bedroom = 5 sets, living room = 1,
laboratory = 1, so in *any* LOEO rotation of this dataset the in-train
environment is either the bedroom (a single environment — invariance
undefinable) or a single capture set (= session identity, not a generalizable
environment). **No rotation of this dataset poses environment-invariance
non-degenerately.** C4 was closed on this basis, without running.

**F4 — Test-time adaptation is neutral at best.** Oriented as TTA − C1: T3A is
indistinguishable from plain C1 (+0.007, CI [−0.047, +0.061]); AdaBN is negative
(−0.059, CI [−0.193, +0.031]) and drags the composition down with it (−0.047).
None of the three resolves. Re-estimating BN statistics on a
single-environment, 11-trace unlabeled test set moves them away from a training
distribution that — per F3 — was not environment-specific to begin with. The
negative result is consistent with the central finding, not in tension with it.

**F5 — Amplitude augmentation is a regularizer, not a cross-domain lever.**
The pre-registered discriminator was: if it acts through cross-domain
invariance, Δtest should exceed Δval. The opposite happened. On P2-lab, where
the baseline was near its val ceiling, Δval = −0.004 and Δtest = −0.028; on
P2-living, whose baseline is much lower, Δval = +0.147 and Δtest = +0.073 — a
gain where there was headroom, of the same order in and out of domain. The
effect is **environment-dependent and its sign flips**; report it as one point
on an unexplored axis, never as "augmentation helps / does not help".

**F6 — The backbone is what matters.** At identical recipe, split, class set,
loss, optimizer and seed, the paper's shallow-wide network loses **23.5**
accuracy points (Δ = C1 − C1_sharplike = +0.235, CI [+0.092, +0.399] — excludes
zero) after already losing ~15 points on in-domain validation. V-B was originally chosen on the **day-2 throughput
gate, not on accuracy**; the accuracy axis now vindicates that choice
independently. Two caveats travel with this row: it measures "deep vs
near-linear wide", not only "the paper's design choice"; and it closes the
**weak-backbone objection** to the CE/SupCon null (our CE baseline does not run
on a strawman backbone) — it does *not* close the SupCon-regime caveat.

**F7 — Selection blindness is not the driver of per-class error.** See Part III
§3. The caveat was pre-declared and is empirically false on the primary rotation.

**F8 — Two variances, side by side (the methodological point).** On C2 the
same-config seed spread (5.45 val points, 4.7 test points) is of the same order
as — and on validation *larger than* — the cross-config gap it was meant to
calibrate (4.6 val points). This is Bouthillier et al.'s phenomenon on our data,
and it retroactively justifies rule 5 (a fixed seed for the controlled
comparison) as a variance-aware strategy rather than a weakness.

## 2. Declared limitations and deviations (the report's declaration list)

Protocol and data:
- Single seed 42 for everything except the two pre-registered seed-43
  replicates; probes, diagnostics and test-time techniques on seed 42 only
  (declared asymmetry).
- Seed declared ≠ bit-exact reproducibility (cuDNN benchmark on).
- **In-domain validation** (15 % of train) for model selection in a DG setting.
- Val macro-F1 is a **5-class** number on both rotations, test is **8-class** —
  the two scales are not comparable; the blindness is common-mode across configs.
- Test sets are small: 11 traces (S7), 15 (S6); 6 of 8 classes have a single
  trace on S7.
- "AR-set invariance", never "environment invariance": room, monitor, subject,
  day and LOS/NLOS are confounded inside a capture set.
- Phase-A epochs are 300 steps against 400 in CE runs (escalation (a)).
- E2′ has its own split file and its own μ/σ.
- Unweighted CE while selecting and reporting macro-F1.
- Probe features are extracted **without** augmentation.
- Declared C2-vs-C4 batch asymmetry for the adversarial head.
- C0 is a **partial, non-faithful** reproduction: 20 % val hold-out (the repo
  trains on all of S1), 5-class set, our preprocessing and backbone, undocumented
  details filled by the common harness. It is a **sanity anchor for our harness
  on P1, not a paper benchmark** — the report must not claim "we reproduced X %".
  Its test sets are held-out *capture sets* (the same bedroom on later campaigns
  plus the two unseen rooms), not "unseen environments".
- C4 closed without running, on evidence.
- The §7 AR-set probe is underpowered on p2_lab; the invariance evidence rests on
  the train-feature domain diagnostics.
- Transductive rows: declared batch-deployment assumption, hyperparameters fixed
  without tuning, T3A as a declared batch variant with two declared deviations
  from the official implementation.
- Percentile bootstrap CI at n = 11 (BCa would be more honest); equal-width ECE
  bins; C0's ECE mildly ill-defined under vote fusion.

## 3. Allowed wording (claims the evidence does and does not support)

This table is binding for the report. Each row is a claim that was stress-tested
and rewritten.

| Do **not** write | Write instead | Why |
|---|---|---|
| "The domain probe proves the GRL is useless" | "Domain is not **linearly** decodable from train features on traces the encoder has seen" + carry the *structural* claim on the **incidence proof** | The probe cannot see an activity-entangled shortcut on seen traces, and the one split where it would matter (held-out S7) is single-domain, so domain readability is undefined there. C4 = pre-registered triage **plus** the incidence proof, not "GRL disproven". |
| "We study domain generalization / AR-set invariance" | "Generalization to n = 2 unseen hostile sessions (S7, and S6)", stating the confound bundle (room + subject + day + LOS/NLOS + monitor) wherever the number appears | With bedroom = 5 sets, living = 1, lab = 1 there are two held-out environments, each a single session. E2′ is the *re-appearance* of the same degenerate confound, not an independent replication. |
| A bare 4-digit macro-F1 as a measurement | Every test row with its explicit `n_traces`; lead comparisons with the paired trace-level CI + a multiplicity caveat; state a priori that small-delta rows are *expected* "comparable" | Effective n ≈ 11. With 16 rows on 11 traces the table realistically resolves 2–3 performance tiers, not 16. |
| "Seven independent instruments agree" | "Two families of evidence": readouts on the same frozen features (agreeing partly by construction), and the full fine-tune (a distinct input path, load-bearing) | Five of the seven consume identical cached features with the same memorization confound; a control pegged at 1.000 cannot calibrate measurements that live near 0.2–0.4. |
| C0 as a comparison baseline | C0 as a **reproduction anchor, causally isolated** from the V-B/P2 path | C0 shares nothing with the path every finding depends on (net, classes, protocol, fusion); its dominant uncertainty is systematic, not seed-driven. Hence: **no C0 seed replicates.** |
| "The GRL destabilizes training" | "On the two seeds run, C2 swung 5.45 val points against C1's 0.87 — an observation, not a variance estimate"; GRL val cost as a **range (≈3.7–10 pts)**, never the single −4.6 | n = 2 supports ordinal/existence claims only. C1 > C2 stays valid as a **direction** (full separation across seeds, minimum gap 3.69 > band). |
| "Augmentation helps cross-domain" | "A label-safe perturbation compatible with the attenuation shift moved / did not move the held-out number" | S7 confounds room + monitor + subject + day; a global amplitude scalar is one label-safe knob, not a physically faithful multipath model (real room change is velocity-selective). |
| "The backbone ablation vindicates V-B" (unqualified) | "Deep beats near-linear-wide in this recipe; V-B was chosen on throughput, so this is one axis" — appendix ablation, never a headline | A ~204 k-param dense head on 25 500 flattened features is a near-linear wide model; and the ground V-B was actually chosen on (throughput) is untested here. |

**The seed-value principle** (governs every seed decision taken): a seed
replicate is worth running **only if** a specific claim needs (a) a calibrated
noise floor or (b) a direction-across-init, **and** the measurement can resolve
it. Never for symmetry, for ± bars, or to *estimate* a variance (n = 2 does not
estimate variances). Applied: C1_s43 / C2_s43 → kept; s44 → not run; C0 seeds →
never; C1_aug_s43 → cut.

---

# PART V — DECISION LOG

Dated pre-registrations and amendments. Every entry was recorded **before** the
outcome it governs was known, except where explicitly marked; git history is the
timestamp of record.

| Date | Decision | Rationale / effect |
|---|---|---|
| 2026-07-15 | **Dataset errata.** The Drive copy is the SHARP TMC dataset (S1–S7, 12 campaigns, 3 environments, identical hardware), not the extended one | Primary rotation redefined to **leave-S7-out**; AR-8/AR-9 do not exist; hardware confound removed |
| 2026-07-16 | **Escalation (a) applied** — phase-A steps/epoch 400 → 300 (C3/C4 only) | Day-3 gate NO-GO (9.16 h > 8 h) → projected 6.87 h. C1/C2 untouched (their gate passed); backbone untouched (escalation (b) would break shared-encoder comparability) |
| 2026-07-16 | Leave-S7-out confirmed as the primary P2 rotation | Small test set and unseen subject P3 stay declared limitations |
| 2026-07-16 | **C0 = the paper's 5-class core** (`["E","J","L","R","W"]`) | The TMC core tables use 5 classes; the pipeline is parametrized on `n_att`, so cost zero |
| 2026-07-16 | C0 test evaluation uses the **repo's decision fusion** inside the harness C0 wrapper; C1–C4 stay on softmax averaging | Faithfulness for the reproduction, one audit trail for everything |
| 2026-07-17 | **C4 closed without running** | The GRL has no readable domain target under either loss family (C1/C2/C3 diagnostics), and C2 shows only measured costs. Expected outcome "C3 plus noise". ~6.9 h freed |
| 2026-07-17 | **§7 AR-set probe declared underpowered** on p2_lab | Structural: 9 val traces, AR-3 absent, AR-1/AR-2 metadata-identical. §9 invariance evidence moves to the train-feature diagnostics |
| 2026-07-17 | **E1′** — seed-43 replicates of C1 and C2 (amendment to rule 5) | Measures the §0.5 noise floor and the robustness of the C2 findings to init. One pre-registered test row each; probes stay on seed 42 |
| 2026-07-17 | **Transductive rows pre-registered** (C1+AdaBN, C1+T3A, C1+both — pinned *unconditional*) | A conditional row inside a frozen list would reopen an in-session choice |
| 2026-07-18 | **Seed-44 trigger rule** pre-registered *before* reading s43 | Launch only if a twin lands outside the ~2-point band or the C1–C2 gap crosses it; otherwise E1′ stays at n = 2 |
| 2026-07-18 | **Twin-binding amendment** — `*alt` twins are bound split units (`bind_alt_twins=True`) | The first S6-out dry run drew `S4a_Lalt` into val with `S4a_L` in train — selection-side quasi-leakage. No retroactive inconsistency: p2_lab already satisfies it by draw |
| 2026-07-18 | **T3A/AdaBN specifications pinned** (batch variant, M = 20, cumulative BN estimator) and the **t-SNE recipe** pinned to train features | Order-dependence removed by construction; train is the only split with all 6 AR-sets and 2 environments |
| 2026-07-18 | NCM/kNN scorers and the domain-probe instrument **promoted to package code** | They are re-run across sessions with declared hyperparameters → pipeline code, not one-offs. Math verified byte-identical to the notebook cells that produced the recorded numbers, so those rows stand |
| 2026-07-19 | **Seed-44 NOT run — E1′ closed at n = 2** | The question the trigger was held for ("pipeline-wide or GRL-specific?") was answered by C1_s43 (GRL-specific); every remaining claim is directional. Standing constraint: report the GRL cost as a range |
| 2026-07-19 | **C3-ft approved** (13th row), recipe fixed a priori | Answers the reviewer question "was the linear probe unfair to SupCon?" with a protected floor. Success pre-defined as *comparable*, not *beats* |
| 2026-07-19 | Candidate B (joint CE+SupCon) **downgraded to future work** | Its information-capture channel is empirically weak (16/349 error-overlap; negative concat number) |
| 2026-07-20 | **E3 (leave-bedroom-out) REJECTED as a run** | Circular: its null is *entailed* by the incidence proof (its own train is two single-set environments); and a 26-trace train confounds "does not generalize" with "had no data". The val problem (2 traces) is the least of it. The incidence argument is kept as a **proposition with proof** |
| 2026-07-20 | **C1-aug arm approved** — variant (b), the amplitude lever, additive `ce_amp` profile, paired at fixed seed, cross-rotation | Time-warp rejected on physics, velocity-warp on label safety; one conceptual lever for clean attribution. Frozen §3 table untouched |
| 2026-07-21 | **C1_sharplike** ablation approved (val-only at first) | C0-vs-C1 confounds architecture + protocol + class count; holding the C1 recipe fixed isolates the backbone axis |
| 2026-07-21 | **C1_sharplike earns a test row (17th)** — decided *after* seeing val | Admissible because val ≠ test and there is **no cherry-pick incentive** (val already shows it ~15 points worse), registered as **pre-register-and-commit-to-report** with an **outcome-independent interpretive key** fixed before the session: (i) gap confirms → V-B wins on the held-out domain too; (ii) gap compresses → on a hostile domain the regime dominates architecture, and the vindication rests on the in-domain gap. Neither branch is an architecture contribution |
| 2026-07-22 | **L6 ratified — `C1_aug_s43` dropped** (list 17 → 16) | The comparison is paired, so init noise is already controlled; the seed twin re-uses the *same* S7 test set, so it cannot touch the dominant uncertainty. The cross-rotation twin is the replication worth keeping. Config and runner stay in the repo, unlaunched |
| 2026-07-22 | **Single §0.7 test session executed** — 16 rows, one shot | Preceded by a full val dry run with zero test contact; 21 accesses logged |
| 2026-07-23 | **Report figure plan resolved** — exactly 2 figures; accuracy bars dropped; forest plot excluded | 6-page IEEE budget; the main table's CI column carries the intervals |
| 2026-07-26 | **Stress-test correction pass on the report** — 10 length-neutral edits, no measured number changed, still exactly 6 pages | Full re-verification of every reported figure against `reports/final/` and `report/tables/`: all 9 table rows and all 6 mirrored CIs are exact, and the paired bootstrap was independently reproduced (C1−C3 = +0.0776 [+0.0295, +0.1282]). Fixed: the `0.13` baseline is a majority-class *accuracy*, not a macro-F1 (macro-F1 of that predictor is 0.029); under-confidence attributed to fusion rather than to label smoothing (§III.5); the linear probe / MLP adversary gap named and closed by the adversary's own majority floor; C3-ft's selected checkpoint disclosed as inside the warm-up, with every later full-LR epoch worse; abstract no longer derives probe unreadability *from* the incidence argument (two claims, different targets); C0's n = 57 added; the augmentation sentence now quotes the largest effect (+0.073) not the smallest; C2 marked as a circular replication of the domain probe; "backbone capacity" → "depth" (L7). Budget measured empirically, not estimated: the tail has < 1 line of slack, so the edit set was chosen greedily by value under a ~150-character ceiling |

**Amendment discipline** (what made all of the above legitimate): the §0.7 row
list may be amended **only by a team call and only while the session is closed**;
a drop tightens the list and is always admissible; a post-val decision is
admissible only as pre-register-and-commit-to-report with the interpretation
fixed in advance. Every amendment is dated in this log and in git.

---

# PART VI — REPOSITORY, REPRODUCTION, REMAINING WORK

## 1. Layout

```
sharp_har/            The versioned package — ALL logic lives here
  inventory.py        Dataset inventory, AR-set metadata, name→AR-set map
  windowing.py        Window enumeration + train-only μ/σ
  splits.py           Frozen split construction (trace-level, rare-cell pins, twin binding)
  data.py             DopplerDataset + file index (windowing, normalization, antennas)
  augment.py          §3 profiles: ce, supcon_view, ce_amp
  sampler.py          P×K sampler (distinct traces, ≥340 reuse offset, per-epoch reseed)
  models/             resnet_vb.py (V-B), sharp_like.py, heads.py, build_backbone factory
  losses.py           CE + label smoothing, SupCon, GRL + λ ramp
  train.py            Config-driven train_run (checkpoints, auto-resume, monitoring, init_ckpt)
  harness.py          The single checkpoint→CSV eval path, C0 wrapper, AdaBN flag, test logging
  probe.py            Frozen linear-probe recipe, phase-B selection, §7 probes
  diagnostics.py      domain_probe, inner_trace_split, NCM/kNN, concat_caches
  transductive.py     T3A (declared batch variant, numpy on cached features)
  session.py          §0.7 row dispatch behind notebook 05 (readiness, run_row, finalize)
  bench.py            Throughput/memory gate measurements
  viz.py              Plots and comparison tables from run artifacts
  utils.py            Seeding, YAML/JSON I/O, git hash, logging
configs/              One YAML per run (c0…c4 + every arm) + Colab paths
notebooks/            Thin runner templates (output-free on Git)
notebooks/runs/       Executed run notebooks, committed verbatim with outputs
notebooks/diagnostics/  Executed investigation sessions
splits/               Frozen split JSONs + CHANGELOG (pre-registration audit trail)
reports/              Measured artifacts: inventory, contingency, gates, embeddings
reports/final/        The single test session's CSVs + test_invocations.jsonl
report/               IEEEtran paper: *.tex, tables/ (8 measured CSVs), figures/
docs/archive/         Superseded documents, kept for the record only
```

**Notebook map:** `00_setup_smoke` (mount + staging + frozen-artifact asserts) ·
`01_inventory_splits` (day 1, splits already frozen) · `02_smoke_gate` ·
`02b_phase_a_gate` · `03_train` (config-driven runner) · `04_probe` (val-only
probe sessions) · `05_test_final` (the single §0.7 session) ·
`06_final_analysis` (post-session analysis, no test access of its own). Arm-specific
pinned runners live in `notebooks/{c1_aug,c3_ft,e1_seed_replicates,e2_living_out,backbone_ablation}/`.

**Two structural rules.** *Thin notebooks:* notebooks mount, stage, call
`sharp_har` and display — no logic, because the dataloader needs cross-review and
notebooks cannot be diffed for it. A diagnostic graduates into the package when
its numbers enter the report **and** it is re-run across sessions.
*Frozen artifacts flow through Git:* `splits/*.json`, `reports/*.csv`,
`reports/gate_*.json`, `reports/final/` and executed notebooks are committed
unmodified and never edited afterwards. Data (~762 MB) and checkpoints never
enter the repo.

## 2. Reproduction

```bash
pip install -r requirements.txt
```

There is no test suite, linter or build step. Verification is by the **blocking
asserts** built into the pipeline (split disjointness, axes check, AR-set
coverage, NaN policy, rare-cell coverage, sampler invariants, cache alignment,
readiness of every §0.7 artifact) and by the committed gate reports. Local work
is code-only; training and evaluation run on Colab through the notebooks. A
session: open the notebook for your phase, set the config selector in the marked
cell, run top to bottom, commit the executed copy under `notebooks/runs/`.

Reproducing a published number needs: the frozen split JSON, the config YAML,
the git hash in `run_meta.json`, and the run's checkpoint. The single test
session is *not* reproducible by re-running it — that would be a second test
contact; it is reproducible from `reports/final/` downstream (notebook 06 is
free to re-run).

## 3. Remaining work

1. Report polish on `report/` (the draft is refined and builds to exactly 6
   pages with Tectonic, zero undefined references, both figures rendering — do
   **not** build in place: the output name `template.pdf` would clobber the
   course template).
2. The presentation.

Nothing else is open. Code freeze 2026-07-28, deadline 2026-07-30.

## 4. Sources

Method and positioning:
- Meneghello et al., *SHARP: Environment and Person-Independent Activity Recognition with Commodity IEEE 802.11 Access Points* (IEEE TMC 2023; arXiv 2103.09924) and the dataset paper (IEEE Comm. Mag. 2023); repo `francescamen/SHARP`
- Khosla et al., *Supervised Contrastive Learning* (NeurIPS 2020) — https://arxiv.org/pdf/2004.11362
- Ganin & Lempitsky, DANN / gradient reversal
- Iwasawa & Matsuo, *T3A* (NeurIPS 2021)
- van der Maaten & Hinton, t-SNE (2008)

Framing of the nulls and of the statistics:
- Bouthillier et al., *Accounting for Variance in ML Benchmarks* (MLSys 2021) — https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
- *A Tale of Two Variances: When Single-Seed Benchmarks Fail* — https://arxiv.org/pdf/2604.23114
- *A Paired Bootstrap Protocol for Evaluating Small Models* — https://www.arxiv.org/pdf/2511.19794
- *Position: Embracing Negative Results in Machine Learning* — https://arxiv.org/pdf/2406.03980
- Semmelrock et al., *Reproducibility in ML-based Research* (AI Magazine 2025) — https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002

Field context (declared out of scope, cited in related work):
- *A Survey on Wi-Fi Sensing Generalizability* (2025) — https://arxiv.org/pdf/2503.08008
- *Wi-Fi Sensing for HAR: Survey, Challenges, Research Directions* (ACM CSUR) — https://dl.acm.org/doi/10.1145/3705893
- *Transferability of Representations from Supervised Contrastive Learning* — https://arxiv.org/pdf/2309.15486
- *Data Augmentation for Cross-Domain WiFi CSI HAR* — https://arxiv.org/abs/2401.00964
- *MU-SHOT-Fi: Source-free UDA for Multi-User Wi-Fi Sensing* — https://arxiv.org/pdf/2605.01369
- RF-Net (meta-learning), AirFi (DG), AutoFi (self-supervised CSI pretraining), DATTA (WACV 2026, domain-adversarial TTA for cross-domain WiFi HAR)

---

# APPENDIX A — Supersession map

Where the content of each archived document went. The archived copies under
`docs/archive/` are the record; nothing needs to be read from them.

| Archived document | Absorbed into | Deliberately not carried over |
|---|---|---|
| `pipeline_wifi_har_v5.md` (spec, v5/v5.1/v5.2) | **Part I §0–§9 in full**, with the same section numbering; §8.4 as a consuntivo | The v5 → v5.1 → v5.2 changelog blocks (dataset reality is now stated as fact, not as a diff); §10 (day-by-day timeline and ownership) and §11 (risk table) — both spent, with their outcomes recorded in §8.3 and Part V |
| `STATUS.md` | Part II (runs, gates, the test session), Part III (every measured number), Part V (decisions) | Session chronicle: who launched what, notebook-archive regularizations, resolved implementation bugs, Drive-shortcut logistics, superseded readings that were corrected in place. All of it is in git history and in the archived notebooks |
| `CONSOLIDATION_REVIEW.md` | §9 statistical protocol, Part IV literature framing of the two nulls, the incidence proof (F3), Part VI sources | The decision table (all directions resolved) and the G1–G12 gap list (all closed 2026-07-23) |
| `CONCEPTUAL_STRESS_TEST.md` | **Part IV §3 "allowed wording"** (L0, L1, L2, L4, L5, L7) and the seed-value principle; L6 and L8 as decisions in Part V and results in Part III | The level-by-level essays and the "where it bends" argumentation, once their conclusions became binding wording |
| `NOTEBOOK_06_REVIEW.md` | Part III (its measured-numbers appendix, in full) and the two declared caveats (C0 ECE under vote fusion, percentile CI at n = 11) | The review process itself and the four dated update passes; the notebook is report-grade and the checklist is closed |
| `README.md` | Kept, trimmed to a repository entry point pointing here | Duplicate protocol and rule listings |
| `splits/CHANGELOG.md` | Kept in place as the pre-registration audit trail next to the frozen splits; summarized in Part V | — |
| `notebooks/*/README.md` | Kept in place (local conventions and per-folder indices) | — |

**Historical name references.** Dated artifacts that must not be rewritten —
the config headers carrying pre-registration rationales, `splits/CHANGELOG.md`,
and the executed notebooks — cite `STATUS.md` and the review documents by their
old names. Those names now resolve to `docs/archive/`; the current statement of
the same content is this document (decisions in Part V, numbers in Part III,
sections §X.Y in Part I).
