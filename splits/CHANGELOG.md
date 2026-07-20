# Split / escalation changelog

§5.2 requires every escalation to be recorded "in the split file
changelog"; §0.1 forbids editing the frozen split JSONs. Resolution:
this file, committed next to the splits, is that changelog — the JSONs
themselves are never touched.

## 2026-07-16 — Escalation (a): phase-A epoch_steps 400 → 300

- **Trigger:** day-3 phase-A gate NO-GO (`reports/gate_day3_phase_a.json`,
  commit `faa0cdb`): measured 1.373 s/step at full batch (512 SupCon
  views, real P×K sampler, augmentation in-step) → projected
  60 × 400 × 1.373 s = **9.16 h > 8 h** (§10.1 pre-committed rule).
- **Action:** first rung of the pre-committed §5.2 ladder — steps/epoch
  400 → 300 for the **phase-A configs only** (`c3_supcon.yaml`,
  `c4_supcon_grl.yaml`, kept identical so C3 vs C4 stay directly
  comparable). Projected phase A: 60 × 300 × 1.373 s = **6.87 h ≤ 8 h**.
- **Not touched:** C1/C2 stay at 400 steps/epoch (their gate passed:
  projected C1 2.34 h ≤ 4 h, `reports/gate_day2.json`); the backbone is
  unchanged (escalation (b) would have broken the shared-encoder
  comparability across C1–C4); max_epochs stays 60, so the pre-committed
  checkpoint grid ⌈2H/3⌉/⌈5H/6⌉/H = 40/50/60 is unchanged (§6-C3).
- **Declared asymmetry:** a phase-A "epoch" is now 300 steps vs 400 in
  the CE runs — epochs are budget units, not comparability units; goes
  in the report.
- **Recovery margin noted:** the measured s/step includes the 512
  augmentations in the main process (declared upper bound in bench.py);
  if the day-4 wiring moves augmentation into DataLoader workers, real
  phase A should land under the 6.87 h projection.
- **Memory (§5.2 verification):** peak 8.57 GiB allocated / 11.37 GiB
  reserved on T4 — activation checkpointing NOT needed.

## 2026-07-17 — E1: seed replicates of C1 and C2 (amendment to §0 rule 5)

- **Trigger:** the C4 gate closed on evidence (GRL has no target under
  either loss family — see STATUS 2026-07-17), freeing C4's ~7 h budget.
  Every comparison in the final table leans on the §0.5 "~2 points =
  comparable" band, which was assumed, never measured.
- **Action:** rule 5 ("single seed 42 everywhere") amended — seed 42
  stays the primary seed for every stream and every technique/probe;
  two pre-registered replicates `C1_s43` / `C2_s43` are added
  (`configs/c1_ce_s43.yaml`, `configs/c2_grl_s43.yaml`, byte-identical
  to the originals except `name`/`seed`; runner notebooks in
  `notebooks/e1_seed_replicates/`).
- **What they measure:** (i) the pipeline's seed noise floor — the
  empirical calibration of the §0.5 band (reported as an observed
  range, no significance claims from n=2); (ii) robustness of the C2
  findings to initialization (adversary at the train majority floor,
  val cost vs C1); (iii) C1_s43's cached features double as the §7
  concat ensemble control C1⊕C1′ (label fixed 2026-07-17: originally
  written "E2", colliding with the E2′ living-out label of §10.3).
- **Budget:** 2 × ~2.3 h ≈ 4.6 h ≤ the freed C4 budget (§8.4 amendment,
  net negative). Each replicate adds ONE pre-registered test row
  (mean±range cell with its seed-42 sibling in the report); probes,
  diagnostics and techniques stay on seed 42 only.
- **Not replicated:** C3 (3× the cost; its findings are already
  replicated across encoders/machines; the flat 40/50/60 grid — 0.7 pt
  span — already evidences representation stability; declared
  single-seed in the report).

## 2026-07-18 — E1 addendum: pre-registered seed-44 trigger rule (+ label fix)

- **Seed-44 rule (pre-registered before reading the s43 results):**
  `C1_s44`/`C2_s44` are launched ONLY if the s43 results make the
  headline claim ambiguous — i.e. a seed twin lands outside the §0.5
  ~2-point band from its seed-42 sibling (val macro-F1, fused), or the
  C1–C2 gap crosses the band. Decision due by table-freeze day and its
  outcome recorded in STATUS either way; otherwise E1 stays at n=2,
  declared. Configs may be prepared in advance; a launch requires the
  trigger, not spare capacity.
- **Label fix:** the 2026-07-17 entry above says C1_s43's features
  "double as the E2 concat ensemble control" — read "concat ensemble
  control C1⊕C1′ (§7 v5.2 val-only diagnostics)". "E2′" in pipeline
  §10.3 is the living-out rotation, unrelated to the concat control.

## 2026-07-18 — E2′ amendment: dual-archive twins are bound split units

- **Rule (pre-registered before the `p2_living.json` freeze):** the
  dual-archive twin pairs (`S4a_L`/`S4a_Lalt`, `S5a_L`/`S5a_Lalt`) are
  two recordings of the same physical session (day-1 finding). §0
  rule 2's rationale — correlated units must never straddle a split
  boundary — applies to them at the recording level, so an `*alt` trace
  is **not an independent split unit**: it is removed from the
  stratification pool and assigned to the same side as its base trace
  (`build_p2_rotation(bind_alt_twins=True)`, the default from this
  date). Trigger: the first S6-out dry-run drew `S4a_Lalt` into val
  with `S4a_L` in train — a selection-side quasi-leakage (test = S6,
  untouched either way).
- **No retroactive discrepancy:** the frozen `p2_lab.json` already
  satisfies the invariant by draw (all four twins in train) and is
  never regenerated; it predates the amendment, so reproducing it
  byte-identically requires `bind_alt_twins=False` (excluding the alts
  from the pool shifts the seed-42 draw stream).
- **Validation record (local dry-run on the frozen trace universe,
  2026-07-18):** (i) with `bind_alt_twins=False`, the AR-7 rotation
  reproduces the frozen p2_lab partition exactly (train/val/test/
  pinned) — the dry-run methodology predicts the real session; (ii)
  amended AR-6 partition: **train=80 val=6 test=15 pinned=41**, both
  twin pairs in train, val = S1b_E, S1b_J2, S1c_S, S2a_R, S4a_C2,
  S4b_J1 (AR-1/2/4; classes {C, E, J, R, S} — **H, L, W absent from
  val**, accepted per §2.2's explicit rare-cell clause, same caveat
  family as p2_lab's 5-class val; selection is within-run, so a
  k-class val metric stays valid for checkpoint selection and is
  never compared across scales).

## 2026-07-19 — C3-ft: pre-registered 13th test row (§0.7 list amendment)

- **Decision:** team-approved 2026-07-19 (recipe package adopted by owner A
  the same day, recorded in STATUS): ONE additional run, `C3_ft` — CE
  fine-tuning of C3's SupCon encoder from the phase-B-selected
  `C3/epoch40.ckpt`. It answers the reviewer-facing fairness question
  ("was the linear probe holding SupCon back?") with a protected floor
  (worst case ≈ a CE run from a different init).
- **Recipe (fixed a priori, no grid, no selection —
  `configs/c3_ft.yaml` header carries the full rationale):** full-network
  fine-tune; fresh ActivityHead (enforced by the train.py `init_ckpt`
  wiring — the head never transfers); 40-epoch cosine + warmup 5 as C1
  with peak LR 1e-4 (10× below C1, so the pre-trained init is not washed
  out); C1 "ce" augmentation profile unchanged. Byte-diff from
  `c1_ce.yaml`: name, `init_ckpt`, `optim.lr`.
- **Pre-registered hypothesis:** comparable to C1 (±1 pt val macro-F1);
  success = comparable, not beats (§0.5 band + measured seed noise).
- **Amendments:** §0.7 frozen row list 12 → 13 (session not yet opened —
  the freeze clause forbids extensions with the session OPEN, and this
  precedes it); §8.4 budget +2 h (extensions total ≈ 9.1 h, still under
  the pre-v5.2 envelope). Wiring (`train.py` init_ckpt) enters the
  pre-freeze cross-review with the rest of the 2026-07-18 pass.

## 2026-07-20 — C1-aug: targeted amplitude augmentation arm (§0.7 list 13 → 16)

- **Decision:** team-approved 2026-07-20 (closes the "C6-aug" proposal
  tabled 2026-07-20, reviewed twice in STATUS): THREE runs of variant
  (b), the minimal cross-rotation package from the deepened review —
  `C1_aug` (P2-lab, seed 42), `C1_aug_s43` (P2-lab, seed 43),
  `C1_s6out_aug` (P2-living, seed 42). Question: does strengthening the
  attenuation component of the §3 augmentation improve cross-AR-set
  generalization?
- **Lever (fixed by the 2026-07-20 physical analysis, recorded in
  STATUS):** amplitude scaling only — s ~ U(0.6, 1.5) at p=0.8 (frozen
  CE: U(0.8, 1.2) at p=0.5). Amplitude/attenuation is the single §3
  lever that is physically coherent on μ-Doppler (global attenuation ≈
  the room/distance component of the shift), label-safe (a global
  scalar preserves the class pattern; velocity manipulations are not,
  §3) and in-scope (no raw CSI). One conceptual lever → clean
  attribution; bundling other channels (= variant (a)) rejected as
  uninterpretable.
- **Implementation (additive, non-mutative):** new `ce_amp` profile in
  `augment.py` (own `_PROFILE_PROBS` entry + `amplitude_range`
  override); the frozen `ce`/`supcon_view` profiles and the §3 width
  table stay byte-identical, so every archived run reproduces
  unchanged. Selected per run via new config key
  `train.augment_profile` (default "ce"; CE path only, blocking
  asserts in `train.py`). Profile name is `ce_amp`, NOT the
  "ce_s7aug" example from the review: the arm replicates on the
  S6-out rotation too, where S7 is in train — the profile names the
  transform, not a rotation. Wiring joins the pre-freeze cross-review.
- **Design (paired, no baseline reruns):** frozen splits and mu/sigma
  reused; baselines are the EXISTING `C1`, `C1_s43`, `C1_s6out` runs.
  At a fixed seed, init and batch order are identical between twin
  runs (verified in train.py: build_backbone after set_seed; shuffle
  stream = f(seed, epoch), profile-independent; the augmenter owns an
  independent RNG) → the paired test delta cancels the seed nuisance.
  Priority: cross-rotation replication (S7-out + S6-out) over a second
  seed; the s43 twin's job is to check the AUGMENTED run stays
  seed-stable (C1's 0.87-pt floor does not automatically transfer),
  reported as a mean±range cell with its s42 sibling (E1' pattern).
  Probes/diagnostics are NOT run on this arm (end-to-end rows only,
  declared E1'-style asymmetry); separate Drive folders `C1_aug`,
  `C1_aug_s43`, `C1_s6out_aug` (auto-resume never crosses runs).
- **Pre-registered hypothesis (fixed BEFORE any run):** stronger
  attenuation augmentation improves cross-AR-set generalization — the
  paired same-seed test delta (aug − baseline, 8-class test macro-F1)
  is positive on BOTH rotations. Framed on the room/attenuation
  component only (§2.2: S7 confounds room + monitor + person + day —
  NEVER read as "compensates the person shift"). Noise floor: C1's
  measured seed swing 0.87 pt; deltas within the §0.5 ~2-pt band read
  "comparable". Whatever the outcome, the report states it as one
  point on an unexplored axis, never "augmentation helps / does not
  help"; the in-domain val CANNOT see the effect (declared), so val
  numbers select checkpoints and nothing else.
- **Amendments:** §3 additive profile row (the frozen table untouched);
  §0.7 frozen row list 13 → 16 (session not yet opened — the freeze
  clause forbids extension only with the session OPEN; the notebook-05
  readiness assert, still deliberately deferred, is now written against
  16 rows); §8.4 reopened by explicit team decision (+3 × ~2.3 ≈ 6.9 h,
  extensions ≈ 16 h, inside the pre-v5.2 15–35 h envelope); §6 note
  (the arm does not reopen the eliminated augmentation ablation);
  §10.3 item 4.
