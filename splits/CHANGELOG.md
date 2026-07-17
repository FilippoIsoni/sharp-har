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
  val cost vs C1); (iii) C1_s43's cached features double as the E2
  concat ensemble control C1⊕C1′.
- **Budget:** 2 × ~2.3 h ≈ 4.6 h ≤ the freed C4 budget (§8.4 amendment,
  net negative). Each replicate adds ONE pre-registered test row
  (mean±range cell with its seed-42 sibling in the report); probes,
  diagnostics and techniques stay on seed 42 only.
- **Not replicated:** C3 (3× the cost; its findings are already
  replicated across encoders/machines; the flat 40/50/60 grid — 0.7 pt
  span — already evidences representation stability; declared
  single-seed in the report).
