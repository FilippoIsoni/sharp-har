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
