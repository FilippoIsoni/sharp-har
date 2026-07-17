# notebooks/runs — archived executed notebooks

Definitive, executed copies of the real training/eval runs, committed
**verbatim with their outputs** — the same policy as the gate reports
(`reports/gate_*.json`): measured artifacts enter Git unmodified.

Convention:

- Name: `YYYY-MM-DD_<config>.ipynb` (config = the `configs/*.yaml` stem,
  e.g. `2026-07-16_c0_sharp.ipynb`). If one run spans several Colab
  sessions (auto-resume, §8.2), suffix the segments `_part1`, `_part2`, …
- Content: the executed notebook exactly as downloaded from Colab
  (`File → Download → .ipynb`). Never re-run, edit, or clear outputs of
  an archived copy; language rules do not apply retroactively to
  captured outputs.
- The templates (`notebooks/0*.ipynb`) stay output-free on Git; this
  folder is where outputs live.
- Authoritative numbers still come from the run directory on Drive
  (`history.csv`, `run_meta.json`, checkpoints) and from the harness
  CSVs — an archived notebook is the human-readable session record,
  not a data source.

Sibling folder: `notebooks/diagnostics/` holds investigation sessions
(one-off analyses outside the §10.2 protocol), not run deliverables.

## Index

| Notebook | Config | Status |
|---|---|---|
| `2026-07-16_c0_sharp_part1.htm` + `_part1_history.csv` | `c0_sharp` | GPU rerun, main session (epochs 1-30); best val macro-F1 0.8916 @ epoch 20. Part 1 is an HTML export + CSV instead of an executed `.ipynb` (the session notebook was not saved — declared exception). |
| `2026-07-16_c0_sharp_part2.ipynb` | `c0_sharp` | GPU rerun, resumed tail: epoch 31 only, early stop 31/60. Its `Train run finished` dict carries the full 31-epoch history — verified identical to `_part1_history.csv` at full float precision (2026-07-17). |
| `2026-07-16_c1_ce.ipynb` | `c1_ce` | Complete, 40/40 epochs; best val macro-F1 0.8871 @ epoch 37. |
| `2026-07-16_c2_grl.ipynb` | `c2_grl` | Complete; best val macro-F1 0.8415 @ epoch 13, early stop 23/40. See the corrected §6-C2 monitoring reading in `STATUS.md`. |
| `2026-07-16_c2_grl_probe.ipynb` | `c2_grl` | C2-lin probe (val macro-F1 0.8410) + §7 diagnostics. |
| `2026-07-17_c1_ce_probe.ipynb` | `c1_ce` | C1-lin probe (val macro-F1 0.8835) + §7 diagnostics (ar_set 0.287, persona 0.928). |
| `2026-07-17_c3_supcon_part1.ipynb` | `c3_supcon` | Phase A, session covering epochs 26-42 (was `c3_epoch42_train.ipynb`). Resumes from the first session (epochs 1-25), whose notebook was not saved — declared exception, see `_part1_history.csv`. |
| `2026-07-17_c3_supcon_part1_history.csv` | `c3_supcon` | Drive `history.csv` snapshot at epoch 42 (epochs 1-42, covering the unarchived first session; was `history.csv`). Verified against part-2's full-history dict: agreement to ≤3.3e-13 relative — float-representation truncation only, the file passed through a tool that shortened the digits. |
| `2026-07-17_phase_b_c3.ipynb` | `c3_supcon` | Phase B: §5.3 probe on the epoch40/50/60 grid → **selected epoch 40** (val macro-F1 0.8190; 0.8150 / 0.8120 — plateau). Includes the §7 ar_set/persona probes on the selected checkpoint (0.289 vs 0.390 / 0.928 = baseline, known-underpowered val). |
| `2026-07-17_c3_supcon_part2.ipynb` | `c3_supcon` | Phase A, final session: epochs 43-60, run complete 60/60. Train loss 5.914 → 4.430, plateaued on the annealed tail. No in-loop selection by design (§6-C3, `best_val_macro_f1` = -1): deliverables are epoch40/50/60.ckpt on Drive. Its `Train run finished` dict carries the full 60-epoch history. |
