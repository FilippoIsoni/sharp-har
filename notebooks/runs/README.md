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
| `2026-07-18_c2_grl_s43.ipynb` | `c2_grl_s43` | E1′ seed replicate, complete single session: best val macro-F1 **0.7870 @ epoch 6**, early stop 16/40. `arset_train_acc` at the majority floor (0.30-0.32) for all 16 epochs — the §6-C2 no-op-adversary finding is robust to init. Seed swing vs C2 (0.8415) = **5.45 pts** → fired the pre-registered seed-44 trigger. (Was `e1_seed_replicates/03_train_c2_grl_s43.ipynb`, moved here per the archive plan.) |
| `2026-07-18_c1_ce_s43_part1.ipynb` | `c1_ce_s43` | E1′ seed replicate, mid-run snapshot (was `e1_seed_replicates/03_train_c1_ce_s43_epoch31.ipynb`, web-uploaded): log covers epochs 9-31 — itself a resume; the first session's notebook (epochs 1-8) was not saved (declared C0-style exception) and epoch 32 ran after the snapshot. Part-2's full-history dict covers all 40 epochs and matches these log lines to print precision (≤4.9e-05). |
| `2026-07-18_c1_ce_s43_part2.ipynb` | `c1_ce_s43` | E1′ seed replicate, final session: epochs 33-40, run complete 40/40 (no early stop). **Best val macro-F1 0.8784 @ epoch 37** — same best epoch as C1 seed-42 (0.8871 @ 37), seed swing **0.87 pts**, inside the §0.5 band: C1 is seed-stable, the E1′ instability is C2/GRL-specific. Full 40-epoch history in its `Train run finished` dict. |
