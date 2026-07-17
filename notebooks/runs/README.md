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
| `c0_sharp_train.ipynb` | `c0_sharp` | GPU rerun, **resumed tail only** (log opens at epoch 31, early stop there); best val macro-F1 0.8916 @ epoch 20. Epochs 1-30 are in the `.htm` export below. Name/segmentation are an open decision in `STATUS.md`. |
| `2026-07-16_c0_sharp.htm` + `_history.csv` | `c0_sharp` | HTML export + CSV of the GPU rerun's main session (epochs 1-30). Non-conforming archive format, declared in `STATUS.md`. |
| `2026-07-16_c1_ce.ipynb` | `c1_ce` | Complete, 40/40 epochs; best val macro-F1 0.8871 @ epoch 37. |
| `2026-07-16_c2_grl.ipynb` | `c2_grl` | Complete; best val macro-F1 0.8415 @ epoch 13, early stop 23/40. See the corrected §6-C2 monitoring reading in `STATUS.md`. |
| `2026-07-16_c2_grl_probe.ipynb` | `c2_grl` | C2-lin probe (val macro-F1 0.8410) + §7 diagnostics. |
| `2026-07-17_c1_ce_probe.ipynb` | `c1_ce` | C1-lin probe (val macro-F1 0.8835) + §7 diagnostics (ar_set 0.287, persona 0.928). |
