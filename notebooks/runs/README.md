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

## Index

| Notebook | Config | Status |
|---|---|---|
| `2026-07-16_c0_sharp.ipynb` | `c0_sharp` | Partial: interrupted at epoch 11/60 (KeyboardInterrupt), CPU runtime, best val macro-F1 0.667 @ epoch 6; resumable from `last.ckpt` on Drive. |
