# sharp-har

Human Activity Recognition from WiFi CSI (Doppler) traces on the
**SHARP** dataset (*Environment and Person-Independent Activity
Recognition with Commodity IEEE 802.11 Access Points*, TMC), evaluated
under a **LOEO** (Leave-One-Environment-Out) protocol. Five experimental
configurations share one backbone, one data pipeline, and one evaluation
harness:

| Config | Encoder trained with | Adversary | Eval | Protocol |
|---|---|---|---|---|
| C0 | CE (SHARP-like network) | — | end-to-end, SHARP-repo decision fusion | P1 (train S1, test S2–S7) |
| C1 | CE | — | end-to-end (+ linear probe as C1-lin) | P2 (leave-S7-out) |
| C2 | CE | CE-GRL | end-to-end (+ linear probe as C2-lin) | P2 |
| C3 | SupCon | — | linear probe | P2 |
| C4 | SupCon | CE-GRL | linear probe | P2 |

The dataset (SHARP TMC): sets S1–S7 (≡ AR-1…AR-7), 12 campaigns,
3 environments (bedroom, living room, laboratory), 3 persons, identical
hardware everywhere. The primary rotation leaves the laboratory (S7)
out.

## Reference documents

- **`pipeline_wifi_har_v5.md`** — the single source of truth. Every
  module docstring cites the section it implements (`Ref. §X.Y`); when
  code and doc disagree, the doc wins or the discrepancy gets discussed.
- **`STATUS.md`** — where the project currently stands (done / in
  progress / next). Read it before starting work; this README does not
  track status.
- **`CLAUDE.md`** — working conventions for code assistants.

## Repository layout

```
sharp_har/            The versioned package — ALL logic lives here
  inventory.py        Dataset inventory, AR-set metadata, name→AR-set map
  windowing.py        Window enumeration + train-only μ/σ
  splits.py           Frozen split construction (trace-level, rare-cell pins)
  data.py             DopplerDataset + file index (windowing, normalization, antennas)
  augment.py          §3 augmentation profiles (CE + SupCon two-view)
  sampler.py          P×K sampler with distinct-trace / ≥340-offset constraints
  models/             V-B ResNet backbone, SHARP-like net, heads, shared factory
  losses.py           CE + label smoothing, SupCon, GRL + λ ramp
  train.py            Config-driven train_run (checkpoints, auto-resume, monitoring)
  harness.py          The single checkpoint→CSV eval path + test-access logging
  probe.py            Frozen linear-probe recipe, phase-B grid selection, §7 diagnostics
  bench.py            Throughput/memory gate measurements
  viz.py              Plots and comparison tables from run artifacts
  utils.py            Seeding, YAML/JSON I/O, git hash, logging
configs/              One YAML per run (c0…c4) + Colab paths (paths.yaml)
notebooks/            Thin runner templates (see map below), output-free on Git
notebooks/runs/       Executed run notebooks, committed verbatim with outputs
splits/               Frozen split JSONs (trace ids + μ/σ + metadata) — never edited
reports/              Committed measured artifacts: inventory, contingency, gate results
```

## Principle: thin notebooks, logic in the package

The notebooks are **thin runners**: they mount Drive, stage the data,
call `sharp_har` functions, and display output — no logic of their own.
The pipeline demands cross-review of the dataloader and full
reproducibility (YAML config, seed, git hash), which a notebook diff
can't be reviewed for.

| Notebook | Purpose |
|---|---|
| `00_setup_smoke.ipynb` | Mount Drive + staging, frozen-artifact asserts |
| `01_inventory_splits.ipynb` | Data inventory + split freezing (already run: splits are frozen) |
| `02_smoke_gate.ipynb` | End-to-end smoke test + throughput gate |
| `02b_phase_a_gate.ipynb` | SupCon phase-A full-batch throughput gate |
| `03_train.ipynb` | Config-driven training runner over a `configs/*.yaml` |
| `04_probe.ipynb` | Val-only probe sessions: phase-B grid selection, C1-lin/C2-lin, §7 diagnostics |
| `05_test_final.ipynb` | The single §0.7 test session across all streams |

Executed copies of real runs are committed verbatim (with outputs)
under `notebooks/runs/` as `YYYY-MM-DD_<config>.ipynb` and never edited;
the templates above stay output-free.

## Non-negotiable rules (pipeline §0)

1. **Splits are frozen** on Git and never regenerated or edited.
2. **Split by trace, never by window** — windows of one trace in both
   train and test is leakage.
3. **μ/σ come from train only**, per rotation, stored in the split file,
   reused identically on val/test.
4. Every run saves its full YAML config, seed, git hash, checkpoints,
   and per-AR-set CSV metrics; one shared harness (`checkpoint → CSV`)
   serves every evaluation stream.
5. **A single seed = 42** everywhere; differences under ~2 points are
   "comparable", not improvements.
6. No training run outside the §8.4 budget table or before its
   throughput gate passed.
7. **The test set is evaluated once, at the end**, with the val-selected
   checkpoint only — and every test access goes through the logging
   harness (`test_invocations.jsonl`), including C0's SHARP-repo-style
   eval via wrapper.

## Data: never in the repo

The Doppler CSI data (two zip archives, ~762 MB total) live in the
shared Google Drive folder referenced by `configs/paths.yaml` — never
re-downloaded from IEEE DataPort. Notebooks stage the zips locally on
Colab (`/content/data`); training reads only from local staging, never
from mounted Drive. Checkpoints, feature caches and probe heads live on
Drive per run and are never committed (see `.gitignore`).

## Getting started

```bash
pip install -r requirements.txt
```

Local work is code-only; training and evaluation run on Google Colab
through the notebooks. A typical session: open the notebook for the
phase you own (`03` for training, `04` for probes, `05` only for the
final test session), set the config/run selector in the marked cell,
run top to bottom, then commit the executed copy under
`notebooks/runs/`. There is no test suite or linter; verification
happens through the blocking asserts built into the pipeline code and
the committed gate reports.
