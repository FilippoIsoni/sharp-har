# sharp-har

Human Activity Recognition from WiFi CSI (Doppler) traces on the
**SHARP** dataset (*Environment and Person-Independent Activity
Recognition with Commodity IEEE 802.11 Access Points*), evaluated under
the **LOEO** (Leave-One-Environment-Out) protocol, with a progression of
core runs C0→C4 (reproduction baseline, cross-entropy, adversarial GRL,
contrastive SupCon, and the SupCon+GRL combination).

## Principle: thin notebook, logic lives in the package

All logic lives in the versioned Python package `sharp_har/`. The
notebooks in `notebooks/` are **thin runners**: they mount Drive, stage
the data, call the package's functions, and display inspectable output.
They contain no logic of their own. This is required because the
pipeline demands cross-review of the dataloader and full reproducibility
(YAML config, seed, git hash) — things a notebook diff can't be reviewed
for.

`STATUS.md` is the single source for where the project currently stands
(done / in progress / next) — read it before starting work.

## Non-negotiable principles

1. **Split by trace, never by window** — split lists contain trace-ids,
   not window indices, to avoid leakage between train/val/test.
2. **μ/σ come from train only**, computed for the current rotation and
   stored in the split file. Never recomputed on val/test.
3. **A single seed = 42** for every stochastic choice in the project.
4. **Once frozen, splits are committed to Git and never touched again.**
5. **Data and checkpoints never enter the repo.** They live on Drive
   (~762 MB); the repo contains only code, configs, frozen splits
   (`splits/*.json`), and reports (`reports/*.csv`).
6. **The training architecture isn't decided until it needs to be.**
   Each module stays a stub (a signature and `NotImplementedError`)
   until the corresponding pipeline gate passes — review is possible
   before implementation lands.
7. **Every invocation on the test set is logged**, including the
   SHARP-repo-style evaluation wrapper for C0 — the test set doesn't get
   accidentally consumed during development iterations.

## Notebook → day map

| Notebook | Day | Purpose |
|---|---|---|
| `00_setup_smoke.ipynb` | — | Mount Drive + staging, frozen-artifact asserts |
| `01_inventory_splits.ipynb` | 1 | Data inventory + frozen splits (`p2_lab`, `p1_sharp`) |
| `02_smoke_gate.ipynb` | 2 | Model smoke test + throughput gate |
| `02b_phase_a_gate.ipynb` | 3 | SupCon phase-A full-batch throughput gate |
| `03_train.ipynb` | 3+ | Config-driven training runner over a `configs/*.yaml` |
| `04_probe.ipynb` | §10.2 | Val-only probe sessions (frozen-encoder linear probe, §7 diagnostics, phase-B grid selection) |
| `05_test_final.ipynb` | §10.2 | Single §0.7 test session across all streams, once every stream has a val-selected checkpoint |

Executed run notebooks are committed verbatim (with outputs) under
`notebooks/runs/` as `YYYY-MM-DD_<config>.ipynb`; the templates above
stay output-free on Git.

## Data: never in the repo

The Doppler CSI data (two zip archives, ~762 MB total) live on Google
Drive, with the path defined in `configs/paths.yaml`. The notebooks mount
and stage them locally on Colab (`/content/data`); training reads only
from local staging, never from Drive. Checkpoints and feature caches
follow the same rule: never committed (see `.gitignore`).

## Getting started: day 1 (inventory + splits)

```bash
pip install -r requirements.txt
```

Open `notebooks/01_inventory_splits.ipynb` (on Colab, or locally with
the data already staged) and run the cells in order. The notebook:

1. mounts Drive and stages the two zips locally, timing the staging;
2. inspects the real file names and asks for confirmation of the regex
   pattern before proceeding;
3. builds `reports/inventory.csv` (one row per file-stream) and
   `reports/name_to_arset.json`;
4. runs the day-1 gate checks (axes, AR-set coverage — sets S1–S7 ≡
   AR-1…AR-7, 12 campaigns —, activity×AR-set contingency, NaN policy
   ≤5%, window-count sanity check);
5. freezes `splits/p2_lab.json` (primary rotation for C1–C4:
   leave-S7-out, laboratory) and `splits/p1_sharp.json` (SHARP
   reproduction for C0: train S1, test S2–S7).

The produced artifacts (`splits/*.json`, `reports/*.csv`,
`reports/name_to_arset.json`) get committed to Git: they are the frozen
day-1 deliverables.
