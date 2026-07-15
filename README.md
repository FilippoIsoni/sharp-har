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
   Day 2–9 modules are stubs with a signature and `NotImplementedError`:
   review is possible, implementation lands at the corresponding gate.
7. **Every invocation on the test set is logged**, including the
   SHARP-repo-style evaluation wrapper for C0 — the test set doesn't get
   accidentally consumed during development iterations.

## Notebook → day map

| Notebook | Day | Status | Purpose |
|---|---|---|---|
| `00_setup_smoke.ipynb` | — | stub | Mount Drive + staging, quick environment check |
| `01_inventory_splits.ipynb` | 1 | **implemented** | Data inventory + frozen splits (`p2_office`, `p1_sharp`) |
| `02_smoke_gate.ipynb` | 2 | stub | Model smoke test + throughput gate |
| `03_train.ipynb` | 3+ | stub | Generic training runner over a `configs/*.yaml` |

## Data: never in the repo

The Doppler CSI data (two zip archives, ~762 MB total) live on Google
Drive, with the path defined in `configs/paths.yaml`. The notebooks mount
and stage them locally on Colab (`/content/data`); training reads only
from local staging, never from Drive. Checkpoints and feature caches
follow the same rule: never committed (see `.gitignore`).

## Running Day 1

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
4. runs the day-1 gate checks (axes, AR-set coverage, activity×AR-set
   contingency, NaN policy ≤5%, window-count sanity check);
5. freezes `splits/p2_office.json` (primary rotation for C1–C4) and
   `splits/p1_sharp.json` (SHARP reproduction for C0).

The produced artifacts (`splits/*.json`, `reports/*.csv`,
`reports/name_to_arset.json`) get committed to Git: they are the frozen
day-1 deliverables.
