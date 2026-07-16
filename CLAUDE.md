# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WiFi CSI Human Activity Recognition on the SHARP Doppler dataset, evaluated under a Leave-One-Environment-Out protocol. Five experimental configurations (C0 reproduction, C1 CE baseline, C2 CE+GRL, C3 SupCon, C4 SupCon+GRL) share one backbone, one data pipeline, and one evaluation harness.

**Dataset reality (v5.1 errata):** the shared Drive copy holds the SHARP TMC dataset — sets S1–S7 (internally named AR-1…AR-7, 1:1), 12 campaigns (S1a/b/c, S2a/b, S3a, S4a/b, S5a, S6a/b, S7a), 3 environments (bedroom S1–S5, living room S6, laboratory S7), 3 persons, identical hardware everywhere. AR-8/AR-9 do not exist. Primary P2 rotation = leave-S7-out (`splits/p2_lab.json`); C0 uses the paper's 5-class set. Training data comes exclusively from the shared Drive folder — never re-download from IEEE DataPort.

**`STATUS.md` tracks where we are.** Read it at the start of a session; when work completes a milestone or changes the plan, update `STATUS.md` in the same commit (move items between Done/In progress/Next, keep it synthetic — one line per item). Do not duplicate status information here.

**`pipeline_wifi_har_v5.md` is the single source of truth.** Every module docstring references its sections (e.g. §4.2, §5.3). Before implementing or changing anything, read the relevant section; if code and pipeline doc disagree, the doc wins — or the discrepancy gets discussed, never silently resolved.

## Language and style conventions

- **All code, comments, docstrings, notebook text, and commit messages are in English.** The team communicates in Italian, but nothing Italian goes into the repo (exception: the pipeline doc itself and Italian domain terms already used as data labels, e.g. `attivita`, `campagna` — keep those column names as they are, they are part of the frozen artifact schema).
- Docstrings cite the pipeline section they implement (`Ref. §X.Y`), as the existing modules do. Keep this pattern.
- Everything is parametrized on `d_enc` and `n_att` — no hardcoded feature or class counts anywhere.
- Type hints + `from __future__ import annotations` in every module, as in the existing code.

## Commands

```bash
pip install -r requirements.txt   # numpy/pandas for day 1; torch etc. for training days
```

There is no test suite, linter config, or build step. Verification happens through the blocking asserts built into the pipeline code (split disjointness, axes check, AR-set coverage, NaN policy, rare-cell coverage) and through the day-2 smoke gate. Training runs happen on Google Colab via the notebooks; local work is code-only.

## Architecture

**Thin notebooks, logic in the package.** `notebooks/*.ipynb` only mount Drive, stage data, call `sharp_har` functions, and display output. No logic is ever added to a notebook — the dataloader requires cross-review and notebooks can't be diffed for that.

**Gated implementation.** Modules are deliberately stubs (`NotImplementedError`) until their pipeline day arrives — the day-2 throughput gate can change the architecture (escalation §5.2), so nothing downstream is implemented before its gate passes. Do not "helpfully" fill in a stub without being asked.

Current status:
- **Implemented (day 1):** `utils.py`, `inventory.py`, `windowing.py` (window enumeration + μ/σ only), `splits.py`, notebook `01_inventory_splits`.
- **Implemented (day 2):** `data.py` (`DopplerDataset` + `build_file_index`).
- **Stubs (day 2+):** `augment.py`, `sampler.py`, `losses.py`, `harness.py`, `probe.py`, `train.py`, `models/` (V-B ResNet, heads, SHARP-like net), notebooks `00`, `02`, `03`.

**One config per experimental run.** `configs/c*.yaml` fully describe a run (backbone, loss, adversary, sampler, optimizer, horizon); `train.py` consumes a config, never per-run flags. `configs/paths.yaml` holds Colab paths.

**Frozen artifacts flow through Git.** `splits/*.json` (trace-id lists + μ/σ + axes/classes metadata) and `reports/*.csv` (inventory, contingency) are day-1 deliverables committed once and never modified. Data (~762 MB on shared Drive) and checkpoints never enter the repo.

## Non-negotiable rules (pipeline §0 — violations invalidate results)

1. Splits are frozen on day 1 as JSON on Git and never edited afterwards.
2. **Split by trace, never by window** — windows of one trace in both train and test is leakage.
3. Normalization μ/σ computed on train only, per rotation, stored in the split file, reused identically on val/test.
4. Every run saves: full YAML config, seed, git hash, checkpoints, per-set CSV metrics. One shared eval harness (`checkpoint → CSV`) for all streams.
5. Single seed 42 everywhere (init, samplers reseeded per epoch, augmentation). Differences under ~2 points are "comparable", not improvements.
6. No training run outside the budget table (§8.4) or before the day-2 throughput gate passes.
7. **The test set is evaluated once, at the end, with the val-selected checkpoint only.** Every test invocation goes through the logging harness — including C0's SHARP-repo-style eval, via wrapper.

Domain-specific constraints worth remembering: velocity-axis flip and time flip are forbidden augmentations (§3); gradient accumulation is forbidden in SupCon phase A (§4.2); the P×K sampler enforces distinct traces with a ≥340 offset on reuse (§4.2).
