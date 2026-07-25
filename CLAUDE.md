# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project

WiFi CSI Human Activity Recognition on the SHARP TMC Doppler dataset, evaluated
under Leave-One-Environment-Out. **`PROJECT.md` is the single source of truth**
— specification (§0–§9), execution record, measured results, findings and
limitations, decision log, repository map. Read the sections you need before
changing anything; if code and `PROJECT.md` disagree, the document wins, or the
discrepancy gets discussed — never silently resolved. Module docstrings cite it
as `Ref. §X.Y`; Part I preserves that numbering. Superseded documents live in
`docs/archive/` and must not be read or updated.

**Phase: the project is closed experimentally.** The single test session ran
2026-07-22; the analysis closed 2026-07-23. What remains is the IEEE report in
`report/` and the presentation. **No new training run, no new test contact, no
split regeneration** — code freeze 2026-07-28, deadline 2026-07-30.

## Language and style conventions

- **All code, comments, docstrings, notebook text and commit messages are in
  English.** The team communicates in Italian; nothing Italian goes into the
  repo. Exception: Italian domain terms already used as data labels
  (`attivita`, `campagna`) — those column names are part of the frozen artifact
  schema and stay as they are.
- Docstrings cite the section they implement (`Ref. §X.Y`). Keep this pattern.
- Everything is parametrized on `d_enc` and `n_att` — no hardcoded feature or
  class counts anywhere.
- Type hints + `from __future__ import annotations` in every module.

## Commands

```bash
pip install -r requirements.txt
```

No test suite, linter config or build step. Verification happens through the
blocking asserts built into the pipeline code (split disjointness, axes check,
AR-set coverage, NaN policy, rare-cell coverage, sampler invariants, cache
alignment, §0.7 readiness) and through the committed gate reports. Training runs
on Google Colab via the notebooks; local work is code-only.

## Architecture

**Thin notebooks, logic in the package.** `notebooks/*.ipynb` mount Drive, stage
data, call `sharp_har` functions and display output. No logic is ever added to a
notebook — the dataloader needs cross-review and notebooks can't be diffed for
it. Templates stay output-free on Git; the executed copy of every real run is
committed **verbatim with outputs** under `notebooks/runs/` as
`YYYY-MM-DD_<config>.ipynb` and never edited afterwards. Investigation sessions
go to `notebooks/diagnostics/`; a diagnostic graduates into the package when its
numbers enter the report *and* it is re-run across sessions.

**Every module is live — there are no stubs left.** `inventory` · `windowing` ·
`splits` · `data` · `augment` · `sampler` · `models/{resnet_vb,sharp_like,heads}`
· `losses` · `train` · `harness` · `probe` · `diagnostics` · `transductive` ·
`session` · `bench` · `viz` · `utils`. See `PROJECT.md` Part VI for what each one
owns.

**One config per experimental run.** `configs/*.yaml` fully describe a run
(backbone, loss, adversary, sampler, optimizer, horizon, augmentation profile);
`train.py` consumes a config, never per-run flags. `configs/paths.yaml` holds
Colab paths.

**Frozen artifacts flow through Git** and are never modified: `splits/*.json`,
`reports/*.csv`, `reports/gate_*.json`, `reports/final/`, `report/tables/*.csv`
and the executed notebooks. Data (~762 MB on shared Drive) and checkpoints never
enter the repo.

## Non-negotiable rules (§0 — violations invalidate results)

1. Splits are frozen as JSON on Git and never edited afterwards.
2. **Split by trace, never by window** — windows of one trace on two sides of a
   split is leakage. All 4 antennas of a trace stay on the same side.
3. μ/σ computed on train only, per rotation, stored in the split file, reused
   identically on val/test.
4. Every run saves: full YAML config, seed, git hash, checkpoints, per-set CSV
   metrics. One shared eval harness (`checkpoint → CSV`) for all streams.
5. Seed 42 everywhere (init, per-epoch sampler reseed, augmentation); the only
   exception is the two pre-registered seed-43 replicates. Differences under ~2
   points are "comparable", not improvements.
6. No training run outside the budget table (§8.4).
7. **The test set was evaluated once, on 2026-07-22, with val-selected
   checkpoints.** That session is closed and the row list is frozen. Any further
   test access would invalidate the protocol.

Domain constraints worth remembering: velocity-axis flip and time flip are
forbidden augmentations (§3); gradient accumulation is forbidden in SupCon phase
A (§4.2); the P×K sampler enforces distinct traces with a ≥340 offset on reuse
(§4.2); the frozen `ce`/`supcon_view` augmentation profiles are byte-identical
and new profiles are added additively (§3).
