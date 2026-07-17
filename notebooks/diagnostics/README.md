# notebooks/diagnostics — one-off investigation sessions

Executed notebooks for **investigations**, not for protocol runs: sessions
that answer "is this instrument measuring what we think it measures?".
Committed **verbatim with their outputs**, same policy as
`notebooks/runs/` — measured artifacts enter Git unmodified.

Why a separate folder: `notebooks/runs/` is the record of the §10.2
pipeline (one archived session per config, per the frozen protocol). The
sessions here are outside that protocol — they are throwaway analyses
whose *findings* may feed a team decision, and mixing them into `runs/`
would blur what is a deliverable and what is a diagnostic.

Rules:

- Name: `YYYY-MM-DD_<what>.ipynb`, date = when the session ran.
- The analysis code lives in the notebook, not in `sharp_har/`: these are
  not part of the pipeline and must not create the impression that they
  are. They may **import** frozen package functions (that is the point —
  the recipe under test stays untouched), never redefine them.
- Train/val only. A diagnostic never touches the test set (§0.7); if one
  ever needs to, it goes through the logging harness like everything else.
- Findings that change the plan land in `STATUS.md`; the notebook is the
  evidence, not the conclusion.

## Index

| Notebook | Question | Outcome |
|---|---|---|
| `2026-07-17_c1_ce_domain_probe.ipynb` | Is the domain (AR-set / environment / person / monitor / LOS) linearly readable from C1's frozen CE encoder? Imports the frozen §5.3 recipe (`probe.linear_probe`) and runs it on an inner trace-disjoint split of the 81 train traces, against each target's own majority baseline. | **No.** Positive control `y` = 1.000 (delta +0.803); every domain target lands at or below its majority baseline (`ambiente`/`persona` are exactly constant predictors). C1's features are a near-pure activity code → the C2/C4 GRL has no target to remove. Caveats and the structural root cause (train has 2 environments, one a single AR-set) are recorded in `STATUS.md`. |
