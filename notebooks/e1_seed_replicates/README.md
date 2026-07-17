# notebooks/e1_seed_replicates — E1: seed replicates of C1 and C2

Pre-configured copies of the `03_train` template (one per replicate,
`RUN` pinned, GPU sanity cell added) — open in Colab, run all, no edits.
The two notebooks can run in parallel in separate sessions.

**What E1 is:** an amendment to §0 rule 5 (single seed 42), recorded in
`splits/CHANGELOG.md`. Configs `c1_ce_s43.yaml` / `c2_grl_s43.yaml` are
byte-identical to their seed-42 originals except `name` and `seed`.

**What it measures:**

- the pipeline's **seed noise floor** — the empirical calibration of the
  §0.5 "~2 points = comparable" band (two seeds give a range, not a CI:
  report as "observed seed-to-seed variation", no significance claims);
- **robustness of the C2 findings** to initialization: adversary at the
  train majority floor (0.2969) again? val cost vs C1 reproduced?
- `C1_s43` features double as the **E2 concat control** C1⊕C1′ (cached
  later in the diagnostics session, not here).

**Scope guards:** techniques and probes stay on seed 42 only. The
replicates contribute end-to-end numbers (one pre-registered test row
each, collapsed into the stream's mean±range cell in the report) and the
C1_s43 feature cache. C3 is deliberately NOT replicated: 3× the cost,
its findings are already replicated across encoders/machines, and its
40/50/60 grid (0.7-pt span) already evidences representation stability —
declared single-seed in the report.

Executed copies go to `notebooks/runs/` as `YYYY-MM-DD_<config>.ipynb`,
per the usual convention. These templates stay output-free.
