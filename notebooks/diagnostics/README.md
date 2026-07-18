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
  **Declared exception:** a diagnostic that graduates into a §9/§10.4
  report figure or metric (e.g. `viz.plot_embeddings`) belongs in the
  package like any other `viz.py` function, with cross-review before
  freeze (CLAUDE.md: no logic in notebooks) — the notebook stays thin
  and only calls it. One-off numeric probes (NCM/kNN, domain diagnostics)
  stay notebook-local; a plotting function that becomes THE figure does not.
- Train/val only. A diagnostic never touches the test set (§0.7); if one
  ever needs to, it goes through the logging harness like everything else.
- Findings that change the plan land in `STATUS.md`; the notebook is the
  evidence, not the conclusion.

## Index

| Notebook | Question | Outcome |
|---|---|---|
| `2026-07-17_c1_ce_domain_probe.ipynb` | Is the domain (AR-set / environment / person / monitor / LOS) linearly readable from C1's frozen CE encoder? Imports the frozen §5.3 recipe (`probe.linear_probe`) and runs it on an inner trace-disjoint split of the 81 train traces, against each target's own majority baseline. | **No.** Positive control `y` = 1.000 (delta +0.803); every domain target lands at or below its majority baseline (`ambiente`/`persona` are exactly constant predictors). C1's features are a near-pure activity code → the C2/C4 GRL has no target to remove. Caveats and the structural root cause (train has 2 environments, one a single AR-set) are recorded in `STATUS.md`. |
| `2026-07-17_c3_supcon_domain_probe.ipynb` | The C4 gate: do SupCon features retain the domain that CE discards? Same instrument as the C1/C2 sessions (stem parametrized — C3 has no `best.ckpt`), run on the phase-B feature caches over the full epoch40/50/60 grid (maturity sweep, compression hypothesis). Requires `phase_b_selection.json`. | **Gate closed: no.** Control `y` = 0.995/0.993/0.996 across the grid; every domain target at or below its majority baseline on all three checkpoints (`ambiente`/`persona` are the exact constant predictors, third replication). No maturity trend — the compression hypothesis does not hold for domain. Coherent with the objective: P×K puts same-class views from different AR-sets in every batch and SupCon pulls them together, so domain suppression is built in. The C4 GRL has no target under either loss family. |
| `2026-07-17_c2_grl_domain_probe.ipynb` | Confirmatory: the same diagnostic (identical code) on C2's cached train features — is the GRL encoder any more domain-invariant than plain CE, and did the GRL's 4.6-pt val cost come from transfer only or from train fit too? Run in a session where only C2's cache was staged (C1 SKIP; the C1 session above covers it). | **Verdict confirmed.** Every domain target sits at its majority baseline on C2 too (largest delta +0.015; `ambiente`/`persona` are the exact same constant predictors as C1) — the adversary removed nothing; on macro-F1 the domain is even slightly *more* readable than C1 (ar_set 0.144 vs 0.066). New finding: `y` control = 0.893 on traces the encoder trained on, vs C1's 1.000 → the GRL cost train fit itself, not just transfer. |
| `2026-07-18_ncm_knn_c1_c2_c3.ipynb` | §7 v5.2: are C1/C2/C3's activity features robust to readout choice? NCM (per-class re-normalized centroids) and kNN (k=20, vote-fraction + similarity tie-break) on the same cached features the linear probes used, fused via `harness.fuse_windows` (reused, not reimplemented), printed against the frozen linear-probe numbers already on record. | *(pending — template committed output-free)* |
| `2026-07-18_embeddings_c1_vs_c3.ipynb` | §9 v5.2 key figure: qualitative PCA→t-SNE picture of C1 (CE) vs C3 (SupCon) train-feature geometry, colored by activity and by AR-set. Thin notebook — logic lives in `viz.plot_embeddings` (package code, cross-review pending). C1 vs C3 only (C2's story is already the diagnostic table; C4 never trained) — declared scope, not a default to extend without a team call. | **Visual match to the diagnostic table.** By-activity: 8 tight clusters in both encoders (consistent with C1-lin 0.8835 / C3 0.8190); C3's L/S/E clusters visibly chain into one continuous shape rather than 3 discrete blobs, a plausible geometric reading of why a *linear* probe scores lower on C3 even though the structure is arguably still there. By-AR-set: colors are uniformly mixed within every cluster, in BOTH encoders — no visible domain sub-structure, echoing the delta≈0 diagnostic result qualitatively. Train-only/seen-traces caveat applies (declared). Asset: `reports/embeddings_c1_vs_c3.png`. |
