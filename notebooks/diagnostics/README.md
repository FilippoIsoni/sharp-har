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
  and only calls it. The dividing line (sharpened 2026-07-18): truly
  one-off probes stay notebook-local; anything with its own new math
  whose numbers enter the report and that is re-run across sessions is
  pipeline code — the NCM/kNN scorers were promoted to
  `sharp_har/diagnostics.py` on this basis (cross-review recorded in
  STATUS, math verified identical to the notebook-local original), and
  the domain-probe instrument followed the same day
  (`diagnostics.domain_probe`): the pre-registered E2′ S6-out
  replication (§10.3) turned it from a one-off into a re-run
  instrument. The three executed domain-probe sessions predate the
  promotion, stay archived verbatim, and their recorded rows stand
  (math verified byte-identical on synthetic data).
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
| `2026-07-18_probe_c1_s6out.ipynb` | E2′ (§10.3) replication of the domain diagnostic on the **leave-S6-out** rotation's C1 train features (`C1_s6out`, `p2_living`) — the 2nd environment in train here is the **laboratory** (S7), not living-room. Same promoted `diagnostics.domain_probe`, inner trace-disjoint split (53 fit / 27 eval). | **Verdict replicates on a second rotation.** Control `y` acc 0.870 (delta +0.660; lower than the S7-rotation controls because the inner eval traces are held-out *within* train → generalization, not memorization — plumbing sound). Every domain target at/below its majority baseline (ar_set +0.011, `ambiente`/`persona` exact constant predictors, direct_path −0.029, monitor +0.002). The "no readable domain in CE features" finding holds with a different 2-environment train composition — no counterexample. |
| `2026-07-18_ncm_knn_c1_c2_c3.ipynb` | §7 v5.2: are C1/C2/C3's activity features robust to readout choice? NCM (per-class re-normalized centroids) and kNN (k=20, vote-fraction + similarity tie-break) on the same cached features the linear probes used, scorers imported from `sharp_har.diagnostics` (promoted 2026-07-18; first C1 numbers came from the byte-identical notebook-local cell), fused via `harness.fuse_windows` (reused, not reimplemented), printed against the frozen linear-probe numbers already on record. | **C1-only session (owner A).** NCM 0.8653/0.8888, kNN 0.8453/0.8563 — both ≈ the linear probe (0.8711/0.8835) → readout-robust. C2/C3 were blocked here on Drive shortcuts; the full stream set is the sibling row below. |
| `2026-07-18_ncm_knn_c1_c2_c3_full.ipynb` | Same instrument, **full stream set** (Melissa's session with the C2/C3 shortcuts available + the C1_s43 footnote). Baseline 0.3209 throughout. | **Complete.** NCM (acc/F1): C1 0.8653/0.8888, C2 0.7765/0.8176, C3 0.6963/0.7178, C1_s43 0.8567/0.8707. kNN: C1 0.8453/0.8563, C2 0.8424/0.8663, C3 0.7937/0.8047, C1_s43 0.8281/0.8497. **C3: kNN ≫ NCM (the t-SNE chaining as a non-linear gain) but still < C3's linear probe 0.8190** → the linear recipe did NOT understate SupCon; C3 is lowest under every readout. C1 readout-robust; C1_s43 confirms seed robustness on NCM/kNN too. |
| `2026-07-19_concat_c1_c3.ipynb` | §7 v5.2 concat: does SupCon capture activity information CE misses (C1⊕C3, 512-d, frozen §5.3 probe recipe unchanged) or does any second encoder help the same way (control C1⊕C1′ = seed 42 ⊕ seed 43)? Alignment across caches asserted by `diagnostics.concat_caches`, never repaired. | **No complementarity.** C1⊕C3 val macro-F1 0.8684 < C1-lin 0.8835 and < control C1⊕C1′ 0.8882; **candidate − control = −0.0197** (needed > +0.02, the §0.5 band). SupCon is a *worse* concat partner than a same-loss twin — coherent with C3 losing on every readout and the 16/349 error-overlap. Both probes select epoch 1/6 on the fragile 5-class val (magnitude noisy, direction robust across all C3 evidence). Val-only, no test contact. → Candidate B (joint CE+SupCon) stays future-work with a negative concat number. |
| `2026-07-18_embeddings_c1_vs_c3.ipynb` | §9 v5.2 key figure: qualitative PCA→t-SNE picture of C1 (CE) vs C3 (SupCon) train-feature geometry, colored by activity and by AR-set. Thin notebook — logic lives in `viz.plot_embeddings` (package code, cross-review pending). C1 vs C3 only (C2's story is already the diagnostic table; C4 never trained) — declared scope, not a default to extend without a team call. | **Visual match to the diagnostic table.** By-activity: 8 tight clusters in both encoders (consistent with C1-lin 0.8835 / C3 0.8190); C3's L/S/E clusters visibly chain into one continuous shape rather than 3 discrete blobs, a plausible geometric reading of why a *linear* probe scores lower on C3 even though the structure is arguably still there. By-AR-set: colors are uniformly mixed within every cluster, in BOTH encoders — no visible domain sub-structure, echoing the delta≈0 diagnostic result qualitatively. Train-only/seen-traces caveat applies (declared). Asset: `reports/embeddings_c1_vs_c3.png`. |
