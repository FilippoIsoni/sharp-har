# CONSOLIDATION REVIEW

> Literature-grounded review of directions to **consolidate the project and
> its results** (not to expand scope with new models). Produced 2026-07-20,
> while the pipeline is in the v5.2 tail, ~8 days before code freeze
> (2026-07-28, deadline 2026-07-30). This is a planning document: collaborators
> should start here. It changes no frozen artifact and proposes no run that is
> not pre-registered. Where it touches the plan, `STATUS.md` remains the source
> of truth for *where we are*; this file is *what is worth doing next and why*.

## 0. Framing

The project's value is a **methodologically airtight negative result** about
loss families for WiFi CSI HAR under Leave-One-Environment-Out:

- SupCon does not beat CE on this dataset (seven independent instruments agree
  on the ~0.82 SupCon ceiling: linear probe, NCM, kNN, t-SNE, concat, full
  fine-tune, and the domain diagnostics).
- The GRL/domain-adversarial branch has **no readable domain target** under
  either loss family (replicated on C1/C2/C3/C1_s43/C3-ft and on two rotations),
  and this is **structural** — forced by the dataset's set/environment incidence,
  not by our training (see §3.C).

Consolidation therefore means: make the null **more rigorous, more general, and
better positioned**, not "keep trying techniques until something wins" (that
would destroy the honest-null contribution). Every direction below is evaluated
against the project's non-negotiable indications: pre-registration, single test
contact (§0.7), frozen splits and seed, one shared backbone, and the freeze
runway.

## 1. Method

Literature was searched (2026-07-20) on six axes central to the project:
cross-environment generalization in WiFi sensing; failure modes of
domain-adversarial training (DANN/GRL); SupCon vs CE on small data; test-time
adaptation for CSI; benchmark variance / significance with few runs; and the
value of negative results. The generalizability taxonomy was extracted from the
2025 Wi-Fi Sensing Generalizability survey. Sources are listed in §5.

## 2. Decision table

| # | Direction | Value | Cost | Extra test contact | Lit. support | Verdict |
|---|---|---|---|---|---|---|
| A | Statistical rigor: paired bootstrap + variance framing | High | ~0 (analysis) | None | Strong | **Do** |
| B | Literature positioning of the two nulls (GRL, SupCon) | High | 0 (writing) | None | Strong | **Do** |
| C | E3 (leave-bedroom-out) + incidence argument | High | 0 (run rejected 07-20) | None | Strong (as proof) | **Do — proof only, no run** |
| D | C1 model characterization (calibration + error structure) | Med-high | Low | In-session only | Medium | **Do** |
| E | TTA framing (T3A/AdaBN) vs current SOTA | Medium | 0 | Already pre-reg. | Strong | **Do** |
| I | Reproducibility package for the null | High | Low | None | Strong | **Do** |
| G | Data augmentation as a cross-domain lever | Medium | Many runs | New rows | Strong | Future work |
| F | New backbone (ViT/TCN/diffusion) | Low | ~5 days | New rows | Weak (here) | Excluded |
| H | Self-supervised / source-free UDA | Low-med | Very high | — | Medium | Future work |

## 3. Directions to do (pros/cons vs project indications)

### A. Statistical rigor: paired bootstrap + variance framing
The project already discovered, empirically, that the **seed swing (5.45 pts,
C2 vs C2_s43) exceeds the config gap (4.6 pts, C1 vs C2)**. This is exactly the
phenomenon in Bouthillier et al., *Accounting for Variance in ML Benchmarks*
(seed-only differences can exceed the gap between competing SOTA methods).
Citing it converts our ad hoc observation into a principled, referenced result,
and retroactively justifies rule-5 (fixed seed for the controlled comparison)
plus the n=2 range as a variance-aware strategy, not a weakness.

For the model **comparison**, the significance literature (paired-bootstrap
protocol) recommends a **paired** bootstrap: resample the same test traces and
read the *difference* C1−C2 / C1−C3 per resample. The pairing cancels
per-trace difficulty, giving a defensible p-value/CI for "C1 beats C2/C3" even
with a small test set. Note this supersedes the earlier informal dismissal of
"bootstrap on a tiny test set": the **marginal per-model** CI is weak on ~11
traces, but the **paired** form for comparison is the correct, more powerful
tool.

- **Pro:** no runs; no extra test contact (uses per-trace predictions logged in
  the single §0.7 session); closes the "is the gap distinguishable from noise?"
  thread rigorously; strong literature backing.
- **Con:** with ~11 test traces the interval stays wide — but *declaring* that
  is the correct result. Bootstrap must be at **trace (cluster) level**, never
  window level (windows within a trace are correlated; window-level bootstrap
  would violate the split-by-trace principle and understate variance).

### B. Literature positioning of the two nulls
- **GRL-null.** Literature documents that DANN "aligns sources but may shrink
  feature-space diversity, risking over-alignment," and that the reversal
  struggles to balance alignment and diversity. Our verdict (GRL pays a cost
  with no target) is the extreme case: when the domain is not even readable,
  only the cost remains. Combined with the **structural incidence argument**
  (§3.C), this is a contribution to the "standardized evaluation protocols"
  open challenge the survey names.
- **SupCon-null.** SupCon's edge over CE is reported as small in some cases and
  demonstrated at **large scale with many classes** (ImageNet/CIFAR); its known
  benefits are on **transfer and corruption robustness**, not necessarily
  in-domain accuracy at small scale. Our setting (~81 traces, 8 classes) is the
  opposite regime. Honest framing: our null is *consistent with* the literature,
  and we did **not** test the axis where SupCon is strong (transfer/robustness)
  — itself a precise future-work sentence.
- **Pro:** zero cost; turns "found on our data" into "consistent with / extends
  known results." **Con:** writing only; needs accurate citation.

### C. E3 (leave-bedroom-out) + incidence argument
Already agreed as positive expected value. The set/environment incidence is:

```
bedroom    = AR-1,2,3,4,5   (5 sets)
living_room= AR-6           (1 set)
laboratory = AR-7           (1 set)
```

**No environment other than bedroom is represented by more than one AR-set.**
Combinatorial consequence, provable on paper, run-independent: in *any* LOEO
rotation of this dataset, the in-train environment is either the bedroom (a
single environment — invariance undefinable) or a single AR-set (= session
identity, not a generalizable "environment"). **No rotation of this dataset
poses environment-invariance non-degenerately.** Hence "the GRL has no target"
is not an outcome of our runs — it is forced by the dataset design.

Honest note: E3's own train (living AR-6 + lab AR-7) also has each environment =
a single set — which is precisely what makes the argument a *proof* rather than
luck.

**Feasibility check DONE (2026-07-20) → the E3 run is REJECTED; the incidence
argument is kept as a proof.** The check this section required as a pre-req was
run against the frozen artifacts and `_stratified_val_split`:
- train pool = 26 traces (living 15 + lab 11), test = bedroom 76; §2.2 pinning
  makes 15 of 16 `(ar_set, attivita)` cells rare → **val = 2 traces / 2 classes**,
  no blocking assert (guard rejects only empty val).
- **Decisive reason is not the val (that is fixable via no-val pre-registration)
  but circularity:** E3's null is *entailed* by the incidence proof — its own
  single-set-per-environment train cannot pose invariance non-degenerately — so
  the run would illustrate a theorem, not test it; and its 26-trace train confounds
  "does not generalize" with "had no data". A diagnostic-only variant does not
  escape either objection (the domain probe recovers session identity).
- **Corrected framing:** the incidence argument is the capstone, as a §9
  proposition-with-proof. The LOEO matrix is complete as far as it can be
  non-degenerately — **P2-lab (S7-out) + E2′ (S6-out) done**; the bedroom-out
  rotation is the degenerate one and is covered by the proof, not by a run.
- Still true: do **not** run C2/C3 on E3 (would reconfirm a degenerate target).
  See STATUS "Blockers / open decisions" for the full rejection record.

### D. C1 model characterization (calibration + error structure)
C1 is the deliverable model but has only ever been summarized by a scalar
macro-F1. Inside the **single §0.7 test session** (no extra test contact),
extract:
- **Calibration** (reliability diagram / ECE): is the CSI classifier
  well-calibrated or overconfident?
- **Error structure** per class / per AR-set / per trace. macro-F1 averages 8
  classes equally, so it hides whether the model is uniformly mediocre or strong
  on 6 classes and failing on 2 confusable ones. Sensible confusions (activities
  with similar Doppler signatures) validate real learning; the per-class cut
  also **empirically tests the declared caveat** that val is blind to C/E/S —
  do those classes actually underperform on test?
- **Pro:** report depth at ~zero cost; the harness already emits per-AR-set
  confusion CSVs. **Con:** per-AR-set cut is degenerate on the primary test
  (single set S7); it is rich on E3 (test = 5 bedroom sets) and C0 (test
  S2–S7).

### E. TTA framing of the transductive rows
The T3A/AdaBN rows are already pre-registered. Recent literature (DATTA, WACV
2026 — domain-adversarial test-time adaptation for cross-domain WiFi HAR;
MetaFormer — one-sample adaptation) shows TTA is the live direction of the
field. Position AdaBN/T3A as simple TTA baselines; if they recover part of the
S7 cross-environment gap, report it as consistent with that direction. Zero
extra runs.

### I. Reproducibility package for the null
The negative-results and reproducibility literature is explicit: a surprising
null needs a meticulous, accessible setup to show it is not an implementation
artifact. The project is already unusually rigorous (frozen artifacts, git,
pre-registration, pipeline doc). Packaging it (reproduction README, the §10.4
declaration list, pipeline doc) is the highest-certainty consolidation and
plays to an existing strength — and it matters *more* precisely because the
result is a null.

## 4. Excluded / future work (with reasons)

- **G — Cross-domain data augmentation.** Reviewed in depth 2026-07-20 (full
  record in STATUS "Blockers / open decisions"). The literature (2401.00964)
  supports amplitude augmentation for cross-scenario transfer, but weakly. On our
  μ-Doppler data the lever question resolves cleanly: **amplitude/attenuation is
  the only coherent, label-safe, in-scope lever** — time-warp is physically
  incoherent (time and velocity are coupled in μ-Doppler, so stretching time alone
  models no real motion), velocity-warp is label-unsafe (velocity separates
  walking/running, §3), and channel-sim / generative CSI need raw CSI we do not
  hold. Since S7 confounds room + monitor + person + day (§2.2), any gain is
  unattributable and must be framed on the room component only. If ever pursued it
  is a **minimal additive arm** (a new config-driven `ce_s7aug` profile — the
  frozen §3 table stays byte-identical — paired at fixed seed against existing
  baselines, replicated across the S7-out and S6-out rotations, ~3 runs), **not**
  the "many runs + new backbone" shape. Still future work, not runway work — but if
  run, this is the correct and only shape.
- **F — New backbone (ViT/TCN/diffusion).** Breaks the shared-backbone design
  that gives the comparison its integrity; cannot move the GRL verdict (proven
  structural in §3.C); needs its own throughput gate and budget line; ViT is
  data-hungry for ~81 train traces and diffusion is off-axis (generative, does
  not plug into the discriminative harness) and infeasible in the runway.
  Future work / v2 — as an explicitly declared, pre-registered robustness arm on
  the loss-family verdict (success = "verdict replicates", not "higher number"),
  never folded into the frozen shared-backbone table.
- **H — Self-supervised / source-free UDA.** A real direction of the field
  (e.g. MU-SHOT-Fi) but out of scope and out of runway.

## 5. Recommended plan for the freeze runway (~8 days)

None of these is "build something new for 5 days"; all are consolidation that
raises the rigor and generality of the existing null.

1. **A + B + I, in parallel with report writing** (zero-run): paired bootstrap
   on the single-session predictions; Bouthillier variance framing; positioning
   of both nulls; reproduction package.
2. **The incidence proof** (§3.C) as the LOEO capstone that makes the null
   *structural* — **zero runs.** The feasibility gate ran (2026-07-20) and
   **rejected the E3 run**: its null is entailed by the proof and its 26-trace
   train confounds the result (val = 2 traces besides). So no additional run
   enters the runway; the matrix stays P2-lab + E2′.
3. **D + E** inside the single §0.7 test session already planned.

## 6. Post-test analysis audit — is the result set report-grade? (2026-07-20)

Directions **A** and **D** above are implemented as `notebooks/06_final_analysis.ipynb`
(trace-level paired bootstrap + calibration/ECE + per-class/per-AR-set error). The
notebook is methodologically sound — bootstrap on accuracy not macro-F1 (macro-F1
ill-defined on ~11 traces, 6 of 8 classes with a single trace each), test-sampling
variance kept separate from seed variance, per-stream labels, a synthetic self-test.
But an audit against the §9 report spec (2026-07-20) finds the post-test **analysis
pipeline does not yet produce a report-grade result set**. Every gap below is a
pre-registered §0.7 row whose result is uncomputed, a pre-registered §9 key figure
with no producer, or a named direction-D cut that is missing — **none is a new
post-hoc comparison**. Listed here (planning only, no frozen artifact touched) so the
pre-freeze work closes a known list rather than an ad hoc one.

**Coverage coupling (read first).** Notebook 06 is row-count-agnostic: it analyses
whatever `*_test_*_windows.csv` the single §0.7 session writes, keyed by the filename
prefix, so its output is bounded by notebook 05. Today 05 (i) evaluates 7 streams —
C0, C1, C2, C1-lin, C2-lin, C3 and **the closed C4** — i.e. covers 6 of the frozen 16
rows plus a stream v5.2 closed, and misses 10 (C1_s43, C2_s43, C1+AdaBN, C1+T3A,
C1+both, C1_s6out, C3-ft, C1_aug, C1_aug_s43, C1_s6out_aug); and (ii) copies artifacts
under checkpoint-stem keys (`C1_best`, `C1_C1_lin`) that the naming contract and 06's
`FOCUS`/`PAIRS` (`C1`, `C1_lin`, `C1_AdaBN`, `C3_ft`, …) do not match. A report-grade
set needs the notebook-05 rewrite (already on the §10.4 checklist) **and** the 06/viz
extensions below; neither alone suffices. The 05 side is tracked with the freeze
checklist; this section is the 06/viz side.

### Must-fix — pre-registered results with no output
- **G1 — C1-aug paired test deltas.** The augmentation arm's *only* deliverable
  (§10.3 item 4) is a paired Δ on test: `C1_aug−C1` (s42), `C1_aug_s43−C1_s43`,
  `C1_s6out_aug−C1_s6out` (the last on P2-living, its own units — not poolable with
  the P2-lab pairs). `PAIRS` in 06 has none of them. Without this the arm has no
  number and "one point on an unexplored axis" cannot be written.
- **G2 — E1′ seed range on test.** §9 mandates "media ± (min–max)" once seeds are
  added; C1/C1_s43/C2/C2_s43 are all frozen test rows. The report table needs
  C1 = mean±range, C2 = mean±range, and the C1–C2 gap as a *range* — the val story
  STATUS already closed (direction-consistent, 3.7–10 pt) reproduced on test. 06
  computes no seed table.
- **G3 — C1+AdaBN+T3A (row 11).** `PAIRS` compares C1 to AdaBN and to T3A singly but
  not to the composed row, the 3rd pre-registered transductive row (§9).

### Must-fix — pre-registered §9 key figures with no producer
- **G4 — accuracy bars per config × domain (§9 key figure #1).** `viz.py` has
  `plot_history`/`plot_confusion`/`compare_runs`/`metrics_table`/`plot_embeddings` —
  **no grouped bar chart.** The report's *figure #1* has no code. Needs a `viz`
  function over the harness metrics CSVs (fused rows), degenerate-aware (P2-lab test =
  single set S7; rich on C0 S2–S7 and the S6-out row).
- **G5 — domain-diagnostics consolidated table (§9 key figure #2).** The invariance
  evidence (target × encoder, delta-vs-own-baseline) is the figure that *replaces* the
  underpowered §7 probe, but the numbers still live as prose across five diagnostic
  notebooks + STATUS (C1, C2, C3, C1_s6out, C3-ft). Nothing assembles them into the
  one grid the report shows. Train/val-only, no test contact — pure assembly, yet no
  artifact today.

### Should-do — direction-D depth and the report headline
- **G6 — per-trace error cut.** Direction D names per class / per AR-set / **per
  trace**; 06 has the first two. On the primary test per-AR-set is degenerate (single
  S7), so **per-trace (11 traces) is the informative granularity** and it is absent.
- **G7 — cross-stream ECE.** Calibration runs for `FOCUS=C1` only; an ECE-per-stream
  line (C1/C2/C3/C0) is a cheap, genuinely interesting comparison — are the loss
  families differently calibrated?
- **G8 — master results table + macro-F1 in 06.** 06 reads only `*_windows.csv`,
  never the `*_metrics.csv`, so the report's headline metric (macro-F1, with the
  `absent_classes` list) and a re-runnable master table (acc + macro-F1 + n_traces per
  stream) live only in the run-once notebook 05. The post-session analysis notebook
  should own the master table.

### Should-do — framing that makes the null excel
- **G9 — the two variances, juxtaposed (Bouthillier on our data).** §3.A's hook —
  seed swing (5.45 pt) > config gap (4.6 pt) — is the report's strongest
  methodological point. On test we can now show, in one figure/table, the C1–C2 gap +
  its test-sampling CI (paired bootstrap, direction A) + the seed range (G2). 06
  correctly keeps them "separate and labelled" but never places both numbers side by
  side, which is exactly what instantiates Bouthillier concretely.
- **G10 — error-discordance (McNemar-style) C1 vs C3/C2.** The paired bootstrap gives
  the aggregate gap; a per-window agreement table (windows C3 rescues that C1 misses,
  and vice versa) gives the *mechanism* behind "SupCon buys nothing / encoders
  redundant" (STATUS: 16/349 val error-overlap). Cheap test-side complement,
  strengthens the SupCon-null.
- **G11 — multiple-comparisons caveat.** With the expanded `PAIRS` (10+ comparisons on
  ~11 traces) the report needs a declared multiplicity caveat (STATUS already flags the
  pressure). Not a correction — the honest wide interval is the result — a sentence.

### Sequencing
None of G1–G11 is a run or a new test contact; all are analysis on the single
session's CSVs (plus assembly of already-measured val/train diagnostics for G5) and
one `viz` figure (G4). Order: notebook-05 rewrite (16 clean-keyed rows) → 06
`PAIRS`/`FOCUS` extension + per-trace / master-table / seed-range / discordance cells
(G1–G3, G6–G11) → `viz` accuracy-bars (G4) + domain-diagnostics table assembly (G5).
Do them with the report draft, after the single §0.7 test session has written its CSVs.

## 7. Sources

- A Survey on Wi-Fi Sensing Generalizability (2025) — https://arxiv.org/pdf/2503.08008
- Wi-Fi Sensing for HAR: Survey, Challenges, Research Directions (ACM CSUR) — https://dl.acm.org/doi/10.1145/3705893
- Bouthillier et al., Accounting for Variance in ML Benchmarks (MLSys 2021) — https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
- A Tale of Two Variances: When Single-Seed Benchmarks Fail — https://arxiv.org/pdf/2604.23114
- A Paired Bootstrap Protocol for Evaluating Small Models — https://www.arxiv.org/pdf/2511.19794
- Khosla et al., Supervised Contrastive Learning (NeurIPS 2020) — https://arxiv.org/pdf/2004.11362
- Transferability of Representations from Supervised Contrastive Learning — https://arxiv.org/pdf/2309.15486
- Data Augmentation for Cross-Domain WiFi CSI HAR — https://arxiv.org/abs/2401.00964
- Position: Embracing Negative Results in Machine Learning — https://arxiv.org/pdf/2406.03980
- Semmelrock et al., Reproducibility in ML-based Research (AI Magazine 2025) — https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002
- MU-SHOT-Fi: Source-free UDA for Multi-User Wi-Fi Sensing — https://arxiv.org/pdf/2605.01369
