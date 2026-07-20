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

- **G — Cross-domain data augmentation.** Strongly supported in the literature
  (simple amplitude augmentations improve cross-scenario generalization), **but**
  the §3 augmentation set is frozen and deliberately restricted (velocity-axis
  and time flips forbidden). Changing it = new artifact + many runs + a new
  axis. A good future-work sentence, not work for now.
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

## 6. Sources

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
