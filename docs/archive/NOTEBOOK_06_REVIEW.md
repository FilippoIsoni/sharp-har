# NOTEBOOK 06 — DEEP REVIEW

> Multi-level review of `notebooks/06_final_analysis.ipynb`, produced 2026-07-22.
> Unlike the 2026-07-20 audit in `CONSOLIDATION_REVIEW.md` §6 (written before the
> test session, against a planned notebook), this review was run **against the real
> committed test session** — every number below was computed by re-executing the
> notebook's own functions over `reports/final/*_test_*_windows.csv` (the §0.7
> session of 2026-07-22; audit trail `reports/final/test_invocations.jsonl`;
> executed notebook `notebooks/runs/2026_07_22_test_final.ipynb`). It changes no
> frozen artifact. `STATUS.md` remains the source of truth for *where we are*; this
> file is *what the notebook-06 review found* and *what remains to make it
> report-grade*, so a later session can restart without re-deriving anything.

## 0. Verdict

The notebook is **correct and runs clean on the real session** — all 16 streams
load, the notebook-05→06 naming contract holds, every `paired_bootstrap`
alignment assert passes, and the three analysis functions do exactly what they
document. The gaps are **not bugs**: they are analysis cells still missing
against the team's own G1–G12 list (`CONSOLIDATION_REVIEW.md` §6), and some of
those missing cells would surface results that change the report's wording.

Since the 2026-07-20 §6 snapshot the notebook has been **extended** (it now reads
macro-F1 from the metrics CSVs and its `PAIRS` cover the aug/AdaBN/T3A/sharplike
streams), so **G1, G3, G8 largely landed** and **direction A + direction-D core**
are implemented. **Still open in notebook-06 scope: G6, G7, G9, G10, G11, G12**
(plus G2 as presentation). **G4, G5 are `viz`/assembly, outside notebook 06.**

> **Update 2026-07-23 — the notebook-06 open items are now implemented and verified.**
> The follow-up cells for **G2, G6, G7, G9, G10, G11, G12** landed in commit
> `6b30b9d`; a subsequent pass then added: (a) synthetic **self-tests** for the new
> `mcnemar` / `top_confusions` / `seen_blind_f1` (refactored to pure functions in the
> analysis-functions cell, tested in the self-test cell before any data load); (b)
> **CSV persistence** of every report table to `reports/tables/` (`summary`,
> `paired_bootstrap` with `resolved`/`primary` flags, `ece`, `class_coverage`,
> `per_trace_accuracy`) so no number is retyped from stdout; (c) report **figures**
> to `reports/figures/` — per-stream confusion + per-class F1 for C0/C1/C2/C3/C1_s6out
> (no longer FOCUS-only), a cross-stream **reliability** overlay, and a **forest plot**
> of the paired deltas (pre-registered primaries highlighted); and (d) explicit
> framing (per-trace≈per-class confound on the single-set test, "CI crosses 0 =
> under-powered not null" with a resolution-floor read, label-smoothing→underconfidence
> shown computed). The notebook runs **clean end to end** on the committed session (16
> streams; every number below reproduced). Only **G4/G5** remain, and they are outside
> notebook 06 as noted. The §3 status table and §5 checklist below are updated to match.

> **Update 2026-07-23 (second pass) — report-foregrounding + figure/framing fixes.**
> A follow-up deep read flagged that several findings were *computed but buried*. Fixed
> in the notebook (still zero test contact, runs clean, output-free): (1) confusion +
> per-class figures now also cover the **backbone ablation `C1_sharplike`** and the
> SupCon fine-tune **`C3_ft`** (`FIG_STREAMS`), not just the loss families; (2) the
> **forest plot** now shows **both rotations** (the pre-registered P2-living augmentation
> test is on it, group-tagged, n=11 vs n=15 noted); (3) the **reliability overlay**
> excludes **C0** — its `sharp_c0` majority-VOTE fusion makes "confidence" ≠ P(y_pred),
> so mixing it on the calibration axes was misleading (its ECE stays in the ece table
> with the footnote); (8) **C0 is now framed as a PARTIAL, NON-FAITHFUL reproduction**
> (backbone/preprocessing/fusion differ) — a sanity anchor for our harness on P1, **not**
> a paper benchmark; the report must not claim "we reproduced X%". New **`## F — report
> headlines`** section foregrounds four syntheses, each with a persisted CSV: **F1**
> augmentation effect is environment-dependent (sign flips −0.028 lab / +0.073 living →
> `augmentation_effect.csv`); **F2** GRL damage localization (probe−e2e +0.002 for C1 vs
> +0.073 for C2 → harm is in the classifier HEAD, not the representation); **F3** TTA
> verdict (AdaBN net-negative, T3A ~0, none resolves → `tta_effect.csv`); **F4** worst-
> trace drill-down (C1−C2 gap concentrates on S7a_H/W/S → `worst_trace_c1_vs_c2.csv`).
> The two statistical notes deliberately **not** actioned: BCa vs percentile CI at n=11,
> and equal-mass vs equal-width ECE bins — declared as limitations, not changed.

> **Update 2026-07-23 (third pass) — G4/G5 settled against the actual report draft.**
> Cross-checked against the committed `report/results.tex`. **G4 (accuracy bars) is
> DROPPED** — the report conveys the main results via `tab:test`, and per-class/accuracy
> bars were cut for the 6-page IEEE budget (`report/figures/README.md`). **G5
> (domain-diagnostics consolidated table) is already DELIVERED** as `tab:domainprobe`
> (the C1 domain probe) + a prose sentence covering C2/C3/2nd-rotation. So **neither is
> pending report work** — the earlier "G4/G5 remain / outside nb-06 / report critical
> path" notes above are superseded. The report references exactly **2 figures**:
> `embeddings_c1_vs_c3.pdf` (t-SNE, from the committed PNG — **NOT** an nb-06 output) and
> `confusion_c1_test.pdf` (an nb-06 output). nb-06's other figures (perclass, reliability,
> forest) are produced but **unreferenced** — held OPTIONAL pending the page-budget build;
> the **forest plot** is the lead add-if-space candidate (it could replace the CI column
> of `tab:test`). The §3 table and §5 line below are updated to match.

---

## 1. Level 1 — correctness, completeness, coherence with the pipeline

### Verified by execution (not by reading)
- **Naming contract 05→06 holds.** `session.finalize_csvs` writes
  `<key>_test_<fusion>_<kind>.csv`; notebook 06's `split("_test_")[0]` recovers the
  clean keys (`C1`, `C1_lin`, `C1_AdaBN_T3A`, …). No collision: the metrics glob
  `f"{k}_test_*_metrics.csv"` is unique even for prefix keys (`C1` does not pick up
  `C1_s43`/`C1_aug`/`C1_lin`), confirmed on all 16 files.
- **`paired_bootstrap`** — every `a.index.equals(b.index)` in `PAIRS`, `PAIRS_S6`
  and the seed cell passes; the cluster (trace-level) resample is arithmetically a
  window-weighted paired accuracy difference, correct.
- **`per_class_report` / `calibration`** — positional class indexing matches the
  `p_<label>` column order the harness writes; the `(lo,hi]` binning has no
  double-count; ECE tracks the |acc−conf| gap, not the error rate (the synthetic
  self-test blinds this).
- **macro-F1 in the summary table is *read* from the harness** (`fused_* / ALL`
  row), not recomputed — stays identical to the reported value. Verified: C1
  0.8038, C0 0.6053 read correctly.

### Two real caveats (to declare in the report — not bugs)
1. **Calibration under `sharp_c0` (C0).** `conf = max(mean_softmax)` but C0's
   `y_pred` is the majority vote, *not* the argmax of the mean softmax, so for C0
   the confidence is not the probability of the predicted class → C0's ECE (0.103)
   is mildly ill-defined. The default `FOCUS="C1"` (softmax_avg) avoids it and the
   docstring acknowledges it, but if the report shows C0 calibration this needs a
   footnote.
2. **Percentile CI on 11 clusters is coarse.** Correct and pre-registered, but with
   n=11 traces and skew a BCa interval would be more honest. Declare the width as
   the result (it is), optionally note BCa as a refinement.

No other divergence from the pipeline: fusion, §0.7 audit trail, per-stream labels,
and the deliberate separation of the two variances are all conformant.

---

## 2. Level 2 — conceptual assessment (are the choices optimal + motivated?)

**Sound and well-argued:** bootstrap on *accuracy* not macro-F1 (a trace-level
macro-F1 bootstrap is ill-defined here — only ~2.8 % of resamples contain all 8
classes); **trace-level cluster** bootstrap (window-level would violate
split-by-trace and understate variance); the **two variances kept separate**
(test-sampling vs seed) — the notebook's strongest methodological choice; a fixed
seed shared across all `PAIRS` (desirable — same resample enables the G9
juxtaposition).

**Sub-optimal / incomplete choices:**
- **`FOCUS` is single-stream.** Calibration + error-structure run for one stream at
  a time. But direction D says the per-AR-set cut is *degenerate on the primary
  test* (single set S7) and *rich on C0* (S2–S7). C0's per-AR-set is genuinely
  informative (§4: AR-2 0.66 … AR-7 0.73), so the most valuable direction-D
  instrument is exactly the one the `FOCUS="C1"` default never activates — the
  notebook should also run the D-cuts for C0.
- **Confidence definition + label smoothing.** The systematic underconfidence of C1
  (§4: every reliability bin has acc>conf) is largely *expected* from the label
  smoothing in the CE loss; the calibration cells do not connect the two.

---

## 3. Level 3 — report-readiness: done-vs-open map + what the missing cells reveal

### Done-vs-open against the G1–G12 list

| G | item | status in nb-06 today (updated 2026-07-23) |
|---|---|---|
| A | trace-level paired bootstrap | **done** (`PAIRS`, `PAIRS_S6`; persisted + forest plot) |
| D | calibration + per-class/per-AR-set error | **done** (per-stream figures loop, not FOCUS-only) |
| G1 | C1-aug paired test deltas | **done** (`C1_aug`−`C1`; `C1_s6out_aug`−`C1_s6out`) |
| G2 | E1′ seed range on test | **done** — seed cell prints `mean ± half-range` + gap range |
| G3 | C1+AdaBN+T3A comparison | **done** (in `PAIRS`) |
| G8 | master table + macro-F1 in 06 | **done** (summary reads metrics CSVs → `summary.csv`) |
| G6 | **per-trace** error cut | **done** — per-trace grid + per-trace≈per-class note (`per_trace_accuracy.csv`) |
| G7 | **cross-stream ECE** | **done** — all-stream ECE (`ece.csv`) + reliability overlay figure |
| G9 | **two variances juxtaposed** | **done** — seed range vs test-sampling CI side by side |
| G10 | **McNemar-style discordance** C1 vs C3/C2 | **done** — `mcnemar` (self-tested), counts + anti-conservative caveat |
| G11 | **multiplicity caveat** sentence | **done** — primary/secondary flags in table + declared caveat |
| G12 | **class-coverage decomposition** | **done** — `seen_blind_f1` (self-tested) → `class_coverage.csv` |
| G4 | accuracy bars per config×domain | **dropped** — report uses `tab:test`; bars cut for the 6-page budget |
| G5 | domain-diagnostics consolidated table | **delivered** — present in `results.tex` as `tab:domainprobe` (C1) + prose |

### The findings the missing cells would surface (measured on the real session)

- **G12 — the caveat is empirically *false* on p2_lab, and non-uniform across
  rotations.** cell-11 asks "val was blind to C/E/S — do they underperform?" but
  never computes the answer. On p2_lab the val-*blind* classes are the *best*
  (mean F1 {C,E,S}=0.904 vs val-seen {H,J,L,R,W}=0.744); on p2_living the picture
  flips — a val-*seen* class collapses (S F1=0.131) while the blind {H,L,W} sit
  mid-pack. The story is not uniform, which is exactly why the computed
  decomposition (+ a val-visible test macro-F1 restricted to each rotation's val
  classes) is needed, not an eyeball. Note: on test **no class is absent**, so the
  `absent_classes` column is empty everywhere — val-blindness shows up as
  per-class *performance*, confirming G12 (not the absent list) is the right
  instrument.
- **G9/G11 — even the headline gap does not resolve on 11 traces.** C1 vs C2:
  observed **+0.158** but paired-bootstrap CI **[−0.034, +0.323]**, P(C1>C2)=0.946
  — a 16-point gap not significant at 95 % two-sided. C1 vs C3 by contrast **does**
  resolve (+0.078, CI [+0.030, +0.128], P=1.000). This is the honest wide-interval
  result and it must sit next to the seed range (G9) with a multiplicity sentence
  (G11).
- **G7 — the loss families are differently calibrated.** ECE: C2 0.070 (best) vs
  C1 0.243 — C2/GRL is far better calibrated but much less accurate; TTA
  (C1_T3A 0.345, C1_AdaBN_T3A 0.306) *worsens* calibration. A genuinely
  interesting cross-stream comparison the notebook computes for FOCUS only.
- **TTA narrative (direction E) — computed but not foregrounded.** T3A ≈ C1
  (0.8118 vs 0.8047, Δ within noise, CI crosses 0); **AdaBN *hurts*** (0.7459).
  A clean neutral/negative TTA result the notebook produces but does not narrate.
- **G6/G10 — the informative granularity is absent.** On a single-set test the
  11 traces are the granularity; a trace×config correctness grid + a C1↔C3
  agreement table would give the *mechanism* behind "SupCon buys nothing".
- **"Sensible confusions" (direction D) — extractable, not extracted.** C1's
  dominant off-diagonals are W↔R (18+18, walking↔running — adjacent Doppler),
  H→J (14), then low-motion S→L/H/C. Physically plausible confusions validate
  real learning; one cell would surface this from the confusion CSV.

---

## 4. Measured-numbers appendix (source of truth — reproducible from `reports/final/`)

**Per-stream (fused, test).** Accuracy recomputed from windows CSV; macro-F1 read
from the metrics CSV; ECE from the notebook's `calibration`.

| stream | split | n_tr | n_win | acc | macro-F1 | ECE |
|---|---|---|---|---|---|---|
| C0 | P1 5-cls | 57 | 2717 | 0.6117 | 0.6053 | 0.103 |
| C1 | P2-lab | 11 | 425 | 0.8047 | 0.8038 | 0.243 |
| C1_s43 | P2-lab | 11 | 425 | 0.8000 | 0.7990 | 0.237 |
| C2 | P2-lab | 11 | 425 | 0.6471 | 0.6006 | 0.070 |
| C2_s43 | P2-lab | 11 | 425 | 0.6000 | 0.5618 | 0.067 |
| C1_lin | P2-lab | 11 | 425 | 0.8071 | 0.8059 | 0.153 |
| C2_lin | P2-lab | 11 | 425 | 0.7200 | 0.7080 | 0.173 |
| C3 | P2-lab | 11 | 425 | 0.7271 | 0.7286 | 0.107 |
| C3_ft | P2-lab | 11 | 425 | 0.7200 | 0.7059 | 0.206 |
| C1_T3A | P2-lab | 11 | 425 | 0.8118 | 0.8101 | 0.345 |
| C1_AdaBN | P2-lab | 11 | 425 | 0.7459 | 0.7221 | 0.214 |
| C1_AdaBN_T3A | P2-lab | 11 | 425 | 0.7576 | 0.7525 | 0.306 |
| C1_aug | P2-lab | 11 | 425 | 0.7765 | 0.7720 | 0.235 |
| C1_sharplike | P2-lab | 11 | 425 | 0.5694 | 0.5434 | 0.187 |
| C1_s6out | P2-living | 15 | 716 | 0.7430 | 0.6842 | 0.172 |
| C1_s6out_aug | P2-living | 15 | 716 | 0.8156 | 0.8227 | 0.224 |

**Paired bootstrap** (accuracy, N=10000, seed 42; diff = A−B, positive favours A):

| comparison | diff | 95 % CI | P(A>B) | resolved? |
|---|---|---|---|---|
| C1 vs C2 | +0.1576 | [−0.0336, +0.3233] | 0.946 | no (crosses 0) |
| C1 vs C3 | +0.0776 | [+0.0295, +0.1282] | 1.000 | **yes** |
| C1 vs C3_ft | +0.0847 | [−0.0582, +0.2212] | 0.864 | no |
| C1 vs C1_sharplike | +0.2353 | [+0.0924, +0.3986] | 1.000 | **yes** |
| C1 vs C1_aug | +0.0282 | [−0.0233, +0.0924] | 0.779 | no |
| C1 vs C1_AdaBN | +0.0588 | [−0.0307, +0.1929] | 0.791 | no |
| C1 vs C1_T3A | −0.0071 | [−0.0613, +0.0474] | 0.408 | no (T3A ≈ C1) |
| C1 vs C1_AdaBN_T3A | +0.0471 | [−0.0354, +0.1419] | 0.839 | no |
| C1_s6out_aug vs C1_s6out | +0.0726 | [−0.0289, +0.2025] | 0.894 | no |

**Seed spread on test** (n=2, a spread not a CI): C1 vs C1_s43 \|Δ\|=0.0047;
C2 vs C2_s43 \|Δ\|=0.0471 (GRL seed-instability replicates on test).

**C1 per-class F1 (p2_lab test):** C 0.933 · E 1.000 · H 0.713 · J 0.812 ·
L 0.871 · R 0.673 · S 0.780 · W 0.649. Blind {C,E,S} mean 0.904, seen
{H,J,L,R,W} mean 0.744.

**C1_s6out per-class F1 (p2_living test):** seen {C 0.796, E 0.908, J 0.888,
R 0.626, S 0.131} · blind {H 0.647, L 0.731, W 0.748}.

**C1 top confusions (true→pred, count):** W→R 18 · R→W 18 · H→J 14 · S→L 7 ·
S→H 5 · S→C 5.

**C0 per-AR-set accuracy (test S2–S7, non-degenerate):** AR-2 0.662 · AR-3 0.721 ·
AR-4 0.549 · AR-5 0.499 · AR-6 0.573 · AR-7 0.732.

---

## 5. Restart checklist (what to add, prioritized — all zero test contact)

> **Status 2026-07-23: items 1–8 below are all implemented and verified** in
> `notebooks/06_final_analysis.ipynb` (see the §0 Update). They are kept here as the
> record of what was added and why. Notebook 06 itself is report-grade. **G4/G5, once
> listed here as the remaining report-critical work, are now settled** (see the
> 2026-07-23 third-pass update in §0): **G4 dropped** (budget — `tab:test` covers the
> results), **G5 already delivered** as `tab:domainprobe` + prose. The only report work
> still outstanding is placing the 2 referenced figures + the page-budget build, all
> outside this notebook.

All items are analysis on the already-committed `reports/final/*_windows.csv`
(+ `*_metrics.csv`), plus one `viz` figure. None is a rerun or a new §0.7 row.

1. **G12** — computed class-coverage decomposition per stream: seen-vs-blind
   per-class F1 aggregate + a val-visible test macro-F1 restricted to each
   rotation's val classes (p2_lab seen {H,J,L,R,W}; p2_living seen {C,E,J,R,S}).
   *Highest report value — the caveat's answer is counterintuitive and non-uniform.*
2. **G6** — per-trace correctness cut (11 traces on the primary test), the
   informative granularity where per-AR-set is degenerate.
3. **G7** — cross-stream ECE line (C0/C1/C2/C3 + TTA rows); optionally a reliability
   figure and the label-smoothing note.
4. **G9 + G11** — one table placing the C1–C2 gap, its paired-bootstrap CI, and the
   seed range side by side (Bouthillier made concrete), with a one-line multiplicity
   caveat for the ~10 comparisons.
5. **G10** — C1↔C3 (and C1↔C2) window-agreement / discordance table.
6. **G2 presentation** — turn the seed cell's \|Δ\| into `C1 = mean±range`,
   `C2 = mean±range`, and the C1–C2 gap as a range.
7. **FOCUS for C0** — run the direction-D cuts for C0 too (its per-AR-set is rich).
8. **Confusion interpretation** — extract dominant off-diagonals (the "sensible
   confusions" sentence).

Outside notebook 06, both now settled (see the 2026-07-23 third-pass update in §0):
**G4** (`viz` accuracy bars per config×domain) is **dropped** — `tab:test` conveys the
results and the bars were cut for the 6-page budget; **G5** (domain-diagnostics
consolidated table) is **already delivered** in `results.tex` as `tab:domainprobe`
(the C1 probe) + prose covering C2/C3/2nd-rotation. Neither is pending.

Process note: notebook 06 is notebook-local math by the diagnostics dividing line
(`notebooks/diagnostics/README.md`); keep the "synthetic self-test first" pattern
when adding cells, and if any of these become re-run pipeline metrics, promote to
`sharp_har/diagnostics.py` as NCM/kNN did.
