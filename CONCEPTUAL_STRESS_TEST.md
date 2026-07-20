# CONCEPTUAL STRESS TEST

> Multi-level conceptual pressure test of the project (2026-07-20) — an
> internal adversarial read of the **load-bearing claims**, not an
> implementation review and not a caveat inventory (the team already
> declares the obvious caveats). It changes no frozen artifact and proposes
> no un-pre-registered run. `STATUS.md` stays the source of truth for
> *where we are*; this file is *what the conceptual foundations can and
> cannot bear*. It complements `CONSOLIDATION_REVIEW.md` (literature-grounded
> consolidation) and in one place (L4) **pushes back on that document's own
> confidence**.
>
> Levels are ordered most load-bearing first. **L0–L2, L4, L5, L7 are
> conceptual** — they are fixed in the report's prose (each ends with the
> exact "Report wording" to adopt), no code changes. **L6 and L8 carry a
> precise implementation.** (Level 3 — epistemic process — is deliberately
> out of scope for this write-up, by request.)

---

## Verdict in one line

The project is rigorous *inside* every local decision but rests on three
conceptual beams that do not carry the weight the framing puts on them:
(L0) a **non-sequitur** at the centre of the C4 closure, (L1) a **dataset
that cannot support the domain-generalization claim** its protocol name
evokes, and (L2) a **measurement resolving power** (≈11 traces, singleton
classes) two orders of magnitude below the method sophistication. L4–L8 are
consequences or local corrections. None of this invalidates the honest-null
contribution; all of it is fixable in framing, and two items (L6, L8) in
scope/analysis.

---

## The unifying seed-value principle (governs L5, L6, L7)

> A seed replicate is worth running **only if** a specific report claim needs
> **(a)** a calibrated noise floor **or (b)** a direction-across-init — **and**
> the measurement can resolve it.
>
> It is **not** worth running for aesthetic symmetry, to attach ± bars, or to
> **estimate a variance** (n=2 does not estimate variances).

Applied: C1_s43 / C2_s43 serve claim (b) → kept (done); s44 serves a magnitude
no claim uses → not run (correct); **C0 seeds** serve nothing → **no** (L5);
**C1_aug_s43** serves the arm's own variance → **cut** (L6).

---

## L0 — The C4 closure conflates two different things (most load-bearing)

**Claim under stress.** *"Domain is not linearly readable from train features
→ the GRL has no target → C4 = C3 + noise"*, reported as a **structural proof**
("a theorem we already hold").

**Where it bends.** *"Domain is not linearly decodable from train features on
seen traces"* ≠ *"the representation carries no domain-dependent shortcut that
harms transfer."* On train, invariant and domain-specific features are **both**
present and **both** predictive (the `y` control saturates at 1.000). The DG
literature lives precisely in the gap: adversarial/contrastive methods work (when
they work) by preventing *reliance* on domain-correlated, non-transferable cues —
a linear domain probe on **seen train traces** cannot see an activity-entangled
domain shortcut, and the one split where it would matter (held-out domain S7) is
**undefined** for domain readability (single domain). So "GRL useless" is a
stronger statement than the probe evidence supports.

**What survives.** Two things, and they are enough — just not via the probe step:
- **C4-skip as budget triage** is sound: C2 paid a measured val cost with no
  gain, so re-spending ~7 h on C4 was correctly declined.
- **The structural claim has an independent, airtight carrier** — the
  incidence argument (`CONSOLIDATION_REVIEW.md` §3.C): bedroom = 5 sets,
  living = 1, lab = 1, so no LOEO rotation poses environment-invariance
  non-degenerately. That proof does **not** route through the probe and does
  not commit the non-sequitur.

**Report wording.** Present the probe result as *"domain is not **linearly**
decodable from train features on traces the encoder has seen"* (with the
memorization confound quantified via the `y` gap). **Carry the "structural"
claim on the incidence proof, not on probe readability.** Frame C4 as a
*pre-registered non-run on triage + the incidence proof*, never as "the GRL is
proven useless by the diagnostics." Decoupling the two arguments removes the
only inferential overreach in the GRL story.

---

## L1 — The dataset cannot support a *domain*-generalization claim

**Claim under stress.** P2 LOEO measures "AR-set / environment invariance."

**Where it bends.** Environment incidence is bedroom = 5 sets, living = 1, lab =
1. There are only **two held-out environments** (S7, S6), each a **single
session**, and at test "environment" is inextricable from person P3 + day +
LOS/NLOS + monitor M4. This is a **2-point hard-held-out-session transfer study**,
not a domain-generalization study. E2′'s "second-environment replication" is the
**re-appearance of the same degenerate confound** (one environment is a
singleton), not an independent replication of the finding.

**What survives.** The LOEO hygiene (split-by-trace, train-only μ/σ, freeze) is
impeccable and stays. This is the same structural fact as `CONSOLIDATION_REVIEW.md`
§3.C — cross-reference it.

**Report wording.** Downgrade "AR-set invariance" scope to *"generalization to a
single unseen hostile session (S7; and S6), n=2 held-out environments"*; state the
confound bundle (room + person + day + LOS/NLOS + monitor) explicitly wherever the
S7 number appears. The rigor is real; it is rigor applied to a narrower question
than "DG."

---

## L2 — Method sophistication ≫ measurement resolving power

**Claim under stress.** A 16-row test table with 4-digit macro-F1 point estimates
is informative.

**Where it bends.** The independent unit for a generalization claim is the
**trace**, not the window: effective n ≈ 11 (S7), 15 (S6), not ~600 windows. A
macro-F1 printed to 4 digits on 11 traces **visually oversells** a precision the
trace count cannot support. The tell is your own: the macro-F1 bootstrap is
**ill-defined** here (only ~2.8 % of trace resamples contain all 8 classes), which
is why notebook 06 switched to accuracy — the test set cannot support a macro-F1 CI.
With **16 rows on 11 traces** (6 singleton classes), the multiplicity surface is
large relative to the information: the table realistically resolves **2–3
performance tiers, not 16**.

**What survives.** The large, concordant gaps (C1 vs C3 ≈ 6.9 pt; C1 > C2
directional). `CONSOLIDATION_REVIEW.md` §3.A (paired trace-level bootstrap) and
§6 G11 (multiplicity caveat) are the correct tools — cross-reference.

**Report wording.** Report every test row with its **explicit `n_traces`**; never
present a bare 4-digit macro-F1 as a measurement. Lead comparisons with the
**paired, trace-level** bootstrap CI plus a declared multiplicity caveat, and state
a priori that most small-delta rows (transductive, aug) are **expected
"comparable"** — the wide honest interval is the result.

---

## L4 — "Seven instruments agree" is shared-input redundancy, not independent confirmation

**Claim under stress.** `CONSOLIDATION_REVIEW.md` §0: *"seven independent
instruments agree on the ~0.82 SupCon ceiling."*

**Where it bends.** Five of the seven — linear probe, NCM, kNN, concat, t-SNE —
consume the **same cached features** (train, seen traces) with the **same failure
mode** (memorization confound; `y` saturates at 1.000). Their agreement is largely
**shared-input redundancy**, not seven independent measurements. A positive control
pegged at 1.000 also cannot calibrate a measurement whose interesting values sit
near baseline (0.2–0.4). By input path there are really **two families**: (i)
readouts on the same frozen features, and (ii) **full fine-tune (C3-ft)** — a
different input path.

**What survives.** The **conclusion** (C3 < C1) is robust — because family (ii),
the one genuinely different-input instrument, independently confirms it, and it is
the one that most cleanly **falsified** the hypothesis (0.8183 ≈ C3-lin, geometry
un-chaining toward C1). It is the "seven independent instruments" *framing* that
oversells independence, not the ceiling result.

**Report wording.** Count **two families of evidence** — multiple readouts on the
same frozen features (which agree partly by construction), and full fine-tune (a
distinct input path, the load-bearing confirmation) — not "seven independent
instruments." Note the `y`-control saturation (the near-baseline regime is
uncalibrated). **Recommend softening `CONSOLIDATION_REVIEW.md` §0 accordingly.**

---

## L5 — C0 is causally isolated; do **not** seed it

**Claim under stress.** C0 is a baseline (open question: run more seeds?).

**Where it bends.** C0 shares **nothing** of the V-B/P2 path — SHARP-like net, 5
classes, P1, decision fusion — so it cannot cross-check the components every finding
depends on. Its dominant uncertainty is **systematic** (declared deviations: val 3
traces, 5-class, val-20 %, undocumented fusion), not seed; seeds sharpen the least
important term. Its 3-trace val makes any seed replicate's **selection** dominate
the spread, so C0 seed replicates would measure selection noise and could make the
reproduction look **falsely unstable**.

**Decision (per the seed-value principle).** **No C0 seeds.** No report claim needs
a C0 noise floor or a direction-across-init; C0 is the *reproduction anchor*, and
the comparison denominator is C1 ("C1 è il denominatore di tutti i confronti", §6).

**Report wording.** Frame C0 explicitly as a **"reproduction anchor, not a
comparison baseline; causally isolated from the V-B/P2 path"** (declared construct
limitation, this L5). If spare *parallel* compute exists **and** an anchor to the
real path is wanted, a **single V-B-under-P1 bridge run** is strictly more
informative than a C0 seed replicate (it shares one axis with C1–C4) — but neither
is recommended in the freeze runway.

---

## L7 — A variance claim from n=2

**Claim under stress.** *"The GRL destabilizes training run-to-run"* (5.45 pt vs
0.87 pt, n=2) — stated as a property of the method.

**Where it bends.** You cannot estimate a variance from 2 points, nor compare two
2-point spreads as inference — that is two anecdotes. n=2 supports **ordinal /
existence** claims only.

**What survives.** (i) The **directional** claim C1 > C2 (full separation across
seeds; min gap 3.69 > band) — valid. (ii) The **existence** statement "on C2's two
seeds we observed a 5.45-pt swing" — a fact. (iii) n=2 is the **replication
threshold**: the 1→2 jump is the big one (it *revealed* C2's fragility; n=1 would
have reported 0.8415 as stable). This is exactly Bouthillier (seed swing > config
gap) — `CONSOLIDATION_REVIEW.md` §3.A.

**Report wording.** Keep C1 > C2 as **directional**; state the GRL val cost as a
**range (≈3.7–10 pt)**, never the single −4.6; downgrade "GRL destabilizes" to *"on
the two seeds run, C2 swung 5.45 pt vs C1's 0.87 — reported as an observation, not a
variance estimate."* No s44 (the rule's own default branch; no claim uses the
magnitude). This ratifies the existing no-s44 decision on principle, not budget.

---

## L6 — C1-aug: the package contains the **wrong twin** (implementation)

**Finding.** The pre-registered 3-run aug package (CHANGELOG 2026-07-20) adds a
**seed** twin where the design calls for a **rotation** twin. The comparison is
**paired** (init + batch order identical between C1 and C1-aug, only the transform
differs — verified in `train.py`), so the pairing **already controls the init
noise** a second seed would otherwise buy. The single paired run gives a clean
effect estimate; the seed twin `C1_aug_s43` **re-evaluates on the same S7 test set**,
so it (a) cannot address the dominant uncertainty (test-sampling — same 11 traces)
and (b) only weakly checks "effect replicates across init," which is **redundant
with and weaker than** the cross-rotation twin `C1_s6out_aug` (a *different* test set,
S6). The seed twin evaluates the identical test set; the rotation twin does not.

| Run | init controlled? | test set | what it actually adds |
|---|---|---|---|
| **C1_aug** (vs C1) | yes (pairing) | S7 (11 tr) | **the effect estimate — this is the experiment** |
| **C1_aug_s43** (vs C1_s43) | yes (pairing) | **S7, same 11 tr** | only "replicates across init" — but on the same test → does not touch the dominant nuisance |
| **C1_s6out_aug** (vs C1_s6out) | yes (pairing) | **S6, 15 different tr** | "replicates on a different test" → the only replication that touches test-sampling |

**Recommended scope: reduce 3 → 2 runs.** Keep `C1_aug` (s42, P2-lab) +
`C1_s6out_aug` (s42, P2-living); **drop `C1_aug_s43`**. This is the arm's own
declared priority ("cross-rotation replication > 2nd seed") followed to its
conclusion. A single paired run per rotation is near-sufficient; the rotation twin
is the replication worth having.

**Conditionality.** Run the 2 **only if** GPU is idle *in parallel* with the
critical path (cross-review → notebook-05 rewrite → single test session → report,
freeze 2026-07-28). If the hours come *out of* the critical path, **drop the arm
entirely** (future work — `CONSOLIDATION_REVIEW.md` §4 G). The multiplicity
externality (each low-information row on 11 traces dilutes the family-wise
confidence of the C1/C3 rows that matter) argues for the drop even at free GPU.

**Process (this is an amendment to a team-approved pre-registration — needs a team
call, not a silent edit).** If ratified:
- **CHANGELOG:** add a dated 2026-07-20 amendment entry to the C1-aug pre-registration
  — drop `C1_aug_s43`, keep the two cross-rotation runs; rationale = paired design ⇒
  seed twin is the wrong twin (re-uses the S7 test set); §8.4 −2.3 h (extensions ≈
  13.7 h).
- **§0.7 frozen row list 16 → 15** (valid only while the session is **not** open; the
  deferred notebook-05 readiness assert is then written against 15).
- **`configs/c1_ce_aug_s43.yaml` + `notebooks/c1_aug/` s43 runner:** leave in the repo
  (byte-identical artifacts, no harm) but **not launched**; note in the folder README.
- **notebook-06 `PAIRS`:** keep the two paired deltas (`C1_aug − C1` on P2-lab;
  `C1_s6out_aug − C1_s6out` on P2-living, its own units — not poolable across
  rotations), drop the `C1_aug_s43 − C1_s43` pair (aligns with
  `CONSOLIDATION_REVIEW.md` §6 G1).

**If the team keeps s43 instead:** the report must **not** cite it as an independent
"effect replicates across seeds" check — it shares the S7 test set with s42; frame it
only as *the augmented run's own seed stability*, never as strengthening the effect
estimate.

**No `augment.py` / `train.py` / config code change is needed either way** — the
additive `ce_amp` profile is already in place. L6 is a run-scope decision + notebook-06
`PAIRS` + docs, nothing more. Physical-story caveat for the report (unchanged from the
deepened review): the amplitude lever is chosen by **label-safety elimination**, and a
global scalar does **not** faithfully model multipath room change (which is
velocity-selective) — declare it as "one label-safe knob," not "physically faithful."

---

## L8 — 5-class val selection / 8-class test: not critical, but a measured decomposition is owed (implementation)

**Finding.** Checkpoint selection uses a **5-class** val macro-F1 (p2_lab val is
blind to {C, E, S}); test is **8-class**. Because **every** config selects on the
same 5-class val with the same absent classes, the blindness is **common-mode** and
**cancels to first order in the comparisons** — so the headline C1/C2/C3 ordering is
**not critical** (it is also confirmed by readouts that do not depend on val
selection: domain probe, NCM, kNN, concat, C3-ft). It **is** a real confound for
(i) **absolute scale** (5-class val ≈0.88 vs 8-class test ≈0.7, not scale-comparable)
and (ii) **per-class test analysis** (a config's edge in a selection-blind class is
chosen blindly). The bigger root is the same tiny skewed val (9 traces, H = 1 trace
yet 1/5 of macro-F1) — also not pre-freeze-fixable.

**Not fixable via the split.** Bringing {C, E, S} into val requires editing the
frozen split (AR-3 absence + rare-cell pinning are deterministic: round(0.15·1)=0),
which violates §0.1 and **invalidates every baseline already run**. Locked in — by
the (correct) freeze rule.

**Implementation — report-time only, in notebook-06, post-hoc on already-logged
`*_test_*_windows.csv` (`y_true`/`y_pred` per window), NO new test contact, NO split
change, NO new §0.7 row:**
1. **Class-consistent (val-visible) test macro-F1**, per stream, **alongside** the
   8-class number — restricted to *that rotation's own* val-visible class set, so the
   selection metric and the reported metric cover the same label space:
   - **p2_lab (S7 test):** val-visible = {H, J, L, R, W}; selection-blind = {C, E, S}.
   - **p2_living (S6 test):** val-visible = {C, E, J, R, S}; selection-blind = {H, L, W}.
   The 8-class number is still reported, with the selection-blind classes flagged.
2. **Per-class test F1**, explicitly surfacing the selection-blind classes, to
   **empirically test the declared caveat** ("do the blind classes actually
   underperform on test?" — the exact question `CONSOLIDATION_REVIEW.md` §3.D poses).
   This is the per-class cut already named in Direction D / §6 G6.
3. **Placement:** a "class-coverage decomposition" cell in notebook 06 (which reads
   only `*_windows.csv` → no test access of its own, free to re-run), a natural
   extension of the master-table item (§6 G8). Restriction/`groupby` on logged
   predictions — no test invocation, so the frozen 16-row list and "test once" are
   untouched.
4. **Not available:** a test-side *selection-sensitivity* check via the C3 grid
   (epoch 40/50/60) — evaluating all three on test would be extra invocations outside
   the frozen row list (violates §0.7). The decomposition above is the contact-free
   mitigation.

**Verdict.** Not critical for the headline comparisons; the decomposition converts
the confound into a **measured, declared** result. Added to the notebook-06 do-list
(`CONSOLIDATION_REVIEW.md` §6 G12).

---

## Summary — what to do with each level

| Level | Kind | Action |
|---|---|---|
| **L0** | conceptual | Carry "structural" on the incidence proof, not the probe; C4 = triage + proof, not "GRL proven useless." |
| **L1** | conceptual | Downgrade "AR-set invariance" → "transfer to n=2 unseen hostile sessions"; state the confound bundle. |
| **L2** | conceptual | Report `n_traces` per row; paired trace-level bootstrap + multiplicity caveat; no bare 4-digit macro-F1. |
| **L4** | conceptual | "Two families of evidence," not "seven instruments"; soften `CONSOLIDATION_REVIEW.md` §0. |
| **L5** | conceptual + decision | **No C0 seeds**; frame C0 as reproduction anchor, causally isolated. |
| **L7** | conceptual | C1>C2 directional; GRL cost as a range; downgrade "GRL destabilizes"; no s44. |
| **L6** | implementation | **Reduce aug 3 → 2** (drop `C1_aug_s43`, keep S7 + S6 twins); conditional on parallel GPU; team-ratified CHANGELOG/§0.7 amendment; notebook-06 `PAIRS`. |
| **L8** | implementation | notebook-06 class-coverage decomposition (val-visible test macro-F1 + per-class blind-class F1); no rerun, no split change, no test row. |
