# notebooks/e2_living_out — E2′: living-out rotation (C1 only)

Pre-configured runners for the E2′ extension (v5.2 §10.3): the full
leave-one-domain-out rotation with **test = S6 (living room), train =
S1–S5 + S7**, run for **C1 only**, seed 42. Pre-declared artifact names
(§10.3): `splits/p2_living.json`, `configs/c1_ce_s6out.yaml`, Drive run
folder `C1_s6out`.

**What E2′ measures:**

- C1 on a second, different cross-domain regime: S6 is an unseen room
  with a **seen person** (P1) — complementary to the primary rotation's
  S7 (environment, day AND person all unseen). One pre-registered test
  row in the frozen §0.7 list.
- The input for **replicating the §9 domain diagnostics** on a train set
  whose non-bedroom environment is the laboratory instead of the living
  room (structural caveat carries over symmetrically: the second
  environment is still a single AR-set).

## Order of operations

1. **`01_s6out_split.ipynb`** — one-shot split session: day-1 inventory
   rebuild + gates (reports to a session scratch dir, frozen day-1
   reports untouched), contingency inspection, then
   `build_p2_rotation(test_ar_set="AR-6", reference=p2_lab.json)` —
   the reference check aborts without writing if the trace universe or
   the class/axes/window metadata diverge from the frozen primary
   rotation, or if μ/σ were not computed fresh (§0.3). Commit
   `splits/p2_living.json` (frozen, §0.1) + the archived executed copy
   + the STATUS line, push.
2. **`03_train_c1_ce_s6out.ipynb`** — C1 S6-out (~2.3 h, seed 42);
   its readiness cell asserts the frozen split is present in the clone.
   Archive the executed copy per the usual convention.
3. Domain-diagnostic replication on the run's **train** features
   (separate diagnostics session, frozen §5.3 recipe, no val selection,
   no test contact).

## Declared expectations and caveats (local dry-run, 2026-07-18)

The id assignment depends only on the trace ids and seed 42 (never on
data values), so a local dry-run on the frozen universe predicts the
real partition exactly; only μ/σ are new in the Colab session. Expected:
**train=79, val=7, test=15, pinned=41**, with:

- **val = 7 traces over AR-1/2/4/5 only** (AR-3 absent as in p2_lab;
  AR-6 is the test; AR-7's single campaign is fully absorbed into
  train) — selection is noisy by construction, declared;
- **val classes = {E, H, J, L, R, S}: C and W absent** → this
  rotation's val macro-F1 is a **6-class** number, not scale-comparable
  to p2_lab val (5-class) nor to the 8-class test macro-F1;
- **dual-archive twin pair split across sides: S4a_L in train,
  S4a_Lalt in val** — two distinct recordings of the same session, so a
  declared selection-side quasi-leakage; the S6 test set is untouched
  (in p2_lab all four twins sat in train — this rotation is the first
  where the pair separates);
- test = all 15 AR-6 traces (S6a ×9, S6b ×6), single domain.

## Scope guards

C1 only (no C2/C3 replication on this rotation, §10.3); seed 42 only;
one pre-registered test row (`C1 S6-out` in the frozen §0.7 list); the
final test session needs an Editor shortcut to the `C1_s6out` Drive
folder from the session account — verify before the session day.

Executed copies go to `notebooks/runs/` (`YYYY-MM-DD_s6out_split.ipynb`,
`YYYY-MM-DD_c1_ce_s6out.ipynb`). These templates stay output-free.
