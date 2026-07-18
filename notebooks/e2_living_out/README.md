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
3. **`04_domain_probe_c1_s6out.ipynb`** — domain-diagnostic replication
   on the run's **train** features: caches `C1_s6out/best.ckpt` train
   features over the frozen `p2_living` split (50948 samples — no
   probe session exists on this rotation, so the cache is produced
   here), then runs `diagnostics.domain_probe` (the instrument
   promoted 2026-07-18, byte-identical to the three executed
   C1/C2/C3 sessions). Frozen §5.3 recipe, no val selection, no test
   contact. Executed copy →
   `notebooks/diagnostics/YYYY-MM-DD_c1_s6out_domain_probe.ipynb`
   (next to the other domain-probe sessions).

## Declared expectations and caveats (local dry-run, 2026-07-18)

The id assignment depends only on the trace ids and seed 42 (never on
data values), so a local dry-run on the frozen universe predicts the
real partition exactly; only μ/σ are new in the Colab session. The same
dry-run validated the methodology by reproducing the frozen p2_lab
partition exactly under the pre-amendment mechanics. Expected:
**train=80, val=6, test=15, pinned=41**, with:

- **twin-binding amendment applied** (`splits/CHANGELOG.md` 2026-07-18,
  pre-registered before the freeze): the dual-archive `*alt` twins —
  two recordings of the same physical session — are not independent
  split units and follow their base trace's side (§0 rule 2's rationale
  at the recording level). The first dry-run had drawn S4a_Lalt into
  val with S4a_L in train, a selection-side quasi-leakage; under the
  amendment **both twin pairs land in train with their bases**. p2_lab
  already satisfies the invariant by draw (all four twins in train), so
  the amendment introduces no cross-rotation inconsistency;
- **val = S1b_E, S1b_J2, S1c_S, S2a_R, S4a_C2, S4b_J1** (AR-1 ×3,
  AR-2 ×1, AR-4 ×2; AR-3/5/7 absent, AR-6 is the test) — selection is
  noisy by construction, declared;
- **val classes = {C, E, J, R, S}: H, L and W absent** → this
  rotation's val macro-F1 is a **5-class** number, not scale-comparable
  to p2_lab val (itself 5-class, different classes) nor to the 8-class
  test macro-F1. Accepted, not amended: §2.2 explicitly pre-accepts
  rare cells missing from val ("la macro-F1 di val è comunque definita
  sulle classi presenti"), and checkpoint selection is a within-run
  comparison where a k-class metric stays valid — forcing class
  coverage in val would change the §2.2 stratification itself, and the
  doc wins;
- test = all 15 AR-6 traces (S6a ×9, S6b ×6), single domain.

## Scope guards

C1 only (no C2/C3 replication on this rotation, §10.3); seed 42 only;
one pre-registered test row (`C1 S6-out` in the frozen §0.7 list); the
final test session needs an Editor shortcut to the `C1_s6out` Drive
folder from the session account — verify before the session day.

Executed copies go to `notebooks/runs/` (`YYYY-MM-DD_s6out_split.ipynb`,
`YYYY-MM-DD_c1_ce_s6out.ipynb`) — except the domain probe (`04`), whose
executed copy goes to `notebooks/diagnostics/` with the other three
domain-probe sessions (it is an investigation, not a protocol run).
These templates stay output-free.
