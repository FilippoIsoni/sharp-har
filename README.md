# sharp-har

Human Activity Recognition from WiFi CSI (micro-Doppler) on the **SHARP** TMC
dataset (*Environment and Person-Independent Activity Recognition with Commodity
IEEE 802.11 Access Points*), evaluated under a **Leave-One-Environment-Out**
protocol. Five configurations — a SHARP reproduction (C0), a cross-entropy
baseline (C1), a domain-adversarial GRL variant (C2), supervised contrastive
pre-training (C3) and SupCon+GRL (C4) — share one backbone, one data pipeline,
one evaluation harness and one seed. C4 was closed on evidence without running.
The study is a **controlled comparison with an audit trail**, not a new
architecture: the test set was touched **once**, on 2026-07-22, with a row list
frozen beforehand, and every number below traces to a committed artifact.

**Status — closed.** Delivery is two artifacts:
[`report/Isoni_Schiabel_Dedej_report.pdf`](report/Isoni_Schiabel_Dedej_report.pdf)
(6-page IEEE paper) and `Isoni_Schiabel_Dedej_code.zip` (this tree, minus the
paper build and the archive). Nothing is open; no run, test contact or split
regeneration will happen.

## → `PROJECT.md` is the single source of truth

[`PROJECT.md`](PROJECT.md) holds the specification (§0–§9 — the numbering the
package's `Ref. §X.Y` docstrings cite), the execution record, **all** measured
results, the findings with their declared limitations, the decision log and the
repository map. This README orients; where the two ever disagree, `PROJECT.md`
wins. Superseded documents are archived under [`docs/archive/`](docs/archive/)
and are not to be read. `CLAUDE.md` holds working conventions for code
assistants.

## Results

Test rows, each with its effective n (`n_tr` = **traces**, the independent unit).
Extract of the 16-row table in `PROJECT.md` Part III / [`reports/tables/summary.csv`](reports/tables/summary.csv):

| Stream | What it is | Split | n_tr | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| **C1** | CE baseline — **the deliverable model** | P2-lab, 8 cls | 11 | **0.8047** | **0.8038** |
| C3 | SupCon pre-training + linear probe | P2-lab, 8 cls | 11 | 0.7271 | 0.7286 |
| C2 | CE + domain-adversarial GRL | P2-lab, 8 cls | 11 | 0.6471 | 0.6006 |
| C1_sharplike | the paper's shallow net in C1's exact recipe | P2-lab, 8 cls | 11 | 0.5694 | 0.5434 |
| C1_s6out | C1 on the second rotation | P2-living, 8 cls | 15 | 0.7430 | 0.6842 |
| C0 | SHARP reproduction anchor | P1, 5 cls | 57 | 0.6117 | 0.6053 |

The three blocks are **not** mutually comparable (different class sets, splits
and μ/σ). Differences are read with a **trace-level paired bootstrap**
(N = 10⁴); at n = 11 the table realistically resolves 2–3 tiers, not 16 rows.
Two comparisons resolve — C1 − C3 = **+0.078** [+0.030, +0.128] and
C1 − C1_sharplike = **+0.235** [+0.092, +0.399]; C1 − C2 = +0.158 does **not**
([−0.034, +0.323]). Full set: [`reports/tables/paired_bootstrap.csv`](reports/tables/paired_bootstrap.csv).

**What the study found** (stated as `PROJECT.md` Part IV permits):

- **The plain CE baseline wins** on the held-out session and no variant resolves
  an improvement over it; its seed-43 replicate lands 0.5 points away.
- **SupCon does not help, and the readout is not the reason** — C3 is below C1
  under linear, NCM, kNN and concatenation readouts, and a full fine-tune of its
  encoder (a genuinely different input path) does not recover.
- **The adversary had no target.** No domain attribute is *linearly* decodable
  from train features on any of five encoders and two rotations, and C2's own
  adversary head never left its majority floor. This is structural: the dataset's
  environment incidence (bedroom = 5 capture sets, living room = 1, laboratory =
  1) means **no** LOEO rotation of it poses environment-invariance
  non-degenerately — which is why C4 was closed without running.
- **Depth beats near-linear-wide at equal recipe.** Holding split, classes, loss,
  optimizer and seed fixed, the shallow reference network loses 23.5 accuracy
  points. This says nothing about the axis the backbone was actually chosen on
  (throughput).
- **Test-time adaptation is neutral at best** (AdaBN negative, T3A
  indistinguishable), consistent with the finding above.

**Honest framing.** "Generalization to two unseen, hostile capture sessions",
never "domain generalization": inside a capture set, room, monitor, subject, day
and LOS/NLOS are confounded. Selection ran on **in-domain** validation whose
macro-F1 is a 5-class number against an 8-class test. Full declaration list:
`PROJECT.md` Part IV §2.

## Protocol in brief

- **Data** — SHARP TMC: capture sets S1–S7 ≡ AR-1…AR-7, 12 campaigns, identical
  hardware; 101 traces × 4 antennas; 8 activity classes. Doppler windows of
  340 time steps × 100 velocity bins (train stride 100, val/test stride 340, so
  evaluation windows are disjoint). Never re-downloaded: the frozen inventory in
  [`reports/inventory.csv`](reports/inventory.csv) is the reference.
- **Splits** — leave-S7-out (laboratory) as the primary rotation, leave-S6-out
  (living room) as the replication, plus P1 for the C0 reproduction. Frozen as
  JSON in [`splits/`](splits/) with their own μ/σ, with the pre-registration
  trail in [`splits/CHANGELOG.md`](splits/CHANGELOG.md).
- **Models** — a width-halved ResNet-18 variant (V-B, `d_enc = 256`, chosen on a
  measured throughput gate) shared by C1–C4; the SHARP-like net for C0 and the
  backbone ablation.
- **Evaluation** — prediction = argmax of the antenna-averaged softmax on each
  disjoint window; accuracy, macro-F1 and ECE per set; comparisons by paired
  trace-level bootstrap. One harness (`checkpoint → CSV`) for every stream,
  logging every test access.

## Repository

```
sharp_har/     the versioned package — ALL logic lives here
configs/       one YAML per run: backbone, loss, adversary, sampler, horizon
notebooks/     thin runners (templates output-free; executed copies in runs/)
splits/        frozen split JSONs + pre-registration changelog
reports/       measured artifacts: day-1 inventory, throughput gates,
               final/ (the single test session), tables/ (the 8 report tables),
               figures/ (notebook-06 vector figures)
report/        the IEEE paper: LaTeX sources, IEEEtran, the delivered PDF
docs/archive/  superseded documents, kept for the record only
```

**Following a number to its evidence.** A claim in the paper cites a table in
[`reports/tables/`](reports/tables/) (each file indexed by that folder's README);
each table is computed by `notebooks/06_final_analysis.ipynb` from the per-window
CSVs in [`reports/final/`](reports/final/), which are the raw output of the
single test session — whose 21 logged accesses sit in the same folder as
`test_invocations.jsonl`. Every run archived under `notebooks/runs/` is committed
verbatim with its outputs.

## Running the code

```bash
pip install -r requirements.txt
```

Local work is code-only; training and evaluation ran on Google Colab through the
notebooks (data ~762 MB and checkpoints live on shared Drive and never enter the
repository). There is no test suite or linter: verification happens through the
**blocking asserts** built into the pipeline — split disjointness, axes, AR-set
coverage, NaN policy, rare-cell coverage, sampler invariants, cache alignment,
artifact readiness — and through the committed gate reports. Reproducing a
published number needs the frozen split JSON, the config YAML, the git hash in
that run's `run_meta.json` and its checkpoint; the test session itself is *not*
re-runnable by design, but everything downstream of `reports/final/` is.

**Three rules that are never bent** (full list: `PROJECT.md` §0): splits are
frozen on Git and never edited; the split is by **trace**, never by window; the
test set is evaluated **once**, at the end, through the logging harness.
