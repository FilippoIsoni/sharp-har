# sharp-har

Human Activity Recognition from WiFi CSI (micro-Doppler) on the **SHARP**
dataset (*Environment and Person-Independent Activity Recognition with
Commodity IEEE 802.11 Access Points*, IEEE TMC), evaluated under a **LOEO**
(Leave-One-Environment-Out) protocol. Five configurations — a SHARP
reproduction, a CE baseline, a domain-adversarial GRL variant, SupCon, and
SupCon+GRL — share one backbone, one data pipeline and one evaluation harness.

**Headline result:** the plain CE baseline wins on the held-out environment
(test macro-F1 0.804); SupCon is measurably worse under every readout; the GRL
costs accuracy because it has **no readable domain target** — a structural
consequence of the dataset's environment incidence, not of our training.

## → Read `PROJECT.md`

[`PROJECT.md`](PROJECT.md) is the **single source of truth**: specification
(§0–§9, the numbering the package's `Ref. §X.Y` docstrings cite), execution
record, all measured results, findings and declared limitations, the decision
log, and the repository map. Everything else that used to be documentation is
superseded and archived under [`docs/archive/`](docs/archive/).

`CLAUDE.md` holds working conventions for code assistants.

## Quick orientation

```
sharp_har/     the versioned package — ALL logic lives here
configs/       one YAML per run
notebooks/     thin runners (templates output-free; executed copies in runs/)
splits/        frozen split JSONs + pre-registration changelog
reports/       measured artifacts (inventory, gates, the single test session,
               the 8 report tables, the report figures)
report/        the IEEE paper build tree: LaTeX sources, IEEEtran, the copies
               of the tables and figures it typesets
docs/archive/  superseded documents, kept for the record only
```

```bash
pip install -r requirements.txt
```

Local work is code-only; training and evaluation run on Google Colab through
the notebooks. There is no test suite or linter — verification happens through
the blocking asserts built into the pipeline and the committed gate reports.

**Three rules that are never bent** (the full list is `PROJECT.md` §0): splits
are frozen on Git and never edited; the split is by *trace*, never by window;
the test set is evaluated **once**, at the end, through the logging harness.
Data (~762 MB, shared Drive) and checkpoints never enter the repository.
