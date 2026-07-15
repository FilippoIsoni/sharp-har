# STATUS

> Single synthetic source for "where we are" in the pipeline. Update it
> **in the same commit** as the work that changes it (one line moved per
> milestone, no essays). Timeline days refer to `pipeline_wifi_har_v5.md` §10.

**Last update:** 2026-07-15 · **Phase: day 0 → day 1** (foundations, all together)

## Done

- Repo scaffold: package `sharp_har/`, configs C0–C4, thin notebooks, split/report dirs.
- Day-1 code implemented (not yet run on real data): `inventory.py`, `windowing.py` (μ/σ), `splits.py`, notebook `01`.
- **v5.1 dataset errata** applied everywhere: real coverage = S1–S7 ≡ AR-1…AR-7 (12 campaigns, TMC dataset); primary rotation = **leave-S7-out (lab)** → `splits/p2_lab.json`; P1 = train S1(a/b/c), test S2–S7; C0 = 5 paper classes; Tc ≈ 6 ms confirmed from the paper.
- Local dry-run of the full day-1 flow on synthetic S1a…S7a data: caught and fixed an empty-val bug in P2 stratification (with 1–3 traces per cell every cell is "rare"; now rare cells degrade to AR-set-level stratification per §2.2, plus a blocking assert on empty val); P2 split file now records `c0_paper_set` as in §2.3.

## In progress

- (nothing running)

## Next steps (in order)

1. **Day 1 on Colab (blocker, all together):** run notebook `01` on the Drive data — staging timed, inspect real file names inside `S1a…S7a` dirs → confirm/fix `FILENAME_PATTERN`, coverage + axes + NaN gates, contingency table, freeze `splits/p2_lab.json` + `splits/p1_sharp.json`, commit artifacts.
2. **Day 2 (gate):** implement `data.py`, `resnet_vb.py`, CE loss, `train.py` skeleton (checkpoint/resume); end-to-end smoke test + throughput gate (s/step, staging time) → written go/no-go, recalibrate §8.4 budget.
3. **Day 3:** `harness.py` (test-invocation logging), `sampler.py` P×K, `augment.py`, `probe.py`, feature caching; phase-A full-batch memory test (512 views).
4. **Days 4–9 (vertical ownership, §10.2):** A → C0 + C1 · B → C3, then C4 · C → C2, then C4 + probes + C1-lin/C2-lin.

## Blockers / open decisions

- File naming inside the set directories unverified (last unverified assumption; notebook 01 inspects before trusting the regex).
- Team to ratify: leave-S7-out as primary rotation (small test set, ~1 campaign; person P3 unseen — declared in §2.2).
