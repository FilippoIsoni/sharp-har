# notebooks/backbone_ablation — sharp_like vs ResNet-VB

One pinned runner for the backbone ablation (approved 2026-07-21): the
SHARP-paper architecture in the exact C1 recipe on p2_lab, to isolate the
backbone axis. Rationale + caveats live in `configs/c1_sharplike.yaml`.

The run itself is val-only — it produces `C1_sharplike/best.ckpt`, selected
on val, and touches no test data. **The checkpoint was then promoted to a
§0.7 test row** the same day, after the val run: `splits/CHANGELOG.md`
2026-07-21 (addendum) carries the pre-registration, as
pre-register-AND-commit-to-report with an outcome-independent interpretive
key fixed before the session (admissible after seeing val because val ≠ test
and the ~15-pt val gap leaves no cherry-pick incentive; see also
`PROJECT.md` Part V). The row was the 17th at that moment; the frozen list
closed at **16** once L6 dropped `C1_aug_s43` on 2026-07-22.

Measured outcome: val macro-F1 0.7384 (≈15 pts below C1), test 0.5694 /
0.5434 — one of the two comparisons the paired bootstrap resolves. It reads
as "deep beats near-linear-wide **at equal recipe**", never as "V-B is the
right design" (the axis V-B was actually chosen on is throughput, untested
here): the binding wording is `PROJECT.md` Part IV §3.

Executed copy -> `notebooks/runs/2026-07-21_c1_sharplike.ipynb`. This
template stays output-free.
