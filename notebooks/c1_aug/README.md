# notebooks/c1_aug — the C1-aug arm session runners

Three pinned runners for the targeted-amplitude augmentation arm
(team-approved 2026-07-20): `c1_ce_aug` (S7-out, seed 42),
`c1_ce_aug_s43` (S7-out, seed 43 — seed-stability twin) and
`c1_ce_s6out_aug` (S6-out, seed 42 — cross-rotation replication).
The lever, its rationale and the a-priori hypothesis live in
`configs/c1_ce_aug.yaml`'s header and `splits/CHANGELOG.md`
(2026-07-20 entry) — this folder only holds ready-to-run copies of the
`03_train` template (RUN pinned, GPU sanity cell, ce_amp launch-check
cell), same pattern as `c3_ft/` and `e1_seed_replicates/`.

Baselines are the EXISTING runs (`C1`, `C1_s43`, `C1_s6out`) — nothing
is retrained; each aug run writes its own Drive folder (`C1_aug`,
`C1_aug_s43`, `C1_s6out_aug`), never a baseline's.

Executed copies -> `notebooks/runs/YYYY-MM-DD_<config>.ipynb` + STATUS
line, same commit. These templates stay output-free.
