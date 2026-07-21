# notebooks/backbone_ablation — sharp_like vs ResNet-VB

One pinned runner for the **val-only** backbone ablation (proposed
2026-07-21): the SHARP-paper architecture in the exact C1 recipe on
p2_lab, to isolate the backbone axis. Rationale + caveats live in
`configs/c1_sharplike.yaml`.

Val-only: produces `C1_sharplike/best.ckpt`, no test contact. A §0.7
test row for this checkpoint is a separate, still-open team decision
(admissible only as pre-register-and-commit-to-report). Executed copy
-> `notebooks/runs/YYYY-MM-DD_c1_sharplike.ipynb` + STATUS line.
