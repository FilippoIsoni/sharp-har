# docs/archive — superseded documents

These five documents were the project's working documentation until
2026-07-25. They are **superseded by `PROJECT.md`** at the repository root,
which is now the single source of truth, and are kept here **only as the
historical record**: nothing should be read from them for current information,
and none of them is updated any more. Each carries a supersession banner
saying where its content went.

| File | What it was | Where its content lives now |
|---|---|---|
| `pipeline_wifi_har_v5.md` | The experimental specification (v5 → v5.1 errata → v5.2 tail), in Italian | `PROJECT.md` **Part I**, which preserves the §0–§9 numbering so the ~380 `Ref. §X.Y` docstring citations in `sharp_har/` still resolve |
| `STATUS.md` | The running "where we are" log | `PROJECT.md` **Parts II, III, V** (runs and gates, measured results, decisions). The session chronicle was deliberately dropped — it is in git history and in `notebooks/runs/` |
| `CONSOLIDATION_REVIEW.md` | Literature-grounded consolidation plan + the post-test analysis audit (G1–G12) | `PROJECT.md` **§9** (statistical protocol), **Part IV** (framing of the two nulls, incidence proof), **Part VI** (sources). The gap list closed 2026-07-23 |
| `CONCEPTUAL_STRESS_TEST.md` | Adversarial read of the load-bearing claims (levels L0–L8) | `PROJECT.md` **Part IV §3**, the binding "allowed wording" table, plus the seed-value principle; L6/L8 as entries in **Part V** and results in **Part III** |
| `NOTEBOOK_06_REVIEW.md` | Deep review of the analysis notebook, run against the real test session | `PROJECT.md` **Part III** (its measured-numbers appendix in full) and the two declared caveats |

The supersession map is also in `PROJECT.md`, Appendix A.
