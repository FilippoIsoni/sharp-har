# reports/figures/

Vector figures produced by `notebooks/06_final_analysis.ipynb` from the
per-window CSVs of the single test session (`reports/final/`). The notebook
writes them here; the two that are typeset in the report were copied into the
paper's build tree.

| File | In the report |
|---|---|
| `embeddings_c1_vs_c3.pdf` | yes — t-SNE geometry, C1 vs C3 |
| `confusion_c1_test.pdf` | yes — confusion matrix of the deliverable model |
| `confusion_*.pdf`, `perclass_*.pdf` (the other configurations) | no |
| `forest_paired.pdf`, `reliability_test.pdf` | no |

The figures not typeset were cut for the 6-page budget; their numbers are
reported in the text instead. The report's two remaining figures (the
processing pipeline and the learning framework) are TikZ block diagrams drawn
inline in the paper source, so they have no file here.
