# figures/

Vector PDFs referenced by the report (`\graphicspath{{./figures/}}`).
Only the two data figures live here; the report's other two figures (the
processing pipeline of Section III and the learning framework of Section V)
are TikZ block diagrams drawn inline in `model.tex`.

| File | Source | Referenced in |
|---|---|---|
| `embeddings_c1_vs_c3.pdf` | convert existing `reports/embeddings_c1_vs_c3.png` to PDF (e.g. `img2pdf` or re-render) | results.tex, `\fig{fig:tsne}` |
| `confusion_c1_test.pdf` | notebook 06 figures cell (writes it to `reports/figures/`) | results.tex, `\fig{fig:confusion}` |

The per-class bar chart and the paired-bootstrap forest plot were cut for the
6-page budget; their numbers are reported in the text of Section VI.
