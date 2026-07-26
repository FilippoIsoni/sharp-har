# reports/tables/

The eight measured tables behind the report, written by
`notebooks/06_final_analysis.ipynb` from the per-window CSVs of the single
test session (`reports/final/`). Nothing here is hand-edited: re-running the
notebook on the same CSVs reproduces every file.

| File | Contents |
|---|---|
| `summary.csv` | main test table — accuracy, macro-F1 and ECE per configuration |
| `paired_bootstrap.csv` | trace-level paired bootstrap, per comparison, with 95% CI and `resolved` flag |
| `class_coverage.csv` | macro-F1 decomposed over classes seen / not seen in validation |
| `per_trace_accuracy.csv` | accuracy of each test trace under each configuration |
| `worst_trace_c1_vs_c2.csv` | the traces where C1 and C2 diverge most |
| `ece.csv` | calibration error per configuration |
| `tta_effect.csv` | test-time adaptation arms (AdaBN, T3A) against C1 |
| `augmentation_effect.csv` | amplitude-augmentation arm against C1, both rotations |

`resolved` is the pre-registered decision rule: a difference counts only when
its confidence interval excludes zero. It is `False` on most rows, which is
the report's main quantitative caveat.
