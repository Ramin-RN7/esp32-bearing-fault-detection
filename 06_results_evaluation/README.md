# Results from the supplementary package

These are the summary tables and figures that were shipped with the thesis.

| File | Note |
|---|---|
| `variants.csv` | Copy of the experiment log. The original, with its exact column names, is [`records/logs/variants.csv`](../records/logs/variants.csv). |
| `ablation_summary.csv` | **Known errors in the configuration columns.** For example, A8, A16, A17 and A31 are marked "no PCA". Use [`results/ablation_runs.csv`](../results/ablation_runs.csv) instead. |
| `best_of_table.csv` | **Mixes evaluation levels.** The A42-RF and A43-ET values are window-level F1 scores, but they are listed under `file_F1`. |
| `cm_*.png`, `figs/` | Confusion matrices and summary figures used in the thesis. |
| `make_bestof_table.py`, `plot_*.py`, `audit_runs_integrity.py`, `make_report_bundle.py` | The scripts that produced these files. They expect the original workspace layout. |

All scores here are test-set scores from a single 9-recording split. For stricter evaluations, see [`results/protocol_comparison.md`](../results/protocol_comparison.md).
