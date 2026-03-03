# Outputs — Repository Organization

This folder documents the organization of submission files and experiment outputs.
**All original files remain in the project root** — this README serves as an index.

## Submissions

| File (in project root) | Version | Nov AUC | Kaggle | Notes |
|------------------------|---------|---------|--------|------|
| `submission_v5.csv` | v5 Cowork | 0.63171 | **Best** | LightGBM+RF 70/30, 118 features |
| `submission_v6.csv` | v6 | 0.63408 | Worse | Higher Nov AUC, overfit to Nov |
| `submission_v8.csv` | v8 | 0.61476 | — | CatBoost+LGB ensemble |
| `submission_v8.1.csv` | v8.1 | — | — | Latest v8 variant |
| `submission_v4.csv` | v4 NN | — | 0.597 | NN pipeline milestone |
| `submission_v1.csv` | v1 | 0.5554 | — | XGBoost baseline |
| `submission.csv` | — | — | — | Generic/default output |
| `modal_outputs/submission_v5_gpu_turbo.csv` | v5 GPU | — | — | From v5_gpu_turbo.py |

## Experiment Outputs

| File (in project root) | Purpose |
|------------------------|---------|
| `v6_experiments_results.csv` | v6 config sweep (pruning, LOCK_NAME/CURRENCY, HP, ensemble) |
| `v6_permutation_importance.csv` | Feature importance from v6 |
| `v6_per_feature_auc.csv` | Per-feature temporal AUC |
| `v6_error_analysis.txt` | Error analysis |
| `v8_experiments_results.csv` | v8 config sweep results |
| `modal_outputs/*.csv` | Remote Modal run outputs |
