# PayJoy FPD_15 Prediction Challenge

Stanford OIT367 — Predicting First Payment Default (FPD) within 15 days for PayJoy phone financing. Kaggle competition: [predicting-fraud-in-phone-financing](https://www.kaggle.com/competitions/predicting-fraud-in-phone-financing).

**Metric:** AUC-ROC on November held-out data.

---

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the following files in the project root (obtain from course materials):

- `Orders.csv`
- `Payment_History.csv`
- `Test_OrderIDs.csv` (included in repo)

**Note:** `Orders.csv` and `Payment_History.csv` are large (~710 MB total) and confidential; do not share except with OIT367 students and course staff.

## How to Run

| Script / Notebook | Purpose |
|-------------------|---------|
| `payjoy_model_v8.py` | v8 pipeline → `submission_v8.csv` (CatBoost+LGB ensemble, 92 features) |
| `v5_gpu_turbo.py` | v5 GPU variant → `submission_v5_gpu_turbo.csv` (CatBoost+LGB+XGB, expanding-window rates) |
| `_run_v8_experiments.py` | v8 config sweep — single models, ensembles, feature ablations |
| `modal_v8.py` | v8 on Modal (GPU T4) — run experiments or production remotely, download logs + CSVs |
| `best_model_pipeline.ipynb` | v4 NN pipeline — generates `submission.csv` |
| `model_experiments_run.py` | Experimentation — XGBoost, LightGBM, NN, ensemble (v1–v3 style) |
| `ensemble_submissions.py` | Blend v8 + v5_gpu_turbo predictions → `submission_ensemble.csv` |

**Note:** The v5 Cowork script (`_run_v5_cowork.py`) and v6 script (`_run_v6_cowork.py`) referenced in RESULTS.md are not in this repo. `v5_gpu_turbo.py` is the closest v5 variant available.

## Current Best

**v5 Cowork** — LightGBM + RF ensemble (70/30), 118 features, expanding-window entity rates, payment history features. Best Kaggle performance.

- **November AUC:** 0.63171
- **Submission:** `submission_v5.csv` (produced by cowork pipeline; use for Kaggle)
- **Closest reproducible:** `v5_gpu_turbo.py` → `submission_v5_gpu_turbo.csv`

**Note:** v6 had higher Nov AUC (0.63408) but underperformed v5 on Kaggle; see EXPERIMENTS.md and V7_PROPOSAL.md.

## Repository Layout

- **Submissions:** `submission_v5.csv` (best), `submission_v6.csv`, `submission_v8.csv`, etc. — see [outputs/README.md](outputs/README.md)
- **Report summary:** [REPORT_SUMMARY.md](REPORT_SUMMARY.md) — structured summary for the final report

See `RESULTS.md` and `EXPERIMENTS.md` for details. Legacy experiment scripts are in `archive/`.

---

## Sharing with Teammates (GitHub)

1. Create a **private** repository on GitHub (e.g. `oit367-payjoy-prediction`).
2. Add teammates as collaborators (Settings → Collaborators).
3. Push this repo:

```bash
git remote add origin https://github.com/YOUR_USERNAME/oit367-payjoy-prediction.git
git branch -M main
git push -u origin main
```

Teammates clone the repo and place `Orders.csv` and `Payment_History.csv` in the project folder (from course download).
