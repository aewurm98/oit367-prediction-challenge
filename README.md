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
| `_run_v6_cowork.py` | **Current best** — v6 (+lock_currency) → `submission_v6.csv` |
| `_run_v5_cowork.py` | v5 pipeline — generates `submission_v5.csv` |
| `_run_v6_diagnostics.py` | v6 diagnostics — per-feature temporal AUC, permutation importance |
| `_run_v6_experiments.py` | v6 config sweep — pruning, LOCK_NAME/CURRENCY, HP, ensemble |
| `best_model_pipeline.ipynb` | v4 NN pipeline — generates `submission.csv` |
| `model_experiments.ipynb` | Experimentation — model comparison, hyperparameter tuning |

## Current Best

**v6 Cowork** — +lock_currency config: LOCK_NAME + CURRENCY FPD rates, pruned base (no dormancy, PMT_CORE), 95 features.

- **November AUC:** 0.63408
- **Submission:** `submission_v6.csv`
- **Run:** `python _run_v6_cowork.py`

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
