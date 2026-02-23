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

| Notebook | Purpose |
|----------|---------|
| `best_model_pipeline.ipynb` | Production pipeline — generates `submission.csv` for Kaggle |
| `model_experiments.ipynb` | Experimentation — model comparison, hyperparameter tuning |

## Current Best

**v4 NN** — FPDNet [128,64,32], lr=0.0005, v4 features (smoothed encoding, CITY, MODEL, days_to_first_due).

- **November AUC:** 0.5661
- **Submission:** `submission_v4.csv`

See `RESULTS.md` and `EXPERIMENTS.md` for details.
