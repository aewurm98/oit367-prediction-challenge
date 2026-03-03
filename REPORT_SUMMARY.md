# PayJoy FPD_15 Prediction Challenge — Report Summary

**Stanford OIT367 — Business Intelligence with Big Data**  
This document summarizes all work for the final report. Use it to draft the 2-page report and technical appendix.

---

## 1. Executive Summary

**Task:** Predict First Payment Default (FPD) within 15 days for December 2025 phone-financing orders.  
**Metric:** AUC-ROC (competition grading: full 50 pts at AUC ≥ 0.61, 0 pts at AUC ≤ 0.50287).

**Best model:** **v5 Cowork** — LightGBM + Random Forest (70/30 blend), 118 features, expanding-window entity FPD rates, Bayesian smoothing k=10. November held-out AUC 0.63171; best Kaggle performance.

**Key insight:** Temporal validation (train Jan–Oct, validate Nov) is essential. Random 5-fold CV inflated AUC to 0.76 while true temporal generalization was ~0.56; the gap revealed that entity FPD rates computed on the full training set leaked validation labels.

---

## 2. Methods Tried (Chronological)

| Version | Description | Nov AUC | Kaggle | Key change |
|---------|-------------|---------|--------|------------|
| v1 | XGBoost, 20 features, temporal val | 0.5554 | — | Baseline |
| v2 | +LOCK_NAME, CURRENCY, state_fpd_rate, entity counts | 0.5556 | 0.548 | Feature expansion |
| EXP-003 | Feature pruning (8 temporal-noise features) | 0.5533 | — | Worse; kept v2 |
| EXP-004 | HP tuning via expanding-window CV | 0.5533 | — | Ceiling ~0.556 |
| v3 (EXP-009) | +currency norm, payment history, KYC interactions, geo cross | 0.5574 | — | +0.0018 |
| v4 | NN + high-cardinality encodings | — | 0.597 | Overfit; dropped |
| **v5** | **LightGBM+RF, 118 features, expanding-window rates, k=10** | **0.63171** | **Best** | **Full rebuild** |
| v6 | v5 + LOCK_NAME/CURRENCY FPD rates, pruned | 0.63408 | Worse | Overfit to Nov |
| v8 | CatBoost+LGB ensemble, 92 features | 0.61476 | — | Alternative pipeline |

**Scripts producing submissions:**
- **v5:** `_run_v5_cowork.py` (referenced in docs; not in repo). Closest available: `v5_gpu_turbo.py` → `submission_v5_gpu_turbo.csv`
- **v6:** `_run_v6_cowork.py` (referenced; not in repo)
- **v8:** `payjoy_model_v8.py` → `submission_v8.csv`

---

## 3. Feature Selection Logic

### (a) Features to Include

| Category | Features | Rationale |
|----------|----------|-----------|
| **Financial** | FINANCE_AMOUNT, PURCHASE_AMOUNT, TOTAL_DUE, DOWN_PAYMENT_AMOUNT, down_payment_ratio, finance_ratio, total_due_ratio, log transforms | Core loan terms; ratios normalize across amounts |
| **Currency normalization** | Country-level z-scores for financial amounts | Raw amounts span MXN, BRL, PEN, etc.; z-scores make comparable across markets |
| **Temporal** | txn_hour, txn_dow, txn_month, txn_day, is_weekend, is_night, is_month_end, days_to_first_payment | Time-of-sale and payment timing signals |
| **Merchant** | merchant_tenure_days, log_merchant_tenure, is_new_merchant, days_since_last_sale | Channel maturity and activity |
| **KYC** | FACE/ID/LIVENESS/OVERALL scores, kyc_score_mean/min/std/range, low_face_score, any_low_score, all_scores_perfect, high_finance_low_kyc | Identity verification; deficit transforms (100 − score) for fraud signal |
| **Entity FPD rates** | MERCHANTID, CLERK_ID, ADMINID, STATE, CITY, MODEL (v5); LOCK_NAME, CURRENCY (v6, overfit) | Historical default rates; **expanding window** (month M uses only months 1..M−1) to avoid leakage |
| **Payment history** | Entity-level overdue aggregates, pmt_fpd_proxy_rate, zero_recovery_rate | Portfolio behavior of merchants/clerks; available at prediction time |
| **Geographic** | state_mismatch (user vs merchant state) | Out-of-state transactions may carry different risk |
| **Market payment stats** | COUNTRY/MANUFACTURER/MERCHANT_STATE pay mean/std | Market-level payment track record |

### (b) Features to Exclude

| Excluded | Reason |
|----------|--------|
| Raw MERCHANTID/CLERK_ID/CITY as one-hot | High cardinality (thousands); overfit in random CV; v4 collapsed on Kaggle |
| tx_month (early experiments) | Near-random temporal AUC (0.50); later kept for context |
| LOCK_NAME, LOCK_PRODUCT (early) | Extreme within-month FPD variation but near-random across time boundary |
| Full PMT_FEATS_RICH (v6) | PMT_CORE sufficient; full set slightly worse (step K vs J in v5 ablation) |
| Dormancy (v6) | Zero delta in v5 feature ablation |

---

## 4. Model Architecture

| Model | Use | Hyperparameters (representative) |
|-------|-----|----------------------------------|
| **LightGBM** | Primary (v5, v8) | num_leaves=63, min_child_samples=50, lr=0.03, feature_fraction=0.8, bagging_fraction=0.8, reg_alpha/lambda=0.1 |
| **Random Forest** | Ensemble (v5) | 70/30 LGB/RF blend; RF adds diversity |
| **CatBoost** | v8 ensemble | depth=6, l2_leaf_reg=3.0, lr=0.05 |
| **XGBoost** | Early (v1–v3) | max_depth=4, lr=0.1, n_estimators=500, scale_pos_weight=5 |

**Ensemble strategy:** v5 uses 70/30 LGB/RF; v8 uses CatBoost+LGB with scipy-optimized weights on November validation.

---

## 5. Hyperparameters

- **Tuning method:** Expanding-window temporal CV (not random); config selection by November holdout AUC
- **Key choices:** Bayesian smoothing k=10 (v5; k=50 over-smoothed); scale_pos_weight ≈ n_neg/n_pos (~9–10 for ~9.6% FPD rate)
- **v6 HP grid:** num_leaves ∈ {31, 63, 127}, min_child_samples ∈ {30, 50, 100}; best near baseline

---

## 6. Training Process and Optimizations

1. **Temporal validation:** Train Jan–Oct, validate Nov (matches competition: predict Dec from Jan–Nov)
2. **Expanding-window entity rates:** For month M, compute FPD rates using only orders from months 1..M−1; prevents target leakage
3. **Bayesian smoothing:** (sum + k×prior) / (n + k) with k=10; shrinks rare-entity estimates toward global mean
4. **Imputation:** Median (numeric) and mode (categorical) from training set only
5. **Class imbalance:** scale_pos_weight; no resampling (AUC is threshold-invariant)
6. **Early stopping:** Used in LightGBM/CatBoost to prevent overfitting

---

## 7. Data Cleanup and Preprocessing

- **Missing values:** Median imputation for numeric; mode for categorical (fit on training only)
- **Categorical encoding:** One-hot for low-cardinality (COUNTRY, LOCK_PRODUCT, MANUFACTURER); target encoding with smoothing for high-cardinality (STATE, CITY, MODEL, entity IDs)
- **Payment history:** December test rows have no payment history; entity-level aggregates from Payment_History used (merchant/clerk/admin overdue stats)
- **Temporal filtering:** 2025 data only; test set = December orders per Test_OrderIDs.csv

---

## 8. Lessons Learned

- **Nov AUC ≠ Kaggle AUC:** v6 had higher Nov AUC (0.634) but worse Kaggle; avoid overfitting to November
- **Temporal shift:** CV AUC (0.76) vs Nov AUC (0.56) gap showed random CV is misleading
- **Expanding-window rates:** Critical fix for entity FPD rates; full Jan–Nov rates leaked validation labels
- **v7 proposal:** 4-fold forward validation (Aug–Nov) to reduce Nov-centric selection; recency-weighted entity rates

---

## 9. Reproducibility

- **Best submission:** `submission_v5.csv` (best Kaggle)
- **Scripts:** `payjoy_model_v8.py` (v8), `v5_gpu_turbo.py` (v5 GPU variant). The exact v5 Cowork script (`_run_v5_cowork.py`) is referenced in RESULTS.md but not present in repo.
- **Data:** Orders.csv, Payment_History.csv, Test_OrderIDs.csv
- **Dependencies:** requirements.txt

---

## 10. References

- [RESULTS.md](RESULTS.md) — Version-by-version results
- [EXPERIMENTS.md](EXPERIMENTS.md) — Experiment log and scorecard
- [DECISION_LOG.md](DECISION_LOG.md) — Analytical decisions
- [V7_PROPOSAL.md](V7_PROPOSAL.md) — Lessons from v6 overfitting
