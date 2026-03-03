# PayJoy FPD_15 Prediction Challenge — Report Summary

**Stanford OIT367 — Business Intelligence with Big Data**  
This document summarizes all work for the final report. Use it to draft the 2-page report and technical appendix.

---

## 1. Executive Summary

**Task:** Predict First Payment Default (FPD) within 15 days for December 2025 phone-financing orders.  
**Metric:** AUC-ROC (competition grading: full 50 pts at AUC ≥ 0.61, 0 pts at AUC ≤ 0.50287).

**Best Kaggle result:** **v5 Cowork** — LightGBM + Random Forest (70/30 blend), 118 features, expanding-window entity FPD rates, temporal validation. November held-out AUC 0.63171.

**Final submitted scripts:** **`payjoy_model_clean_v5.py`** and **`v5_model.py`** — CatBoost or LightGBM, ~50–60 features, low-cardinality only, 5-fold stratified CV with fold-level target encoding. Outputs: `improved_submission.csv`, `submission_v5_mismatch.csv`.

**Key insight:** Entity FPD rates (MERCHANTID, CLERK_ID, etc.) computed on the full training set leak validation labels when using random CV — inflating AUC to 0.76 while true temporal generalization was ~0.56. Two mitigation paths: (1) temporal validation + expanding-window rates (v5 Cowork), or (2) drop high-cardinality entity rates entirely and use only low-cardinality/continuous features (Clean v5 / v5_model).

---

## 2. Approaches Investigated (Chronological)

| Version | Description | Validation | Nov AUC | Kaggle | Key lesson |
|---------|-------------|------------|---------|--------|------------|
| v1 | XGBoost, 20 features | Temporal (Jan–Oct train, Nov val) | 0.5554 | — | Baseline |
| v2 | +LOCK_NAME, CURRENCY, state_fpd_rate, entity counts | Temporal | 0.5556 | 0.548 | CV AUC 0.81 vs Nov 0.56 — random CV misleading |
| EXP-003 | Feature pruning (8 temporal-noise features) | Temporal | 0.5533 | — | Worse; kept v2 |
| EXP-004 | HP tuning via expanding-window CV | Expanding-window | 0.5533 | — | Ceiling ~0.556 with current features |
| v3 (EXP-009) | +currency norm, payment history, KYC, geo cross | Temporal | 0.5574 | — | +0.0018; new info helps |
| **v4** | **NN + high-cardinality encodings (MERCHANTID, CLERK_ID, CITY)** | **Random 5-fold CV** | **—** | **0.597** | **Best Kaggle so far, but overfit; December has unseen entities** |
| **v5 Cowork** | **LightGBM+RF, 118 features, expanding-window rates, k=10** | **Temporal + expanding-window** | **0.63171** | **Best** | **Full rebuild; fixes entity leakage** |
| v6 | v5 + LOCK_NAME/CURRENCY FPD rates | Temporal | 0.63408 | Worse | Nov AUC ≠ Kaggle; overfit to November |
| v8 | CatBoost+LGB ensemble, 92 features | Temporal | 0.61476 | — | Alternative pipeline |
| **Clean v5** | **Low-cardinality only, CatBoost/LGB, no entity rates** | **5-fold stratified CV** | **—** | **Submitted** | **Mitigation: avoid leakage source** |
| **v5_model** | **Clean v5 + mismatch features (face_liveness_gap, finance z-score)** | **5-fold stratified CV** | **—** | **Submitted** | **Same mitigation; extra signal** |

---

## 3. Validation: Temporal vs. 5-Fold — Why Both Appear

### The Leakage Problem

Entity FPD rates (MERCHANTID, CLERK_ID, ADMINID, STATE, CITY, MODEL) are strong predictors. When computed from the **full** training set and used with **random 5-fold CV**, validation-fold labels leak into the feature: the entity rate for a merchant includes that merchant’s validation-row outcomes. Result: CV AUC ~0.76, but true temporal generalization ~0.56.

### Two Mitigation Paths

| Approach | Validation | Entity rates | Rationale |
|----------|------------|--------------|------------|
| **v5 Cowork** | Temporal (Jan–Oct train, Nov val) + expanding-window | Yes — month M uses only months 1..M−1 | Eliminates leakage by design; rates never use future or validation labels |
| **Clean v5 / v5_model** | 5-fold StratifiedKFold (shuffle=True) | **No** — explicitly dropped | Avoids leakage by **removing the source**. No entity rates → no entity-level leak. Fold-level target encoding for categoricals (COUNTRY, LOCK_NAME, etc.) uses only `tr_df_fold` per fold. |

### Why Clean v5 Uses 5-Fold Safely

1. **No high-cardinality entity rates** — MERCHANTID, CLERK_ID, CITY, ADMINID, MODEL are excluded. December test has many unseen entities; entity rates would require cold-start handling and are the main leakage source.
2. **Fold-level target encoding** — For low-cardinality categoricals (COUNTRY, LOCK_NAME, LOCK_PRODUCT, CURRENCY, etc.), `target_encode_fold()` computes encodings from `train_fold` only; validation and test never leak.
3. **Low cardinality** — LOCK_NAME (~10–20 levels), CURRENCY (~5), LOCK_PRODUCT, etc. generalize better to December than thousands of merchant/clerk IDs.

### Why the Final Approach Was Selected

- **v4 (0.597 Kaggle)** used high-cardinality encodings; OOF looked good but Kaggle suffered from December cold-start.
- **v5 Cowork** achieved best Kaggle via temporal validation and expanding-window rates but depends on `_run_v5_cowork.py` (not in repo) and a more complex pipeline.
- **Clean v5 / v5_model** were chosen as the **reproducible, in-repo submission scripts** because they:
  1. Avoid the leakage problem by design (no entity rates).
  2. Use only features that generalize to unseen December entities.
  3. Are self-contained, runnable, and do not require the Cowork pipeline.
  4. Build on v4’s 0.597 baseline by adding safe low-cardinality features (LOCK_NAME, LOCK_PRODUCT, CURRENCY, merchant tenure, KYC interactions) and, in v5_model, mismatch features (face_liveness_gap, finance_amount_country_zscore).

---

## 4. Final Submitted Models — Detailed Specs

### (a) payjoy_model_clean_v5.py → improved_submission.csv

| Aspect | Specification |
|--------|----------------|
| **Validation** | 5-fold StratifiedKFold, shuffle=True, random_state=42 |
| **Models** | CatBoost (preferred) or LightGBM; single model, no ensemble |
| **Features** | ~50–60: financial (amounts, ratios, logs), temporal (hour, dow, month, day, is_weekend, is_night, is_month_end/start, days_to_first_payment), merchant tenure (tenure_days, log_tenure, is_new_merchant, days_since_last_sale), KYC (scores, mean/min/std/range, low_face_score, any_low_score, all_scores_perfect, high_finance_low_kyc), market payment stats (COUNTRY, MANUFACTURER, MERCHANT_STATE mean/std), target-encoded categoricals (COUNTRY, USER_STATE, MERCHANT_STATE, MANUFACTURER, LOCK_NAME, LOCK_PRODUCT, CURRENCY) |
| **Excluded** | MERCHANTID, CLERK_ID, ADMINID, CITY, MODEL (high cardinality) |
| **Data prep** | 2025 only; median imputation; target encoding k=20 per fold |

### (b) v5_model.py → submission_v5_mismatch.csv

| Aspect | Specification |
|--------|----------------|
| **Validation** | Same 5-fold StratifiedKFold |
| **Models** | Same CatBoost or LightGBM |
| **Features** | All of Clean v5 **plus**: face_liveness_gap, face_liveness_gap_signed, face_liveness_gap_flag, finance_amount_country_zscore, finance_amount_country_high, new_merchant_high_value, user_merchant_state_mismatch |
| **Data prep** | Same; mismatch features computed from KYC scores and country-level finance stats |

---

## 5. Feature Selection Logic

### (a) Features to Include

| Category | Features | Rationale |
|----------|----------|-----------|
| **Financial** | FINANCE_AMOUNT, PURCHASE_AMOUNT, TOTAL_DUE, DOWN_PAYMENT_AMOUNT, down_payment_ratio, finance_ratio, total_due_ratio, log transforms | Core loan terms; ratios normalize across amounts |
| **Currency normalization** | Country-level z-scores (v5_model: finance_amount_country_zscore) | Raw amounts span MXN, BRL, PEN; z-scores make comparable |
| **Temporal** | txn_hour, txn_dow, txn_month, txn_day, is_weekend, is_night, is_month_end, days_to_first_payment | Time-of-sale and payment timing |
| **Merchant** | merchant_tenure_days, log_merchant_tenure, is_new_merchant, days_since_last_sale | Channel maturity and activity |
| **KYC** | FACE/ID/LIVENESS/OVERALL scores, kyc_score_mean/min/std/range, low_face_score, any_low_score, all_scores_perfect, high_finance_low_kyc | Identity verification |
| **Low-cardinality categoricals** | COUNTRY, LOCK_NAME, LOCK_PRODUCT, CURRENCY, USER_STATE, MERCHANT_STATE, MANUFACTURER (target-encoded) | ~5–20 levels; generalize to December |
| **Mismatch (v5_model)** | face_liveness_gap, finance_amount_country_zscore, new_merchant_high_value, user_merchant_state_mismatch | KYC divergence, loan-size anomaly, geo mismatch |
| **Market payment stats** | COUNTRY/MANUFACTURER/MERCHANT_STATE pay mean/std | Market-level payment track record |

### (b) Features to Exclude (and Why)

| Excluded | Reason |
|----------|--------|
| MERCHANTID, CLERK_ID, ADMINID, CITY, MODEL | High cardinality; overfit in random CV; v4 collapsed on Kaggle; December has unseen entities |
| Entity FPD rates (with random CV) | Leak validation labels when computed from full training set |
| LOCK_NAME/CURRENCY as FPD rates (v6) | Overfit to November; v6 had higher Nov AUC but worse Kaggle |

---

## 6. Model Architecture (Final Scripts)

| Model | Use | Hyperparameters |
|-------|-----|-----------------|
| **CatBoost** | Primary | iterations=1000, lr=0.05, depth=6, l2_leaf_reg=3, scale_pos_weight, early_stopping_rounds=50 |
| **LightGBM** | Fallback | num_leaves=63, min_child_samples=50, lr=0.05, feature_fraction=0.8, bagging_fraction=0.8, scale_pos_weight, early_stopping=50 |

No ensemble in the final scripts; single model per run.

---

## 7. Data Cleanup and Preprocessing

- **Missing values:** Median imputation (numeric), mode (categorical); fit on training only
- **Categorical encoding:** Target encoding with Bayesian smoothing (k=20) per fold; only low-cardinality columns
- **Temporal filtering:** 2025 data only; test set = December orders per Test_OrderIDs.csv
- **Payment history:** Market-level stats (COUNTRY, MANUFACTURER, MERCHANT_STATE) from Payment_History; no entity-level rates in Clean v5 / v5_model

---

## 8. Lessons Learned

- **Random CV + entity rates = leakage:** CV AUC 0.76 vs Nov AUC 0.56; entity rates computed on full set leak validation labels
- **Nov AUC ≠ Kaggle AUC:** v6 had higher Nov AUC (0.634) but worse Kaggle; avoid overfitting to November
- **Two valid mitigations:** (1) Temporal validation + expanding-window rates (v5 Cowork), or (2) Drop entity rates, use low-cardinality only (Clean v5 / v5_model)
- **v4 lesson:** High-cardinality encodings (MERCHANTID, CLERK_ID, CITY) overfit; December has unseen entities → cold-start collapse

---

## 9. Reproducibility

- **Final submission scripts:** `payjoy_model_clean_v5.py` → `improved_submission.csv`; `v5_model.py` → `submission_v5_mismatch.csv`
- **Best Kaggle (v5 Cowork):** `submission_v5.csv`; script `_run_v5_cowork.py` referenced but not in repo; closest: `v5_gpu_turbo.py`
- **Other pipelines:** `payjoy_model_v8.py` → `submission_v8.csv`; `payjoy_model_v9.py` → `submission_v9.csv`
- **Data:** Orders.csv, Payment_History.csv, Test_OrderIDs.csv
- **Dependencies:** requirements.txt

---

## 10. References

- [RESULTS.md](RESULTS.md) — Version-by-version results
- [EXPERIMENTS.md](EXPERIMENTS.md) — Experiment log and scorecard
- [DECISION_LOG.md](DECISION_LOG.md) — Analytical decisions
- [V7_PROPOSAL.md](V7_PROPOSAL.md) — Lessons from v6 overfitting

---

## Key Points Summary

- **Task:** Predict FPD within 15 days for December 2025 orders; metric is AUC-ROC.
- **Best Kaggle:** v5 Cowork (LightGBM+RF, 118 features, temporal validation, expanding-window entity rates).
- **Final submitted scripts:** `payjoy_model_clean_v5.py` and `v5_model.py` — CatBoost/LightGBM, low-cardinality features only.
- **Leakage:** Entity FPD rates (MERCHANTID, CLERK_ID, etc.) computed on full training set leak into random CV → inflated AUC (0.76 vs true ~0.56).
- **Mitigation 1:** Temporal validation + expanding-window rates (v5 Cowork).
- **Mitigation 2:** Drop entity rates; use only low-cardinality/continuous features; fold-level target encoding (Clean v5 / v5_model).
- **Why Clean v5:** Avoids leakage by design, generalizes to unseen December entities, reproducible and self-contained.
- **Features excluded:** MERCHANTID, CLERK_ID, ADMINID, CITY, MODEL — high cardinality, overfit, cold-start on test.
- **v5_model adds:** face_liveness_gap, finance_amount_country_zscore, new_merchant_high_value, user_merchant_state_mismatch.
- **Nov AUC ≠ Kaggle:** v6 had higher Nov AUC but worse Kaggle; avoid overfitting to November.
