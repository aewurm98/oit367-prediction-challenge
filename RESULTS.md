# PayJoy FPD_15 Prediction Challenge — Experiment Results

Each version snapshots the full model configuration and results so any
prior state can be restored exactly.

---

## v1 — Baseline XGBoost (2025-02-22)

**Description:** Initial XGBoost model with 20 features, temporal validation
(train Jan-Oct, validate Nov), and `RandomizedSearchCV` tuning.

**Features (20):**

```
FINANCE_AMOUNT, PURCHASE_AMOUNT, TOTAL_DUE, DOWN_PAYMENT_AMOUNT,
FACE_RECOGNITION_SCORE, IDVALIDATION_OVERALL_SCORE, LIVENESS_SCORE,
OVERALL_SCORE, down_payment_ratio, finance_ratio, tx_hour, tx_dayofweek,
tx_month, merchant_tenure_days, merchantid_fpd_rate, clerk_id_fpd_rate,
adminid_fpd_rate, COUNTRY, LOCK_PRODUCT, MANUFACTURER
```

**XGBoost hyperparameters:**

| Parameter | Value |
|---|---|
| max_depth | 4 |
| learning_rate | 0.1 |
| n_estimators | 500 |
| subsample | 0.9 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 5 |

**Results:**

| Metric | Value |
|---|---|
| CV AUC (within Jan-Oct, 5-fold) | 0.7604 |
| November held-out AUC | 0.5554 |
| Train shape (Jan-Oct) | (998102, 59) |
| Val shape (Nov) | (99442, 59) |
| Train FPD rate | 0.0969 |
| Val FPD rate | 0.0894 |
| Submission mean pred | 0.2596 |

**Submission file:** `submission_v1.csv` (copy of initial `submission.csv`)

**Leaderboard context:** Top teams at 0.62; this score is near the bottom.

---

## v2 — Expanded Features (2025-02-22)

**Description:** Added 7 new features to the v1 baseline: `LOCK_NAME` (one-hot,
7 levels with extreme FPD variation), `CURRENCY` (one-hot, 8 levels),
`state_fpd_rate` (target-encoded STATE), and entity order volume counts
(`merchantid_order_count`, `clerk_id_order_count`, `adminid_order_count`).
Same XGBoost hyperparameters as v1 (tuning selected identical params).

**Features (27 pre-OHE):**

```
FINANCE_AMOUNT, PURCHASE_AMOUNT, TOTAL_DUE, DOWN_PAYMENT_AMOUNT,
FACE_RECOGNITION_SCORE, IDVALIDATION_OVERALL_SCORE, LIVENESS_SCORE,
OVERALL_SCORE, down_payment_ratio, finance_ratio, tx_hour, tx_dayofweek,
tx_month, merchant_tenure_days, merchantid_fpd_rate, clerk_id_fpd_rate,
adminid_fpd_rate, state_fpd_rate, merchantid_order_count,
clerk_id_order_count, adminid_order_count,
COUNTRY, LOCK_PRODUCT, MANUFACTURER, LOCK_NAME, CURRENCY
```

**XGBoost hyperparameters:** (identical to v1 — tuning converged to same config)

| Parameter | Value |
|---|---|
| max_depth | 4 |
| learning_rate | 0.1 |
| n_estimators | 500 |
| subsample | 0.9 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 5 |

**Results:**

| Metric | v2 Value | v1 Value | Delta |
|---|---|---|---|
| CV AUC (within Jan-Oct, 5-fold) | 0.8066 | 0.7604 | +0.0462 |
| November held-out AUC | 0.5556 | 0.5554 | +0.0002 |
| Train shape (Jan-Oct) | (998102, 77) | (998102, 59) | +18 cols |
| Val shape (Nov) | (99442, 77) | (99442, 59) | +18 cols |
| Train FPD rate | 0.0969 | 0.0969 | — |
| Val FPD rate | 0.0894 | 0.0894 | — |
| Submission mean pred | 0.1545 | 0.2596 | -0.1051 |

**Submission file:** `submission.csv`

**Kaggle public AUC:** 0.54797

**Analysis:** The large CV AUC improvement (+0.046) without corresponding Nov AUC
improvement indicates the new features (especially LOCK_NAME and target-encoded
rates) are highly predictive within the training distribution but do not
generalize well across the month boundary. The temporal distribution shift
(Jan-Oct → Nov) remains the core challenge. The mean prediction dropped from
0.26 to 0.15 because the full Jan-Nov training set includes November data,
giving the model more accurate entity rates. Kaggle public AUC (0.548) is
close to the Nov held-out AUC (0.556), confirming the temporal validation
setup is a reasonable proxy for leaderboard performance.

---

## v3 — Phase 3 Combined Features (2025-02-23)

**Description:** Added 13 new features from four groups: currency normalization
(country-level z-scores), entity payment history aggregates (from Payment_History),
KYC interaction features, and a geographic cross-feature. Same hyperparameters
as v2 (confirmed near-optimal by EXP-004's 25-config expanding-window search).

**Features (40 pre-OHE):**

```
[v2's 27 features]
+ FINANCE_AMOUNT_country_zscore, PURCHASE_AMOUNT_country_zscore,
  TOTAL_DUE_country_zscore, DOWN_PAYMENT_AMOUNT_country_zscore,
+ merchantid_avg_days_overdue, merchantid_max_days_overdue,
  clerk_id_avg_days_overdue, clerk_id_max_days_overdue,
  adminid_avg_days_overdue, adminid_max_days_overdue,
+ kyc_min_score, perfect_kyc_flag,
+ user_merchant_same_state
```

**XGBoost hyperparameters:** (identical to v1/v2)

| Parameter | Value |
|---|---|
| max_depth | 4 |
| learning_rate | 0.1 |
| n_estimators | 500 |
| subsample | 0.9 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 5 |

**Results:**

| Metric | v3 Value | v2 Value | Delta |
|---|---|---|---|
| November held-out AUC | 0.5574 | 0.5556 | +0.0018 |

**Individual group contributions (all on top of v2 baseline):**

| Group | Features added | Nov AUC | Delta |
|---|---|---|---|
| A — Currency norm | 4 | 0.5566 | +0.0010 |
| B — Payment history | 6 | 0.5558 | +0.0002 |
| C — KYC interactions | 2 | 0.5563 | +0.0007 |
| D — Geo cross | 1 | 0.5561 | +0.0005 |
| **All combined** | **13** | **0.5574** | **+0.0018** |

**Submission file:** Not yet generated — awaiting user decision.

**Analysis:** All four feature groups improved individually and compounded when
combined (+0.0018 total > any single group), indicating minimal redundancy.
Currency normalization was the strongest group — making financial amounts
comparable across countries resolved a cross-market confound. The combined
v3 model achieves the best Nov AUC to date. Importantly, EXP-003 and EXP-004
proved that feature pruning and HP tuning cannot break past ~0.556, so the
improvement here represents genuine new information, not better optimization.

---

## v5 — Cowork Rebuild (2026-02-25)

**Description:** Full feature rebuild with expanding-window entity FPD rates,
Bayesian smoothing k=10, 8 entity types, KYC deficit transforms, payment
history features, and LightGBM temporal CV.

**Features:** 118 (KYC, financial ratios, temporal, entity FPD rates, excess/weighted,
trend, order size anomaly, interactions, payment history aggregates)

**LightGBM hyperparameters:** learning_rate=0.03, num_leaves=63, min_child_samples=50,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1

**Results:**

| Metric | Value |
|---|---|
| CV AUC (3-fold temporal) | 0.63898 ± 0.00668 |
| November held-out AUC (Fold 3) | 0.63171 |
| RF blend used | True (70/30 LGB/RF) |
| Submission mean pred | 0.44323 |

**Submission file:** `submission_v5.csv`

**Run:** `python _run_v5_cowork.py`

---

## v6 — +lock_currency (2026-02-26)

**Description:** v5 pipeline with v6 config: LOCK_NAME + CURRENCY FPD rates enabled,
dormancy and PMT_CORE disabled (pruned base). Best config from v6 experiments.

**Config:** V5_USE_LOCK_NAME=1, V5_USE_CURRENCY=1, V5_USE_DORMANCY=0, V5_USE_PMT_RICH=0

**Features:** 95 (entity FPD rates including LOCK_NAME, CURRENCY; no dormancy, PMT_CORE)

**Results:**

| Metric | Value |
|---|---|
| CV AUC (3-fold temporal) | 0.63747 ± 0.00245 |
| November held-out AUC (Fold 3) | 0.63408 |
| RF blend used | True (70/30 LGB/RF) |
| Top feature | LOCK_NAME_fpd_rate |

**Submission file:** `submission_v6.csv`

**Run:** `python _run_v6_cowork.py`

**Audit note:** v6 underperformed v5 on Kaggle despite higher Nov AUC. Use v5 for submission. See V7_PROPOSAL.md for lessons learned.
