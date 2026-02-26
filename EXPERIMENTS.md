# PayJoy FPD_15 Prediction Challenge — Experiments Tracker

Living document for proposing, running, and recording model improvement experiments.
Update the scorecard and experiment log after every run.

---

## Scorecard

| # | Experiment | Nov Holdout AUC | Delta vs Previous Best | Date |
|---|---|---|---|---|
| 001 | XGBoost baseline (20 features, RandomizedSearchCV) | 0.5554 | — (first run) | 2025-02-22 |
| 002 | XGBoost + expanded features (27 features: +LOCK_NAME, CURRENCY, state_fpd_rate, entity counts) | 0.5556 | +0.0002 | 2025-02-22 |
| 003 | Feature pruning (dropped 8 temporal-noise features, 19 remaining) | 0.5533 | -0.0023 | 2025-02-23 |
| 004 | HP tuning via expanding-window CV (25 configs, pruned features) | 0.5533 | -0.0023 | 2025-02-23 |
| 005 | v2 + currency normalization (country z-scores for amounts) | 0.5566 | +0.0010 | 2025-02-23 |
| 006 | v2 + entity payment history (merchant/clerk/admin overdue stats) | 0.5558 | +0.0002 | 2025-02-23 |
| 007 | v2 + KYC interactions (kyc_min_score, perfect_kyc_flag) | 0.5563 | +0.0007 | 2025-02-23 |
| 008 | v2 + geographic cross (user_merchant_same_state) | 0.5561 | +0.0005 | 2025-02-23 |
| **009** | **v2 + ALL new feature groups (A+B+C+D combined)** | **0.5574** | **+0.0018** | **2025-02-23** |
| **010** | **v5 Cowork (LightGBM+RF, 118 features, expanding window)** | **0.63171** | **+0.0656** | **2026-02-25** |

**Current best: EXP-010 — Nov AUC 0.63171**

**Kaggle competition slug:** `predicting-fraud-in-phone-financing`

---

## Priority 1 — Validation & Diagnostics

The 5-fold CV AUC (0.76) vs November holdout AUC (0.555) gap is the most urgent
problem. The model appears to learn patterns that do not generalize across time.
Diagnose the root cause before investing in new features or tuning.

### 1A. Temporal distribution analysis

**Hypothesis:** Feature distributions shift significantly between months, making
cross-validated scores misleading.

**Test:** For each feature in `selected_features`, plot the distribution (or mean)
per month (1-11). Flag features whose November distribution diverges sharply from
the training months. Pay special attention to entity FPD rates, country mix, and
manufacturer mix.

**Status:** Not started

---

### 1B. Per-feature temporal AUC

**Hypothesis:** Some features carry strong signal within random CV but zero signal
across time boundaries.

**Test:** For each feature individually, train a single-feature XGBoost (or
logistic regression) on Jan-Oct data and evaluate AUC on November. Rank features
by temporal AUC. Features with temporal AUC near 0.50 are noise in the temporal
setting even if they score well in random CV.

**Status:** Not started

---

### 1C. Expanding-window temporal CV

**Hypothesis:** The 5-fold random CV inflates AUC because it lets the model train
on future months and validate on past months.

**Test:** Replace `RandomizedSearchCV` with a manual expanding-window scheme:

| Fold | Train months | Val month |
|---|---|---|
| 1 | Jan-Jun | Jul |
| 2 | Jan-Jul | Aug |
| 3 | Jan-Aug | Sep |
| 4 | Jan-Sep | Oct |
| 5 | Jan-Oct | Nov |

Average AUC across the five folds. This should produce a number much closer to
the true November holdout AUC and give an honest signal for hyperparameter
selection.

**Status:** Not started

---

### 1D. Leakage audit on entity FPD rates

**Hypothesis:** Within the random 5-fold CV, entity FPD rates computed from the
full Jan-Nov set leak validation labels into training features, inflating AUC.

**Test:** Run the existing XGBoost pipeline twice on the same random CV:
1. Entity rates computed from the full training set (current behavior in CV).
2. Entity rates recomputed per-fold (only from that fold's training rows).

Compare mean CV AUC. A large drop in (2) confirms leakage is a major contributor
to the inflated 0.76.

**Status:** Not started

---

## Priority 2 — Feature Engineering

Catalog of untapped features. Each idea references specific columns from the data.
Available columns: 33 in `Orders.csv`, 8 in `Payment_History.csv`.

### 2A. Geographic features

**Columns:** USER_STATE, MERCHANT_STATE, STATE, CITY

- Target-encode at the state level (lower cardinality than city).
- Cross-feature: `user_merchant_same_state = (USER_STATE == MERCHANT_STATE)`.
- Consider grouping rare states into an "OTHER" bucket before encoding.

**Status:** Partially done (EXP-002) — `state_fpd_rate` added. Cross-features
and city encoding still pending.

---

### 2B. Device and product signals

**Columns:** MODEL, LOCK_NAME, MANUFACTURER

- MODEL has high cardinality — needs frequency or target encoding rather than OHE.
- LOCK_NAME may carry info about the specific financing product/terms.
- Cross MODEL with MANUFACTURER for brand+model risk profiles.

**Status:** Partially done (EXP-002) — `LOCK_NAME` and `CURRENCY` added as OHE.
MODEL encoding and cross-features still pending.

---

### 2C. Temporal enrichment

**Columns:** TRANSACTIONTIME, FIRST_PAYMENT_DUE_TIMESTAMP, LOAN_START_DATE,
MERCHANT_LAST_SALE_DATE

- `days_to_first_due = FIRST_PAYMENT_DUE_TIMESTAMP - TRANSACTIONTIME` (longer
  gaps may correlate with default risk).
- `merchant_recency = TRANSACTIONTIME - MERCHANT_LAST_SALE_DATE` (recently
  active merchants vs dormant ones).
- Day-of-month, weekend/holiday flag.
- Already have: tx_hour, tx_dayofweek, tx_month.

**Status:** Not started

---

### 2D. Merchant/clerk/admin volume features

**Columns:** MERCHANTID, CLERK_ID, ADMINID (aggregated from Orders)

- `merchant_order_count`: total orders placed through the merchant in the
  training window. High-volume entities have more stable rate estimates.
- `clerk_order_count`, `admin_order_count`: same logic.
- Ratio: `clerk_orders / merchant_orders` (does one clerk dominate a merchant?).

**Status:** Partially done (EXP-002) — entity order counts added. Ratio
feature still pending.

---

### 2E. Payment history — entity-level aggregates

**Columns:** TOTAL_CUMPAID, DAYSOVERDUE from `Payment_History.csv`

December test rows have no payment history themselves, but their merchants and
clerks do. Aggregate payment behavior at the entity level:

- `merchant_avg_days_overdue`: mean DAYSOVERDUE across all of a merchant's
  historical orders.
- `merchant_avg_cumpaid`: mean TOTAL_CUMPAID across a merchant's orders.
- `clerk_avg_days_overdue`, `admin_avg_days_overdue`: same pattern.

These features are genuinely available at prediction time (PayJoy knows the
payment track record of its sales channels).

**Status:** Not started

---

### 2F. KYC interaction features

**Columns:** FACE_RECOGNITION_SCORE, IDVALIDATION_OVERALL_SCORE, LIVENESS_SCORE,
OVERALL_SCORE

- `kyc_min_score = min(FACE, ID, LIVENESS)` — weakest-link signal.
- `kyc_score_product = FACE * ID * LIVENESS` — joint verification strength.
- `perfect_kyc_flag = (FACE >= 99) & (ID >= 99) & (LIVENESS >= 99)` — suspiciously
  perfect scores may indicate spoofed verification.

**Status:** Not started

---

### 2G. Currency normalization

**Columns:** FINANCE_AMOUNT, PURCHASE_AMOUNT, TOTAL_DUE, DOWN_PAYMENT_AMOUNT,
COUNTRY, CURRENCY

Dollar amounts are in local currencies (MXN, PEN, BRL, etc.). A 4000 MXN loan is
~$200 USD while a 4000 BRL loan is ~$700 USD. Options:

- Z-score normalize financial features within each COUNTRY.
- Convert to a common currency using approximate exchange rates.
- Add a `country_finance_percentile` feature (rank within country).

**Status:** Not started

---

## Priority 3 — Modeling & Tuning

### 3A. LightGBM comparison

LightGBM is often faster than XGBoost with comparable or better AUC on tabular
data. It also supports native categorical features, avoiding one-hot explosion
for COUNTRY, MANUFACTURER, etc.

**Test:** Swap XGBoost for LightGBM with equivalent hyperparameters. Compare
November holdout AUC and training time.

**Status:** Not started

---

### 3B. Targeted XGBoost re-tune

The top-5 configurations from EXP-001 all cluster around similar values. Lock in
the clear winners and search only the uncertain parameters:

- Fix: `max_depth=4`, `n_estimators=500`, `colsample_bytree=0.8`
- Search: `learning_rate` in [0.05, 0.1, 0.15], `subsample` in [0.8, 0.9],
  `scale_pos_weight` in [3, 5, 7]

This reduces the grid from 972 combinations to 18, cutting runtime dramatically.

**Status:** Not started

---

### 3C. Early stopping

Add `early_stopping_rounds=20` with an eval set to stop training trees once
validation AUC plateaus. This prevents wasted computation on configs with high
`n_estimators` and lets the model self-select the optimal number of rounds.

**Status:** Not started

---

### 3D. Ensemble methods

Blend predictions from multiple model families. Even a simple weighted average
often lifts AUC:

- XGBoost + LightGBM (diversity from different tree implementations)
- XGBoost + NN (diversity from different inductive biases)
- Stacking: train a logistic regression on out-of-fold predictions from base models.

Defer until at least two models achieve reasonable standalone AUC on November.

**Status:** Not started

---

### 3E. Neural network revisit

Defer until XGBoost temporal AUC is stabilized. When revisited:

- Try wider, shallower architectures: [256, 128] instead of [128, 64, 32].
- Increase dropout to 0.4-0.5 for stronger regularization.
- Experiment with learning rate scheduling (e.g., cosine annealing).
- Consider entity embeddings for high-cardinality categoricals.

**Status:** Not started

---

## Priority 4 — Runtime Optimization

Practical knobs for faster iteration cycles.

| Knob | Current | Proposed (exploration) | Proposed (final) |
|---|---|---|---|
| `n_iter` (RandomizedSearchCV) | 20 | 5-10 | 20+ |
| `cv_folds` | 5 | 3 | 5 |
| Training data | 100% (~998K rows) | 25% stratified sample | 100% |
| `early_stopping_rounds` | None | 20 | 20 |
| `tree_method` | default (exact) | `hist` | `hist` |
| NN grid combos | 12 (3x2x2) | 3-4 targeted | full grid |

Using exploration settings, an XGBoost search should take ~5 minutes instead
of ~40 minutes.

---

## Experiment Log

### EXP-001: XGBoost baseline with 20 features

- **Date:** 2025-02-22
- **Hypothesis:** XGBoost with engineered features (financial ratios, temporal
  signals, entity FPD rates) and one-hot categoricals will significantly beat
  the logistic regression benchmark (AUC ~0.51).
- **What changed:** First non-benchmark model. 20 features (8 raw numeric,
  7 engineered numeric, 3 categorical via OHE → 59 columns after encoding).
  RandomizedSearchCV with 20 iterations, 5-fold CV, `n_jobs=-1`.
- **Best hyperparameters:**
  - `max_depth`: 4
  - `learning_rate`: 0.1
  - `n_estimators`: 500
  - `subsample`: 0.9
  - `colsample_bytree`: 0.8
  - `scale_pos_weight`: 5
- **5-fold CV AUC (Jan-Oct):** 0.7604
- **Nov holdout AUC:** 0.5554
- **Compared to previous best:** +0.043 vs benchmark (0.5128 validation AUC)
- **Notes:** Massive gap between CV AUC and temporal holdout AUC indicates the
  random 5-fold CV is unreliable. Entity FPD rates likely leak within CV folds.
  The November holdout of 0.555 is the number to trust. Priority is diagnosing
  the generalization failure before adding features or tuning further.
- **Runtime:** ~40 minutes (100 fits on ~998K rows)

---

### EXP-002: XGBoost with expanded features (27 features)

- **Date:** 2025-02-22
- **Hypothesis:** Adding high-signal unused columns (LOCK_NAME, CURRENCY,
  state_fpd_rate, entity order volume counts) will improve temporal
  generalization since these features capture product-type and maturity
  signals that are more stable across months than entity FPD rates alone.
- **What changed:** Added 7 new features to the v1 baseline (20 → 27 pre-OHE,
  59 → 77 post-OHE):
  - `LOCK_NAME` (OHE, 7 levels — sideload types ~50% FPD vs standard ~8%)
  - `CURRENCY` (OHE, 8 levels — FPD varies by currency)
  - `state_fpd_rate` (target-encoded, 199 states)
  - `merchantid_order_count`, `clerk_id_order_count`, `adminid_order_count`
- **Best hyperparameters:** Same as EXP-001 (tuning converged to identical config)
- **5-fold CV AUC (Jan-Oct):** 0.8066 (+0.046 vs EXP-001)
- **Nov holdout AUC:** 0.5556
- **Compared to previous best:** +0.0002 vs EXP-001 (0.5554)
- **Submission mean pred:** 0.1545 (vs 0.2596 in v1)
- **Kaggle public AUC:** 0.54797
- **Notes:** The CV AUC jumped significantly (+0.046) indicating the new features
  are highly predictive within the training distribution, especially LOCK_NAME.
  However, the Nov AUC barely moved, confirming that temporal distribution shift
  is the core problem — not feature coverage. The new features overfit to
  within-month patterns. Kaggle public AUC (0.548) is close to Nov held-out
  (0.556), validating the temporal validation setup. Next priorities:
  expanding-window temporal CV (1C), per-feature temporal AUC analysis (1B),
  and LightGBM (3A).
- **Runtime:** ~40 minutes (100 fits on ~998K rows × 77 features)

---

### EXP-003: Feature pruning (19 features)

- **Date:** 2025-02-23
- **Hypothesis:** Removing 8 features flagged as temporal noise (temporal AUC
  <0.52) would reduce overfitting and improve temporal generalization.
- **What changed:** Dropped `tx_month`, `FACE_RECOGNITION_SCORE`,
  `IDVALIDATION_OVERALL_SCORE`, `LIVENESS_SCORE`, `OVERALL_SCORE`, `LOCK_NAME`,
  `LOCK_PRODUCT`, `clerk_id_order_count` (27 → 19 pre-OHE features).
- **Nov holdout AUC:** 0.5533 (EW mean: 0.5558)
- **Compared to previous best:** -0.003 vs EXP-002 (0.5556)
- **Notes:** Pruning hurt slightly. The "noise" features contribute marginal
  signal in combination even though they're individually near-random. Keeping
  the v2 feature set.

---

### EXP-004: Hyperparameter tuning via expanding-window CV

- **Date:** 2025-02-23
- **Hypothesis:** Tuning hyperparameters under expanding-window temporal CV
  (which scored 0.557 vs random CV's inflated 0.807) would find configs that
  generalize better across time.
- **What changed:** Searched 25 random configs over `max_depth` [2,3,4,6],
  `learning_rate` [0.05,0.1], `n_estimators` [200,500], `subsample` [0.8,0.9],
  `colsample_bytree` [0.7,0.8], `scale_pos_weight` [3,5,7],
  `min_child_weight` [1,5,10]. Evaluated on both expanding-window mean and
  single Nov holdout.
- **Nov holdout AUC:** 0.5533 (best config: max_depth=6, lr=0.05, n_est=500,
  sub=0.8, col=0.7, spw=3, mcw=10)
- **Compared to previous best:** -0.002 vs EXP-002 (0.5556)
- **Notes:** All 25 configs clustered in 0.552-0.556 Nov AUC. The original
  v2 params are already near-optimal. The ceiling for the current feature set
  is ~0.556 regardless of hyperparameters. The path to 0.61 requires adding
  genuinely new information, not tuning existing features differently.

---

### EXP-005 to EXP-008: Phase 3 new feature groups (individual evaluation)

- **Date:** 2025-02-23
- **Hypothesis:** Adding genuinely new information (currency normalization,
  entity-level payment history, KYC interactions, geographic cross-features) to
  the v2 baseline would improve temporal generalization since these capture
  cross-entity and cross-market risk signals.
- **What changed per group:**
  - **A (EXP-005):** 4 features — country-level z-scored financial amounts
    (`FINANCE_AMOUNT_country_zscore`, etc.)
  - **B (EXP-006):** 6 features — entity-level payment overdue aggregates
    (`merchantid_avg_days_overdue`, `merchantid_max_days_overdue`, etc.)
  - **C (EXP-007):** 2 features — `kyc_min_score` (weakest KYC link),
    `perfect_kyc_flag` (all KYC ≥99, potential spoofing)
  - **D (EXP-008):** 1 feature — `user_merchant_same_state`
- **Nov holdout AUC:** A: 0.5566, B: 0.5558, C: 0.5563, D: 0.5561
- **Notes:** All 4 groups improve over v2 baseline (0.5556). Currency
  normalization was the strongest individual group (+0.001).

---

### EXP-009: All Phase 3 feature groups combined (v2 + A+B+C+D)

- **Date:** 2025-02-23
- **Hypothesis:** Combining all positive feature groups would compound gains
  since they capture different risk dimensions (financial context, payment
  behavior, identity verification, geography).
- **What changed:** Added all 13 new features on top of v2 (27 → 40 pre-OHE).
- **Nov holdout AUC:** 0.5574
- **Compared to previous best:** +0.0018 vs EXP-002 (0.5556)
- **Notes:** Best Nov AUC achieved so far. The compound gain (+0.0018) is
  larger than any individual group, confirming minimal redundancy. This is now
  our best model. Ready for submission whenever the user decides to use one
  of their 3 daily slots.

---

### EXP-005 — v5 Feature Architecture Evaluation (2026-02-25)

- **Goal:** Quantify the AUC contribution of each new feature group in the Cowork rebuild, using the fixed Jan-Oct → Nov temporal holdout.
- **Method:** Sequential addition of feature groups (steps A→L), each evaluated with a fast LightGBM (200 trees, lr=0.1, 31 leaves) on the Jan-Oct training set validated on November.
- **Feature group results:**

| Step | Group | # Features | Nov AUC | Delta |
|------|-------|------------|---------|-------|
| A | KYC original | 4 | 0.52368 | baseline |
| B | + KYC transforms (deficit/min/missing) | 15 | 0.52216 | -0.00152 |
| C | + financial ratios + temporal + geo | 34 | 0.59997 | +0.07781 |
| D | + merchant dormancy | 37 | 0.59997 | +0.00000 |
| E | + all entity FPD rates | 40 | 0.61317 | +0.01320 |
| F | + excess/weighted risk/new flags | 48 | 0.61426 | +0.00109 |
| G | + temporal trend features | 54 | 0.61270 | -0.00155 |
| H | + order size anomaly vs merchant | 56 | 0.61349 | +0.00078 |
| I | + interaction variables | 65 | 0.61385 | +0.00037 |
| J | + pmt FPD proxy + zero recovery | 77 | 0.62723 | +0.01338 |
| K | + all payment history features | 101 | 0.62388 | -0.00335 |
| L | + raw financials + full temporal | 118 | 0.62941 | +0.00553 |

- **Final CV AUC (3-fold temporal):** 0.63898 ± 0.00668
- **Fold 3 (Jan-Oct → Nov) AUC:** 0.63171
- **Key findings:** Financial/temporal/geo (step C) added +0.078 — largest single jump. Entity rates (E) +0.013, payment proxy (J) +0.013. Dormancy (D) added nothing. Full payment features (K) slightly worse than J.
- **Decisions:** Dormancy droppable. Consider PMT_CORE instead of full PMT_FEATS_RICH. RF blend (70/30) used; within 0.02 of LGB.

---

### EXP-006 — Hyperparameter Tuning on v5 Feature Set (TBD)

- **Goal:** Tune `num_leaves` and `min_child_samples` for the v5 feature set.
- **Configuration grid:** num_leaves ∈ [31, 63, 127], min_child_samples ∈ [30, 50, 100]
- **Method:** Jan-Oct → Nov holdout. Status: not yet run.

---

### EXP-NNN: (template)

- **Date:**
- **Hypothesis:**
- **What changed** (features / model / params):
- **Nov holdout AUC:**
- **Compared to previous best:**
- **Notes / next steps:**

---

## v6 Infrastructure (2026-02-26)

v6 improvement framework implemented per action plan. Target: ~0.62 AUC (from 0.60086).

### Scripts

| Script | Purpose |
|--------|---------|
| `_run_v6_diagnostics.py` | Per-feature temporal AUC, permutation importance, error analysis |
| `_run_v6_experiments.py` | Config sweep: pruning, LOCK_NAME/CURRENCY, HP grid, ensemble weights |

### v5 Configurable Flags (env vars)

| Env var | Default | Effect |
|---------|---------|--------|
| `V5_USE_DORMANCY` | 1 | Include dormancy features (D) |
| `V5_USE_TREND` | 1 | Include temporal trend features (G) |
| `V5_USE_PMT_RICH` | 1 | Use PMT_FEATS_RICH; 0 = PMT_CORE only |
| `V5_USE_LOCK_NAME` | 0 | Add LOCK_NAME_fpd_rate |
| `V5_USE_CURRENCY` | 0 | Add CURRENCY_fpd_rate |
| `V5_DIAGNOSTICS` | 0 | Save pipeline to pickle and exit (for v6) |

### v6 Experiment Configs

1. **baseline_v5** — Full v5 feature set
2. **no_dormancy** — Drop D
3. **no_trend** — Drop G (keep clerk_excess_risk)
4. **pmt_core** — PMT_CORE instead of PMT_FEATS_RICH
5. **no_dorm_no_trend** — Drop D + G
6. **prune_all_pmt_core** — Drop D + G + PMT_CORE
7. **+lock_name** — Pruned + LOCK_NAME_fpd_rate
8. **+lock_currency** — Pruned + LOCK_NAME + CURRENCY_fpd_rate
9. **hp_31_30**, **hp_31_100**, **hp_127_50** — num_leaves × min_child_samples
10. **lgb80_rf20**, **lgb90_rf10**, **lgb_only** — Ensemble weight sweep

### v6 Results (to be filled after run)

Run `python _run_v6_experiments.py` and record best config in scorecard.

---

## Post-v5 Next Steps

Read the Section 7 AUC table from `run_v5.log` before choosing a path.

### How to read the Section 7 table

After running `_run_v5_cowork.py`, the log prints a cumulative AUC table (steps A→L).
Use these thresholds to decide what to keep:

| Delta vs. previous step | Action |
|---|---|
| > +0.010 | Critical — never drop |
| +0.003 – 0.010 | Meaningful — keep |
| +0.001 – 0.003 | Minor — keep unless causing overfit |
| < +0.001 | Likely noise — consider dropping |

**Key diagnostic: compare step J vs. step I** — J adds pmt_fpd_proxy_rate and zero_recovery_rate. If J > I by > +0.001, payment history adds independent signal. If J ≈ I, entity FPD rates already captured the same signal.

### Tier 1 — Try regardless of current AUC

- **T1-A:** Prune low-signal features — drop groups where Section 7 delta < +0.001 (DORMANCY, MEDIAN, or replace PMT_FEATS_RICH with PMT_CORE if step K ≈ step J).
- **T1-B:** Tune num_leaves and min_child_samples — grid [31,63,127] × [30,50,100] on Jan-Oct → Nov holdout.
- **T1-C:** Lower learning rate + more rounds if best_iter near 3000 cap.

### Tier 2 — Try if still below 0.61 after Tier 1

- **T2-A:** Country-level stratified models if one country > 60% of test set.
- **T2-B:** Payment history portfolio trend (recent vs overall merchant payment behavior).
- **T2-C:** Stricter expanding window for payment history (no future-month leakage).
- **T2-D:** Smoothing parameter k sensitivity — test k ∈ {5, 10, 20}.

### Tier 3 — Experimental

- **T3-A:** XGBoost ensemble blend.
- **T3-B:** Permutation importance-based feature pruning.
- **T3-C:** Platt scaling for probability calibration.

### Decision matrix (quick reference)

| Feature group | Keep if | Drop if |
|---|---|---|
| Entity FPD rates | Always | Never |
| ADMINID/STATE/CITY rates | Step F adds > +0.001 | Near-zero delta |
| MODEL/MANUFACTURER/LOCK_PRODUCT | Step G adds > +0.001 | Near-zero delta |
| KYC transforms | Step B adds > +0.001 | Raw KYC ≈ transforms |
| PMT core | Step J adds > +0.001 | J ≈ I |
| Full payment features | Step K > step J by +0.001 | K ≈ J |
| Dormancy + order size anomaly | Feature importance > 50 | Near zero |
