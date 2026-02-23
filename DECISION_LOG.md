# PayJoy FPD_15 Prediction Challenge — Decision Log

This is a living document that captures every significant analytical decision
made during the competition. Each entry records the date, the decision, the
rationale behind it, and its current status or measured outcome. It is intended
to feed directly into the two-page competition report.

---

## 2025-02-22 — Two-file architecture

**Decision:** Separate experimentation (`model_experiments.ipynb`) from the
final submission pipeline (`best_model_pipeline.ipynb`).

**Rationale:** Keeping tuning/exploration code away from the production
pipeline reduces the risk of accidentally shipping debug artefacts or stale
parameters. The pipeline notebook is a clean, reproducible path from raw CSVs
to `submission.csv`.

**Status:** Both notebooks built. `best_model_pipeline.ipynb` v1 uses winning
XGBoost config from initial experiment.

---

## 2025-02-22 — Feature selection (initial 20 features)

**Decision:** Use 20 features in three groups:

| Group | Features |
|---|---|
| Raw numeric (8) | FINANCE_AMOUNT, PURCHASE_AMOUNT, TOTAL_DUE, DOWN_PAYMENT_AMOUNT, FACE_RECOGNITION_SCORE, IDVALIDATION_OVERALL_SCORE, LIVENESS_SCORE, OVERALL_SCORE |
| Engineered numeric (7) | down_payment_ratio, finance_ratio, tx_hour, tx_dayofweek, tx_month, merchant_tenure_days, entity FPD rates (merchant/clerk/admin) |
| Categorical (3) | COUNTRY, LOCK_PRODUCT, MANUFACTURER |

**Rationale:** These cover the main signal families identified in the PayJoy
problem description — financial terms, KYC verification scores, temporal
patterns, merchant/clerk risk profiles, and product/geography segmentation.
Payment-history aggregations (pay_max_overdue, etc.) are available but excluded
from the default list because they are NaN for all December test rows and
collapse to a constant after imputation.

**Status:** Baseline set. Additional features (entity-level payment aggregations,
city/state encoding, merchant volume) deferred to later iterations.

---

## 2025-02-22 — Entity FPD rate target encoding

**Decision:** Compute the mean FPD_15 rate per MERCHANTID, CLERK_ID, and
ADMINID from labelled training rows and map it onto all orders as a feature.

**Rationale:** Entity fraud rates are among the strongest predictors in
lending-fraud models. Merchants and clerks with historically high default
rates are more likely to originate fraudulent orders. This mirrors information
that would genuinely be available at prediction time (PayJoy knows historical
performance of its sales channels).

**Leakage risk:** When computing rates on the full Jan–Nov set and then
validating on a random subset, the validation rows' labels leak into their
own features. Mitigated in the next decision (temporal validation).

**Status:** Implemented. Leakage fix applied (see 2025-02-22 — Temporal
validation entry).

---

## 2025-02-22 — Imputation strategy

**Decision:** Fill missing numeric values with the training-set median; fill
missing categoricals with the training-set mode. Apply these same fill values
to the test set.

**Rationale:** Median is robust to outliers (loan amounts span multiple
currencies and magnitudes). Computing fill values from training only prevents
test-set information from leaking into the pipeline.

**Status:** Implemented in `prep_data`.

---

## 2025-02-22 — One-hot encoding for categoricals

**Decision:** Use `pd.get_dummies` for COUNTRY, LOCK_PRODUCT, and
MANUFACTURER, then align columns between train and test sets.

**Rationale:** These categoricals have low cardinality (5–20 unique values),
so one-hot encoding is tractable and preserves interpretability. High-
cardinality columns (CITY, STATE) are excluded for now to avoid feature
explosion.

**Status:** Implemented. Column alignment ensures unseen test categories
are dropped and missing train categories are zero-filled.

---

## 2025-02-22 — Model families: XGBoost and PyTorch MLP

**Decision:** Evaluate two model architectures — gradient-boosted trees
(XGBoost) and a multi-layer perceptron (PyTorch).

**Rationale:** XGBoost is the standard top performer on tabular data with
heterogeneous features and class imbalance. The MLP provides a
complementary inductive bias and is a candidate for ensembling. Both are
required by the project blueprint.

**Status:** Both architectures implemented. XGBoost is the priority for the
initial submission; NN comparison deferred.

---

## 2025-02-22 — Class imbalance handling

**Decision:** Address the ~9.6% FPD positive rate via XGBoost's
`scale_pos_weight` parameter (searched over {1, 5, 9}).

**Rationale:** AUC-ROC is the evaluation metric, which is threshold-
invariant, so resampling is unnecessary. `scale_pos_weight` upweights the
loss on positive examples during training, encouraging the model to learn
the minority class without altering the data distribution.

**Status:** Included in the hyperparameter search grid.

---

## 2025-02-22 — Temporal validation strategy

**Decision:** Replace the random 80/20 train/val split with a temporal split:
train on January–October, validate on November. Entity FPD rates are computed
from January–October only when validating.

**Rationale:** The competition task is to predict December from historical
months. A random split allows the model to "see the future" (e.g., train on
October, validate on March). Training on Jan–Oct and validating on November
mimics the real prediction boundary and provides a more honest estimate of
generalisation to the unseen December test set. Computing entity rates from
the training fold only eliminates target leakage during validation.

**Status:** Implemented via `val_month` parameter in `prep_data`.

---

## 2025-02-22 — AUC-ROC as sole evaluation metric

**Decision:** Use AUC-ROC for all model selection, hyperparameter tuning,
and validation decisions.

**Rationale:** This is the competition's official metric. The grading formula
awards full marks (50/50) at AUC >= 0.61 and zero at AUC <= 0.50287. No
secondary metrics (F1, precision, recall) are considered for model selection.

**Status:** All tuning functions score by `roc_auc`.

---

## 2025-02-22 — XGBoost initial experiment results (temporal validation)

**Decision:** First XGBoost hyperparameter search with temporal split (train
Jan-Oct, validate Nov). 20 random configurations x 5-fold CV on Jan-Oct, then
evaluated on held-out November.

**Results:**

| Metric | Value |
|---|---|
| Best CV AUC (within Jan-Oct) | 0.7604 |
| November held-out AUC | 0.5554 |
| Best max_depth | 4 |
| Best learning_rate | 0.1 |
| Best n_estimators | 500 |
| Best subsample | 0.9 |
| Best colsample_bytree | 0.8 |
| Best scale_pos_weight | 5 |

**Rationale:** The large gap between CV AUC (0.76) and Nov AUC (0.56) is
expected and informative. Within-month random CV mixes temporal patterns;
the Nov held-out AUC is the honest estimate of generalisation to unseen
future data. The Nov AUC of 0.5554 exceeds the logistic regression benchmark
(0.5124) but falls short of the 0.61 target. This confirms that temporal
distribution shift is the main challenge.

**Status:** First submission model. Parameters promoted to
`best_model_pipeline.ipynb`.

---

## 2025-02-22 — Kaggle CLI setup

**Decision:** Configure the Kaggle CLI (v2.0.0) for command-line submissions
via the `oit367-challenge` conda environment.

**Setup details:**
- Kaggle v2 uses an environment variable (`KAGGLE_API_TOKEN`) instead of the
  legacy `~/.kaggle/kaggle.json` file.
- The pyenv shim (`~/.pyenv/shims/kaggle`) resolves to an outdated v1.5.13
  that does not support the new `KGAT_` token format. An alias was added to
  `~/.zshrc` pointing `kaggle` to the conda env binary.
- Persistent config added to `~/.zshrc`:
  ```
  export KAGGLE_API_TOKEN=<token>
  alias kaggle="/opt/anaconda3/envs/oit367-challenge/bin/kaggle"
  ```

**Submission commands (reference):**

| Command | Purpose |
|---|---|
| `kaggle competitions submit -c <slug> -f submission.csv -m "msg"` | Upload |
| `kaggle competitions submissions -c <slug>` | View past submissions |
| `kaggle competitions leaderboard -c <slug>` | Public leaderboard |

**Competition slug:** `predicting-fraud-in-phone-financing`

**Status:** Configured and verified.

---

## 2025-02-22 — First submission.csv generated

**Decision:** Generate initial `submission.csv` by retraining the best XGBoost
model on all Jan-Nov data (entity rates from full Jan-Nov) and predicting
December.

**Validation results:**
- Rows: 98,932 (matches Test_OrderIDs.csv)
- No duplicate IDs
- All test IDs present
- Probabilities in [0, 1]
- No missing values
- Mean predicted FPD probability: 0.2596

**Status:** Ready for Kaggle upload. Public AUC to be recorded after submission.

---

## 2025-02-22 — Feature selection (initial 20 features)

**Update:** Promoted to v2 feature set (27 features). See entry below.

---

## 2025-02-22 — v2 feature expansion (27 features)

**Decision:** Add 7 new features to the model:

| Feature | Type | Rationale |
|---|---|---|
| `LOCK_NAME` | Categorical (OHE, 7 levels) | Extreme FPD variation — sideload lock types show ~50% FPD vs. ~8% for standard locks. Different signal from LOCK_PRODUCT. |
| `CURRENCY` | Categorical (OHE, 8 levels) | FPD varies by currency (BRL 12.8%, PHP 13.4% vs. COP 7.5%, MXN 8.3%). Not redundant with COUNTRY. |
| `state_fpd_rate` | Numeric (target-encoded) | 199 states — too many for OHE. Target-encoded mean FPD rate captures regional risk. Same leakage-safe pattern as entity rates. |
| `merchantid_order_count` | Numeric | Small merchants (1-5 orders) default at 11.5% vs. 9.4% for mid-size. More temporally stable than FPD rates. |
| `clerk_id_order_count` | Numeric | Same rationale as merchant count. |
| `adminid_order_count` | Numeric | Same rationale as merchant count. |

**Results:**

| Metric | v2 | v1 | Delta |
|---|---|---|---|
| CV AUC (Jan-Oct, 5-fold) | 0.8066 | 0.7604 | +0.046 |
| Nov held-out AUC | 0.5556 | 0.5554 | +0.0002 |
| Features (post-OHE) | 77 | 59 | +18 |
| Submission mean pred | 0.1545 | 0.2596 | -0.105 |

**Analysis:** The CV AUC jump (+0.046) shows the new features are highly
discriminative within the training distribution. However, the Nov AUC barely
moved, indicating these features do not improve temporal generalisation.
The core challenge remains distribution shift across the month boundary.
The lower mean prediction (0.15 vs 0.26) is expected — the v2 submission model
trains on full Jan-Nov (including Nov data), giving it more accurate entity
rates for November entities that also appear in December.

**Kaggle public AUC:** 0.54797

**Status:** Submitted. Public AUC (0.548) is close to Nov held-out AUC (0.556),
confirming the temporal validation setup is a reasonable proxy. The gap between
CV AUC (0.807) and public AUC (0.548) reinforces that within-month CV is
unreliable and the core challenge is temporal generalization.

---

## 2025-02-23 — Phase 1 diagnostics (per-feature temporal AUC, drift, expanding-window CV)

**Decision:** Ran three diagnostic analyses to understand the temporal
generalization gap before making further model changes.

### Finding 1: Per-feature temporal AUC ranking

Trained single-feature XGBoost models on Jan-Oct, evaluated on November.
Results split into three clear tiers:

| Tier | Features | Temporal AUC range |
|---|---|---|
| KEEP | `adminid_fpd_rate` (0.576), `merchantid_fpd_rate` (0.567), `state_fpd_rate` (0.564) | >0.55 |
| INVESTIGATE | Financial amounts/ratios, CURRENCY, COUNTRY, `clerk_id_fpd_rate`, temporal features, `adminid_order_count` | 0.52-0.55 |
| DROP | `tx_month` (0.500), all 4 KYC scores (0.506-0.516), `LOCK_NAME` (0.515), `LOCK_PRODUCT` (0.519), `clerk_id_order_count` (0.519) | <0.52 |

Key surprise: `LOCK_NAME` — which had extreme within-month FPD variation
(sideload ~50% vs standard ~8%) — is essentially noise across the time
boundary. KYC scores are near-random temporally despite being intuitively
meaningful.

### Finding 2: Feature distribution drift

Numeric features show minimal drift (only `tx_month` flags, trivially).
Categorical drift in COUNTRY/CURRENCY (South Africa share shifted 9.1pp)
explains why those features retain temporal signal.

### Finding 3: Expanding-window temporal CV

| Fold | Train | Val month | AUC |
|---|---|---|---|
| 1 | Jan-Jun | Jul | 0.559 |
| 2 | Jan-Jul | Aug | 0.555 |
| 3 | Jan-Aug | Sep | 0.560 |
| 4 | Jan-Sep | Oct | 0.554 |
| 5 | Jan-Oct | Nov | 0.556 |
| **Mean** | | | **0.557** |

The expanding-window AUC (0.557) closely matches the Nov holdout (0.556),
confirming the current hyperparameters are already reasonable for temporal
generalization. The massive 0.807 random CV AUC was entirely inflated by
within-month leakage. The model generalizes consistently (~0.554-0.560)
across all month boundaries.

### Action items

1. Drop 8 features flagged as temporal noise (EXP-003)
2. Re-tune hyperparameters under expanding-window CV (EXP-004)
3. Add temporally-stable new features (currency normalization, entity payment
   history, KYC interactions, geographic cross-features)

**Status:** Diagnostics complete. Proceeding to Phase 2.

---

## 2025-02-23 — Phase 2: Feature pruning and HP tuning (EXP-003 + EXP-004)

**Decision:** Tested two "foundation fix" hypotheses before adding new features.

### EXP-003: Feature pruning

Removed 8 features flagged as temporal noise (`tx_month`, 4 KYC scores,
`LOCK_NAME`, `LOCK_PRODUCT`, `clerk_id_order_count`). Result: Nov AUC 0.553,
*worse* than v2 baseline (0.556). The noise features contribute marginal
combined signal despite being individually weak.

### EXP-004: HP tuning via expanding-window CV

Searched 25 random configurations across 7 hyperparameters using 5-fold
expanding-window temporal CV. All configs clustered tightly in 0.552-0.556
Nov AUC. The v2 baseline params (`max_depth=4, lr=0.1, n_est=500, sub=0.9,
col=0.8, spw=5`) are already near-optimal.

**Key insight:** Neither feature pruning nor hyperparameter tuning can break
past ~0.556 with the current feature set. The ceiling is an information
constraint, not an optimization constraint. Improving AUC requires injecting
genuinely new information.

**Status:** Complete. Keeping v2 feature set and original hyperparameters.

---

## 2025-02-23 — Phase 3: New feature groups (EXP-005 to EXP-009)

**Decision:** Tested 4 new feature groups, each capturing a different risk
dimension, then combined them.

| Group | Features added | Nov AUC | Delta vs v2 |
|---|---|---|---|
| A — Currency norm | 4 country-level z-scores for financial amounts | 0.5566 | +0.0010 |
| B — Payment history | 6 entity-level overdue aggregates (mean/max) | 0.5558 | +0.0002 |
| C — KYC interactions | `kyc_min_score`, `perfect_kyc_flag` | 0.5563 | +0.0007 |
| D — Geo cross | `user_merchant_same_state` | 0.5561 | +0.0005 |
| **All combined** | **13 new features** | **0.5574** | **+0.0018** |

**Key insight:** All groups improve individually and compound when combined,
suggesting they capture orthogonal risk signals. The combined gain (+0.0018)
is the largest single-experiment improvement so far.

**Rationale for each group:**
- **Currency normalization:** Raw financial amounts span different currencies
  (MXN, BRL, PEN, etc.) so a "4000" loan means vastly different risk in
  different countries. Z-scoring within country makes amounts comparable.
- **Entity payment history:** Uses the Payment_History table to compute
  how overdue a merchant's/clerk's/admin's historical customers have been.
  This is genuinely available at prediction time.
- **KYC interactions:** Min-score captures the weakest verification link;
  the perfect flag captures suspiciously flawless KYC (potential spoofing).
- **Geographic cross:** Whether the user and merchant are in the same state;
  out-of-state transactions may carry different risk profiles.

**Status:** Best model is now EXP-009 with Nov AUC 0.5574. Not yet submitted;
awaiting user decision on submission slot usage.

---

## 2025-02-23 — Phase 4: LightGBM + NN + Ensemble implementation

**Decision:** Implement three-model ensemble (XGBoost + LightGBM + Neural Network)
with feature engineering upgrades.

### Feature engineering upgrades

| Change | Rationale |
|---|---|
| Smoothed (Bayesian) target encoding (m=50) for MERCHANTID, CLERK_ID, ADMINID, STATE | Raw means are noisy for rare entities; smoothing shrinks toward global mean. Evidence: smoothed ADMINID temporal AUC 0.5812 vs raw 0.5765. |
| CITY smoothed target encoding | 6,826 values with temporal AUC 0.5746 — a top-3 feature previously ignored. |
| MODEL smoothed target encoding | 606 values with temporal AUC 0.5478. FPD rates range 7.5%–14% across top models. |
| `days_to_first_due` feature | Temporal AUC 0.5427; available for all orders including December test set. |

### LightGBM implementation

- **Native categoricals**: LightGBM handles CITY (6,826 values), MODEL (606 values),
  MERCHANTID, CLERK_ID, ADMINID as native categoricals without OHE explosion.
- **`prep_data_lgbm`**: Parallel to `prep_data` but preserves category dtype instead
  of one-hot encoding. Returns DataFrames.
- **Expanding-window temporal CV**: 4-fold (train 1–6/val 7, ..., train 1–9/val 10)
  with 20 random hyperparameter configurations.

### Neural network updates

- Increased dropout from 0.3 to 0.4 (stronger regularization).
- Switched from BCELoss + Sigmoid to BCEWithLogitsLoss with pos_weight (numerically
  stable training with class imbalance handling).
- Added cosine annealing LR scheduler.
- Added early stopping (patience=5 epochs on val AUC).
- Replaced random 80/20 internal split with temporal split (caller provides X_val/y_val).
- Updated grid: `[256, 128]`, `[512, 256]`, `[128, 64, 32]` architectures; LR
  `[1e-3, 5e-4, 1e-4]`; epochs=50.

### Ensemble strategy

- Simple average of all three models' Nov predictions.
- Weighted grid search over XGB/LGBM/NN weights (step 0.1, min weight 0.05).
- Best configuration selected by Nov holdout AUC.

**Status:** All code implemented in `model_experiments.ipynb`. Awaiting execution
to obtain results. `update-pipeline` and `update-docs` deferred until results.
