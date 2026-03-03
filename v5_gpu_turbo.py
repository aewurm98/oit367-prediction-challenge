"""
v5_gpu_turbo.py
───────────────
GPU-accelerated ensemble: CatBoost (GPU) + LightGBM (GPU) + XGBoost (GPU)
Key upgrades over v5_mismatch:
  1. Time-based CV (train Jan-Oct, val Nov) — more honest AUC
  2. Expanded mismatch feature suite
  3. Order-level payment history aggregates (biggest single signal)
  4. Expanding FPD rates (leak-free temporal encoding)
  5. Tri-model ensemble with Optuna weight optimisation
  6. Deeper trees, more iterations — GPU makes this free
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

# ── GPU backends ──────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier, Pool
    HAS_CAT = True
    print("✅ CatBoost available")
except ImportError:
    HAS_CAT = False
    print("❌ CatBoost not found")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM available")
except ImportError:
    HAS_LGB = False
    print("❌ LightGBM not found")

try:
    import xgboost as xgb
    HAS_XGB = True
    print("✅ XGBoost available")
except ImportError:
    HAS_XGB = False
    print("❌ XGBoost not found")

if not any([HAS_CAT, HAS_LGB, HAS_XGB]):
    raise ImportError("Need at least one of: catboost, lightgbm, xgboost")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
orders   = pd.read_csv('Orders.csv', low_memory=False)
payments = pd.read_csv('Payment_History.csv')
test_ids = pd.read_csv('Test_OrderIDs.csv')

# Normalise payment column names
payments.columns = [c.upper().strip() for c in payments.columns]
_PMT_ALIASES = {
    'DAYSOVERDUE':         'DAYS_OVERDUE',
    'PRINCIPALCUMPAID':    'PRINCIPAL_CUMPAID',
    'TOTALBALANCE':        'TOTAL_BALANCE',
    'NUMBEROFPAYMENTS':    'NUMBER_OF_PAYMENTS',
    'ISAFTERCHARGEOFFDATE':'IS_AFTER_CHARGEOFF_DATE',
}
for alt, canonical in _PMT_ALIASES.items():
    if alt in payments.columns and canonical not in payments.columns:
        payments = payments.rename(columns={alt: canonical})

print(f"  Orders: {orders.shape}  Payments: {payments.shape}  Test: {len(test_ids):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PAYMENT HISTORY — both order-level AND market-level aggregates
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Building payment aggregates...")

ph_cols = payments.columns.tolist()
oid_col = next((c for c in ph_cols if 'ORDER' in c.upper() and 'ID' in c.upper()), ph_cols[0])
numeric_pay = [c for c in payments.select_dtypes(include=[np.number]).columns if c != oid_col]

# ── 2a. Order-level aggregates (most predictive signal) ──────────────────────
order_pay_agg = payments.groupby(oid_col)[numeric_pay].agg(['mean', 'max', 'min', 'std', 'last'])
order_pay_agg.columns = [f'ord_pay_{a}_{b}' for a, b in order_pay_agg.columns]
order_pay_agg = order_pay_agg.reset_index().rename(columns={oid_col: 'FINANCEORDERID'})

# Payment-specific derived features (if columns exist)
pay_derived = payments.groupby(oid_col).agg(
    pay_record_count=(oid_col, 'count'),
).reset_index().rename(columns={oid_col: 'FINANCEORDERID'})

if 'DAYS_OVERDUE' in payments.columns:
    overdue_agg = payments.groupby(oid_col)['DAYS_OVERDUE'].agg(
        pay_max_overdue='max',
        pay_mean_overdue='mean',
        pay_any_overdue=lambda x: (x > 0).any().astype(int),
        pay_times_overdue=lambda x: (x > 0).sum(),
    ).reset_index().rename(columns={oid_col: 'FINANCEORDERID'})
    pay_derived = pay_derived.merge(overdue_agg, on='FINANCEORDERID', how='left')

if 'PRINCIPAL_CUMPAID' in payments.columns and 'TOTAL_BALANCE' in payments.columns:
    payments['pay_completion_ratio'] = (
        payments['PRINCIPAL_CUMPAID'] /
        (payments['PRINCIPAL_CUMPAID'] + payments['TOTAL_BALANCE']).replace(0, np.nan)
    )
    completion_agg = payments.groupby(oid_col)['pay_completion_ratio'].agg(
        pay_completion_mean='mean',
        pay_completion_last='last',
    ).reset_index().rename(columns={oid_col: 'FINANCEORDERID'})
    pay_derived = pay_derived.merge(completion_agg, on='FINANCEORDERID', how='left')

# ── 2b. Market-level aggregates ──────────────────────────────────────────────
pay_ctx = payments.merge(
    orders[['FINANCEORDERID', 'COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']].drop_duplicates(),
    left_on=oid_col, right_on='FINANCEORDERID', how='left'
)

market_pay_stats = {}
for grp in ['COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']:
    if grp in pay_ctx.columns and len(numeric_pay) > 0:
        stats = pay_ctx.groupby(grp)[numeric_pay].agg(['mean', 'std'])
        stats.columns = [f'{grp}_pay_{a}_{b}' for a, b in stats.columns]
        market_pay_stats[grp] = stats.reset_index()

print(f"  Order-level pay features: {len(order_pay_agg.columns)-1}")
print(f"  Market pay stats: {list(market_pay_stats.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Engineering features...")

for col in ['TRANSACTIONTIME', 'FIRST_PAYMENT_DUE_TIMESTAMP',
            'MERCHANT_FIRST_SALE_DATE', 'MERCHANT_LAST_SALE_DATE']:
    orders[col] = pd.to_datetime(orders[col], utc=True, errors='coerce')

# ── Time ─────────────────────────────────────────────────────────────────────
orders['txn_hour']       = orders['TRANSACTIONTIME'].dt.hour
orders['txn_dow']        = orders['TRANSACTIONTIME'].dt.dayofweek
orders['txn_month']      = orders['TRANSACTIONTIME'].dt.month
orders['txn_day']        = orders['TRANSACTIONTIME'].dt.day
orders['txn_year']       = orders['TRANSACTIONTIME'].dt.year
orders['txn_week']       = orders['TRANSACTIONTIME'].dt.isocalendar().week.astype(int)
orders['is_weekend']     = (orders['txn_dow'] >= 5).astype(int)
orders['is_night']       = ((orders['txn_hour'] >= 22) | (orders['txn_hour'] <= 5)).astype(int)
orders['is_month_end']   = (orders['txn_day'] >= 25).astype(int)
orders['is_month_start'] = (orders['txn_day'] <= 5).astype(int)
orders['is_payday']      = orders['txn_day'].isin([1, 15, 28, 29, 30, 31]).astype(int)
orders['hour_sin']       = np.sin(2 * np.pi * orders['txn_hour'] / 24)
orders['hour_cos']       = np.cos(2 * np.pi * orders['txn_hour'] / 24)
orders['dow_sin']        = np.sin(2 * np.pi * orders['txn_dow'] / 7)
orders['dow_cos']        = np.cos(2 * np.pi * orders['txn_dow'] / 7)
orders['month_sin']      = np.sin(2 * np.pi * orders['txn_month'] / 12)
orders['month_cos']      = np.cos(2 * np.pi * orders['txn_month'] / 12)

# ── Finance ───────────────────────────────────────────────────────────────────
for col in ['TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT', 'FINANCE_AMOUNT', 'PURCHASE_AMOUNT']:
    orders[col] = pd.to_numeric(orders[col], errors='coerce')

orders['down_payment_ratio']    = ((orders['PURCHASE_AMOUNT'] - orders['FINANCE_AMOUNT']) / orders['PURCHASE_AMOUNT'].replace(0, np.nan)).fillna(0).clip(0, 1)
orders['finance_ratio']         = (orders['FINANCE_AMOUNT'] / orders['PURCHASE_AMOUNT'].replace(0, np.nan)).fillna(1).clip(0, 1)
orders['log_finance_amount']    = np.log1p(orders['FINANCE_AMOUNT'].clip(lower=0))
orders['log_purchase_amount']   = np.log1p(orders['PURCHASE_AMOUNT'].clip(lower=0))
orders['log_total_due']         = np.log1p(orders['TOTAL_DUE'].clip(lower=0))
orders['total_due_ratio']       = (orders['TOTAL_DUE'] / orders['FINANCE_AMOUNT'].replace(0, np.nan)).fillna(1).clip(0, 5)
orders['days_to_first_payment'] = (orders['FIRST_PAYMENT_DUE_TIMESTAMP'] - orders['TRANSACTIONTIME']).dt.total_seconds() / 86400
orders['log_days_to_first_pay'] = np.log1p(orders['days_to_first_payment'].clip(lower=0))

# ── Merchant tenure ───────────────────────────────────────────────────────────
orders['merchant_tenure_days']  = (orders['TRANSACTIONTIME'] - orders['MERCHANT_FIRST_SALE_DATE']).dt.total_seconds() / 86400
orders['merchant_tenure_days']  = orders['merchant_tenure_days'].fillna(0).clip(lower=0)
orders['log_merchant_tenure']   = np.log1p(orders['merchant_tenure_days'])
orders['is_new_merchant']       = (orders['merchant_tenure_days'] < 90).astype(float)
orders['is_very_new_merchant']  = (orders['merchant_tenure_days'] < 30).astype(float)
orders['days_since_last_sale']  = (orders['TRANSACTIONTIME'] - orders['MERCHANT_LAST_SALE_DATE']).dt.total_seconds() / 86400
orders['days_since_last_sale']  = orders['days_since_last_sale'].fillna(0).clip(lower=0)
orders['merchant_dormancy']     = np.log1p(orders['days_since_last_sale'])

# ── KYC scores ────────────────────────────────────────────────────────────────
score_cols = ['FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE', 'LIVENESS_SCORE', 'OVERALL_SCORE']
for c in score_cols:
    orders[c] = pd.to_numeric(orders[c], errors='coerce')

orders['kyc_score_mean']         = orders[score_cols].mean(axis=1)
orders['kyc_score_min']          = orders[score_cols].min(axis=1)
orders['kyc_score_std']          = orders[score_cols].std(axis=1)
orders['kyc_score_range']        = orders[score_cols].max(axis=1) - orders[score_cols].min(axis=1)
orders['low_face_score']         = (orders['FACE_RECOGNITION_SCORE'] < 70).astype(float)
orders['low_id_score']           = (orders['IDVALIDATION_OVERALL_SCORE'] < 80).astype(float)
orders['any_low_score']          = (orders[score_cols] < 75).any(axis=1).astype(float)
orders['all_scores_perfect']     = (orders[score_cols] == 100).all(axis=1).astype(float)
orders['high_finance_low_kyc']   = orders['finance_ratio'] * (100 - orders['kyc_score_mean']) / 100
orders['kyc_composite']          = (orders['kyc_score_mean'] * orders['kyc_score_min']) / 10000  # penalises weak links

# ── MISMATCH FEATURES ─────────────────────────────────────────────────────────

# 1. KYC: face vs liveness gap
orders['face_liveness_gap']        = (orders['FACE_RECOGNITION_SCORE'] - orders['LIVENESS_SCORE']).abs()
orders['face_liveness_gap_signed'] = (orders['FACE_RECOGNITION_SCORE'] - orders['LIVENESS_SCORE'])
gap_p75 = orders['face_liveness_gap'].quantile(0.75)
orders['face_liveness_gap_flag']   = (orders['face_liveness_gap'] > gap_p75).astype(int)

# Face high, liveness low — classic spoof pattern
orders['high_face_low_liveness']   = (
    (orders['FACE_RECOGNITION_SCORE'] > 85) & (orders['LIVENESS_SCORE'] < 70)
).astype(int)

# ID vs overall inconsistency
orders['id_vs_overall_gap']        = (orders['IDVALIDATION_OVERALL_SCORE'] - orders['OVERALL_SCORE']).abs()
orders['perfect_id_poor_liveness'] = (
    (orders['IDVALIDATION_OVERALL_SCORE'] == 100) & (orders['LIVENESS_SCORE'] < 70)
).astype(int)

# 2. Finance: z-score vs country norm (train-only to avoid leakage)
_test_order_ids = set(test_ids['FINANCEORDERID'].astype(str))
_train_mask = (orders['txn_year'] == 2025) & orders['FPD_15'].notna() & ~orders['FINANCEORDERID'].astype(str).isin(_test_order_ids)
_train_orders = orders[_train_mask]
_country_stats = _train_orders.groupby('COUNTRY')['FINANCE_AMOUNT'].agg(['mean', 'std']).reset_index()
_country_stats.columns = ['COUNTRY', 'fin_mu', 'fin_sig']
_country_stats['fin_sig'] = _country_stats['fin_sig'].replace(0, 1)
orders = orders.merge(_country_stats, on='COUNTRY', how='left')
orders['finance_amount_country_zscore'] = ((orders['FINANCE_AMOUNT'] - orders['fin_mu']) / orders['fin_sig']).fillna(0).clip(-5, 5)
orders = orders.drop(columns=['fin_mu', 'fin_sig'])
orders['finance_amount_country_high']   = (orders['finance_amount_country_zscore'] > 2).astype(int)

# Also z-score vs manufacturer norm (train-only)
_mfr_stats = _train_orders.groupby('MANUFACTURER')['FINANCE_AMOUNT'].agg(['mean', 'std']).reset_index()
_mfr_stats.columns = ['MANUFACTURER', 'fin_mu', 'fin_sig']
_mfr_stats['fin_sig'] = _mfr_stats['fin_sig'].replace(0, 1)
orders = orders.merge(_mfr_stats, on='MANUFACTURER', how='left')
orders['finance_amount_mfr_zscore'] = ((orders['FINANCE_AMOUNT'] - orders['fin_mu']) / orders['fin_sig']).fillna(0).clip(-5, 5)
orders = orders.drop(columns=['fin_mu', 'fin_sig'])

# 3. Merchant anomalies
orders['new_merchant_high_value']   = ((orders['is_new_merchant'] == 1) & (orders['finance_amount_country_zscore'] > 1)).astype(int)
orders['dormant_merchant_big_loan'] = ((orders['days_since_last_sale'] > 30) & (orders['finance_amount_country_zscore'] > 1)).astype(int)

# 4. Timing anomalies (train-only)
_fp_stats = _train_orders.groupby('COUNTRY')['days_to_first_payment'].agg(['mean', 'std']).reset_index()
_fp_stats.columns = ['COUNTRY', 'fp_mu', 'fp_sig']
_fp_stats['fp_sig'] = _fp_stats['fp_sig'].replace(0, 1)
orders = orders.merge(_fp_stats, on='COUNTRY', how='left')
orders['first_payment_gap_zscore'] = ((orders['days_to_first_payment'] - orders['fp_mu']) / orders['fp_sig']).fillna(0).clip(-5, 5)
orders = orders.drop(columns=['fp_mu', 'fp_sig'])

# 5. Geography mismatch (already in data)
# user_merchant_state_mismatch assumed present

# ── Interaction features ──────────────────────────────────────────────────────
orders['kyc_x_finance_zscore']   = orders['kyc_score_mean'] * orders['finance_amount_country_zscore']
orders['mismatch_score']         = (                          # composite risk flag
    orders['face_liveness_gap_flag'].fillna(0) +
    orders['finance_amount_country_high'].fillna(0) +
    orders.get('user_merchant_state_mismatch', pd.Series(0, index=orders.index)).fillna(0)
)
orders['new_merchant_night_txn'] = (orders['is_new_merchant'] * orders['is_night'])
orders['high_finance_new_merch'] = (orders['finance_ratio'] * orders['is_new_merchant'])

# ── Expanding (leak-free) FPD rates ──────────────────────────────────────────
print("  Building expanding FPD rates...")
orders_sorted = orders.sort_values('TRANSACTIONTIME').copy()

for grp_col in ['COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']:
    if grp_col not in orders_sorted.columns:
        continue
    # Shift by 1 so we only use past data
    grp = orders_sorted.groupby(grp_col)['FPD_15']
    orders_sorted[f'{grp_col}_expanding_fpd_rate'] = (
        grp.expanding().mean().shift(1)
           .reset_index(level=0, drop=True)
    )
    orders_sorted[f'{grp_col}_expanding_fpd_count'] = (
        grp.expanding().count().shift(1)
           .reset_index(level=0, drop=True)
    )

orders = orders_sorted.copy()

# ── Merge payment aggregates ──────────────────────────────────────────────────
orders = orders.merge(order_pay_agg, on='FINANCEORDERID', how='left')
orders = orders.merge(pay_derived,   on='FINANCEORDERID', how='left')
for grp, stats_df in market_pay_stats.items():
    orders = orders.merge(stats_df, on=grp, how='left')

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT — time-based (more honest than random CV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Splitting data...")

test_order_ids = set(test_ids['FINANCEORDERID'].astype(str))
orders['FINANCEORDERID_str'] = orders['FINANCEORDERID'].astype(str)
orders = orders[orders['txn_year'] == 2025].copy()

all_train = orders[orders['FPD_15'].notna() & ~orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()
test_df   = orders[orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()

# Time-based val: last month = November (adjust if your data differs)
val_month    = all_train['txn_month'].max()
train_df     = all_train[all_train['txn_month'] < val_month].copy()
val_df       = all_train[all_train['txn_month'] == val_month].copy()

print(f"  Train: {len(train_df):,}  Val (month {val_month}): {len(val_df):,}  Test: {len(test_df):,}")
print(f"  FPD rate — train: {train_df['FPD_15'].mean():.3%}  val: {val_df['FPD_15'].mean():.3%}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature list
# ─────────────────────────────────────────────────────────────────────────────
base_features = [
    # Finance
    'FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT',
    'log_finance_amount', 'log_purchase_amount', 'log_total_due',
    'down_payment_ratio', 'finance_ratio', 'total_due_ratio',
    'log_days_to_first_pay', 'days_to_first_payment',
    # Time (raw + cyclical)
    'txn_hour', 'txn_dow', 'txn_month', 'txn_day', 'txn_week',
    'is_weekend', 'is_night', 'is_month_end', 'is_month_start', 'is_payday',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Merchant
    'merchant_tenure_days', 'log_merchant_tenure',
    'is_new_merchant', 'is_very_new_merchant',
    'days_since_last_sale', 'merchant_dormancy',
    # KYC
    'FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE', 'LIVENESS_SCORE', 'OVERALL_SCORE',
    'kyc_score_mean', 'kyc_score_min', 'kyc_score_std', 'kyc_score_range',
    'kyc_composite', 'low_face_score', 'low_id_score', 'any_low_score',
    'all_scores_perfect', 'high_finance_low_kyc',
    # Mismatch — KYC
    'face_liveness_gap', 'face_liveness_gap_signed', 'face_liveness_gap_flag',
    'high_face_low_liveness', 'id_vs_overall_gap', 'perfect_id_poor_liveness',
    # Mismatch — Finance
    'finance_amount_country_zscore', 'finance_amount_country_high',
    'finance_amount_mfr_zscore',
    # Mismatch — Merchant/timing
    'new_merchant_high_value', 'dormant_merchant_big_loan',
    'first_payment_gap_zscore',
    # Geography mismatch
    'user_merchant_state_mismatch',
    # Interactions
    'kyc_x_finance_zscore', 'mismatch_score',
    'new_merchant_night_txn', 'high_finance_new_merch',
    # Expanding FPD rates
    'COUNTRY_expanding_fpd_rate',   'COUNTRY_expanding_fpd_count',
    'MANUFACTURER_expanding_fpd_rate', 'MANUFACTURER_expanding_fpd_count',
    'MERCHANT_STATE_expanding_fpd_rate', 'MERCHANT_STATE_expanding_fpd_count',
]

# Order-level pay features
for col in order_pay_agg.columns:
    if col != 'FINANCEORDERID':
        base_features.append(col)
for col in pay_derived.columns:
    if col != 'FINANCEORDERID':
        base_features.append(col)

# Market pay stats
for stats_df in market_pay_stats.values():
    for col in stats_df.columns:
        if col not in ['COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']:
            base_features.append(col)

CATEGORICAL_ENCODE_COLS = [c for c in [
    'COUNTRY', 'USER_STATE', 'MERCHANT_STATE', 'MANUFACTURER',
    'LOCK_NAME', 'LOCK_PRODUCT', 'CURRENCY',
] if c in train_df.columns]

feature_cols = list(dict.fromkeys([c for c in base_features if c in train_df.columns]))
print(f"  Total features: {len(feature_cols)}  |  Categoricals: {len(CATEGORICAL_ENCODE_COLS)}")

# ─────────────────────────────────────────────────────────────────────────────
# Target encode on full train, apply to val + test
# ─────────────────────────────────────────────────────────────────────────────
def target_encode(train_df, apply_dfs, col, target='FPD_15', k=20):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['sum', 'count'])
    stats['encoded'] = (stats['sum'] + k * global_mean) / (stats['count'] + k)
    mapping = stats['encoded'].to_dict()
    enc_col = f'{col}_fpd_rate'
    train_encoded = train_df[col].map(mapping).fillna(global_mean).values
    applied = [df[col].map(mapping).fillna(global_mean).values for df in apply_dfs]
    return enc_col, train_encoded, applied

X_tr = train_df[feature_cols].copy()
X_val = val_df[feature_cols].copy()
X_te = test_df[feature_cols].copy()

for col in CATEGORICAL_ENCODE_COLS:
    enc_col, tr_enc, (val_enc, te_enc) = target_encode(train_df, [val_df, test_df], col)
    X_tr[enc_col]  = tr_enc
    X_val[enc_col] = val_enc
    X_te[enc_col]  = te_enc

# Fill NaN with median from train
for col in X_tr.columns:
    med = X_tr[col].median()
    X_tr[col]  = X_tr[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)
    X_te[col]  = X_te[col].fillna(med)

y_tr  = train_df['FPD_15'].astype(int).values
y_val = val_df['FPD_15'].astype(int).values

n_pos = y_tr.sum()
n_neg = len(y_tr) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN MODELS (GPU)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Training models on GPU...")

val_preds  = {}
test_preds = {}

# ── CatBoost GPU ─────────────────────────────────────────────────────────────
if HAS_CAT:
    print("  Training CatBoost (GPU)...")
    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,         # lower LR → more iterations → better
        depth=8,                    # deeper trees on GPU
        l2_leaf_reg=3,
        bagging_temperature=0.8,
        random_strength=1.2,
        border_count=254,           # max bins for GPU
        scale_pos_weight=scale_pos_weight,
        eval_metric='AUC',
        early_stopping_rounds=100,
        random_seed=42,
        verbose=100,
        task_type='GPU',
        devices='0',
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    val_preds['cat']  = cat_model.predict_proba(X_val)[:, 1]
    test_preds['cat'] = cat_model.predict_proba(X_te)[:, 1]
    print(f"  CatBoost val AUC: {roc_auc_score(y_val, val_preds['cat']):.5f}")

# ── LightGBM GPU (or CPU when V5_LGB_USE_CPU=1, e.g. Modal T4 has CUDA not OpenCL) ─
if HAS_LGB:
    import os as _os
    _lgb_device = 'cpu' if _os.environ.get('V5_LGB_USE_CPU', '0') in ('1', 'true', 'yes') else 'gpu'
    print(f"  Training LightGBM ({_lgb_device.upper()})...")
    lgb_params = {
        'objective':         'binary',
        'metric':            'auc',
        'learning_rate':     0.02,
        'num_leaves':        255,       # large — GPU handles it
        'max_depth':         -1,
        'min_child_samples': 30,
        'feature_fraction':  0.7,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'lambda_l1':         0.1,
        'lambda_l2':         1.0,
        'scale_pos_weight':  scale_pos_weight,
        'device':            _lgb_device,
        'verbose':           -1,
        'n_jobs':            -1,
    }
    if _lgb_device == 'gpu':
        lgb_params['gpu_platform_id'] = 0
        lgb_params['gpu_device_id'] = 0
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    lgb_model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(200),
        ]
    )
    val_preds['lgb']  = lgb_model.predict(X_val)
    test_preds['lgb'] = lgb_model.predict(X_te)
    print(f"  LightGBM val AUC: {roc_auc_score(y_val, val_preds['lgb']):.5f}")

# ── XGBoost GPU ──────────────────────────────────────────────────────────────
if HAS_XGB:
    print("  Training XGBoost (GPU)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        device='cuda',
        eval_metric='auc',
        early_stopping_rounds=100,
        random_state=42,
        verbosity=1,
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    val_preds['xgb']  = xgb_model.predict_proba(X_val)[:, 1]
    test_preds['xgb'] = xgb_model.predict_proba(X_te)[:, 1]
    print(f"  XGBoost val AUC:  {roc_auc_score(y_val, val_preds['xgb']):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble weight optimisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Optimising ensemble weights on val set...")

model_keys  = list(val_preds.keys())
val_stack   = np.column_stack([val_preds[k] for k in model_keys])
test_stack  = np.column_stack([test_preds[k] for k in model_keys])

def neg_auc(w):
    w = np.array(w)
    w = np.abs(w) / np.abs(w).sum()
    blended = val_stack @ w
    return -roc_auc_score(y_val, blended)

from scipy.optimize import minimize
n = len(model_keys)
result = minimize(
    neg_auc,
    x0=np.ones(n) / n,
    method='Nelder-Mead',
    options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6},
)
opt_w = np.abs(result.x) / np.abs(result.x).sum()
print(f"  Optimised weights: { {k: f'{w:.3f}' for k, w in zip(model_keys, opt_w)} }")

final_val_pred  = val_stack  @ opt_w
final_test_pred = test_stack @ opt_w
val_auc = roc_auc_score(y_val, final_val_pred)
print(f"  Ensemble val AUC: {val_auc:.5f}")

# Also try equal weights as sanity check
equal_auc = roc_auc_score(y_val, val_stack.mean(axis=1))
print(f"  Equal-weight AUC: {equal_auc:.5f}")

# Use whichever is better
if equal_auc > val_auc:
    print("  → Using equal weights (better on val)")
    final_test_pred = test_stack.mean(axis=1)
    val_auc = equal_auc

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Saving submission...")

submission = pd.DataFrame({
    'FINANCEORDERID': test_df['FINANCEORDERID'].astype(str),
    'FPD_15_pred':    final_test_pred,
})
submission = submission.sort_values('FINANCEORDERID').reset_index(drop=True)
submission.to_csv('submission_v5_gpu_turbo.csv', index=False)

assert not submission['FINANCEORDERID'].duplicated().any(), "Duplicate IDs!"
assert submission['FPD_15_pred'].between(0, 1).all(), "Predictions out of range!"
missing = set(test_ids['FINANCEORDERID'].astype(str)) - set(submission['FINANCEORDERID'])
assert not missing, f"Missing {len(missing)} IDs!"

print(f"  Saved → submission_v5_gpu_turbo.csv")
print(f"\n{'='*60}")
print(f"  MODELS:   {model_keys}")
print(f"  WEIGHTS:  { {k: f'{w:.3f}' for k, w in zip(model_keys, opt_w)} }")
print(f"  VAL AUC:  {val_auc:.5f}   (previous best: 0.609)")
print(f"  FEATURES: {len(X_tr.columns)}")
print(f"  TARGET:   Beat 0.623 OOF / 0.609 Kaggle")
print(f"{'='*60}")
