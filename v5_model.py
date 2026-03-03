import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Try CatBoost first, then LightGBM, then XGBoost
MODEL_BACKEND = None
try:
    from catboost import CatBoostClassifier
    MODEL_BACKEND = 'catboost'
    print("✅ Using CatBoost (best for this problem)")
except ImportError:
    pass

if MODEL_BACKEND is None:
    try:
        import lightgbm as lgb
        MODEL_BACKEND = 'lgb'
        print("Using LightGBM")
    except ImportError:
        pass

if MODEL_BACKEND is None:
    raise ImportError("Install catboost: pip install catboost  OR  pip install lightgbm")

print(f"Backend: {MODEL_BACKEND}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")
orders   = pd.read_csv('Orders.csv', low_memory=False)
payments = pd.read_csv('Payment_History.csv')
test_ids = pd.read_csv('Test_OrderIDs.csv')

print(f"  Orders: {orders.shape}  Payments: {payments.shape}  Test: {len(test_ids):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PAYMENT HISTORY — market-level stats only (same as original)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Building market-level payment track record...")

ph_cols     = payments.columns.tolist()
oid_col     = next((c for c in ph_cols if 'ORDER' in c.upper() and 'ID' in c.upper()), ph_cols[0])
numeric_pay = [c for c in payments.select_dtypes(include=[np.number]).columns if c != oid_col]

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

print(f"  Payment stats: {list(market_pay_stats.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Engineering features...")

orders['TRANSACTIONTIME'] = pd.to_datetime(orders['TRANSACTIONTIME'], utc=True, errors='coerce')
orders['FIRST_PAYMENT_DUE_TIMESTAMP'] = pd.to_datetime(
    orders['FIRST_PAYMENT_DUE_TIMESTAMP'], utc=True, errors='coerce')
orders['MERCHANT_FIRST_SALE_DATE'] = pd.to_datetime(
    orders['MERCHANT_FIRST_SALE_DATE'], utc=True, errors='coerce')
orders['MERCHANT_LAST_SALE_DATE'] = pd.to_datetime(
    orders['MERCHANT_LAST_SALE_DATE'], utc=True, errors='coerce')

# Time
orders['txn_hour']       = orders['TRANSACTIONTIME'].dt.hour
orders['txn_dow']        = orders['TRANSACTIONTIME'].dt.dayofweek
orders['txn_month']      = orders['TRANSACTIONTIME'].dt.month
orders['txn_day']        = orders['TRANSACTIONTIME'].dt.day
orders['txn_year']       = orders['TRANSACTIONTIME'].dt.year
orders['is_weekend']     = (orders['txn_dow'] >= 5).astype(int)
orders['is_night']       = ((orders['txn_hour'] >= 22) | (orders['txn_hour'] <= 5)).astype(int)
orders['is_month_end']   = (orders['txn_day'] >= 25).astype(int)
orders['is_month_start'] = (orders['txn_day'] <= 5).astype(int)

# Finance
orders['TOTAL_DUE'] = pd.to_numeric(orders['TOTAL_DUE'], errors='coerce')
orders['DOWN_PAYMENT_AMOUNT'] = pd.to_numeric(orders['DOWN_PAYMENT_AMOUNT'], errors='coerce')
orders['down_payment_ratio'] = (
    (orders['PURCHASE_AMOUNT'] - orders['FINANCE_AMOUNT'])
    / orders['PURCHASE_AMOUNT'].replace(0, np.nan)
).fillna(0).clip(0, 1)
orders['finance_ratio'] = (
    orders['FINANCE_AMOUNT'] / orders['PURCHASE_AMOUNT'].replace(0, np.nan)
).fillna(1).clip(0, 1)
orders['log_finance_amount']  = np.log1p(orders['FINANCE_AMOUNT'].clip(lower=0))
orders['log_purchase_amount'] = np.log1p(orders['PURCHASE_AMOUNT'].clip(lower=0))
orders['log_total_due']       = np.log1p(orders['TOTAL_DUE'].clip(lower=0))
orders['total_due_ratio']     = (
    orders['TOTAL_DUE'] / orders['FINANCE_AMOUNT'].replace(0, np.nan)
).fillna(1).clip(0, 5)
orders['days_to_first_payment'] = (
    orders['FIRST_PAYMENT_DUE_TIMESTAMP'] - orders['TRANSACTIONTIME']
).dt.total_seconds() / 86400

# Merchant tenure
orders['merchant_tenure_days'] = (
    orders['TRANSACTIONTIME'] - orders['MERCHANT_FIRST_SALE_DATE']
).dt.total_seconds() / 86400
orders['merchant_tenure_days']  = orders['merchant_tenure_days'].fillna(0).clip(lower=0)
orders['log_merchant_tenure']   = np.log1p(orders['merchant_tenure_days'])
orders['is_new_merchant']       = (orders['merchant_tenure_days'] < 90).astype(float)
orders['days_since_last_sale']  = (
    orders['TRANSACTIONTIME'] - orders['MERCHANT_LAST_SALE_DATE']
).dt.total_seconds() / 86400
orders['days_since_last_sale']  = orders['days_since_last_sale'].fillna(0).clip(lower=0)

# KYC
score_cols = ['FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
              'LIVENESS_SCORE', 'OVERALL_SCORE']
for c in score_cols:
    orders[c] = pd.to_numeric(orders[c], errors='coerce')

orders['kyc_score_mean']     = orders[score_cols].mean(axis=1)
orders['kyc_score_min']      = orders[score_cols].min(axis=1)
orders['kyc_score_std']      = orders[score_cols].std(axis=1)
orders['kyc_score_range']    = orders[score_cols].max(axis=1) - orders[score_cols].min(axis=1)
orders['low_face_score']     = (orders['FACE_RECOGNITION_SCORE'] < 70).astype(float)
orders['low_id_score']       = (orders['IDVALIDATION_OVERALL_SCORE'] < 80).astype(float)
orders['any_low_score']      = (orders[score_cols] < 75).any(axis=1).astype(float)
orders['all_scores_perfect'] = (orders[score_cols] == 100).all(axis=1).astype(float)
orders['high_finance_low_kyc'] = orders['finance_ratio'] * (100 - orders['kyc_score_mean']) / 100

# ─────────────────────────────────────────────────────────────────────────────
# ✅ NEW: MISMATCH FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# 1. KYC mismatch: face vs liveness gap
#    Large gap suggests spoofing — face photo passes but liveness check diverges
orders['face_liveness_gap'] = (
    orders['FACE_RECOGNITION_SCORE'] - orders['LIVENESS_SCORE']
).abs()

# Directional version: positive = face higher than liveness (more suspicious)
orders['face_liveness_gap_signed'] = (
    orders['FACE_RECOGNITION_SCORE'] - orders['LIVENESS_SCORE']
)

# Flag extreme gaps (top quartile of divergence)
gap_threshold = orders['face_liveness_gap'].quantile(0.75)
orders['face_liveness_gap_flag'] = (
    orders['face_liveness_gap'] > gap_threshold
).astype(int)

# 2. Finance mismatch: loan amount vs country-level norm (z-score)
#    Unusually large loans relative to the country's typical deal size
market_finance_mean = orders.groupby('COUNTRY')['FINANCE_AMOUNT'].transform('mean')
market_finance_std  = orders.groupby('COUNTRY')['FINANCE_AMOUNT'].transform('std')
orders['finance_amount_country_zscore'] = (
    (orders['FINANCE_AMOUNT'] - market_finance_mean)
    / market_finance_std.replace(0, np.nan)
).fillna(0).clip(-5, 5)  # clip extreme outliers

# Flag loans more than 2 std above country mean
orders['finance_amount_country_high'] = (
    orders['finance_amount_country_zscore'] > 2
).astype(int)

# Interaction: high z-score loan AND new merchant (double risk signal)
orders['new_merchant_high_value'] = (
    (orders['is_new_merchant'] == 1) &
    (orders['finance_amount_country_zscore'] > 1)
).astype(int)

print("  ✅ Mismatch features added: face_liveness_gap, finance_amount_country_zscore")

# Merge market payment stats
for grp, stats_df in market_pay_stats.items():
    orders = orders.merge(stats_df, on=grp, how='left')

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT
# ─────────────────────────────────────────────────────────────────────────────
test_order_ids = set(test_ids['FINANCEORDERID'].astype(str))
orders['FINANCEORDERID_str'] = orders['FINANCEORDERID'].astype(str)
orders = orders[orders['txn_year'] == 2025].copy()

train_df = orders[orders['FPD_15'].notna() & ~orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()
test_df  = orders[orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()

print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}  FPD rate: {train_df['FPD_15'].mean():.3%}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature list
# ─────────────────────────────────────────────────────────────────────────────
base_features = [
    # Finance
    'FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT',
    'log_finance_amount', 'log_purchase_amount', 'log_total_due',
    'down_payment_ratio', 'finance_ratio', 'total_due_ratio',
    # Time
    'txn_hour', 'txn_dow', 'txn_month', 'txn_day',
    'is_weekend', 'is_night', 'is_month_end', 'is_month_start',
    'days_to_first_payment',
    # Merchant tenure
    'merchant_tenure_days', 'log_merchant_tenure',
    'is_new_merchant', 'days_since_last_sale',
    # KYC
    'FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
    'LIVENESS_SCORE', 'OVERALL_SCORE',
    'kyc_score_mean', 'kyc_score_min', 'kyc_score_std', 'kyc_score_range',
    'low_face_score', 'low_id_score', 'any_low_score',
    'all_scores_perfect', 'high_finance_low_kyc',
    # ✅ NEW: Mismatch features
    'face_liveness_gap',               # absolute KYC divergence
    'face_liveness_gap_signed',        # directional (face > liveness = suspicious)
    'face_liveness_gap_flag',          # binary flag: top-quartile gap
    'finance_amount_country_zscore',   # loan size vs country norm
    'finance_amount_country_high',     # binary flag: 2+ std above norm
    'new_merchant_high_value',         # interaction: new merchant + large loan
    # Geography mismatch (already present in data per user confirmation)
    'user_merchant_state_mismatch',
]

# Add market payment stats
for stats_df in market_pay_stats.values():
    for col in stats_df.columns:
        if col not in ['COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']:
            base_features.append(col)

CATEGORICAL_ENCODE_COLS = [
    'COUNTRY', 'USER_STATE', 'MERCHANT_STATE', 'MANUFACTURER',
    'LOCK_NAME',
    'LOCK_PRODUCT',
    'CURRENCY',
]
CATEGORICAL_ENCODE_COLS = [c for c in CATEGORICAL_ENCODE_COLS if c in train_df.columns]

feature_cols = [c for c in base_features if c in train_df.columns]
feature_cols = list(dict.fromkeys(feature_cols))
print(f"  Numeric features: {len(feature_cols)}  |  Cat encode: {CATEGORICAL_ENCODE_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def target_encode_fold(train_fold, val_fold, test_fold, col, target='FPD_15', k=20):
    global_mean = train_fold[target].mean()
    stats = train_fold.groupby(col)[target].agg(['sum', 'count'])
    stats['encoded'] = (stats['sum'] + k * global_mean) / (stats['count'] + k)
    mapping = stats['encoded'].to_dict()
    return (
        train_fold[col].map(mapping).fillna(global_mean).values,
        val_fold[col].map(mapping).fillna(global_mean).values,
        test_fold[col].map(mapping).fillna(global_mean).values,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. 5-FOLD STRATIFIED CV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training with 5-fold CV using {MODEL_BACKEND}...")

y_train_all      = train_df['FPD_15'].astype(int).values
n_pos            = y_train_all.sum()
n_neg            = len(y_train_all) - n_pos
scale_pos_weight = n_neg / n_pos

kf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
fold_aucs  = []

X_all_base  = train_df[feature_cols].copy()
X_test_base = test_df[feature_cols].copy()

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all_base, y_train_all), 1):
    tr_df_fold  = train_df.iloc[tr_idx]
    val_df_fold = train_df.iloc[val_idx]

    X_tr  = X_all_base.iloc[tr_idx].copy()
    X_val = X_all_base.iloc[val_idx].copy()
    X_te  = X_test_base.copy()
    y_tr  = y_train_all[tr_idx]
    y_val = y_train_all[val_idx]

    for col in CATEGORICAL_ENCODE_COLS:
        if col in train_df.columns:
            enc_col = f'{col}_fpd_rate'
            X_tr[enc_col], X_val[enc_col], X_te[enc_col] = target_encode_fold(
                tr_df_fold, val_df_fold, test_df, col)

    for col in X_tr.columns:
        med = X_tr[col].median()
        X_tr[col]  = X_tr[col].fillna(med)
        X_val[col] = X_val[col].fillna(med)
        X_te[col]  = X_te[col].fillna(med)

    if MODEL_BACKEND == 'catboost':
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
            task_type='CPU',
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        val_pred  = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_te)[:, 1]

    else:  # LightGBM fallback
        params = {
            'objective': 'binary', 'metric': 'auc',
            'learning_rate': 0.05, 'num_leaves': 63,
            'min_child_samples': 50, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1, 'n_jobs': -1,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model  = lgb.train(params, dtrain, num_boost_round=1000,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(50, verbose=False),
                                      lgb.log_evaluation(False)])
        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_te)

    auc = roc_auc_score(y_val, val_pred)
    fold_aucs.append(auc)
    oof_preds[val_idx] = val_pred
    test_preds        += test_pred / 5
    print(f"  Fold {fold} AUC: {auc:.4f}")

overall_auc = roc_auc_score(y_train_all, oof_preds)
print(f"\n  OOF AUC : {overall_auc:.4f}")
print(f"  Mean±Std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  Previous best Kaggle: 0.609  |  Target to beat OOF: 0.623")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving submission...")

submission = pd.DataFrame({
    'FINANCEORDERID': test_df['FINANCEORDERID'].astype(str),
    'FPD_15_pred':    test_preds
})
submission = submission.sort_values('FINANCEORDERID').reset_index(drop=True)
submission.to_csv('submission_v5_mismatch.csv', index=False)

assert not submission['FINANCEORDERID'].duplicated().any()
assert submission['FPD_15_pred'].between(0, 1).all()
missing = set(test_ids['FINANCEORDERID'].astype(str)) - set(submission['FINANCEORDERID'])
assert not missing, f"Missing {len(missing)} IDs!"

print(f"  Saved → submission_v5_mismatch.csv")
print(f"\n{'='*55}")
print(f"  OOF AUC ({MODEL_BACKEND}): {overall_auc:.4f}")
print(f"  New features: face_liveness_gap, finance_amount_country_zscore")
print(f"  Only submit if OOF > 0.623 (previous best)")
print(f"{'='*55}")
