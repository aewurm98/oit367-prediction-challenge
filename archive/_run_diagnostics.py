"""Phase 1 diagnostics: per-feature temporal AUC, distribution drift, expanding-window CV.

Outputs:
  _diagnostic_feature_auc.json — per-feature temporal AUC ranking
  _diagnostic_drift.json       — per-feature distribution drift analysis
  _diagnostic_expanding_cv.json — expanding-window CV results
  All results also printed to stdout.
"""
import json, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1 DIAGNOSTICS")
print("=" * 70)
print("\nLoading data...")
orders = pd.read_csv('Orders.csv', low_memory=False)
payments = pd.read_csv('Payment_History.csv')
print(f'Orders: {orders.shape}  Payments: {payments.shape}')

# ── Shared feature engineering (identical to model_experiments.ipynb) ────────
def _aggregate_payments(payments_df):
    return payments_df.groupby('FINANCEORDERID').agg(
        pay_max_cumpaid=('TOTAL_CUMPAID', 'max'),
        pay_max_overdue=('DAYSOVERDUE', 'max'),
        pay_mean_overdue=('DAYSOVERDUE', 'mean'),
        pay_num_snapshots=('CALENDAR_DATE', 'nunique'),
    ).reset_index()


def _engineer_features(df, rate_mask=None):
    df['down_payment_ratio'] = df['DOWN_PAYMENT_AMOUNT'] / df['PURCHASE_AMOUNT'].replace(0, np.nan)
    df['finance_ratio'] = df['FINANCE_AMOUNT'] / df['PURCHASE_AMOUNT'].replace(0, np.nan)
    tx = pd.to_datetime(df['TRANSACTIONTIME'], utc=True)
    df['tx_hour'] = tx.dt.hour
    df['tx_dayofweek'] = tx.dt.dayofweek
    df['tx_month'] = tx.dt.month
    mfsd = pd.to_datetime(df['MERCHANT_FIRST_SALE_DATE'], utc=True)
    df['merchant_tenure_days'] = (tx - mfsd).dt.days
    if rate_mask is None:
        rate_mask = df['FPD_15'].notna()
    for entity in ['MERCHANTID', 'CLERK_ID', 'ADMINID']:
        rates = df.loc[rate_mask].groupby(entity)['FPD_15'].mean()
        df[f'{entity.lower()}_fpd_rate'] = df[entity].map(rates)
    state_rates = df.loc[rate_mask].groupby('STATE')['FPD_15'].mean()
    df['state_fpd_rate'] = df['STATE'].map(state_rates)
    for entity in ['MERCHANTID', 'CLERK_ID', 'ADMINID']:
        counts = df.loc[rate_mask].groupby(entity)['FPD_15'].count()
        df[f'{entity.lower()}_order_count'] = df[entity].map(counts)
    return df


def prepare_full(orders_df, payments_df, val_month):
    """Prepare data with temporal split, returning DataFrames (not numpy) for diagnostics."""
    df = orders_df.copy()
    pay_agg = _aggregate_payments(payments_df)
    df = df.merge(pay_agg, on='FINANCEORDERID', how='left')
    labelled = df['FPD_15'].notna()
    tx_months = pd.to_datetime(df['TRANSACTIONTIME'], utc=True).dt.month
    rate_mask = labelled & (tx_months != val_month)
    df = _engineer_features(df, rate_mask=rate_mask)
    df['_tx_month'] = tx_months
    return df, labelled


selected_features = [
    'FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT',
    'FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
    'LIVENESS_SCORE', 'OVERALL_SCORE',
    'down_payment_ratio', 'finance_ratio',
    'tx_hour', 'tx_dayofweek', 'tx_month',
    'merchant_tenure_days',
    'merchantid_fpd_rate', 'clerk_id_fpd_rate', 'adminid_fpd_rate',
    'state_fpd_rate',
    'merchantid_order_count', 'clerk_id_order_count', 'adminid_order_count',
    'COUNTRY', 'LOCK_PRODUCT', 'MANUFACTURER',
    'LOCK_NAME', 'CURRENCY',
]

cat_features = ['COUNTRY', 'LOCK_PRODUCT', 'MANUFACTURER', 'LOCK_NAME', 'CURRENCY']
num_features = [f for f in selected_features if f not in cat_features]

# ── Prepare data once (val_month=11) ────────────────────────────────────────
print("\nPreparing data (val_month=11)...")
df, labelled = prepare_full(orders, payments, val_month=11)
tx_months = df['_tx_month']
train_mask = labelled & (tx_months != 11)
val_mask = labelled & (tx_months == 11)

y_tr = df.loc[train_mask, 'FPD_15'].values
y_val = df.loc[val_mask, 'FPD_15'].values
print(f'Train rows: {train_mask.sum():,}  Val rows: {val_mask.sum():,}')

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1.1: Per-feature temporal AUC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1.1: Per-feature temporal AUC (Jan-Oct → Nov)")
print("=" * 70)

feature_auc_results = []

for feat in selected_features:
    try:
        feat_df_tr = df.loc[train_mask, [feat]].copy()
        feat_df_val = df.loc[val_mask, [feat]].copy()

        is_cat = feat in cat_features
        if is_cat:
            feat_df_tr = pd.get_dummies(feat_df_tr, columns=[feat])
            feat_df_val = pd.get_dummies(feat_df_val, columns=[feat])
            feat_df_tr, feat_df_val = feat_df_tr.align(
                feat_df_val, join='left', axis=1, fill_value=0
            )

        num_cols = feat_df_tr.select_dtypes(include='number').columns.tolist()
        medians = feat_df_tr[num_cols].median()
        feat_df_tr[num_cols] = feat_df_tr[num_cols].fillna(medians)
        feat_df_val[num_cols] = feat_df_val[num_cols].fillna(medians)

        X_tr_feat = feat_df_tr.values.astype(np.float64)
        X_val_feat = feat_df_val.values.astype(np.float64)

        model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc', random_state=42,
            max_depth=2, n_estimators=50, scale_pos_weight=5,
            verbosity=0, n_jobs=-1,
        )
        model.fit(X_tr_feat, y_tr)
        preds = model.predict_proba(X_val_feat)[:, 1]
        auc = roc_auc_score(y_val, preds)

        if auc < 0.52:
            verdict = "DROP"
        elif auc < 0.55:
            verdict = "INVESTIGATE"
        else:
            verdict = "KEEP"

        feature_auc_results.append({
            'feature': feat, 'temporal_auc': round(auc, 5),
            'verdict': verdict, 'n_columns': X_tr_feat.shape[1],
        })
        print(f'  {feat:35s}  AUC={auc:.5f}  [{verdict}]')
    except Exception as e:
        print(f'  {feat:35s}  ERROR: {e}')
        feature_auc_results.append({
            'feature': feat, 'temporal_auc': None, 'verdict': 'ERROR', 'error': str(e),
        })

feature_auc_results.sort(key=lambda x: x['temporal_auc'] or 0, reverse=True)
print("\n--- Ranked by temporal AUC ---")
for r in feature_auc_results:
    print(f"  {r['temporal_auc']:.5f}  {r['verdict']:12s}  {r['feature']}")

with open('_diagnostic_feature_auc.json', 'w') as f:
    json.dump(feature_auc_results, f, indent=2)
print("\nSaved to _diagnostic_feature_auc.json")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1.2: Feature distribution drift analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1.2: Feature distribution drift (Nov vs Jan-Oct)")
print("=" * 70)

labelled_df = df[labelled].copy()
drift_results = []

print("\n--- Numeric features ---")
print(f"  {'Feature':35s}  {'Jan-Oct mean':>14s}  {'Nov mean':>14s}  {'Drift z':>8s}  {'Flag':>6s}")
for feat in num_features:
    jan_oct = labelled_df.loc[labelled_df['_tx_month'] != 11, feat]
    nov = labelled_df.loc[labelled_df['_tx_month'] == 11, feat]
    jo_mean = jan_oct.mean()
    jo_std = jan_oct.std()
    nov_mean = nov.mean()
    drift_z = (nov_mean - jo_mean) / jo_std if jo_std > 0 else 0.0
    flag = "DRIFT" if abs(drift_z) > 1.0 else ""
    drift_results.append({
        'feature': feat, 'type': 'numeric',
        'jan_oct_mean': round(float(jo_mean), 4) if pd.notna(jo_mean) else None,
        'nov_mean': round(float(nov_mean), 4) if pd.notna(nov_mean) else None,
        'drift_zscore': round(float(drift_z), 4),
        'flagged': abs(drift_z) > 1.0,
    })
    print(f"  {feat:35s}  {jo_mean:14.4f}  {nov_mean:14.4f}  {drift_z:8.4f}  {flag:>6s}")

print("\n--- Categorical features ---")
for feat in cat_features:
    jan_oct_dist = labelled_df.loc[labelled_df['_tx_month'] != 11, feat].value_counts(normalize=True)
    nov_dist = labelled_df.loc[labelled_df['_tx_month'] == 11, feat].value_counts(normalize=True)
    all_vals = set(jan_oct_dist.index) | set(nov_dist.index)
    max_diff = 0.0
    max_diff_val = None
    for v in all_vals:
        jo_share = jan_oct_dist.get(v, 0.0)
        nov_share = nov_dist.get(v, 0.0)
        diff = abs(nov_share - jo_share)
        if diff > max_diff:
            max_diff = diff
            max_diff_val = v
    flag = "DRIFT" if max_diff > 0.05 else ""
    drift_results.append({
        'feature': feat, 'type': 'categorical',
        'max_shift_value': str(max_diff_val),
        'max_shift_pp': round(float(max_diff * 100), 2),
        'flagged': max_diff > 0.05,
    })
    print(f"  {feat:35s}  max shift: {max_diff_val} ({max_diff*100:.1f}pp)  {flag}")

for r in drift_results:
    for k, v in r.items():
        if isinstance(v, (np.bool_, np.integer)):
            r[k] = int(v)
        elif isinstance(v, np.floating):
            r[k] = float(v)
with open('_diagnostic_drift.json', 'w') as f:
    json.dump(drift_results, f, indent=2)
print("\nSaved to _diagnostic_drift.json")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1.3: Expanding-window temporal CV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1.3: Expanding-window temporal CV")
print("=" * 70)

best_params = dict(
    objective='binary:logistic', eval_metric='auc', random_state=42,
    max_depth=4, learning_rate=0.1, n_estimators=500,
    subsample=0.9, colsample_bytree=0.8, scale_pos_weight=5,
    verbosity=0, n_jobs=-1,
)

folds = [
    (list(range(1, 7)), 7),   # Jan-Jun → Jul
    (list(range(1, 8)), 8),   # Jan-Jul → Aug
    (list(range(1, 9)), 9),   # Jan-Aug → Sep
    (list(range(1, 10)), 10), # Jan-Sep → Oct
    (list(range(1, 11)), 11), # Jan-Oct → Nov
]

def prep_fold(df_full, labelled_mask, train_months, val_month, features, cat_cols):
    """Prepare a single expanding-window fold."""
    tx_m = df_full['_tx_month']
    rate_mask = labelled_mask & tx_m.isin(train_months)

    df_fold = df_full.copy()
    # Recompute entity rates and counts using only this fold's training months
    for entity in ['MERCHANTID', 'CLERK_ID', 'ADMINID']:
        rates = df_fold.loc[rate_mask].groupby(entity)['FPD_15'].mean()
        df_fold[f'{entity.lower()}_fpd_rate'] = df_fold[entity].map(rates)
        counts = df_fold.loc[rate_mask].groupby(entity)['FPD_15'].count()
        df_fold[f'{entity.lower()}_order_count'] = df_fold[entity].map(counts)
    state_rates = df_fold.loc[rate_mask].groupby('STATE')['FPD_15'].mean()
    df_fold['state_fpd_rate'] = df_fold['STATE'].map(state_rates)

    t_mask = labelled_mask & tx_m.isin(train_months)
    v_mask = labelled_mask & (tx_m == val_month)

    y_t = df_fold.loc[t_mask, 'FPD_15'].values
    y_v = df_fold.loc[v_mask, 'FPD_15'].values
    tr_feat = df_fold.loc[t_mask, features].copy()
    val_feat = df_fold.loc[v_mask, features].copy()

    cat_c = [c for c in cat_cols if c in tr_feat.columns]
    num_c = tr_feat.select_dtypes(include='number').columns.tolist()

    medians = tr_feat[num_c].median()
    tr_feat[num_c] = tr_feat[num_c].fillna(medians)
    val_feat[num_c] = val_feat[num_c].fillna(medians)
    for c in cat_c:
        fill = tr_feat[c].mode().iloc[0] if not tr_feat[c].mode().empty else 'UNKNOWN'
        tr_feat[c] = tr_feat[c].fillna(fill)
        val_feat[c] = val_feat[c].fillna(fill)

    if cat_c:
        tr_feat = pd.get_dummies(tr_feat, columns=cat_c)
        val_feat = pd.get_dummies(val_feat, columns=cat_c)
        tr_feat, val_feat = tr_feat.align(val_feat, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    X_t = scaler.fit_transform(tr_feat.values.astype(np.float64))
    X_v = scaler.transform(val_feat.values.astype(np.float64))
    return X_t, X_v, y_t, y_v


# We need the raw df BEFORE feature engineering for recomputing per-fold
print("\nReloading raw data for per-fold feature engineering...")
df_raw = orders.copy()
pay_agg = _aggregate_payments(payments)
df_raw = df_raw.merge(pay_agg, on='FINANCEORDERID', how='left')

# Basic features that don't depend on rate_mask
tx = pd.to_datetime(df_raw['TRANSACTIONTIME'], utc=True)
df_raw['down_payment_ratio'] = df_raw['DOWN_PAYMENT_AMOUNT'] / df_raw['PURCHASE_AMOUNT'].replace(0, np.nan)
df_raw['finance_ratio'] = df_raw['FINANCE_AMOUNT'] / df_raw['PURCHASE_AMOUNT'].replace(0, np.nan)
df_raw['tx_hour'] = tx.dt.hour
df_raw['tx_dayofweek'] = tx.dt.dayofweek
df_raw['tx_month'] = tx.dt.month
mfsd = pd.to_datetime(df_raw['MERCHANT_FIRST_SALE_DATE'], utc=True)
df_raw['merchant_tenure_days'] = (tx - mfsd).dt.days
df_raw['_tx_month'] = tx.dt.month
labelled_raw = df_raw['FPD_15'].notna()

expanding_cv_results = []
print(f"\n{'Fold':<6} {'Train months':<20} {'Val month':<10} {'AUC':<10} {'Train size':<12} {'Val size':<10}")
print("-" * 70)

for fold_idx, (train_months, val_month) in enumerate(folds, 1):
    X_t, X_v, y_t, y_v = prep_fold(
        df_raw, labelled_raw, train_months, val_month,
        selected_features, cat_features,
    )
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_t, y_t)
    preds = model.predict_proba(X_v)[:, 1]
    auc = roc_auc_score(y_v, preds)
    train_str = f"{train_months[0]}-{train_months[-1]}"
    print(f"{fold_idx:<6} {train_str:<20} {val_month:<10} {auc:<10.5f} {X_t.shape[0]:<12,} {X_v.shape[0]:,}")
    expanding_cv_results.append({
        'fold': fold_idx,
        'train_months': train_str,
        'val_month': val_month,
        'auc': round(auc, 5),
        'train_size': X_t.shape[0],
        'val_size': X_v.shape[0],
    })

mean_auc = np.mean([r['auc'] for r in expanding_cv_results])
print(f"\nMean expanding-window AUC: {mean_auc:.5f}")
print(f"Comparison: random 5-fold CV AUC was 0.8066, Nov holdout was 0.5556")

with open('_diagnostic_expanding_cv.json', 'w') as f:
    json.dump({'folds': expanding_cv_results, 'mean_auc': round(mean_auc, 5)}, f, indent=2)
print("Saved to _diagnostic_expanding_cv.json")

print("\n" + "=" * 70)
print("ALL DIAGNOSTICS COMPLETE")
print("=" * 70)
