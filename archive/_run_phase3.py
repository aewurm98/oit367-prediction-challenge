"""Phase 3: Test new feature groups individually and combined.

Evaluates 4 feature groups on top of the v2 baseline (27 features):
  A) Currency normalization (country-level z-scores for financial amounts)
  B) Entity payment history aggregates (merchant/clerk/admin avg days overdue)
  C) KYC interaction features (min score, perfect flag)
  D) Geographic cross-feature (user_merchant_same_state)

Then tests the combined set of all winners.
No submission generated — results only.
"""
import json, os, time, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...", flush=True)
orders = pd.read_csv('Orders.csv', low_memory=False)
payments = pd.read_csv('Payment_History.csv')
print(f'Orders: {orders.shape}  Payments: {payments.shape}', flush=True)

# ── Base feature engineering (v2) ────────────────────────────────────────────
def _aggregate_payments(payments_df):
    return payments_df.groupby('FINANCEORDERID').agg(
        pay_max_cumpaid=('TOTAL_CUMPAID', 'max'),
        pay_max_overdue=('DAYSOVERDUE', 'max'),
        pay_mean_overdue=('DAYSOVERDUE', 'mean'),
        pay_num_snapshots=('CALENDAR_DATE', 'nunique'),
    ).reset_index()


def engineer_base(df, payments_df):
    pay_agg = _aggregate_payments(payments_df)
    df = df.merge(pay_agg, on='FINANCEORDERID', how='left')
    df['down_payment_ratio'] = df['DOWN_PAYMENT_AMOUNT'] / df['PURCHASE_AMOUNT'].replace(0, np.nan)
    df['finance_ratio'] = df['FINANCE_AMOUNT'] / df['PURCHASE_AMOUNT'].replace(0, np.nan)
    tx = pd.to_datetime(df['TRANSACTIONTIME'], utc=True)
    df['tx_hour'] = tx.dt.hour
    df['tx_dayofweek'] = tx.dt.dayofweek
    df['tx_month'] = tx.dt.month
    mfsd = pd.to_datetime(df['MERCHANT_FIRST_SALE_DATE'], utc=True)
    df['merchant_tenure_days'] = (tx - mfsd).dt.days
    df['_tx_month'] = tx.dt.month
    return df


def add_target_features(df, rate_mask):
    for entity in ['MERCHANTID', 'CLERK_ID', 'ADMINID']:
        rates = df.loc[rate_mask].groupby(entity)['FPD_15'].mean()
        df[f'{entity.lower()}_fpd_rate'] = df[entity].map(rates)
        counts = df.loc[rate_mask].groupby(entity)['FPD_15'].count()
        df[f'{entity.lower()}_order_count'] = df[entity].map(counts)
    state_rates = df.loc[rate_mask].groupby('STATE')['FPD_15'].mean()
    df['state_fpd_rate'] = df['STATE'].map(state_rates)
    return df


# ── New feature groups ──────────────────────────────────────────────────────
def add_currency_norm(df, rate_mask):
    """Group A: Z-score financial amounts within each country (from training rows only)."""
    fin_cols = ['FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT']
    for col in fin_cols:
        stats = df.loc[rate_mask].groupby('COUNTRY')[col].agg(['mean', 'std'])
        means = df['COUNTRY'].map(stats['mean'])
        stds = df['COUNTRY'].map(stats['std']).replace(0, np.nan)
        df[f'{col}_country_zscore'] = (df[col] - means) / stds
    return df


def add_entity_payment_history(df, payments_df, rate_mask):
    """Group B: Entity-level payment behavior from Payment_History."""
    pay = payments_df.copy()
    # Join orders to get entity IDs for each payment row
    entity_map = df[['FINANCEORDERID', 'MERCHANTID', 'CLERK_ID', 'ADMINID']].drop_duplicates()
    pay = pay.merge(entity_map, on='FINANCEORDERID', how='left')

    # Only use payment history from training orders (rate_mask rows)
    train_order_ids = set(df.loc[rate_mask, 'FINANCEORDERID'])
    pay_train = pay[pay['FINANCEORDERID'].isin(train_order_ids)]

    for entity in ['MERCHANTID', 'CLERK_ID', 'ADMINID']:
        agg = pay_train.groupby(entity)['DAYSOVERDUE'].agg(['mean', 'max']).reset_index()
        agg.columns = [entity, f'{entity.lower()}_avg_days_overdue', f'{entity.lower()}_max_days_overdue']
        df = df.merge(agg, on=entity, how='left')
    return df


def add_kyc_interactions(df):
    """Group C: KYC interaction features."""
    df['kyc_min_score'] = df[['FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
                              'LIVENESS_SCORE']].min(axis=1)
    df['perfect_kyc_flag'] = (
        (df['FACE_RECOGNITION_SCORE'] >= 99) &
        (df['IDVALIDATION_OVERALL_SCORE'] >= 99) &
        (df['LIVENESS_SCORE'] >= 99)
    ).astype(int)
    return df


def add_geo_cross(df):
    """Group D: Geographic cross-feature."""
    df['user_merchant_same_state'] = (df['USER_STATE'] == df['MERCHANT_STATE']).astype(int)
    return df


# ── Evaluation function ─────────────────────────────────────────────────────
v2_features = [
    'FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT',
    'FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
    'LIVENESS_SCORE', 'OVERALL_SCORE',
    'down_payment_ratio', 'finance_ratio',
    'tx_hour', 'tx_dayofweek', 'tx_month',
    'merchant_tenure_days',
    'merchantid_fpd_rate', 'clerk_id_fpd_rate', 'adminid_fpd_rate',
    'state_fpd_rate',
    'merchantid_order_count', 'clerk_id_order_count', 'adminid_order_count',
    'COUNTRY', 'LOCK_PRODUCT', 'MANUFACTURER', 'LOCK_NAME', 'CURRENCY',
]
v2_cats = ['COUNTRY', 'LOCK_PRODUCT', 'MANUFACTURER', 'LOCK_NAME', 'CURRENCY']

group_a_features = ['FINANCE_AMOUNT_country_zscore', 'PURCHASE_AMOUNT_country_zscore',
                    'TOTAL_DUE_country_zscore', 'DOWN_PAYMENT_AMOUNT_country_zscore']
group_b_features = ['merchantid_avg_days_overdue', 'merchantid_max_days_overdue',
                    'clerk_id_avg_days_overdue', 'clerk_id_max_days_overdue',
                    'adminid_avg_days_overdue', 'adminid_max_days_overdue']
group_c_features = ['kyc_min_score', 'perfect_kyc_flag']
group_d_features = ['user_merchant_same_state']

best_params = dict(
    objective='binary:logistic', eval_metric='auc', random_state=42,
    max_depth=4, learning_rate=0.1, n_estimators=500,
    subsample=0.9, colsample_bytree=0.8, scale_pos_weight=5,
    verbosity=0, n_jobs=-1,
)


def evaluate_nov(df_prepped, labelled, features, cat_cols, params):
    """Train on Jan-Oct, evaluate on Nov. Returns Nov AUC."""
    tx_m = df_prepped['_tx_month']
    t_mask = labelled & (tx_m != 11)
    v_mask = labelled & (tx_m == 11)

    y_t = df_prepped.loc[t_mask, 'FPD_15'].values
    y_v = df_prepped.loc[v_mask, 'FPD_15'].values
    tr = df_prepped.loc[t_mask, features].copy()
    va = df_prepped.loc[v_mask, features].copy()

    cat_c = [c for c in cat_cols if c in tr.columns]
    num_c = tr.select_dtypes(include='number').columns.tolist()
    medians = tr[num_c].median()
    tr[num_c] = tr[num_c].fillna(medians)
    va[num_c] = va[num_c].fillna(medians)
    for c in cat_c:
        fill = tr[c].mode().iloc[0] if not tr[c].mode().empty else 'UNKNOWN'
        tr[c] = tr[c].fillna(fill)
        va[c] = va[c].fillna(fill)
    if cat_c:
        tr = pd.get_dummies(tr, columns=cat_c)
        va = pd.get_dummies(va, columns=cat_c)
        tr, va = tr.align(va, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    X_t = scaler.fit_transform(tr.values.astype(np.float64))
    X_v = scaler.transform(va.values.astype(np.float64))

    model = xgb.XGBClassifier(**params)
    model.fit(X_t, y_t)
    preds = model.predict_proba(X_v)[:, 1]
    return roc_auc_score(y_v, preds)


# ── Prepare data with all new features ──────────────────────────────────────
print("Preparing data with all new features...", flush=True)
df = engineer_base(orders.copy(), payments)
labelled = df['FPD_15'].notna()
rate_mask = labelled & (df['_tx_month'] != 11)

df = add_target_features(df, rate_mask)
df = add_currency_norm(df, rate_mask)
df = add_entity_payment_history(df, payments, rate_mask)
df = add_kyc_interactions(df)
df = add_geo_cross(df)
print("All features engineered.", flush=True)

# ── Evaluate each group ────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PHASE 3: New feature group evaluation", flush=True)
print("=" * 70, flush=True)

results = {}

# Baseline: v2 features only
print("\n[Baseline] v2 features (27 pre-OHE)...", flush=True)
t0 = time.time()
baseline_auc = evaluate_nov(df, labelled, v2_features, v2_cats, best_params)
print(f"  Nov AUC: {baseline_auc:.5f}  ({time.time()-t0:.0f}s)", flush=True)
results['baseline_v2'] = round(baseline_auc, 5)

# Group A: +currency normalization
print("\n[Group A] v2 + currency normalization...", flush=True)
t0 = time.time()
a_feats = v2_features + group_a_features
a_auc = evaluate_nov(df, labelled, a_feats, v2_cats, best_params)
print(f"  Nov AUC: {a_auc:.5f}  (delta: {a_auc-baseline_auc:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
results['group_a_currency_norm'] = {'auc': round(a_auc, 5), 'delta': round(a_auc - baseline_auc, 5)}

# Group B: +entity payment history
print("\n[Group B] v2 + entity payment history...", flush=True)
t0 = time.time()
b_feats = v2_features + group_b_features
b_auc = evaluate_nov(df, labelled, b_feats, v2_cats, best_params)
print(f"  Nov AUC: {b_auc:.5f}  (delta: {b_auc-baseline_auc:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
results['group_b_payment_hist'] = {'auc': round(b_auc, 5), 'delta': round(b_auc - baseline_auc, 5)}

# Group C: +KYC interactions
print("\n[Group C] v2 + KYC interactions...", flush=True)
t0 = time.time()
c_feats = v2_features + group_c_features
c_auc = evaluate_nov(df, labelled, c_feats, v2_cats, best_params)
print(f"  Nov AUC: {c_auc:.5f}  (delta: {c_auc-baseline_auc:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
results['group_c_kyc'] = {'auc': round(c_auc, 5), 'delta': round(c_auc - baseline_auc, 5)}

# Group D: +geographic cross-feature
print("\n[Group D] v2 + geographic cross-feature...", flush=True)
t0 = time.time()
d_feats = v2_features + group_d_features
d_auc = evaluate_nov(df, labelled, d_feats, v2_cats, best_params)
print(f"  Nov AUC: {d_auc:.5f}  (delta: {d_auc-baseline_auc:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
results['group_d_geo'] = {'auc': round(d_auc, 5), 'delta': round(d_auc - baseline_auc, 5)}

# Combined: all groups that improved
print("\n[Combined] v2 + all new feature groups...", flush=True)
t0 = time.time()
all_feats = v2_features + group_a_features + group_b_features + group_c_features + group_d_features
all_auc = evaluate_nov(df, labelled, all_feats, v2_cats, best_params)
print(f"  Nov AUC: {all_auc:.5f}  (delta: {all_auc-baseline_auc:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
results['all_combined'] = {'auc': round(all_auc, 5), 'delta': round(all_auc - baseline_auc, 5)}

# Combined winners only (those with positive delta)
winners = []
if a_auc > baseline_auc: winners.extend(group_a_features)
if b_auc > baseline_auc: winners.extend(group_b_features)
if c_auc > baseline_auc: winners.extend(group_c_features)
if d_auc > baseline_auc: winners.extend(group_d_features)

if winners and set(winners) != set(group_a_features + group_b_features + group_c_features + group_d_features):
    print(f"\n[Winners only] v2 + {len(winners)} winning features...", flush=True)
    t0 = time.time()
    win_feats = v2_features + winners
    win_auc = evaluate_nov(df, labelled, win_feats, v2_cats, best_params)
    print(f"  Nov AUC: {win_auc:.5f}  (delta: {win_auc-baseline_auc:+.5f})  ({time.time()-t0:.0f}s)", flush=True)
    results['winners_only'] = {'auc': round(win_auc, 5), 'delta': round(win_auc - baseline_auc, 5),
                               'features': winners}

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"  Baseline (v2):          {baseline_auc:.5f}", flush=True)
print(f"  + Currency norm (A):    {a_auc:.5f}  ({a_auc-baseline_auc:+.5f})", flush=True)
print(f"  + Payment history (B):  {b_auc:.5f}  ({b_auc-baseline_auc:+.5f})", flush=True)
print(f"  + KYC interactions (C): {c_auc:.5f}  ({c_auc-baseline_auc:+.5f})", flush=True)
print(f"  + Geo cross (D):        {d_auc:.5f}  ({d_auc-baseline_auc:+.5f})", flush=True)
print(f"  All combined:           {all_auc:.5f}  ({all_auc-baseline_auc:+.5f})", flush=True)

with open('_phase3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to _phase3_results.json", flush=True)
