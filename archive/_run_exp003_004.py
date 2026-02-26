"""EXP-003: Feature pruning + EXP-004: HP tuning with dual evaluation.

For each config, reports BOTH:
  - Expanding-window mean AUC (5 folds)
  - Single Nov-holdout AUC (train Jan-Oct, val Nov)
so we can compare which evaluation strategy produces the best Kaggle result.

Does NOT generate submission.csv — the user will decide which config to submit.
"""
import json, os, itertools, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
orders = pd.read_csv('Orders.csv', low_memory=False)
payments = pd.read_csv('Payment_History.csv')

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

def prep_split(df, train_mask, val_mask, features, cat_cols):
    y_t = df.loc[train_mask, 'FPD_15'].values
    y_v = df.loc[val_mask, 'FPD_15'].values
    tr = df.loc[train_mask, features].copy()
    va = df.loc[val_mask, features].copy()
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
    return X_t, X_v, y_t, y_v

# ── Feature sets ────────────────────────────────────────────────────────────
# v2 full feature set (27 pre-OHE)
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

# Pruned feature set (19 pre-OHE, dropped 8 temporal noise features)
pruned_features = [
    'FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT',
    'down_payment_ratio', 'finance_ratio',
    'tx_hour', 'tx_dayofweek',
    'merchant_tenure_days',
    'merchantid_fpd_rate', 'clerk_id_fpd_rate', 'adminid_fpd_rate',
    'state_fpd_rate',
    'merchantid_order_count', 'adminid_order_count',
    'COUNTRY', 'MANUFACTURER', 'CURRENCY',
]
pruned_cats = ['COUNTRY', 'MANUFACTURER', 'CURRENCY']

# ── Prepare base data ───────────────────────────────────────────────────────
print("Preparing base data...")
df = engineer_base(orders.copy(), payments)
labelled = df['FPD_15'].notna()

expanding_folds = [
    (list(range(1, 7)), 7),
    (list(range(1, 8)), 8),
    (list(range(1, 9)), 9),
    (list(range(1, 10)), 10),
    (list(range(1, 11)), 11),
]

def evaluate_config(df_base, labelled_mask, features, cat_cols, params):
    """Return both expanding-window mean AUC and single Nov-holdout AUC."""
    # Expanding-window CV
    ew_aucs = []
    for train_months, val_month in expanding_folds:
        df_fold = df_base.copy()
        rate_mask = labelled_mask & df_fold['_tx_month'].isin(train_months)
        df_fold = add_target_features(df_fold, rate_mask)
        t_mask = labelled_mask & df_fold['_tx_month'].isin(train_months)
        v_mask = labelled_mask & (df_fold['_tx_month'] == val_month)
        X_t, X_v, y_t, y_v = prep_split(df_fold, t_mask, v_mask, features, cat_cols)
        model = xgb.XGBClassifier(**params)
        model.fit(X_t, y_t)
        preds = model.predict_proba(X_v)[:, 1]
        ew_aucs.append(roc_auc_score(y_v, preds))

    nov_auc = ew_aucs[-1]  # Fold 5 is Jan-Oct → Nov
    ew_mean = np.mean(ew_aucs)
    return ew_mean, nov_auc, ew_aucs

# ── EXP-003: Pruned features vs v2 features (same params) ──────────────────
print("\n" + "=" * 70)
print("EXP-003: Feature pruning comparison (same HP, two feature sets)")
print("=" * 70)

base_params = dict(
    objective='binary:logistic', eval_metric='auc', random_state=42,
    max_depth=4, learning_rate=0.1, n_estimators=500,
    subsample=0.9, colsample_bytree=0.8, scale_pos_weight=5,
    verbosity=0, n_jobs=-1,
)

t0 = time.time()
v2_ew, v2_nov, v2_folds = evaluate_config(df, labelled, v2_features, v2_cats, base_params)
print(f"v2 (27 features): EW mean={v2_ew:.5f}  Nov={v2_nov:.5f}  folds={[f'{a:.4f}' for a in v2_folds]}")

pruned_ew, pruned_nov, pruned_folds = evaluate_config(df, labelled, pruned_features, pruned_cats, base_params)
print(f"Pruned (19 feat):  EW mean={pruned_ew:.5f}  Nov={pruned_nov:.5f}  folds={[f'{a:.4f}' for a in pruned_folds]}")
print(f"  Delta EW: {pruned_ew - v2_ew:+.5f}  Delta Nov: {pruned_nov - v2_nov:+.5f}")
print(f"  Time: {time.time()-t0:.0f}s")

# ── EXP-004: HP tuning ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXP-004: Hyperparameter tuning (pruned features, 25 random configs)")
print("=" * 70)

param_grid = {
    'max_depth': [2, 3, 4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 500],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
    'scale_pos_weight': [3, 5, 7],
    'min_child_weight': [1, 5, 10],
}

all_keys = list(param_grid.keys())
all_combos = list(itertools.product(*[param_grid[k] for k in all_keys]))
n_search = 25
np.random.seed(42)
sampled_indices = np.random.choice(len(all_combos), size=n_search, replace=False)
sampled_combos = [all_combos[i] for i in sampled_indices]

print(f"Searching {n_search} configs × 5 expanding-window folds = {n_search*5} fits")
results_log = []
t0 = time.time()

for idx, combo in enumerate(sampled_combos):
    hp = dict(zip(all_keys, combo))
    full_params = dict(
        objective='binary:logistic', eval_metric='auc', random_state=42,
        verbosity=0, n_jobs=-1, **hp,
    )
    ew_mean, nov_auc, fold_aucs = evaluate_config(df, labelled, pruned_features, pruned_cats, full_params)
    results_log.append({
        'params': hp,
        'ew_mean_auc': round(ew_mean, 5),
        'nov_auc': round(nov_auc, 5),
        'fold_aucs': [round(a, 5) for a in fold_aucs],
    })
    if (idx + 1) % 5 == 0:
        best_so_far = max(results_log, key=lambda x: x['nov_auc'])
        print(f"  [{idx+1}/{n_search}] best Nov AUC so far: {best_so_far['nov_auc']:.5f}  "
              f"(EW: {best_so_far['ew_mean_auc']:.5f})  elapsed: {time.time()-t0:.0f}s")

# Rank by Nov AUC (our closest proxy to Kaggle)
results_log.sort(key=lambda x: x['nov_auc'], reverse=True)

print(f"\nTotal time: {time.time()-t0:.0f}s")
print(f"\nTop 10 by Nov holdout AUC:")
print(f"  {'#':<4} {'Nov AUC':<10} {'EW mean':<10} {'max_d':<6} {'lr':<6} {'n_est':<6} {'sub':<5} {'col':<5} {'spw':<5} {'mcw':<5}")
for i, r in enumerate(results_log[:10]):
    p = r['params']
    print(f"  {i+1:<4} {r['nov_auc']:<10.5f} {r['ew_mean_auc']:<10.5f} "
          f"{p['max_depth']:<6} {p['learning_rate']:<6} {p['n_estimators']:<6} "
          f"{p['subsample']:<5} {p['colsample_bytree']:<5} {p['scale_pos_weight']:<5} {p['min_child_weight']:<5}")

print(f"\nTop 10 by expanding-window mean AUC:")
ew_sorted = sorted(results_log, key=lambda x: x['ew_mean_auc'], reverse=True)
for i, r in enumerate(ew_sorted[:10]):
    p = r['params']
    print(f"  {i+1:<4} EW={r['ew_mean_auc']:<10.5f} Nov={r['nov_auc']:<10.5f} "
          f"{p['max_depth']:<6} {p['learning_rate']:<6} {p['n_estimators']:<6} "
          f"{p['subsample']:<5} {p['colsample_bytree']:<5} {p['scale_pos_weight']:<5} {p['min_child_weight']:<5}")

# ── Save all results ────────────────────────────────────────────────────────
output = {
    'exp003': {
        'v2_ew_mean': round(v2_ew, 5), 'v2_nov': round(v2_nov, 5),
        'pruned_ew_mean': round(pruned_ew, 5), 'pruned_nov': round(pruned_nov, 5),
        'pruned_features': pruned_features,
        'dropped_features': ['tx_month', 'FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE',
                             'LIVENESS_SCORE', 'OVERALL_SCORE', 'LOCK_NAME', 'LOCK_PRODUCT',
                             'clerk_id_order_count'],
    },
    'exp004_top10_by_nov': results_log[:10],
    'exp004_top10_by_ew': ew_sorted[:10],
}
with open('_exp003_004_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved to _exp003_004_results.json")
