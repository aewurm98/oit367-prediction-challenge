"""
v9 Pipeline — Clean build combining v5's entity FPD rates + PMT_CORE with v8's Cat+LGB ensemble.
Single config, no experiment flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CATBOOST = None
_LGB = None


def _get_catboost():
    global _CATBOOST
    if _CATBOOST is None:
        try:
            from catboost import CatBoostClassifier
            _CATBOOST = CatBoostClassifier
        except ImportError:
            _CATBOOST = False
    return _CATBOOST


def _get_lgb():
    global _LGB
    if _LGB is None:
        try:
            import lightgbm as lgb
            _LGB = lgb
        except ImportError:
            _LGB = False
    return _LGB


def _use_gpu() -> bool:
    return os.environ.get("V9_USE_GPU", "0") in ("1", "true", "yes")


K_SMOOTH = 10
ENTITY_RATE_COLS = ['MERCHANTID', 'CLERK_ID', 'ADMINID', 'STATE', 'CITY', 'MODEL']
MARKET_RATE_COLS = ['MANUFACTURER', 'LOCK_PRODUCT', 'STATE']


def _smooth_rate(fpd_sum: float, n: int, prior_rate: float, k: int = K_SMOOTH) -> float:
    return (fpd_sum + prior_rate * k) / (n + k)


@dataclass
class V9Config:
    """Fixed v9 config — Cat+LGB, cat_lr_03, no experiment flags."""
    ensemble_models: tuple = ('cat', 'lgb')
    weight_strategy: str = 'optimize'
    depth: int = 6
    l2_leaf_reg: float = 3.0
    learning_rate: float = 0.03  # cat_lr_03
    iterations: int = 1000
    early_stopping_rounds: int = 50
    random_state: int = 42
    cat_depth: int = 6
    cat_l2_leaf_reg: float = 3.0
    cat_learning_rate: float = 0.03
    lgb_num_leaves: int = 63
    lgb_min_child_samples: int = 50
    lgb_learning_rate: float = 0.05


def _build_pmt_core(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    train_ids: set,
    oid_col: str,
    log: callable | None,
) -> None:
    """Build entity-level PMT_CORE: pmt_fpd_proxy_rate, zero_recovery_rate per MERCHANTID, CLERK_ID, ADMINID.
    Adds columns in-place to orders."""
    pay = payments.copy()
    entity_map = orders[['FINANCEORDERID', 'MERCHANTID', 'CLERK_ID', 'ADMINID']].drop_duplicates()
    pay = pay.merge(entity_map, on='FINANCEORDERID', how='inner')
    pay_train = pay[pay['FINANCEORDERID'].isin(train_ids)]

    days_col = 'DAYS_OVERDUE' if 'DAYS_OVERDUE' in pay.columns else 'DAYSOVERDUE'
    princ_col = 'PRINCIPAL_CUMPAID' if 'PRINCIPAL_CUMPAID' in pay.columns else 'PRINCIPALCUMPAID'

    for entity in ['MERCHANTID', 'CLERK_ID', 'ADMINID']:
        if entity not in pay_train.columns:
            continue

        if days_col in pay_train.columns:
            order_max_overdue = pay_train.groupby('FINANCEORDERID')[days_col].max()
            order_fpd_proxy = (order_max_overdue > 0).astype(float)
            entity_orders = pay_train[['FINANCEORDERID', entity]].drop_duplicates()
            entity_orders = entity_orders.merge(
                order_fpd_proxy.rename('_fpd_proxy').reset_index(),
                on='FINANCEORDERID', how='left'
            )
            grp = entity_orders.groupby(entity)['_fpd_proxy'].mean().reset_index()
            grp.columns = [entity, f'{entity}_pmt_fpd_proxy_rate']
            orders[f'{entity}_pmt_fpd_proxy_rate'] = orders[entity].map(grp.set_index(entity)[f'{entity}_pmt_fpd_proxy_rate'])

        if princ_col in pay_train.columns:
            order_max_princ = pay_train.groupby('FINANCEORDERID')[princ_col].max()
            order_zero_recovery = (order_max_princ <= 0).astype(float)
            entity_orders = pay_train[['FINANCEORDERID', entity]].drop_duplicates()
            entity_orders = entity_orders.merge(
                order_zero_recovery.rename('_zero_rec').reset_index(),
                on='FINANCEORDERID', how='left'
            )
            grp = entity_orders.groupby(entity)['_zero_rec'].mean().reset_index()
            grp.columns = [entity, f'{entity}_pmt_zero_recovery_rate']
            orders[f'{entity}_pmt_zero_recovery_rate'] = orders[entity].map(grp.set_index(entity)[f'{entity}_pmt_zero_recovery_rate'])


def build_features(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    test_ids: pd.DataFrame,
    config: V9Config,
    log: callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """
    Build v9 features: v5 entity FPD rates + PMT_CORE + v8 market features.
    Excludes LOCK_NAME, CURRENCY FPD rates (v6 overfit).
    """
    def _log(msg: str):
        if log:
            log(msg)

    payments = payments.copy()
    payments.columns = [c.upper().strip() for c in payments.columns]
    oid_col = next((c for c in payments.columns if 'ORDER' in c and 'ID' in c), payments.columns[0])
    _PMT_ALIASES = {
        'DAYSOVERDUE': 'DAYS_OVERDUE',
        'PRINCIPALCUMPAID': 'PRINCIPAL_CUMPAID',
        'TOTALBALANCE': 'TOTAL_BALANCE',
    }
    for alt, canonical in _PMT_ALIASES.items():
        if alt in payments.columns and canonical not in payments.columns:
            payments = payments.rename(columns={alt: canonical})
    numeric_pay = [c for c in payments.select_dtypes(include=[np.number]).columns if c != oid_col]

    test_order_ids = set(test_ids['FINANCEORDERID'].astype(str))
    orders = orders.copy()
    orders['FINANCEORDERID_str'] = orders['FINANCEORDERID'].astype(str)
    if 'TRANSACTIONTIME' in orders.columns:
        orders['TRANSACTIONTIME'] = pd.to_datetime(orders['TRANSACTIONTIME'], utc=True, errors='coerce')
        orders['txn_year'] = orders['TRANSACTIONTIME'].dt.year
    if 'txn_year' in orders.columns:
        orders = orders[orders['txn_year'] == 2025].copy()

    train_ids = set(orders[orders['FPD_15'].notna() & ~orders['FINANCEORDERID_str'].isin(test_order_ids)]['FINANCEORDERID'])

    for c in ['FIRST_PAYMENT_DUE_TIMESTAMP', 'MERCHANT_FIRST_SALE_DATE', 'MERCHANT_LAST_SALE_DATE']:
        if c in orders.columns:
            orders[c] = pd.to_datetime(orders[c], utc=True, errors='coerce')

    if 'TRANSACTIONTIME' not in orders.columns or orders['TRANSACTIONTIME'].dtype == object:
        orders['TRANSACTIONTIME'] = pd.to_datetime(orders['TRANSACTIONTIME'], utc=True, errors='coerce')
    orders['txn_hour'] = orders['TRANSACTIONTIME'].dt.hour
    orders['txn_dow'] = orders['TRANSACTIONTIME'].dt.dayofweek
    orders['txn_month'] = orders['TRANSACTIONTIME'].dt.month
    orders['txn_day'] = orders['TRANSACTIONTIME'].dt.day
    orders['txn_year'] = orders['TRANSACTIONTIME'].dt.year
    orders['is_weekend'] = (orders['txn_dow'] >= 5).astype(int)
    orders['is_night'] = ((orders['txn_hour'] >= 22) | (orders['txn_hour'] <= 5)).astype(int)
    orders['is_month_end'] = (orders['txn_day'] >= 25).astype(int)
    orders['is_month_start'] = (orders['txn_day'] <= 5).astype(int)

    orders['TOTAL_DUE'] = pd.to_numeric(orders['TOTAL_DUE'], errors='coerce')
    orders['DOWN_PAYMENT_AMOUNT'] = pd.to_numeric(orders['DOWN_PAYMENT_AMOUNT'], errors='coerce')
    orders['down_payment_ratio'] = (
        (orders['PURCHASE_AMOUNT'] - orders['FINANCE_AMOUNT']) / orders['PURCHASE_AMOUNT'].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    orders['finance_ratio'] = (
        orders['FINANCE_AMOUNT'] / orders['PURCHASE_AMOUNT'].replace(0, np.nan)
    ).fillna(1).clip(0, 1)
    orders['log_finance_amount'] = np.log1p(orders['FINANCE_AMOUNT'].clip(lower=0))
    orders['log_purchase_amount'] = np.log1p(orders['PURCHASE_AMOUNT'].clip(lower=0))
    orders['log_total_due'] = np.log1p(orders['TOTAL_DUE'].clip(lower=0))
    orders['total_due_ratio'] = (
        orders['TOTAL_DUE'] / orders['FINANCE_AMOUNT'].replace(0, np.nan)
    ).fillna(1).clip(0, 5)
    orders['days_to_first_payment'] = (
        orders['FIRST_PAYMENT_DUE_TIMESTAMP'] - orders['TRANSACTIONTIME']
    ).dt.total_seconds() / 86400

    orders['merchant_tenure_days'] = (
        orders['TRANSACTIONTIME'] - orders['MERCHANT_FIRST_SALE_DATE']
    ).dt.total_seconds() / 86400
    orders['merchant_tenure_days'] = orders['merchant_tenure_days'].fillna(0).clip(lower=0)
    orders['log_merchant_tenure'] = np.log1p(orders['merchant_tenure_days'])
    orders['is_new_merchant'] = (orders['merchant_tenure_days'] < 90).astype(float)
    orders['days_since_last_sale'] = (
        orders['TRANSACTIONTIME'] - orders['MERCHANT_LAST_SALE_DATE']
    ).dt.total_seconds() / 86400
    orders['days_since_last_sale'] = orders['days_since_last_sale'].fillna(0).clip(lower=0)

    score_cols = ['FACE_RECOGNITION_SCORE', 'IDVALIDATION_OVERALL_SCORE', 'LIVENESS_SCORE', 'OVERALL_SCORE']
    for c in score_cols:
        if c in orders.columns:
            orders[c] = pd.to_numeric(orders[c], errors='coerce')
    orders['kyc_score_mean'] = orders[[c for c in score_cols if c in orders.columns]].mean(axis=1)
    orders['kyc_score_min'] = orders[[c for c in score_cols if c in orders.columns]].min(axis=1)
    orders['kyc_score_std'] = orders[[c for c in score_cols if c in orders.columns]].std(axis=1)
    orders['kyc_score_range'] = orders[[c for c in score_cols if c in orders.columns]].max(axis=1) - orders[[c for c in score_cols if c in orders.columns]].min(axis=1)
    orders['low_face_score'] = (orders['FACE_RECOGNITION_SCORE'] < 70).astype(float) if 'FACE_RECOGNITION_SCORE' in orders.columns else 0.0
    orders['low_id_score'] = (orders['IDVALIDATION_OVERALL_SCORE'] < 80).astype(float) if 'IDVALIDATION_OVERALL_SCORE' in orders.columns else 0.0
    orders['any_low_score'] = (orders[[c for c in score_cols if c in orders.columns]] < 75).any(axis=1).astype(float)
    orders['all_scores_perfect'] = (orders[[c for c in score_cols if c in orders.columns]] == 100).all(axis=1).astype(float)
    orders['high_finance_low_kyc'] = orders['finance_ratio'] * (100 - orders['kyc_score_mean'].fillna(50)) / 100

    if 'USER_STATE' in orders.columns and 'MERCHANT_STATE' in orders.columns:
        orders['state_mismatch'] = (
            orders['USER_STATE'].fillna('__X__') != orders['MERCHANT_STATE'].fillna('__X__')
        ).astype(float)

    if 'COUNTRY' in orders.columns:
        train_only = orders[orders['FINANCEORDERID'].isin(train_ids)]
        for col in ['FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT']:
            if col in orders.columns:
                stats = train_only.groupby('COUNTRY')[col].agg(['mean', 'std']).reset_index()
                stats.columns = ['COUNTRY', f'{col}_mu', f'{col}_sig']
                stats[f'{col}_sig'] = stats[f'{col}_sig'].replace(0, 1)
                orders = orders.merge(stats, on='COUNTRY', how='left')
                orders[f'{col}_z'] = ((orders[col] - orders[f'{col}_mu']) / orders[f'{col}_sig']).fillna(0)
                orders = orders.drop(columns=[f'{col}_mu', f'{col}_sig'])

    if numeric_pay:
        ord_cols = ['FINANCEORDERID', 'COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']
        ord_cols = [c for c in ord_cols if c in orders.columns]
        pay_ctx = payments.merge(
            orders[ord_cols].drop_duplicates(),
            left_on=oid_col, right_on='FINANCEORDERID', how='inner'
        )
        pay_ctx_train = pay_ctx[pay_ctx['FINANCEORDERID'].isin(train_ids)]
        for grp in ['COUNTRY', 'MANUFACTURER', 'MERCHANT_STATE']:
            if grp in pay_ctx_train.columns:
                stats = pay_ctx_train.groupby(grp)[numeric_pay].agg(['mean', 'std'])
                stats.columns = [f'{grp}_pay_{a}_{b}' for a, b in stats.columns]
                orders = orders.merge(stats.reset_index(), on=grp, how='left')

    _build_pmt_core(orders, payments, train_ids, oid_col, log)

    train_df = orders[orders['FPD_15'].notna() & ~orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()
    test_df = orders[orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()
    global_rate = train_df['FPD_15'].mean()

    entity_cols = [c for c in ENTITY_RATE_COLS if c in train_df.columns]
    market_cols = [c for c in ['MANUFACTURER', 'LOCK_PRODUCT'] if c in train_df.columns]
    rate_cols = list(dict.fromkeys(entity_cols + market_cols))

    train_months = sorted(train_df['txn_month'].unique())

    for col in rate_cols:
        stats = train_df.groupby(col)['FPD_15'].agg(n='count', s='sum').reset_index()
        stats[f'{col}_fpd_rate'] = stats.apply(lambda r: _smooth_rate(r['s'], r['n'], global_rate), axis=1)
        test_df = test_df.merge(stats[[col, f'{col}_fpd_rate']], on=col, how='left')
        test_df[f'{col}_fpd_rate'] = test_df[f'{col}_fpd_rate'].fillna(global_rate)

    train_results = []
    for month in train_months:
        lookback = train_df[train_df['txn_month'] < month]
        current = train_df[train_df['txn_month'] == month].copy()
        lb_rate = lookback['FPD_15'].mean() if len(lookback) > 0 else global_rate
        for col in rate_cols:
            if len(lookback) > 0:
                stats = lookback.groupby(col)['FPD_15'].agg(n='count', s='sum').reset_index()
                stats[f'{col}_fpd_rate'] = stats.apply(lambda r: _smooth_rate(r['s'], r['n'], lb_rate), axis=1)
                current = current.merge(stats[[col, f'{col}_fpd_rate']], on=col, how='left')
            else:
                current[f'{col}_fpd_rate'] = np.nan
            current[f'{col}_fpd_rate'] = current[f'{col}_fpd_rate'].fillna(lb_rate)
        train_results.append(current)
    train_df = pd.concat(train_results, ignore_index=True)

    base_features = [
        'FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT',
        'log_finance_amount', 'log_purchase_amount', 'log_total_due',
        'down_payment_ratio', 'finance_ratio', 'total_due_ratio',
        'txn_hour', 'txn_dow', 'txn_month', 'txn_day',
        'is_weekend', 'is_night', 'is_month_end', 'is_month_start',
        'days_to_first_payment',
        'merchant_tenure_days', 'log_merchant_tenure', 'is_new_merchant', 'days_since_last_sale',
        'kyc_score_mean', 'kyc_score_min', 'kyc_score_std', 'kyc_score_range',
        'low_face_score', 'low_id_score', 'any_low_score', 'all_scores_perfect', 'high_finance_low_kyc',
    ]
    if 'state_mismatch' in orders.columns:
        base_features.append('state_mismatch')
    base_features.extend([c for c in orders.columns if c.endswith('_z')])
    base_features.extend([f'{c}_fpd_rate' for c in rate_cols if c in train_df.columns and f'{c}_fpd_rate' in train_df.columns])
    for c in orders.columns:
        if c.startswith('COUNTRY_pay_') or c.startswith('MANUFACTURER_pay_') or c.startswith('MERCHANT_STATE_pay_'):
            if c not in base_features:
                base_features.append(c)
    for c in orders.columns:
        if any(c.startswith(f'{e}_pmt_') for e in ['MERCHANTID', 'CLERK_ID', 'ADMINID']):
            if c not in base_features:
                base_features.append(c)

    feature_cols = [c for c in base_features if c in train_df.columns and c in test_df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    return train_df, test_df, feature_cols, global_rate


def optimize_ensemble_weights(val_preds: list[np.ndarray], y_val: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize
    n = len(val_preds)
    if n == 1:
        return np.array([1.0])
    def neg_auc(w):
        blend = sum(w[i] * val_preds[i] for i in range(n))
        return -roc_auc_score(y_val, blend)
    result = minimize(
        neg_auc, x0=np.ones(n) / n,
        bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )
    return result.x


def train_model(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_id: str,
    config: V9Config,
    scale_pos_weight: float,
) -> tuple[Any, np.ndarray, np.ndarray]:
    X_tr = X_tr.fillna(X_tr.median())
    X_val = X_val.fillna(X_tr.median())
    for c in X_tr.columns:
        if X_tr[c].dtype == object or X_tr[c].dtype.name == 'category':
            X_tr[c] = pd.to_numeric(X_tr[c], errors='coerce').fillna(X_tr[c].median())
            X_val[c] = pd.to_numeric(X_val[c], errors='coerce').fillna(X_tr[c].median())

    if model_id == 'cat':
        CatBoost = _get_catboost()
        if not CatBoost:
            return train_model(X_tr, y_tr, X_val, y_val, 'lgb', config, scale_pos_weight)
        _eval_metric = 'Logloss' if _use_gpu() else 'AUC'
        model = CatBoost(
            iterations=config.iterations,
            learning_rate=config.cat_learning_rate,
            depth=config.cat_depth,
            l2_leaf_reg=config.cat_l2_leaf_reg,
            scale_pos_weight=scale_pos_weight,
            eval_metric=_eval_metric,
            early_stopping_rounds=config.early_stopping_rounds,
            random_seed=config.random_state,
            verbose=0,
            task_type='GPU' if _use_gpu() else 'CPU',
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        val_pred = model.predict_proba(X_val)[:, 1]
        return model, val_pred, None

    elif model_id == 'lgb':
        lgb = _get_lgb()
        if not lgb:
            raise ImportError("lightgbm required: pip install lightgbm")
        params = {
            'objective': 'binary', 'metric': 'auc',
            'learning_rate': config.lgb_learning_rate,
            'num_leaves': config.lgb_num_leaves,
            'min_child_samples': config.lgb_min_child_samples,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1, 'n_jobs': -1, 'random_state': config.random_state,
            'device': 'cpu',
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(
            params, dtrain, num_boost_round=config.iterations,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred = model.predict(X_val)
        return model, val_pred, None
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    global_rate: float,
    config: V9Config,
    log: callable | None = None,
) -> tuple[float, np.ndarray, list[float]]:
    mask_tr = train_df['txn_month'].isin(range(1, 11))
    mask_val = train_df['txn_month'] == 11
    X_tr = train_df.loc[mask_tr, feature_cols]
    X_val = train_df.loc[mask_val, feature_cols]
    X_te = test_df[feature_cols]
    y_tr = train_df.loc[mask_tr, 'FPD_15'].values
    y_val = train_df.loc[mask_val, 'FPD_15'].values
    scale_pos_weight = (1 - global_rate) / (global_rate + 1e-6)

    val_preds = []
    test_preds_list = []
    for mid in config.ensemble_models:
        model, vp, _ = train_model(X_tr, y_tr, X_val, y_val, mid, config, scale_pos_weight)
        val_preds.append(vp)
        X_te_fill = X_te.fillna(X_tr.median())
        obj_cols = X_te_fill.select_dtypes(include=[object, 'category']).columns.tolist()
        for c in obj_cols:
            X_te_fill[c] = pd.to_numeric(X_te_fill[c], errors='coerce').fillna(X_tr[c].median())
        if hasattr(model, 'predict_proba'):
            tp = model.predict_proba(X_te_fill)[:, 1]
        else:
            tp = model.predict(X_te_fill)
        test_preds_list.append(tp)

    weights = optimize_ensemble_weights(val_preds, y_val)
    test_preds = sum(float(weights[i]) * test_preds_list[i] for i in range(len(weights)))
    blend_val = sum(float(weights[i]) * val_preds[i] for i in range(len(weights)))
    val_auc = roc_auc_score(y_val, blend_val)
    return val_auc, test_preds, list(weights)
