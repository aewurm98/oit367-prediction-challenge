"""
v8 Pipeline — shared feature building and training for payjoy_model_v8 and _run_v8_experiments.
Builds on payjoy_model_clean_v5 and v5 cowork patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Optional imports (lazy)
_CATBOOST = None
_LGB = None
_XGB = None


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


def _get_xgb():
    global _XGB
    if _XGB is None:
        try:
            import xgboost as xgb
            _XGB = xgb
        except ImportError:
            _XGB = False
    return _XGB


@dataclass
class V8Config:
    """Configuration for v8 pipeline."""
    config_id: str = "baseline"
    use_expanding_rates: bool = True
    use_country_z: bool = True
    use_state_mismatch: bool = True
    use_market_pay_stats: bool = True
    use_payment_aggregates: bool = True
    ensemble_models: tuple = ('cat', 'lgb')
    weight_strategy: str = 'optimize'
    include_rf: bool = False
    depth: int = 6
    l2_leaf_reg: float = 3.0
    learning_rate: float = 0.05
    iterations: int = 1000
    early_stopping_rounds: int = 50
    random_state: int = 42
    # Per-model overrides (None = use depth/l2_leaf_reg/learning_rate/iterations)
    cat_depth: int | None = None
    cat_l2_leaf_reg: float | None = None
    cat_learning_rate: float | None = None
    lgb_num_leaves: int | None = None
    lgb_min_child_samples: int | None = None
    lgb_learning_rate: float | None = None
    xgb_max_depth: int | None = None
    xgb_learning_rate: float | None = None
    rf_n_estimators: int | None = None
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int | None = None

    def feature_signature(self) -> str:
        """Hash for feature caching (excludes HP)."""
        return f"exp{self.use_expanding_rates}_cz{self.use_country_z}_sm{self.use_state_mismatch}_mp{self.use_market_pay_stats}_pa{self.use_payment_aggregates}"


K_SMOOTH = 10
EXPANDING_RATE_COLS = ['LOCK_NAME', 'CURRENCY', 'MANUFACTURER', 'LOCK_PRODUCT', 'STATE']
LOW_CARD_PAY_GROUPS = ['LOCK_NAME', 'CURRENCY']


def _smooth_rate(fpd_sum: float, n: int, prior_rate: float, k: int = K_SMOOTH) -> float:
    return (fpd_sum + prior_rate * k) / (n + k)


def build_features(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    test_ids: pd.DataFrame,
    config: V8Config,
    log: callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """
    Build v8 features. Returns (train_df, test_df, feature_cols, global_rate).
    Uses temporal split: train = Jan-Oct, val = Nov. Test = test IDs.
    """
    def _log(msg: str):
        if log:
            log(msg)

    # Normalize payment columns
    payments = payments.copy()
    payments.columns = [c.upper().strip() for c in payments.columns]
    oid_col = next((c for c in payments.columns if 'ORDER' in c and 'ID' in c), payments.columns[0])
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

    # Parse timestamps
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

    if config.use_state_mismatch and 'USER_STATE' in orders.columns and 'MERCHANT_STATE' in orders.columns:
        orders['state_mismatch'] = (
            orders['USER_STATE'].fillna('__X__') != orders['MERCHANT_STATE'].fillna('__X__')
        ).astype(float)

    if config.use_country_z and 'COUNTRY' in orders.columns:
        eps = 1e-6
        for col in ['FINANCE_AMOUNT', 'PURCHASE_AMOUNT', 'TOTAL_DUE', 'DOWN_PAYMENT_AMOUNT']:
            if col in orders.columns:
                mu = orders.groupby('COUNTRY')[col].transform('mean')
                sig = orders.groupby('COUNTRY')[col].transform('std').replace(0, 1)
                orders[f'{col}_z'] = (orders[col] - mu) / sig

    if config.use_market_pay_stats and numeric_pay:
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

    if config.use_payment_aggregates and numeric_pay:
        ord_cols = ['FINANCEORDERID'] + [c for c in LOW_CARD_PAY_GROUPS if c in orders.columns]
        if len(ord_cols) > 1:
            pay_ctx = payments.merge(orders[ord_cols].drop_duplicates(), left_on=oid_col, right_on='FINANCEORDERID', how='inner')
            pay_ctx_train = pay_ctx[pay_ctx['FINANCEORDERID'].isin(train_ids)]
            for grp in LOW_CARD_PAY_GROUPS:
                if grp in pay_ctx_train.columns:
                    stats = pay_ctx_train.groupby(grp)[numeric_pay].agg(['mean', 'std'])
                    stats.columns = [f'{grp}_lcpay_{a}_{b}' for a, b in stats.columns]
                    orders = orders.merge(stats.reset_index(), on=grp, how='left')

    train_df = orders[orders['FPD_15'].notna() & ~orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()
    test_df = orders[orders['FINANCEORDERID_str'].isin(test_order_ids)].copy()
    global_rate = train_df['FPD_15'].mean()

    if config.use_expanding_rates:
        rate_cols = [c for c in EXPANDING_RATE_COLS if c in train_df.columns]
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
    if config.use_state_mismatch and 'state_mismatch' in orders.columns:
        base_features.append('state_mismatch')
    if config.use_country_z:
        base_features.extend([c for c in orders.columns if c.endswith('_z')])
    if config.use_expanding_rates:
        base_features.extend([f'{c}_fpd_rate' for c in EXPANDING_RATE_COLS if c in train_df.columns and f'{c}_fpd_rate' in train_df.columns])
    for c in orders.columns:
        if c.startswith('COUNTRY_pay_') or c.startswith('MANUFACTURER_pay_') or c.startswith('MERCHANT_STATE_pay_'):
            if c not in base_features:
                base_features.append(c)
    for c in orders.columns:
        if any(c.startswith(f'{g}_lcpay_') for g in LOW_CARD_PAY_GROUPS) and c not in base_features:
            base_features.append(c)

    feature_cols = [c for c in base_features if c in train_df.columns and c in test_df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    return train_df, test_df, feature_cols, global_rate


def optimize_ensemble_weights(val_preds: list[np.ndarray], y_val: np.ndarray) -> np.ndarray:
    """Find weights that maximize AUC on validation. Returns normalized weights."""
    from scipy.optimize import minimize

    n = len(val_preds)
    if n == 1:
        return np.array([1.0])

    def neg_auc(w):
        blend = sum(w[i] * val_preds[i] for i in range(n))
        return -roc_auc_score(y_val, blend)

    result = minimize(
        neg_auc,
        x0=np.ones(n) / n,
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
    config: V8Config,
    scale_pos_weight: float,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Train a single model. Returns (model, val_preds, test_preds)."""
    X_tr = X_tr.fillna(X_tr.median())
    X_val = X_val.fillna(X_tr.median())
    for c in X_tr.columns:
        if X_tr[c].dtype == object or X_tr[c].dtype.name == 'category':
            X_tr[c] = pd.to_numeric(X_tr[c], errors='coerce').fillna(X_tr[c].median())
            X_val[c] = pd.to_numeric(X_val[c], errors='coerce').fillna(X_tr[c].median())

    if model_id == 'cat':
        CatBoost = _get_catboost()
        if not CatBoost:
            import warnings
            warnings.warn("catboost not installed, falling back to LightGBM for 'cat'")
            return train_model(X_tr, y_tr, X_val, y_val, 'lgb', config, scale_pos_weight)
        _depth = config.cat_depth if config.cat_depth is not None else config.depth
        _l2 = config.cat_l2_leaf_reg if config.cat_l2_leaf_reg is not None else config.l2_leaf_reg
        _lr = config.cat_learning_rate if config.cat_learning_rate is not None else config.learning_rate
        model = CatBoost(
            iterations=config.iterations,
            learning_rate=_lr,
            depth=_depth,
            l2_leaf_reg=_l2,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            early_stopping_rounds=config.early_stopping_rounds,
            random_seed=config.random_state,
            verbose=0,
            task_type='CPU',
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        val_pred = model.predict_proba(X_val)[:, 1]
        return model, val_pred, None

    elif model_id == 'lgb':
        lgb = _get_lgb()
        if not lgb:
            raise ImportError("lightgbm required: pip install lightgbm")
        _nl = config.lgb_num_leaves
        if _nl is None:
            _nl = 2 ** config.depth - 1 if config.depth <= 10 else 63
        _mcs = config.lgb_min_child_samples if config.lgb_min_child_samples is not None else 50
        _lr = config.lgb_learning_rate if config.lgb_learning_rate is not None else config.learning_rate
        params = {
            'objective': 'binary', 'metric': 'auc',
            'learning_rate': _lr,
            'num_leaves': _nl,
            'min_child_samples': _mcs,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1, 'n_jobs': -1, 'random_state': config.random_state,
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

    elif model_id == 'xgb':
        xgb = _get_xgb()
        if not xgb:
            raise ImportError("xgboost required: pip install xgboost")
        _md = config.xgb_max_depth if config.xgb_max_depth is not None else config.depth
        _lr = config.xgb_learning_rate if config.xgb_learning_rate is not None else config.learning_rate
        model = xgb.XGBClassifier(
            n_estimators=config.iterations,
            learning_rate=_lr,
            max_depth=_md,
            reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = model.predict_proba(X_val)[:, 1]
        return model, val_pred, None

    elif model_id == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        _ne = config.rf_n_estimators if config.rf_n_estimators is not None else 200
        _md = config.rf_max_depth if config.rf_max_depth is not None else 12
        _msl = config.rf_min_samples_leaf if config.rf_min_samples_leaf is not None else 50
        model = RandomForestClassifier(
            n_estimators=_ne, max_depth=_md, min_samples_leaf=_msl,
            max_features='sqrt', class_weight='balanced',
            n_jobs=-1, random_state=config.random_state,
        )
        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)[:, 1]
        return model, val_pred, None

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    global_rate: float,
    config: V8Config,
    log: callable | None = None,
) -> tuple[float, np.ndarray, list[float] | None]:
    """
    Train ensemble, optimize weights, return (val_auc, test_preds, weights).
    Uses Jan-Oct train, Nov val.
    """
    mask_tr = train_df['txn_month'].isin(range(1, 11))
    mask_val = train_df['txn_month'] == 11
    X_tr = train_df.loc[mask_tr, feature_cols]
    X_val = train_df.loc[mask_val, feature_cols]
    X_te = test_df[feature_cols]
    y_tr = train_df.loc[mask_tr, 'FPD_15'].values
    y_val = train_df.loc[mask_val, 'FPD_15'].values
    scale_pos_weight = (1 - global_rate) / (global_rate + 1e-6)

    models_to_train = list(config.ensemble_models)
    if config.include_rf:
        models_to_train.append('rf')

    val_preds = []
    test_preds_list = []

    for mid in models_to_train:
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

    if config.include_rf and len(val_preds) >= 2:
        rf_auc = roc_auc_score(y_val, val_preds[-1])
        best_single = max(roc_auc_score(y_val, vp) for vp in val_preds[:-1])
        if abs(best_single - rf_auc) >= 0.02:
            val_preds = val_preds[:-1]
            test_preds_list = test_preds_list[:-1]

    if len(val_preds) == 1:
        weights = np.array([1.0])
        test_preds = test_preds_list[0]
    else:
        if config.weight_strategy == 'optimize':
            weights = optimize_ensemble_weights(val_preds, y_val)
        else:
            weights = np.ones(len(val_preds)) / len(val_preds)
        test_preds = sum(float(weights[i]) * test_preds_list[i] for i in range(len(weights)))

    blend_val = sum(float(weights[i]) * val_preds[i] for i in range(len(weights)))
    val_auc = roc_auc_score(y_val, blend_val)

    return val_auc, test_preds, list(weights)


# 3-fold expanding-window temporal CV (matches v6 schema)
CV_SPLITS = [
    (list(range(1, 8)), list(range(8, 11))),   # train Jan-Jul, val Aug-Oct
    (list(range(1, 10)), list(range(10, 12))),  # train Jan-Sep, val Oct-Nov
    (list(range(1, 11)), [11]),                 # train Jan-Oct, val Nov
]


def _train_and_blend_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    global_rate: float,
    config: V8Config,
    tr_months: list[int],
    val_months: list[int],
    scale_pos_weight: float,
) -> tuple[float, np.ndarray, list[float]]:
    """Train ensemble on tr_months, evaluate on val_months. Returns (auc, test_preds, weights)."""
    mask_tr = train_df['txn_month'].isin(tr_months)
    mask_val = train_df['txn_month'].isin(val_months)
    X_tr = train_df.loc[mask_tr, feature_cols]
    X_val = train_df.loc[mask_val, feature_cols]
    X_te = test_df[feature_cols]
    y_tr = train_df.loc[mask_tr, 'FPD_15'].values
    y_val = train_df.loc[mask_val, 'FPD_15'].values

    models_to_train = list(config.ensemble_models)
    if config.include_rf:
        models_to_train.append('rf')

    val_preds = []
    test_preds_list = []

    for mid in models_to_train:
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

    if config.include_rf and len(val_preds) >= 2:
        rf_auc = roc_auc_score(y_val, val_preds[-1])
        best_single = max(roc_auc_score(y_val, vp) for vp in val_preds[:-1])
        if abs(best_single - rf_auc) >= 0.02:
            val_preds = val_preds[:-1]
            test_preds_list = test_preds_list[:-1]

    if len(val_preds) == 1:
        weights = np.array([1.0])
        test_preds = test_preds_list[0]
    else:
        if config.weight_strategy == 'optimize':
            weights = optimize_ensemble_weights(val_preds, y_val)
        else:
            weights = np.ones(len(val_preds)) / len(val_preds)
        test_preds = sum(float(weights[i]) * test_preds_list[i] for i in range(len(weights)))

    blend_val = sum(float(weights[i]) * val_preds[i] for i in range(len(weights)))
    auc = roc_auc_score(y_val, blend_val)
    return auc, test_preds, list(weights)


def train_and_eval_with_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    global_rate: float,
    config: V8Config,
    log: callable | None = None,
) -> tuple[float, float, float, list[float], np.ndarray, list[float]]:
    """
    Train ensemble with 3-fold temporal CV, then full Jan-Oct train for Nov/test.
    Returns (nov_auc, cv_mean, cv_std, cv_fold_aucs, test_preds, weights).
    """
    def _log(msg: str):
        if log:
            log(msg)

    scale_pos_weight = (1 - global_rate) / (global_rate + 1e-6)
    cv_fold_aucs = []

    for fold_idx, (tr_months, val_months) in enumerate(CV_SPLITS, 1):
        auc, _, _ = _train_and_blend_fold(
            train_df, test_df, feature_cols, global_rate, config,
            tr_months, val_months, scale_pos_weight,
        )
        cv_fold_aucs.append(auc)
        tr_str = f"{min(tr_months)}-{max(tr_months)}"
        val_str = f"{min(val_months)}-{max(val_months)}"
        _log(f"  Fold {fold_idx} (train {tr_str}, val {val_str}): AUC={auc:.5f}")

    cv_mean = float(np.mean(cv_fold_aucs))
    cv_std = float(np.std(cv_fold_aucs))

    # Full Jan-Oct train, Nov val, test preds
    nov_auc, test_preds, weights = _train_and_blend_fold(
        train_df, test_df, feature_cols, global_rate, config,
        list(range(1, 11)), [11], scale_pos_weight,
    )

    _log(f"  CV mean={cv_mean:.5f} +/- {cv_std:.5f}  Nov AUC={nov_auc:.5f}  weights={[f'{w:.3f}' for w in weights]}")

    return nov_auc, cv_mean, cv_std, cv_fold_aucs, test_preds, weights
