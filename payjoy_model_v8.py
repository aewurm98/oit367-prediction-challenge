"""
PayJoy FPD_15 — v8
==================
Builds on payjoy_model_clean_v5 and v5 cowork.
- Temporal validation (Jan-Oct train, Nov val)
- Market payment stats from TRAIN ONLY (fix leakage)
- Expanding-window FPD rates (LOCK_NAME, CURRENCY, MANUFACTURER, LOCK_PRODUCT)
- Country z-scores, state_mismatch
- Low-cardinality payment aggregates
- Multi-model ensemble (CatBoost + LightGBM) with optimized weights

Usage:
  python payjoy_model_v8.py
  python payjoy_model_v8.py --config-id lgb_only
  python payjoy_model_v8.py --config-id hp_lgb_nl63_mcs50_lr0.05

Output:
  submission_v8.csv
  run_v8.log
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from lib.log_utils import make_logger
from lib.submission_utils import validate_submission
from lib.v8_pipeline import V8Config, build_features, train_and_eval
from _run_v8_experiments import get_config_by_id

np.random.seed(42)

VERSION = os.environ.get('SUBMISSION_VERSION', 'v8')
SUBMISSION_FILE = f'submission_{VERSION}.csv'
LOG_FILE = f'run_{VERSION}.log'

t0 = time.time()
log = make_logger(LOG_FILE, t0)


def _parse_args():
    parser = argparse.ArgumentParser(description='v8 production model for Kaggle submission')
    parser.add_argument('--config-id', type=str, default=None,
                        help='Config from experiments (e.g. lgb_only, hp_lgb_nl63_mcs50_lr0.05). Default: v8_production.')
    return parser.parse_args()


def _default_config() -> V8Config:
    return V8Config(
        config_id='v8_production',
        use_expanding_rates=True,
        use_country_z=True,
        use_state_mismatch=True,
        use_market_pay_stats=True,
        use_payment_aggregates=True,
        ensemble_models=('cat', 'lgb'),
        weight_strategy='optimize',
        include_rf=False,
    )


args = _parse_args()
config = get_config_by_id(args.config_id) if args.config_id else _default_config()

log("Loading data...")
orders = pd.read_csv('Orders.csv', low_memory=False)
payments = pd.read_csv('Payment_History.csv')
test_ids = pd.read_csv('Test_OrderIDs.csv')

payments.columns = [c.upper().strip() for c in payments.columns]
_PMT_ALIASES = {
    'DAYSOVERDUE': 'DAYS_OVERDUE',
    'PRINCIPALCUMPAID': 'PRINCIPAL_CUMPAID',
    'TOTALBALANCE': 'TOTAL_BALANCE',
    'NUMBEROFPAYMENTS': 'NUMBER_OF_PAYMENTS',
    'ISAFTERCHARGEOFFDATE': 'IS_AFTER_CHARGEOFF_DATE',
}
for alt, canonical in _PMT_ALIASES.items():
    if alt in payments.columns and canonical not in payments.columns:
        payments = payments.rename(columns={alt: canonical})

log(f"Orders: {orders.shape[0]:,}  Payments: {payments.shape[0]:,}  Test: {len(test_ids):,}")
log(f"Config: {config.config_id}")

log("Building features...")
train_df, test_df, feature_cols, global_rate = build_features(orders, payments, test_ids, config, log=log)
log(f"Features: {len(feature_cols)}  |  Train: {len(train_df):,}  |  Test: {len(test_df):,}  |  FPD rate: {global_rate:.4f}")

log("Training ensemble (CatBoost + LightGBM, weights optimized on Nov val)...")
val_auc, test_preds, weights = train_and_eval(train_df, test_df, feature_cols, global_rate, config, log=log)
log(f"Nov AUC: {val_auc:.5f}  |  Weights: {[f'{w:.3f}' for w in weights]}")

submission = pd.DataFrame({
    'FINANCEORDERID': test_df['FINANCEORDERID'].astype(str),
    'FPD_15_pred': test_preds,
})
submission = submission.sort_values('FINANCEORDERID').reset_index(drop=True)

ok, errs = validate_submission(submission, test_ids)
if not ok:
    for e in errs:
        log(f"[FAIL] {e}")
    raise SystemExit(1)
log("[OK] All validation checks passed")

submission.to_csv(SUBMISSION_FILE, index=False)
log(f"Saved {SUBMISSION_FILE} ({len(submission):,} rows)")

log("")
log("=" * 60)
log(f"VERSION: {VERSION}")
log(f"Nov AUC: {val_auc:.5f}")
log(f"Features: {len(feature_cols)}")
log(f"Ensemble weights: {weights}")
log(f"Submission: {SUBMISSION_FILE}")
log("=" * 60)
