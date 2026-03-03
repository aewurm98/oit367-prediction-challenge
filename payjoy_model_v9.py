"""
PayJoy FPD_15 — v9 Clean Build
==============================
Combines v5's entity FPD rates + PMT_CORE with v8's Cat+LGB ensemble (cat_lr_03).
Single config, no experiment flags.

Usage:
  python payjoy_model_v9.py

Output:
  submission_v9.csv
  run_v9.log
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from lib.log_utils import make_logger
from lib.submission_utils import validate_submission
from lib.v9_pipeline import V9Config, build_features, train_and_eval

np.random.seed(42)

VERSION = os.environ.get('SUBMISSION_VERSION', 'v9')
SUBMISSION_FILE = f'submission_{VERSION}.csv'
LOG_FILE = f'run_{VERSION}.log'

t0 = time.time()
log = make_logger(LOG_FILE, t0)


def main():
    log("Loading data...")
    orders = pd.read_csv('Orders.csv', low_memory=False)
    payments = pd.read_csv('Payment_History.csv')
    test_ids = pd.read_csv('Test_OrderIDs.csv')

    payments.columns = [c.upper().strip() for c in payments.columns]
    _PMT_ALIASES = {
        'DAYSOVERDUE': 'DAYS_OVERDUE',
        'PRINCIPALCUMPAID': 'PRINCIPAL_CUMPAID',
        'TOTALBALANCE': 'TOTAL_BALANCE',
    }
    for alt, canonical in _PMT_ALIASES.items():
        if alt in payments.columns and canonical not in payments.columns:
            payments = payments.rename(columns={alt: canonical})

    log(f"Orders: {orders.shape[0]:,}  Payments: {payments.shape[0]:,}  Test: {len(test_ids):,}")

    config = V9Config()
    log("Building features (v5 entity rates + PMT_CORE + v8 market)...")
    train_df, test_df, feature_cols, global_rate = build_features(orders, payments, test_ids, config, log=log)
    log(f"Features: {len(feature_cols)}  |  Train: {len(train_df):,}  |  Test: {len(test_df):,}  |  FPD rate: {global_rate:.4f}")

    log("Training Cat+LGB ensemble (cat_lr_03)...")
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


if __name__ == '__main__':
    main()
