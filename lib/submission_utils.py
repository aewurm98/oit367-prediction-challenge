"""
Submission validation utilities for Kaggle FPD_15 predictions.
Shared across _run_v5_cowork.py, best_model_pipeline, model_experiments, and benchmark notebook.
"""

import pandas as pd


def validate_submission(
    submission: pd.DataFrame,
    test_ids: pd.DataFrame,
    pred_col: str = 'FPD_15_pred',
    id_col: str = 'FINANCEORDERID',
) -> tuple[bool, list[str]]:
    """
    Validate a Kaggle submission DataFrame against test IDs.

    Returns:
        (all_passed, list of error messages)
    """
    errors = []

    if submission[id_col].duplicated().any():
        errors.append("Duplicate FINANCEORDERIDs in submission")

    exp = set(test_ids[id_col].astype(str))
    got = set(submission[id_col].astype(str))
    if exp != got:
        missing = len(exp - got)
        extra = len(got - exp)
        if missing:
            errors.append(f"Missing {missing} test IDs in submission")
        if extra:
            errors.append(f"Extra {extra} IDs not in test set")

    if pred_col not in submission.columns:
        errors.append(f"Missing prediction column '{pred_col}'")
    elif not submission[pred_col].between(0, 1, inclusive='both').all():
        errors.append("Predictions must be probabilities in [0, 1]")

    if submission.isnull().any().any():
        errors.append("Submission contains missing (NaN) values")

    if len(submission) != len(test_ids):
        errors.append(f"Row count mismatch: submission has {len(submission)}, expected {len(test_ids)}")

    return (len(errors) == 0, errors)
