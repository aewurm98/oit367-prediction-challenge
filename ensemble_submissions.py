#!/usr/bin/env python3
"""
Ensemble v8 and v5_turbo submission predictions.
Averages FPD_15_pred from both models. Output: submission_ensemble.csv

Usage:
  python ensemble_submissions.py
  python ensemble_submissions.py --v8 submission_v8.csv --v5 modal_outputs/submission_v5_gpu_turbo.csv
  python ensemble_submissions.py --v8 submission_v8.csv --v5 submission_v5_gpu_turbo.csv --weights 0.5 0.5
"""

import argparse
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Ensemble v8 and v5_turbo submission predictions")
    parser.add_argument(
        "--v8",
        type=str,
        default="submission_v8.csv",
        help="Path to v8 submission CSV",
    )
    parser.add_argument(
        "--v5",
        type=str,
        default="modal_outputs/submission_v5_gpu_turbo.csv",
        help="Path to v5_turbo submission CSV",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        metavar=("W_V8", "W_V5"),
        help="Weights for v8 and v5 (default: 0.5 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_ensemble.csv",
        help="Output path",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

    if not os.path.isfile(args.v8):
        print(f"Error: v8 file not found: {args.v8}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.v5):
        print(f"Error: v5 file not found: {args.v5}", file=sys.stderr)
        sys.exit(1)

    w8, w5 = args.weights
    if abs(w8 + w5 - 1.0) > 1e-6:
        w8, w5 = w8 / (w8 + w5), w5 / (w8 + w5)
        print(f"Normalized weights: v8={w8:.3f} v5={w5:.3f}", file=sys.stderr)

    df8 = pd.read_csv(args.v8)
    df5 = pd.read_csv(args.v5)

    if "FINANCEORDERID" not in df8.columns or "FPD_15_pred" not in df8.columns:
        print("Error: v8 must have FINANCEORDERID and FPD_15_pred columns", file=sys.stderr)
        sys.exit(1)
    if "FINANCEORDERID" not in df5.columns or "FPD_15_pred" not in df5.columns:
        print("Error: v5 must have FINANCEORDERID and FPD_15_pred columns", file=sys.stderr)
        sys.exit(1)

    df8["FINANCEORDERID"] = df8["FINANCEORDERID"].astype(str)
    df5["FINANCEORDERID"] = df5["FINANCEORDERID"].astype(str)
    df8 = df8.rename(columns={"FPD_15_pred": "pred_v8"})
    df5 = df5.rename(columns={"FPD_15_pred": "pred_v5"})

    merged = df8[["FINANCEORDERID", "pred_v8"]].merge(
        df5[["FINANCEORDERID", "pred_v5"]],
        on="FINANCEORDERID",
        how="inner",
    )
    merged["FPD_15_pred"] = w8 * merged["pred_v8"] + w5 * merged["pred_v5"]
    merged = merged[["FINANCEORDERID", "FPD_15_pred"]].sort_values("FINANCEORDERID").reset_index(drop=True)

    merged.to_csv(args.output, index=False)
    print(f"Ensemble saved: {args.output} ({len(merged):,} rows)", file=sys.stderr)
    print(f"  Weights: v8={w8:.3f} v5={w5:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
