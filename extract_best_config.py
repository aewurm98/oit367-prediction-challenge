#!/usr/bin/env python3
"""
Extract best config_id from downloaded Modal experiment results.
Reads modal_outputs/*_results.csv, sorts by nov_auc, outputs best config_id.
Usage: python extract_best_config.py [--fallback-config CONFIG]
"""

import argparse
import glob
import os
import sys

import pandas as pd

OUT_DIR = "modal_outputs"
BEST_FILE = "best_config.txt"
RESULT_PATTERNS = [
    "config_sweep_results.csv",
    "cat_grid_results.csv",
    "lgb_grid_results.csv",
    "lgb_subset_results.csv",
    "full_grid_results.csv",
]


def main():
    parser = argparse.ArgumentParser(description="Extract best config from experiment results")
    parser.add_argument(
        "--fallback-config",
        type=str,
        default="cat_lr_03",
        help="Config to use if no results found (default: cat_lr_03)",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    results_dir = os.path.join(os.getcwd(), OUT_DIR)
    if not os.path.isdir(results_dir):
        print(f"No {OUT_DIR}/ directory. Run ./download_modal_outputs.sh first.", file=sys.stderr)
        print(args.fallback_config)
        return

    dfs = []
    for pattern in RESULT_PATTERNS:
        path = os.path.join(results_dir, pattern)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
                if "config_id" in df.columns and "nov_auc" in df.columns:
                    dfs.append(df)
            except Exception as e:
                print(f"  Skipping {pattern}: {e}", file=sys.stderr)

    if not dfs:
        print(f"No valid results. Using fallback: {args.fallback_config}", file=sys.stderr)
        with open(BEST_FILE, "w") as f:
            f.write(args.fallback_config + "\n")
        print(args.fallback_config)
        return

    combined = pd.concat(dfs, ignore_index=True)
    valid = combined[combined["nov_auc"].notna()]
    if valid.empty:
        print(f"No valid nov_auc. Using fallback: {args.fallback_config}", file=sys.stderr)
        print(args.fallback_config)
        return

    best = valid.loc[valid["nov_auc"].idxmax()]
    config_id = str(best["config_id"])
    nov_auc = best["nov_auc"]

    with open(BEST_FILE, "w") as f:
        f.write(config_id + "\n")

    print(f"Best config: {config_id} (nov_auc={nov_auc:.5f})", file=sys.stderr)
    print("Top 5:", file=sys.stderr)
    top5 = valid.nlargest(5, "nov_auc")[["config_id", "nov_auc", "cv_mean"]]
    print(top5.to_string(index=False), file=sys.stderr)
    print(config_id)


if __name__ == "__main__":
    main()
