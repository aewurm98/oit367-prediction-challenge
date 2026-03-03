#!/bin/bash
# Download all Modal experiment outputs to local modal_outputs/ directory.
# Run after run_modal_experiments.sh completes. Pass --force to overwrite existing files.

set -e
cd "$(dirname "$0")"

OUT_DIR="modal_outputs"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "Downloading Modal outputs to $OUT_DIR/..."

# Config sweep (14 configs)
modal volume get payjoy-v8-data output_config_sweep/v8_experiments_results.csv ./config_sweep_results.csv "$@" 2>/dev/null || echo "  config_sweep_results.csv not found (run may have failed)"
modal volume get payjoy-v8-data output_config_sweep/v8_experiments_params.csv ./config_sweep_params.csv "$@" 2>/dev/null || true
modal volume get payjoy-v8-data output_config_sweep/v8_experiments.log ./config_sweep.log "$@" 2>/dev/null || true

# LightGBM grid (27 configs)
modal volume get payjoy-v8-data output_lgb/v8_experiments_results.csv ./lgb_grid_results.csv "$@" 2>/dev/null || echo "  lgb_grid_results.csv not found"
modal volume get payjoy-v8-data output_lgb/v8_experiments_params.csv ./lgb_grid_params.csv "$@" 2>/dev/null || true
modal volume get payjoy-v8-data output_lgb/v8_experiments.log ./lgb_grid.log "$@" 2>/dev/null || true

# CatBoost grid (27 configs)
modal volume get payjoy-v8-data output_cat/v8_experiments_results.csv ./cat_grid_results.csv "$@" 2>/dev/null || echo "  cat_grid_results.csv not found"
modal volume get payjoy-v8-data output_cat/v8_experiments_params.csv ./cat_grid_params.csv "$@" 2>/dev/null || true
modal volume get payjoy-v8-data output_cat/v8_experiments.log ./cat_grid.log "$@" 2>/dev/null || true

# Full grid (729 configs)
modal volume get payjoy-v8-data output_full/v8_experiments_results.csv ./full_grid_results.csv "$@" 2>/dev/null || echo "  full_grid_results.csv not found (run may have timed out)"
modal volume get payjoy-v8-data output_full/v8_experiments_params.csv ./full_grid_params.csv "$@" 2>/dev/null || true
modal volume get payjoy-v8-data output_full/v8_experiments.log ./full_grid.log "$@" 2>/dev/null || true

# LightGBM subset (9 configs, from run_modal_final.sh)
modal volume get payjoy-v8-data output_lgb_subset/v8_experiments_results.csv ./lgb_subset_results.csv "$@" 2>/dev/null || echo "  lgb_subset_results.csv not found"
modal volume get payjoy-v8-data output_lgb_subset/v8_experiments_params.csv ./lgb_subset_params.csv "$@" 2>/dev/null || true
modal volume get payjoy-v8-data output_lgb_subset/v8_experiments.log ./lgb_subset.log "$@" 2>/dev/null || true

# v5 GPU turbo submission (from run_modal_final.sh)
modal volume get payjoy-v8-data output_v5_turbo/submission_v5_gpu_turbo.csv ./submission_v5_gpu_turbo.csv "$@" 2>/dev/null || echo "  submission_v5_gpu_turbo.csv not found"

echo ""
echo "Done. Best config by nov_auc (column 2):"
echo "  tail -n +2 config_sweep_results.csv cat_grid_results.csv lgb_grid_results.csv lgb_subset_results.csv full_grid_results.csv 2>/dev/null | sort -t',' -k2 -rn | head -5"
