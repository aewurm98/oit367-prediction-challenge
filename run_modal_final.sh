#!/bin/bash
# 60-90 minute optimized Modal runs. All detached — no dependency on terminal or run completion.
# Each run writes to a separate volume path. Run from project root.
# Requires: modal token, data uploaded to payjoy-v8-data volume.

set -e
cd "$(dirname "$0")"

echo "Launching 4 parallel Modal runs (60-90 min optimized, all --detach)..."
echo ""

# Run 1: Config sweep (14 configs) — ~45 min
modal run -d --timestamps modal_v8.py --output-suffix config_sweep &
PID1=$!

# Run 2: CatBoost grid (27 configs) — ~30 min
modal run -d --timestamps modal_v8.py --grid-search --grid-mode cat_only --output-suffix cat &
PID2=$!

# Run 3: LightGBM subset (9 configs) — ~40 min (avoids full 27-config timeout)
modal run -d --timestamps modal_v8.py --grid-search --grid-mode lgb_only --config-ids 1,2,3,4,5,6,7,8,9 --output-suffix lgb_subset &
PID3=$!

# Run 4: v5 GPU turbo (alternative model)
modal run -d --timestamps modal_v8.py --mode v5_turbo --output-suffix v5_turbo &
PID4=$!

echo "PIDs: config_sweep=$PID1 cat=$PID2 lgb_subset=$PID3 v5_turbo=$PID4"
echo "All runs use --detach; jobs continue in cloud even if terminal closes."
echo "Waiting for job submission..."
wait $PID1 && echo "config_sweep submitted" || echo "config_sweep failed (exit $?)"
wait $PID2 && echo "cat submitted" || echo "cat failed (exit $?)"
wait $PID3 && echo "lgb_subset submitted" || echo "lgb_subset failed (exit $?)"
wait $PID4 && echo "v5_turbo submitted" || echo "v5_turbo failed (exit $?)"

echo ""
echo "Done. Check Modal dashboard for progress. Download when ready:"
echo "  ./download_modal_outputs.sh --force"