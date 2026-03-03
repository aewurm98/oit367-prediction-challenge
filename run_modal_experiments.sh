#!/bin/bash
# Run all 4 Modal experiment runs in parallel.
# Each writes to a separate volume path (output_config_sweep, output_lgb, output_cat, output_full).
# Run from project root. Requires: modal token, data uploaded to payjoy-v8-data volume.

set -e
cd "$(dirname "$0")"

echo "Launching 4 parallel Modal experiment runs..."
echo ""

# Run 1: Config sweep (14 configs)
modal run -d --timestamps modal_v8.py --output-suffix config_sweep &
PID1=$!

# Run 2: LightGBM grid (27 configs)
modal run -d --timestamps modal_v8.py --grid-search --grid-mode lgb_only --output-suffix lgb &
PID2=$!

# Run 3: CatBoost grid (27 configs)
modal run -d --timestamps modal_v8.py --grid-search --grid-mode cat_only --output-suffix cat &
PID3=$!

# Run 4: Full grid (729 configs) - may exceed 2h timeout
modal run -d --timestamps modal_v8.py --grid-search --grid-mode full --output-suffix full &
PID4=$!

echo "PIDs: config_sweep=$PID1 lgb=$PID2 cat=$PID3 full=$PID4"
echo "All runs launched with --detach (continue even if terminal closes)."
echo "Waiting for local processes to submit jobs..."
wait $PID1 && echo "config_sweep submitted" || echo "config_sweep submit failed (exit $?)"
wait $PID2 && echo "lgb submitted" || echo "lgb submit failed (exit $?)"
wait $PID3 && echo "cat submitted" || echo "cat submit failed (exit $?)"
wait $PID4 && echo "full submitted" || echo "full submit failed (exit $?)"

echo ""
echo "All runs finished. Download outputs with: ./download_modal_outputs.sh"
