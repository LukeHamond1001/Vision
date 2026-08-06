#!/usr/bin/env bash
# Self-driving debug pod: run the phase, push raw results to a
# results-* branch (the pod-ledger pattern), then remove own pod so
# billing stops. Requires env: GIT_TOKEN (push), RUNPOD_POD_ID (auto).
set -uo pipefail

cd /workspace
curl -s https://raw.githubusercontent.com/LukeHamond1001/iga-scale/main/scripts/pod_debug.sh -o pod_debug.sh
bash pod_debug.sh > /workspace/debug_phase.log 2>&1 || true

cd /workspace/iga-scale || exit 1
git config user.email "pod@iga-scale"
git config user.name "iga-pod"
git checkout -b results-v50-debug || git checkout results-v50-debug
cp /workspace/debug_phase.log . 2>/dev/null || true
git add -f debug_phase.log 2>/dev/null || true
git add -f ab_results.txt calib_results.txt 2>/dev/null || true
git add -f results/lm_constants_real.json 2>/dev/null || true
git commit -m "v5.0 debug pod: talk-x-shape A/B + real-data calibration (raw pod output)" || true
git push -f "https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git" results-v50-debug || true

# stop the meter
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
