#!/usr/bin/env bash
# Self-driving debug pod v2: per-stage heartbeat pushes (never blind),
# prebuilt-data branch support, self-removing. Env: GIT_TOKEN,
# RUNPOD_POD_ID (auto-set by RunPod).
set -uo pipefail

cd /workspace
rm -rf iga-scale
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"
git config user.name "iga-pod"
git checkout -b results-v50-debug

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log 2>/dev/null || true
  git add -f ab_results.txt calib_results.txt debug_phase.log 2>/dev/null || true
  git add -f results/lm_constants_real.json 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v50-debug 2>/dev/null || true
}

hb "boot $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pip install -q numpy datasets tokenizers >> debug_phase.log 2>&1
hb "deps installed"

if git ls-remote --exit-code origin data-v50 >/dev/null 2>&1; then
  git fetch --depth 1 origin data-v50
  git checkout FETCH_HEAD -- data/
  hb "data: prebuilt shards fetched"
else
  python -m iga.lm_data_ultrachat prepare --convos 8000 \
    --out data/uc_debug --vocab 16384 >> debug_phase.log 2>&1
  hb "data: train shard built"
  python - >> debug_phase.log 2>&1 <<'EOF'
import iga.lm_data_ultrachat as U
from iga.lm_data_ultrachat import prepare
orig = U.iter_convos
U.iter_convos = lambda limit, skip=0: orig(limit, skip=8000)
prepare("data/uc_calib", n_convos=1500, seed=1, vocab=16384)
EOF
  hb "data: calib shard built"
fi

python -m iga.lm_ab --steps 400 --d 128 --lanes 8 --chunk 512 \
  --data data/uc_debug > ab_results.txt 2>&1
hb "A/B complete"
python -m iga.lm_calibrate --data data/uc_calib --chunks 60 \
  --out results/lm_constants_real.json > calib_results.txt 2>&1
hb "calibration complete"
hb "phase complete"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
