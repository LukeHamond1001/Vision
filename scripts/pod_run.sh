#!/usr/bin/env bash
# v5.0 REGISTERED RUN pod (A12, frozen config). Heartbeats every stage
# + every 15 min during training. Env: GIT_TOKEN, RUNPOD_POD_ID.
#
# Frozen: talk=tick shape=uniform d=256 lanes=32 chunk=512 eager
# TF32 + expandable segments; constants from real-data calibration;
# ~600M fresh tokens (~37k steps); no mid-run resume (a death = rerun).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace
rm -rf iga-scale
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"
git config user.name "iga-pod"
git checkout -b results-v50-run

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log train_tail.log eval_results.txt \
    calib_run.txt results/lm_constants_run.json 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v50-run 2>/dev/null || true
}

hb "boot $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pip install -q numpy tokenizers >> prep.log 2>&1
hb "deps installed"

# --- data: bulk-download raw parts (never the streaming API) ---
mkdir -p data
BASE="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
for i in 0 1 2 3; do
  curl -s -L "$BASE/train_$i.jsonl" >> data/ultrachat_raw.jsonl
  echo "part $i done: $(wc -l < data/ultrachat_raw.jsonl) lines" >> prep.log
  hb "download: part $i ($(wc -l < data/ultrachat_raw.jsonl) lines)"
  if [ "$(wc -l < data/ultrachat_raw.jsonl)" -ge 520000 ]; then break; fi
done
curl -s -L "$BASE/train_9.jsonl" | head -n 6000 > data/ultrachat_heldout.jsonl
hb "download: heldout file ($(wc -l < data/ultrachat_heldout.jsonl) lines)"

# --- prep: train shard (~500k convos), eval + calib from the
#     disjoint held-out file (the memorization law, A12) ---
python - >> prep.log 2>&1 <<'EOF'
import iga.lm_data_ultrachat as U
from iga.lm_data_ultrachat import prepare
prepare("data/uc_train", n_convos=500000, seed=0, vocab=16384)
EOF
hb "prep: train shard built"
python - >> prep.log 2>&1 <<'EOF'
import os
os.environ["ULTRACHAT_JSONL"] = "data/ultrachat_heldout.jsonl"
import iga.lm_data_ultrachat as U
from iga.lm_data_ultrachat import prepare
prepare("data/uc_eval", n_convos=2500, seed=2, vocab=16384)
orig = U.iter_convos
U.iter_convos = lambda limit, skip=0: orig(limit, skip=2500)
prepare("data/uc_calib_run", n_convos=1500, seed=3, vocab=16384)
EOF
hb "prep: eval + calib shards built"

python -m iga.lm_calibrate --data data/uc_calib_run --chunks 60 \
  --out results/lm_constants_run.json > calib_run.txt 2>&1
hb "calibration frozen"

# --- the ONE run ---
( while true; do sleep 900; tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; done ) &
HBPID=$!
python -m iga.lm_train run --data data/uc_train --d 256 --lanes 32 \
  --chunk 512 --steps 36600 --talk tick --device cuda --ckpt run.pt \
  --constants results/lm_constants_run.json > train.log 2>&1
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete"

# --- evaluation on the held-out shard: tables, lesion, talk ---
python -m iga.lm_eval --ckpt run.pt --data data/uc_eval --d 256 \
  --talk tick --lanes 4 --chunks 80 > eval_results.txt 2>&1
hb "evaluation complete"

git add -f run.pt train.log prep.log 2>/dev/null || true
git commit -qm "v5.0 registered run: checkpoint + full logs" || true
git push -qf "$PUSH" results-v50-run || true
hb "phase complete"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
