#!/usr/bin/env bash
# v5.1-LITE binding-curve probe (A16). Debug-width model, dense
# graduated instruments, binding-margin channel. ~2h, ~$0.50.
# All v4 protections: heartbeats, rolling snapshots, eval-first,
# verified ckpt pieces, truthful exit codes.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace
rm -rf iga-scale
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"
git config user.name "iga-pod"
git checkout -b results-v51-lite

mkdir -p /workspace/snap
( cd /workspace/snap && git init -q && git checkout -qb snap \
  && git config user.email pod@iga && git config user.name iga-pod )

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v51-lite 2>/dev/null || true
}

hb "boot lite $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pip install -q numpy tokenizers >> prep.log 2>&1
hb "deps installed"

mkdir -p data
BASE="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
for i in 0 1; do
  curl -s -L "$BASE/train_$i.jsonl" >> data/ultrachat_raw.jsonl
  hb "download: part $i ($(wc -l < data/ultrachat_raw.jsonl) lines)"
done
curl -s -L "$BASE/train_9.jsonl" | head -n 6000 > data/ultrachat_heldout.jsonl
hb "download: heldout"

python - >> prep.log 2>&1 <<'EOF'
from iga.lm_data_ultrachat import prepare
prepare("data/uc_lite", n_convos=150000, seed=0, vocab=16384,
        instrument_every=1)
EOF
hb "prep: dense train shard built"
python - >> prep.log 2>&1 <<'EOF'
import os
os.environ["ULTRACHAT_JSONL"] = "data/ultrachat_heldout.jsonl"
from iga.lm_data_ultrachat import prepare
prepare("data/uc_lite_eval", n_convos=2500, seed=2, instrument_every=1,
        tokenizer_path="data/uc_lite/tokenizer.json")
EOF
hb "prep: dense eval shard built (train tokenizer reused)"

( C=0; while true; do sleep 900; C=$((C+1)); \
    tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; \
    if [ $((C % 4)) -eq 0 ] && [ -f lite.pt ]; then \
      cp lite.pt /workspace/snap/s.pt && \
      ( cd /workspace/snap && rm -f snap_part_* && \
        split -b 25m s.pt snap_part_ && rm -f s.pt && \
        git add -A && git commit -qm "rolling snapshot" && \
        git push -qf "$PUSH" snap:results-v51-ckpt ) && \
      hb "rolling ckpt snapshot pushed"; \
    fi; done ) &
HBPID=$!
python -m iga.lm_train run --data data/uc_lite --d 128 --lanes 32 \
  --chunk 512 --steps 10000 --talk tick --device cuda --ckpt lite.pt \
  > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

if python -m iga.lm_eval --ckpt lite.pt --data data/uc_lite_eval \
    --d 128 --talk tick --lanes 4 --chunks 200 > eval_results.txt 2>&1; then
  hb "evaluation complete"
else
  hb "EVAL FAILED - traceback in eval_results.txt"
fi

split -b 25m lite.pt lite_part_
BASE_SHA=$(git rev-parse HEAD)
CKPT_OK=1
for f in $(ls lite_part_*); do
  git add -f "$f" && git commit -qm "ckpt piece: $f"
  PUSHED=0
  for i in 1 2 3 4; do
    if git push -f "$PUSH" results-v51-lite && \
       [ "$(git ls-remote "$PUSH" results-v51-lite | cut -f1)" = "$(git rev-parse HEAD)" ]; then
      PUSHED=1; break
    fi
    sleep 20
  done
  if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
  hb "ckpt piece landed: $f"
done
# the eval shard comes home too: the binding curve is computed locally
git add -f data/uc_lite_eval/tokens.bin data/uc_lite_eval/events.jsonl \
  data/uc_lite_eval/tokenizer.json train.log prep.log 2>/dev/null || true
git commit -qm "eval shard + logs" 2>/dev/null || true
git push -qf "$PUSH" results-v51-lite 2>/dev/null || true
if [ "$CKPT_OK" = "1" ]; then
  hb "checkpoint FULLY VERIFIED on remote"
else
  git reset --mixed "$BASE_SHA"
  hb "ckpt git pushes incomplete"
fi
hb "phase complete"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
