#!/usr/bin/env bash
# v5.7 (A30): XL chunk carry (attention owns one chunk of lookback) + gated matrix reads (induction protected) + live held-out circuit probes: attention + band latents + live imagination + full drive, past the emergence threshold. Debug-width model, dense
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
git checkout -b results-abl

mkdir -p /workspace/snap
( cd /workspace/snap && git init -q && git checkout -qb snap \
  && git config user.email pod@iga && git config user.name iga-pod )

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt prep.log abl.pt.trace.jsonl; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-abl 2>/dev/null || true
}

hb "boot abl-DRIVE-LAM0 $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ram $(free -g 2>/dev/null | awk '/Mem:/{print $2}')G"
pip install -q numpy tokenizers >> prep.log 2>&1
hb "deps installed"

# CUDA canary (wrapper v5): nvidia-smi can pass while cuInit is wedged —
# run 1's GPU was dead the whole time and we spent 47 CPU-minutes
# finding out. Probe with a real device allocation BEFORE downloads.
cuda_canary() {
  python - <<'EOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda") + 1  # force real allocation + kernel
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
EOF
}
if ! cuda_canary >> prep.log 2>&1; then
  hb "CUDA BROKEN AT BOOT - canary failed, aborting (see prep.log)"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "cuda canary passed"

# A28 arm B is a FRESH run; only the tokenizer comes from v5.4
# (identical ids -> tables comparable head-to-head)
git fetch -q origin results-v54 --depth 1
mkdir -p data/tokref
git show FETCH_HEAD:data/uc_v54_eval/tokenizer.json > data/tokref/tokenizer.json
if [ ! -s data/tokref/tokenizer.json ]; then
  hb "TOKENIZER FETCH FAILED - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "v54 tokenizer fetched"

mkdir -p data
BASE="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
for i in 0 1 2 3 4 5 6 7 8 9; do
  curl -s -L "$BASE/train_$i.jsonl" >> data/ultrachat_raw.jsonl
  hb "download: part $i ($(wc -l < data/ultrachat_raw.jsonl) lines)"
done
curl -s -L "$BASE/train_9.jsonl" | head -n 6000 > data/ultrachat_heldout.jsonl
hb "download: heldout"

# prep runs in the background with 5-min beats — this phase OOM-died
# silently on runs 1-2 while an unconditional heartbeat said "built".
# The success beat now requires rc=0 AND tokens.bin on disk.
python - >> prep.log 2>&1 <<'EOF' &
from iga.lm_data_ultrachat import prepare
prepare("data/uc_abl", n_convos=1500000, seed=0, vocab=16384,
        instrument_every=1, tokenizer_path="data/tokref/tokenizer.json")
EOF
PREP_PID=$!
while kill -0 $PREP_PID 2>/dev/null; do
  sleep 300
  kill -0 $PREP_PID 2>/dev/null && \
    hb "prep beat: $(tail -1 prep.log | cut -c1-90)"
done
wait $PREP_PID; PREP_RC=$?
if [ "$PREP_RC" -ne 0 ] || [ ! -s data/uc_abl/tokens.bin ]; then
  hb "PREP FAILED (rc=$PREP_RC, tokens.bin $( [ -f data/uc_abl/tokens.bin ] && echo present || echo MISSING)) - see prep.log"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
rm -f data/ultrachat_raw.jsonl
hb "prep: train shard built ($(stat -c%s data/uc_abl/tokens.bin) bytes tokens.bin, raw deleted)"
if python - >> prep.log 2>&1 <<'EOF'
import os
os.environ["ULTRACHAT_JSONL"] = "data/ultrachat_heldout.jsonl"
from iga.lm_data_ultrachat import prepare
prepare("data/uc_abl_eval", n_convos=2500, seed=2, instrument_every=1,
        tokenizer_path="data/uc_abl/tokenizer.json")
EOF
then
  hb "prep: dense eval shard built (train tokenizer reused)"
else
  hb "EVAL-SHARD PREP FAILED - see prep.log"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi

# the GPU can also die during the ~45 CPU-minutes above — reprobe
if ! cuda_canary >> prep.log 2>&1; then
  hb "CUDA DIED PRE-TRAIN - canary failed, aborting (see prep.log)"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "cuda canary passed pre-train"

( C=0; while true; do sleep 900; C=$((C+1)); \
    tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; \
    if [ $((C % 4)) -eq 0 ] && [ -f abl.pt ]; then \
      cp abl.pt /workspace/snap/s.pt && \
      ( cd /workspace/snap && rm -f snap_part_* && \
        split -b 25m s.pt snap_part_ && rm -f s.pt && \
        git add -A && git commit -qm "rolling snapshot" && \
        git push -qf "$PUSH" snap:results-abl-ckpt ) && \
      hb "rolling ckpt snapshot pushed"; \
    fi; done ) &
HBPID=$!
python -m iga.lm_train run --data data/uc_abl --d 128 --lanes 32 \
  --chunk 512 --steps 135000 --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lam 0 --eval-data data/uc_abl_eval \
  --ckpt abl.pt > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

if python -m iga.lm_eval --ckpt abl.pt --data data/uc_abl_eval \
    --d 128 --arch hybrid --store matrix --xl off --chunk 512 --lanes 4 --chunks 200 \
    > eval_results.txt 2>&1; then
  hb "evaluation complete"
else
  hb "EVAL FAILED - traceback in eval_results.txt"
fi

# truthfulness (wrapper v5): a missing checkpoint must FAIL this phase,
# not vacuously pass it — run 1 heartbeat "FULLY VERIFIED" with no
# checkpoint in existence because the piece loop ran zero times
BASE_SHA=$(git rev-parse HEAD)
if [ ! -f abl.pt ]; then
  hb "NO CHECKPOINT EXISTS - nothing to verify (train rc=$TRAIN_RC)"
  CKPT_OK=0
else
split -b 25m abl.pt abl_part_
CKPT_OK=1
for f in $(ls abl_part_*); do
  git add -f "$f" && git commit -qm "ckpt piece: $f"
  PUSHED=0
  for i in 1 2 3 4; do
    if git push -f "$PUSH" results-abl && \
       [ "$(git ls-remote "$PUSH" results-abl | cut -f1)" = "$(git rev-parse HEAD)" ]; then
      PUSHED=1; break
    fi
    sleep 20
  done
  if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
  hb "ckpt piece landed: $f"
done
fi
# the eval shard comes home too: the binding curve is computed locally
git add -f data/uc_abl_eval/tokens.bin data/uc_abl_eval/events.jsonl \
  data/uc_abl_eval/tokenizer.json train.log prep.log \
  abl.pt.trace.jsonl 2>/dev/null || true
git commit -qm "eval shard + logs" 2>/dev/null || true
git push -qf "$PUSH" results-abl 2>/dev/null || true
if [ "$CKPT_OK" = "1" ]; then
  hb "checkpoint FULLY VERIFIED on remote"
elif [ -f abl.pt ]; then
  git reset --mixed "$BASE_SHA"
  hb "ckpt git pushes incomplete"
fi
hb "phase complete (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
