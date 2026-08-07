#!/usr/bin/env bash
# v5.0 REGISTERED RUN pod — wrapper v4. Lessons welded in:
#   run 1: eval must share the train tokenizer; big single pushes fail.
#   run 2: reset --hard DELETES the working tree (killed run.pt six
#          hours before the eval needed it); hb must report failures.
# v4 order: train -> EVAL FIRST -> tables pushed -> ckpt pieces
# (verified, reset --mixed on failure, external mirror). Plus a
# rolling ckpt snapshot to a side branch every 2h DURING training,
# from an isolated repo dir — a mid-run pod death can no longer lose
# the artifact either.
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

mkdir -p /workspace/snap
( cd /workspace/snap && git init -q && git checkout -qb snap \
  && git config user.email pod@iga && git config user.name iga-pod )

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
      calib_run.txt results/lm_constants_run.json; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v50-run 2>/dev/null || true
}

hb "boot v4 $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pip install -q numpy tokenizers >> prep.log 2>&1
hb "deps installed"

# --- data: bulk-download raw parts (never the streaming API) ---
mkdir -p data
BASE="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
for i in 0 1 2 3; do
  curl -s -L "$BASE/train_$i.jsonl" >> data/ultrachat_raw.jsonl
  hb "download: part $i ($(wc -l < data/ultrachat_raw.jsonl) lines)"
  if [ "$(wc -l < data/ultrachat_raw.jsonl)" -ge 520000 ]; then break; fi
done
curl -s -L "$BASE/train_9.jsonl" | head -n 6000 > data/ultrachat_heldout.jsonl
hb "download: heldout file"

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
TOK = "data/uc_train/tokenizer.json"
prepare("data/uc_eval", n_convos=2500, seed=2, tokenizer_path=TOK)
orig = U.iter_convos
U.iter_convos = lambda limit, skip=0: orig(limit, skip=2500)
prepare("data/uc_calib_run", n_convos=1500, seed=3, tokenizer_path=TOK)
EOF
hb "prep: eval + calib shards built (train tokenizer reused)"

python -m iga.lm_calibrate --data data/uc_calib_run --chunks 60 \
  --out results/lm_constants_run.json > calib_run.txt 2>&1
hb "calibration frozen"

# --- the ONE run, with 2h rolling ckpt snapshots from isolation ---
( C=0; while true; do sleep 900; C=$((C+1)); \
    tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; \
    if [ $((C % 8)) -eq 0 ] && [ -f run.pt ]; then \
      cp run.pt /workspace/snap/run_snap.pt && \
      ( cd /workspace/snap && rm -f snap_part_* && \
        split -b 25m run_snap.pt snap_part_ && rm -f run_snap.pt && \
        git add -A && git commit -qm "rolling snapshot" && \
        git push -qf "$PUSH" snap:results-v50-ckpt ) && \
      hb "rolling ckpt snapshot pushed"; \
    fi; done ) &
HBPID=$!
python -m iga.lm_train run --data data/uc_train --d 256 --lanes 32 \
  --chunk 512 --steps 36600 --talk tick --device cuda --ckpt run.pt \
  --constants results/lm_constants_run.json > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

# --- EVAL FIRST: the verdict can never again be starved by
#     checkpoint handling ---
if python -m iga.lm_eval --ckpt run.pt --data data/uc_eval --d 256 \
    --talk tick --lanes 4 --chunks 160 > eval_results.txt 2>&1; then
  hb "evaluation complete"
else
  hb "EVAL FAILED - traceback in eval_results.txt"
fi

# --- checkpoint pieces, each push verified; failure keeps the
#     working tree (reset --mixed) and mirrors externally ---
python - <<'EOF'
import torch
s = torch.load("run.pt", map_location="cpu", weights_only=False)
torch.save({"model": {k: v.half() for k, v in s["model"].items()},
            "step": s["step"], "fp16": True}, "run_fp16.pt")
EOF
split -b 25m run.pt run_part_
BASE_SHA=$(git rev-parse HEAD)
CKPT_OK=1
for f in run_fp16.pt $(ls run_part_*); do
  git add -f "$f" && git commit -qm "ckpt piece: $f"
  PUSHED=0
  for i in 1 2 3 4; do
    if git push -f "$PUSH" results-v50-run && \
       [ "$(git ls-remote "$PUSH" results-v50-run | cut -f1)" = "$(git rev-parse HEAD)" ]; then
      PUSHED=1; break
    fi
    sleep 20
  done
  if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
  hb "ckpt piece landed: $f"
done
git add -f train.log prep.log 2>/dev/null || true
git commit -qm "run logs" 2>/dev/null || true
git push -qf "$PUSH" results-v50-run 2>/dev/null || true
if [ "$CKPT_OK" = "1" ]; then
  hb "checkpoint FULLY VERIFIED on remote"
else
  MIRROR=$(curl -s -F "file=@run_fp16.pt" https://0x0.st 2>/dev/null || echo none)
  git reset --mixed "$BASE_SHA"   # keeps the working tree (run-2 lesson)
  hb "ckpt git pushes incomplete - fp16 mirror: $MIRROR"
fi
hb "phase complete"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
