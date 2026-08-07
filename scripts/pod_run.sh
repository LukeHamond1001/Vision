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
  # one add PER path: a single multi-path add refuses everything when
  # any listed file is missing (the silent-heartbeat bug, run 1)
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
      calib_run.txt results/lm_constants_run.json; do
    git add -f "$f" 2>/dev/null || true
  done
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
# REUSE the train tokenizer — a fresh BPE voids every measurement
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

# --- the checkpoint goes home FIRST. Runs 1-2 both lost run.pt to
#     the same systematic failure: a single ~67MB push never lands
#     from these pods. v3: small pieces, separate commits, each push
#     verified; plus an fp16 sidecar and an external mirror.
python - <<'EOF'
import torch
s = torch.load("run.pt", map_location="cpu", weights_only=False)
torch.save({"model": {k: v.half() for k, v in s["model"].items()},
            "step": s["step"], "fp16": True}, "run_fp16.pt")
EOF
split -b 25m run.pt run_part_
BASE=$(git rev-parse HEAD)
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
  hb "checkpoint FULLY VERIFIED on remote (fp16 + fp32 parts)"
else
  MIRROR=$(curl -s -F "file=@run_fp16.pt" https://0x0.st 2>/dev/null || echo none)
  git reset --hard "$BASE"  # unblock small pushes
  hb "CKPT GIT PUSH INCOMPLETE - fp16 mirror: $MIRROR - holding 2h"
  sleep 7200
fi

# --- evaluation on the held-out shard: tables, lesion, talk ---
python -m iga.lm_eval --ckpt run.pt --data data/uc_eval --d 256 \
  --talk tick --lanes 4 --chunks 80 > eval_results.txt 2>&1
hb "evaluation complete"
hb "phase complete"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
