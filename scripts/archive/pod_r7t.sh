#!/usr/bin/env bash
# A56 R7 trainer: BONUS DOSE ESCALATION. R6's exact config with one
# variable changed (--read-drop 0.75): the logit bonus already
# trains under the global read-dropout at 0.5 (read_ok gates the
# bonus block) and v9 atrophied THROUGH that dose at 488k steps —
# R7 asks whether 0.75 preserves the store's certified value
# (cost side); the benefit is only observable at v9.1 duration.
# A54c layout: workdir on CONTAINER disk; the network volume is
# READ-ONLY here (mix_r1 shard) — v9 is live on the same volume
# and nothing of its state may be touched.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/root/w-r7
rm -rf "$W" && mkdir -p "$W" && cd "$W"
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -B results-r7

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
           r7.pt.trace.jsonl canary.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-r7 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-r7 2>/dev/null; } || \
    { sleep 45; git push -qf "$PUSH" results-r7 2>/dev/null; } || true
}
hb "boot r7-DROP075 trainer $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pip install -q numpy tokenizers >> /dev/null 2>&1

cuda_canary() {
  python - <<'CANEOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda") + 1
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"vram free {free/2**30:.1f}/{total/2**30:.1f} GiB")
assert free > 10 * 2**30, f"GPU DIRTY: {free/2**30:.1f} free"
big = torch.empty(int(8e9) // 4, dtype=torch.float32, device="cuda")
del big; torch.cuda.empty_cache()
print("capacity canary ok (8GB claim)")
CANEOF
}
if ! cuda_canary >> canary.log 2>&1; then
  hb "CUDA BROKEN OR DIRTY - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
E=0
until [ -f /workspace/rmix/DONE ]; do
  sleep 60; E=$((E+1))
  [ $((E % 5)) -eq 0 ] && hb "waiting for rmix shard ($E min)"
  if [ $E -gt 180 ]; then hb "SHARD NEVER ARRIVED"; runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; fi
done
hb "shard found ($(stat -c%s /workspace/rmix/mix_r1/tokens.bin)B) — training"

( while true; do sleep 900; tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; \
    if tail -200 train.log | grep -qE "ce (nan|inf)"; then \
      hb "NAN DETECTED - killing training, landing artifacts"; \
      pkill -f "iga.lm_train" || true; break; fi; done ) &
HBPID=$!
python -m iga.lm_train run --data /workspace/rmix/mix_r1 --d 256 --lanes 16 \
  --chunk 1024 --steps 75000 --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lr 1.5e-4 --gate-init -2 --keyed logit \
  --norm-mix --read-drop 0.75 \
  --eval-data /workspace/rmix/mix_r1_eval \
  --ckpt r7.pt > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

for CK in r7.pt r7.pt.best.pt; do
  if [ -f "$CK" ]; then
    echo "===== EVAL $CK =====" >> eval_results.txt
    python -m iga.lm_eval --ckpt "$CK" --data /workspace/rmix/mix_r1_eval \
      --d 256 --arch hybrid --store matrix --xl off --chunk 1024 \
      --keyed logit --norm-mix --lanes 2 --chunks 120 \
      >> eval_results.txt 2>&1 && hb "eval complete: $CK"
  fi
done

BASE_SHA=$(git rev-parse HEAD)
CKPT_OK=0
if [ -f r7.pt ]; then
  split -b 25m r7.pt r7_part_
  [ -f r7.pt.best.pt ] && split -b 25m r7.pt.best.pt r7best_part_
  CKPT_OK=1
  for f in $(ls r7_part_* r7best_part_* 2>/dev/null); do
    git add -f "$f" && git commit -qm "ckpt piece: $f"
    PUSHED=0
    for i in 1 2 3 4; do
      if git push -f "$PUSH" results-r7 && \
         [ "$(git ls-remote "$PUSH" results-r7 | cut -f1)" = "$(git rev-parse HEAD)" ]; then
        PUSHED=1; break
      fi
      sleep 20
    done
    if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
    hb "ckpt piece landed: $f"
  done
fi
git add -f train.log r7.pt.trace.jsonl 2>/dev/null || true
git commit -qm "logs" 2>/dev/null || true
git push -qf "$PUSH" results-r7 2>/dev/null || true
if [ "$CKPT_OK" != "1" ] && [ -f r7.pt ]; then git reset --mixed "$BASE_SHA"; fi
hb "phase complete (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
