#!/usr/bin/env bash
# A54 v9 trainer: THE FULL-ARCHITECTURE SCALE GATE. R5's certified
# design with only the scale axes changed: d=512, T=2048, 6B fresh
# tokens (KD auto-doubles with T — the capacity law). lr 1e-4 per
# the width-LR law. Waits for the v9 shard (DONE_V9), zero prep.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace/w-v9
rm -rf "$W" && mkdir -p "$W" && cd "$W"
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -b results-v9

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
           v9.pt.trace.jsonl canary.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v9 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-v9 2>/dev/null; } || \
    { sleep 45; git push -qf "$PUSH" results-v9 2>/dev/null; } || true
}
hb "boot v9-FULL-ARCH trainer $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
pip install -q numpy tokenizers >> /dev/null 2>&1

cuda_canary() {
  python - <<'CANEOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda") + 1
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"vram free {free/2**30:.1f}/{total/2**30:.1f} GiB")
assert free > 18 * 2**30, f"GPU DIRTY: {free/2**30:.1f} free"
big = torch.empty(int(14e9) // 4, dtype=torch.float32, device="cuda")
del big; torch.cuda.empty_cache()
print("capacity canary ok (14GB claim)")
CANEOF
}
if ! cuda_canary >> canary.log 2>&1; then
  hb "CUDA BROKEN OR DIRTY - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
E=0
until [ -f /workspace/rmix/DONE_V9 ]; do
  sleep 60; E=$((E+1))
  [ $((E % 10)) -eq 0 ] && hb "waiting for v9 shard ($E min)"
  if [ $E -gt 240 ]; then hb "SHARD NEVER ARRIVED"; runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; fi
done
hb "shard found ($(stat -c%s /workspace/rmix/mix_v9/tokens.bin)B) — training"

( while true; do sleep 1800; tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; done ) &
HBPID=$!
python -m iga.lm_train run --data /workspace/rmix/mix_v9 --d 512 --lanes 8 \
  --chunk 2048 --steps 366000 --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lr 1e-4 --gate-init -2 --keyed logit \
  --eval-data /workspace/rmix/mix_r1_eval \
  --ckpt v9.pt > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

for CK in v9.pt v9.pt.best.pt; do
  if [ -f "$CK" ]; then
    echo "===== EVAL $CK =====" >> eval_results.txt
    python -m iga.lm_eval --ckpt "$CK" --data /workspace/rmix/mix_r1_eval \
      --d 512 --arch hybrid --store matrix --xl off --chunk 2048 \
      --keyed logit --lanes 2 --chunks 120 \
      >> eval_results.txt 2>&1 && hb "eval complete: $CK"
  fi
done

BASE_SHA=$(git rev-parse HEAD)
CKPT_OK=0
if [ -f v9.pt ]; then
  split -b 25m v9.pt v9_part_
  [ -f v9.pt.best.pt ] && split -b 25m v9.pt.best.pt v9best_part_
  CKPT_OK=1
  for f in $(ls v9_part_* v9best_part_* 2>/dev/null); do
    git add -f "$f" && git commit -qm "ckpt piece: $f"
    PUSHED=0
    for i in 1 2 3 4; do
      if git push -f "$PUSH" results-v9 && \
         [ "$(git ls-remote "$PUSH" results-v9 | cut -f1)" = "$(git rev-parse HEAD)" ]; then
        PUSHED=1; break
      fi
      sleep 20
    done
    if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
    hb "ckpt piece landed: $f"
  done
fi
git add -f train.log v9.pt.trace.jsonl 2>/dev/null || true
git commit -qm "logs" 2>/dev/null || true
git push -qf "$PUSH" results-v9 2>/dev/null || true
if [ "$CKPT_OK" != "1" ] && [ -f v9.pt ]; then git reset --mixed "$BASE_SHA"; fi
hb "phase complete (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
