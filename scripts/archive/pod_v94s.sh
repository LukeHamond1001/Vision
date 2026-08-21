#!/usr/bin/env bash
# A63: v-scale Phase 1 application — wake v94-best (266k) 30k steps
# on the CERTIFIED v9.4 config with ARM B sleep interleaved at the
# debug-tested dose (2 chunks / 8 wake steps). Seed + shard + eval
# instrument all live on the volume (w-v94 kept by the A61 prune).
# Outputs on lightweight branch results-v94s: heartbeats, trace,
# eval, sleep provenance, paired gate windows, ckpt pieces.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace/w-v94s

for i in $(seq 1 18); do [ -d /workspace/rmix ] && break; sleep 20; done
if [ ! -d /workspace/rmix ]; then sleep 60; runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1; fi

mkdir -p "$W" && cd "$W"
if [ ! -d iga-scale ]; then
  git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
fi
cd iga-scale
git fetch -q origin main && git reset --hard -q origin/main
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -B results-v94s

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
           v94s.pt.trace.jsonl canary.log smoke_tail.log \
           sleep_smoke.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94s 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-v94s 2>/dev/null; } || \
    { sleep 45; git push -qf "$PUSH" results-v94s 2>/dev/null; } || true
}
hb "boot v94s ARM-B sleep $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# preconditions: certified seed + shards on the volume
if [ ! -f /workspace/w-v94/iga-scale/v94.pt.best.pt ]; then
  hb "SEED MISSING (w-v94/iga-scale/v94.pt.best.pt) - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
if [ ! -f /workspace/rmix/DONE_V9 ] || [ ! -d /workspace/rmix/mix_r1_eval ]; then
  hb "SHARDS MISSING (DONE_V9 or mix_r1_eval) - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
pip install -q numpy tokenizers >> /dev/null 2>&1

cuda_canary() {
  python - <<'CANEOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda") + 1
torch.cuda.synchronize()
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"{name}: vram free {free/2**30:.1f}/{total/2**30:.1f} GiB")
assert total > 23 * 2**30, f"CARD TOO SMALL for v9: {name}"
assert free > 22 * 2**30, f"GPU DIRTY: {free/2**30:.1f} free"
big = torch.empty(int(18e9) // 4, dtype=torch.float32, device="cuda")
del big; torch.cuda.empty_cache()
print("capacity canary ok (18GB claim)")
CANEOF
}
if ! cuda_canary >> canary.log 2>&1; then
  hb "CUDA BROKEN, DIRTY, OR SUB-24GB - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi

# sleep-rig smoke on cuda (the harness never ran on GPU before —
# 30s here beats a device bug 2h into the run)
python - > sleep_smoke.log 2>&1 <<'SLEOF'
import torch, sys
sys.path.insert(0, ".")
from iga.lm_hybrid import HybridLM
from iga.lm_drive import Drive
from iga.lm_sleep import Sleeper
torch.manual_seed(0)
m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
             store="matrix", keyed="logit", norm_mix=True,
             aux_trunk=0.2, use_xl=False).cuda()
with torch.no_grad():
    for a in m.alpha.values(): a.fill_(2.0)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
d = Drive(n_lanes=2)
sl = Sleeper(arm="B", every=4, block_chunks=2, seed=0)
sl.bind(d)
for _ in range(8):
    sl.observe(torch.randint(0, 64, (2, 64)))
d.ledger.append({"lane": 0, "band": 3, "key": "recall:b0",
                 "phi0": 0.5, "w": 1.0, "t0": 64, "phi1": 0.1,
                 "pay": 0.4, "t1": 448})
d.step_t = 512
row = sl.maybe_sleep(m, opt, d, step=4)
assert row is not None and sl.audit()["only_paid"], "sleep rig failed"
print("SLEEP SMOKE PASS", row)
SLEOF
if ! grep -q "SLEEP SMOKE PASS" sleep_smoke.log; then
  hb "SLEEP RIG FAILED ON CUDA - aborting (see sleep_smoke)"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
hb "sleep rig cuda pass"

# certified wake-shape smoke (memory + budget guard, v94t verbatim)
python -m iga.lm_train run --data /workspace/rmix/mix_v9 --d 512 \
  --lanes 6 --chunk 2048 --steps 60 --talk tick --arch hybrid \
  --device cuda --store matrix --xl off --lr 1e-4 --gate-init -2 \
  --keyed logit --norm-mix --aux-trunk 0.2 --lam 0.02 --ckpt smoke.pt > smoke.log 2>&1
SMOKE_RC=$?
tail -12 smoke.log > smoke_tail.log
TOKS=$(grep -oE '[0-9,]+ tok/s' smoke.log | tail -1 | tr -d ' ,' | grep -oE '^[0-9]+' || echo 0)
rm -f smoke.pt smoke.pt.tmp smoke.pt.trace.jsonl
if [ "$SMOKE_RC" -ne 0 ]; then
  hb "SMOKE FAILED rc=$SMOKE_RC - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
if [ "${TOKS:-0}" -lt 15000 ]; then
  hb "SMOKE TOO SLOW (${TOKS} tok/s) - aborting per budget"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
hb "smoke pass: ${TOKS} tok/s - starting 30k application"

( LAST_STEP=0; BEAT=0
  while true; do
    sleep 900; BEAT=$((BEAT+1))
    tail -40 train.log > train_tail.log
    SLB=$(grep -c "sleep@" train.log || echo 0)
    hb "beat: $(grep -oE 'step +[0-9]+.*tok/s' train.log | tail -1) | sleep blocks $SLB"
    if tail -200 train.log | grep -qE "ce (nan|inf)"; then
      hb "NAN DETECTED - killing training, landing artifacts"
      pkill -f v94s_driver || true; break
    fi
    CUR=$(grep -oE "step +[0-9]+ " train.log | tail -1 | grep -oE "[0-9]+" || echo 0)
    if [ "$CUR" != "0" ] && [ "$CUR" = "$LAST_STEP" ]; then
      hb "STALL DETECTED at step $CUR - killing training"
      pkill -f v94s_driver || true; break
    fi
    LAST_STEP="$CUR"
    if [ $((BEAT % 2)) -eq 0 ] && [ -f v94s.pt ]; then
      ( rm -rf /workspace/w-v94s/snap && mkdir -p /workspace/w-v94s/snap &&
        cp v94s.pt /workspace/w-v94s/snap/ && cd /workspace/w-v94s/snap &&
        split -b 25m v94s.pt v94sroll_ && rm v94s.pt &&
        git init -q . && git checkout -q -b results-v94s-ckpt &&
        git add . &&
        git -c user.email=pod@iga -c user.name=pod commit -qm "rolling beat $BEAT step $CUR" &&
        git push -qf "$PUSH" results-v94s-ckpt ) >/dev/null 2>&1 \
        && hb "rolling snapshot pushed (step $CUR)"
    fi
  done ) &
WATCHPID=$!
python scripts/v94s_driver.py > train.log 2>&1
TRAIN_RC=$?
kill $WATCHPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

for CK in v94s.pt v94s.pt.best.pt; do
  if [ -f "$CK" ]; then
    echo "===== EVAL $CK =====" >> eval_results.txt
    python -m iga.lm_eval --ckpt "$CK" --data /workspace/rmix/mix_r1_eval \
      --d 512 --arch hybrid --store matrix --xl off --chunk 2048 \
      --keyed logit --norm-mix --aux-trunk 0.2 --lanes 2 --chunks 120 \
      >> eval_results.txt 2>&1 && hb "eval complete: $CK"
  fi
done

BASE_SHA=$(git rev-parse HEAD)
CKPT_OK=0
if [ -f v94s.pt ]; then
  split -b 25m v94s.pt v94s_part_
  [ -f v94s.pt.best.pt ] && split -b 25m v94s.pt.best.pt v94sbest_part_
  [ -f v94s_windows.jsonl ] && gzip -9 -c v94s_windows.jsonl > vw.gz \
    && split -b 25m vw.gz v94swin_ && rm -f vw.gz
  [ -f v94s_sleep.json ] && gzip -9 -c v94s_sleep.json > vs.gz \
    && split -b 25m vs.gz v94sprov_ && rm -f vs.gz
  CKPT_OK=1
  for f in $(ls v94s_part_* v94sbest_part_* v94swin_* v94sprov_* 2>/dev/null); do
    git add -f "$f" && git commit -qm "piece: $f"
    PUSHED=0
    for i in 1 2 3 4; do
      if git push -f "$PUSH" results-v94s && \
         [ "$(git ls-remote "$PUSH" results-v94s | cut -f1)" = "$(git rev-parse HEAD)" ]; then
        PUSHED=1; break
      fi
      sleep 20
    done
    if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
    hb "piece landed: $f"
  done
fi
git add -f train.log 2>/dev/null || true
git add -f v94s.pt.trace.jsonl 2>/dev/null || true
git commit -qm "logs" 2>/dev/null || true
git push -qf "$PUSH" results-v94s 2>/dev/null || true
if [ "$CKPT_OK" != "1" ] && [ -f v94s.pt ]; then git reset --mixed "$BASE_SHA"; fi
hb "DONE (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
