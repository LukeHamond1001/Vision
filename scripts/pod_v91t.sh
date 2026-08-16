#!/usr/bin/env bash
# A57 v9.1: the certified recipe at scale — R6 exact (norm_mix,
# read_drop 0.5) on v9's certified ops config (d=512 T=2048
# lanes 6, 488k steps, cosine, real-shard smoke, snapshots).
# Protection = the PRE-REGISTERED KILL RULE (A56b): local
# watcher reads the landed trace; same(recent) < half banked
# peak for 3 windows after 60k -> killed and landed.
# A54 v9 trainer: THE FULL-ARCHITECTURE SCALE GATE. R5's certified
# design with only the scale axes changed: d=512, T=2048, 6B fresh
# tokens (KD auto-doubles with T), lr 1e-4 COSINE-decayed to 10%
# (A54 audit C3: v8.0 rotted on constant lr at this exact
# width/duration). Survivability (audit C1/C2/M4/H5): resume-aware
# boot, rolling ckpt snapshots every ~2h, NaN/stall watchdog, and a
# paid 20-step smoke on the real card before committing.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace/w-v91
TOTAL_STEPS=488000
RESUME=""
STEPS_LEFT=$TOTAL_STEPS
if [ -f "$W/iga-scale/v91.pt" ]; then
  # A54 audit C1: a surviving checkpoint on the volume is the most
  # valuable object in the account — never rm it; resume instead.
  # A54d: false-start guard — a sub-5k-step stub (crashed launch)
  # is not worth resuming; steps remaining shrink by the resume
  # point so the token budget stays exactly one epoch.
  STEP=$(cd "$W/iga-scale" && python - <<'P'
import torch
try:
    print(torch.load("v91.pt", map_location="cpu",
                     weights_only=False).get("step", 0))
except Exception:
    print(0)
P
)
  if [ "${STEP:-0}" -ge 5000 ]; then
    cd "$W/iga-scale"
    git fetch -q origin main && git reset --hard -q origin/main
    RESUME="--resume v91.pt"
    STEPS_LEFT=$((TOTAL_STEPS - STEP))
  else
    rm -rf "$W"
  fi
fi
if [ ! -d "$W/iga-scale" ]; then
  mkdir -p "$W" && cd "$W"
  git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
  cd iga-scale
fi
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -B results-v9

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
           v91.pt.trace.jsonl canary.log smoke_tail.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v9 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-v9 2>/dev/null; } || \
    { sleep 45; git push -qf "$PUSH" results-v9 2>/dev/null; } || true
}
hb "boot v91-R6-RECIPE trainer $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) resume='${RESUME}'"
# land v9's banked-best (healthy-trunk 38k model, graft candidate)
# from the old workdir if this is the first boot and it exists
if [ -f /workspace/w-v9/iga-scale/v9.pt.best.pt ] && \
   ! git ls-remote --exit-code origin results-v9-best >/dev/null 2>&1; then
  ( rm -rf /tmp/v9best && mkdir -p /tmp/v9best && \
    cp /workspace/w-v9/iga-scale/v9.pt.best.pt /tmp/v9best/ && \
    cd /tmp/v9best && split -b 25m v9.pt.best.pt v9hb_ && \
    rm v9.pt.best.pt && git init -q . && git checkout -q -b results-v9-best && \
    git add . && git -c user.email=pod@iga -c user.name=pod \
      commit -qm "v9 banked best (step ~38k, healthy trunk)" && \
    git push -qf "$PUSH" results-v9-best ) >/dev/null 2>&1 \
    && hb "v9 best.pt landed to results-v9-best"
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
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi

# A54 audit H5/H2 + A54d: paid smoke on the REAL shard — peak
# memory is EVENT-DENSITY dependent (drive pays retain logp graph
# on hold-dense chunks; the quiet mix_r1 smoke passed at shapes
# that OOM'd on mix_v9 by step 800). 60 real steps, real events.
python -m iga.lm_train run --data /workspace/rmix/mix_v9 --d 512 \
  --lanes 6 --chunk 2048 --steps 60 --talk tick --arch hybrid \
  --device cuda --store matrix --xl off --lr 1e-4 --gate-init -2 \
  --keyed logit --norm-mix --ckpt smoke.pt > smoke.log 2>&1
SMOKE_RC=$?
tail -12 smoke.log > smoke_tail.log
TOKS=$(grep -oE '[0-9,]+ tok/s' smoke.log | tail -1 | tr -d ' ,' | grep -oE '^[0-9]+' || echo 0)
rm -f smoke.pt smoke.pt.tmp
if [ "$SMOKE_RC" -ne 0 ]; then
  hb "SMOKE FAILED rc=$SMOKE_RC (see smoke_tail) - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
if [ "${TOKS:-0}" -lt 25000 ]; then
  hb "SMOKE TOO SLOW (${TOKS} tok/s < 25000) - aborting per budget"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "smoke pass: ${TOKS} tok/s at v9 shapes"

E=0
until [ -f /workspace/rmix/DONE_V9 ]; do
  sleep 60; E=$((E+1))
  [ $((E % 10)) -eq 0 ] && hb "waiting for v9 shard ($E min)"
  if [ $E -gt 240 ]; then hb "SHARD NEVER ARRIVED"; runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; fi
done
hb "shard found ($(stat -c%s /workspace/rmix/mix_v9/tokens.bin)B) — training"

( LAST_STEP=0; BEAT=0
  while true; do
    sleep 1800; BEAT=$((BEAT+1))
    tail -40 train.log > train_tail.log
    hb "training heartbeat: $(tail -1 train_tail.log)"
    # A54 audit M4: NaN watchdog — a diverged run must not burn
    # 40 more zombie hours; kill and fall through to landing
    if tail -200 train.log | grep -qE "ce (nan|inf)"; then
      hb "NAN DETECTED - killing training, landing artifacts"
      pkill -f "iga.lm_train" || true
      break
    fi
    CUR=$(grep -oE "step +[0-9]+ " train.log | tail -1 | grep -oE "[0-9]+" || echo 0)
    if [ "$CUR" != "0" ] && [ "$CUR" = "$LAST_STEP" ]; then
      hb "STALL DETECTED at step $CUR - killing training"
      pkill -f "iga.lm_train" || true
      break
    fi
    LAST_STEP="$CUR"
    # A54 audit C1: rolling snapshot every ~2h — a host death costs
    # <=2h, not the run (atomic_save keeps v91.pt always-complete)
    if [ $((BEAT % 4)) -eq 0 ] && [ -f v91.pt ]; then
      ( rm -rf /workspace/w-v91/snap && mkdir -p /workspace/w-v91/snap &&
        cp v91.pt /workspace/w-v91/snap/ && cd /workspace/w-v91/snap &&
        split -b 25m v91.pt v91roll_ && rm v91.pt &&
        git init -q . && git checkout -q -b results-v91-ckpt &&
        git add . &&
        git -c user.email=pod@iga -c user.name=pod commit -qm "rolling beat $BEAT step $CUR" &&
        git push -qf "$PUSH" results-v91-ckpt ) >/dev/null 2>&1 \
        && hb "rolling snapshot pushed (step $CUR)"
    fi
  done ) &
WATCHPID=$!
python -m iga.lm_train run --data /workspace/rmix/mix_v9 --d 512 --lanes 6 \
  --chunk 2048 --steps $STEPS_LEFT --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lr 1e-4 --lr-decay cosine \
  --lr-total-steps $TOTAL_STEPS --gate-init -2 \
  --keyed logit --norm-mix $RESUME \
  --eval-data /workspace/rmix/mix_r1_eval \
  --ckpt v91.pt > train.log 2>&1
TRAIN_RC=$?
kill $WATCHPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

for CK in v91.pt v91.pt.best.pt; do
  if [ -f "$CK" ]; then
    echo "===== EVAL $CK =====" >> eval_results.txt
    python -m iga.lm_eval --ckpt "$CK" --data /workspace/rmix/mix_r1_eval \
      --d 512 --arch hybrid --store matrix --xl off --chunk 2048 \
      --keyed logit --norm-mix --lanes 2 --chunks 120 \
      >> eval_results.txt 2>&1 && hb "eval complete: $CK"
  fi
done

BASE_SHA=$(git rev-parse HEAD)
CKPT_OK=0
if [ -f v91.pt ]; then
  split -b 25m v91.pt v91_part_
  [ -f v91.pt.best.pt ] && split -b 25m v91.pt.best.pt v91best_part_
  CKPT_OK=1
  for f in $(ls v91_part_* v91best_part_* 2>/dev/null); do
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
git add -f train.log v91.pt.trace.jsonl 2>/dev/null || true
git commit -qm "logs" 2>/dev/null || true
git push -qf "$PUSH" results-v9 2>/dev/null || true
if [ "$CKPT_OK" != "1" ] && [ -f v91.pt ]; then git reset --mixed "$BASE_SHA"; fi
hb "phase complete (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
