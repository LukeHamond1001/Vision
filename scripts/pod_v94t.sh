#!/usr/bin/env bash
# A60e v9.4: RESUME THE v9.3 CORPSE AT FULL COVERAGE (practice-
# like-we-play). A60d forensics: --hold-cap 1 cut store payment
# COVERAGE to ~1/13 of probe-time (token-denominated horizons);
# new-write cohort scores decayed 0.74->0.59->0.44 while the
# trunk set family records. v9.4 = resume snapshot 114,350
# (results-v93-ckpt), cap REMOVED, lam 0.25->0.02: pay-gradient
# VOLUME pinned at the proven-trunk-safe dose (13 x 0.02 ~= 1 x
# 0.25) with v9.2's full store-economy coverage restored.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace/w-v94
TOTAL_STEPS=488000
RESUME=""
STEPS_LEFT=$TOTAL_STEPS
if [ -f "$W/iga-scale/v94.pt" ]; then
  # A54 audit C1: a surviving checkpoint on the volume is the most
  # valuable object in the account — never rm it; resume instead.
  # A54d: false-start guard — a sub-5k-step stub (crashed launch)
  # is not worth resuming; steps remaining shrink by the resume
  # point so the token budget stays exactly one epoch.
  STEP=$(cd "$W/iga-scale" && python - <<'P'
import torch
try:
    print(torch.load("v94.pt", map_location="cpu",
                     weights_only=False).get("step", 0))
except Exception:
    print(0)
P
)
  if [ "${STEP:-0}" -ge 5000 ]; then
    cd "$W/iga-scale"
    git fetch -q origin main && git reset --hard -q origin/main
    RESUME="--resume v94.pt"
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
git checkout -B results-v94

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt \
           v94.pt.trace.jsonl canary.log smoke_tail.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-v94 2>/dev/null; } || \
    { sleep 45; git push -qf "$PUSH" results-v94 2>/dev/null; } || true
}
hb "boot v94-FULLCOV-LAM002 trainer $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) resume='${RESUME}'"

# A60e: seed the resume from the v9.3 corpse (verified snapshot,
# step 114,350, on results-v93-ckpt) when the volume has no v94.pt.
# The corpse trunk is the family record; A60d convicts only the
# store's payment coverage, which this config restores.
if [ -z "$RESUME" ]; then
  if git fetch --depth 1 origin results-v93-ckpt 2>/dev/null; then
    for f in $(git ls-tree --name-only FETCH_HEAD | grep '^v93roll_'); do
      git show "FETCH_HEAD:$f" > "$f"
    done
    cat v93roll_* > v94.pt && rm -f v93roll_*
    SEEDSTEP=$(python - <<'P'
import torch
try:
    print(torch.load("v94.pt", map_location="cpu",
                     weights_only=False).get("step", 0))
except Exception:
    print(0)
P
)
    if [ "${SEEDSTEP:-0}" -ge 114000 ]; then
      RESUME="--resume v94.pt"
      STEPS_LEFT=$((TOTAL_STEPS - SEEDSTEP))
      hb "corpse seed ok: resume at step $SEEDSTEP, $STEPS_LEFT left"
    else
      hb "SEED FAILED (step=${SEEDSTEP:-0}) - aborting"
      runpodctl remove pod "$RUNPOD_POD_ID" || true
      sleep 120; exit 1
    fi
  else
    hb "SEED FETCH FAILED (results-v93-ckpt unreachable) - aborting"
    runpodctl remove pod "$RUNPOD_POD_ID" || true
    sleep 120; exit 1
  fi
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
# on hold-dense chunks; full uncapped density here matches v9.2's
# proven memory profile — lam scales the scalar, not the graph).
python -m iga.lm_train run --data /workspace/rmix/mix_v9 --d 512 \
  --lanes 6 --chunk 2048 --steps 60 --talk tick --arch hybrid \
  --device cuda --store matrix --xl off --lr 1e-4 --gate-init -2 \
  --keyed logit --norm-mix --aux-trunk 0.2 --lam 0.02 --ckpt smoke.pt > smoke.log 2>&1
SMOKE_RC=$?
tail -12 smoke.log > smoke_tail.log
TOKS=$(grep -oE '[0-9,]+ tok/s' smoke.log | tail -1 | tr -d ' ,' | grep -oE '^[0-9]+' || echo 0)
rm -f smoke.pt smoke.pt.tmp
if [ "$SMOKE_RC" -ne 0 ]; then
  hb "SMOKE FAILED rc=$SMOKE_RC (see smoke_tail) - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
# floor is a BUDGET guard, not science: 25k was calibrated for
# 4090 pricing; A5000-class at ~$0.25/hr holds economics to
# ~15k (69h ~ $18). Lowered when the 4090 pool went lemon.
if [ "${TOKS:-0}" -lt 15000 ]; then
  hb "SMOKE TOO SLOW (${TOKS} tok/s < 15000) - aborting per budget"
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
    # <=2h, not the run (atomic_save keeps v94.pt always-complete)
    if [ $((BEAT % 4)) -eq 0 ] && [ -f v94.pt ]; then
      ( rm -rf /workspace/w-v94/snap && mkdir -p /workspace/w-v94/snap &&
        cp v94.pt /workspace/w-v94/snap/ && cd /workspace/w-v94/snap &&
        split -b 25m v94.pt v94roll_ && rm v94.pt &&
        git init -q . && git checkout -q -b results-v94-ckpt &&
        git add . &&
        git -c user.email=pod@iga -c user.name=pod commit -qm "rolling beat $BEAT step $CUR" &&
        git push -qf "$PUSH" results-v94-ckpt ) >/dev/null 2>&1 \
        && hb "rolling snapshot pushed (step $CUR)"
    fi
  done ) &
WATCHPID=$!
python -m iga.lm_train run --data /workspace/rmix/mix_v9 --d 512 --lanes 6 \
  --chunk 2048 --steps $STEPS_LEFT --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lr 1e-4 --lr-decay cosine \
  --lr-total-steps $TOTAL_STEPS --gate-init -2 \
  --keyed logit --norm-mix --aux-trunk 0.2 --lam 0.02 $RESUME \
  --eval-data /workspace/rmix/mix_r1_eval \
  --ckpt v94.pt > train.log 2>&1
TRAIN_RC=$?
kill $WATCHPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

for CK in v94.pt v94.pt.best.pt; do
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
if [ -f v94.pt ]; then
  split -b 25m v94.pt v94_part_
  [ -f v94.pt.best.pt ] && split -b 25m v94.pt.best.pt v94best_part_
  CKPT_OK=1
  for f in $(ls v94_part_* v94best_part_* 2>/dev/null); do
    git add -f "$f" && git commit -qm "ckpt piece: $f"
    PUSHED=0
    for i in 1 2 3 4; do
      if git push -f "$PUSH" results-v94 && \
         [ "$(git ls-remote "$PUSH" results-v94 | cut -f1)" = "$(git rev-parse HEAD)" ]; then
        PUSHED=1; break
      fi
      sleep 20
    done
    if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
    hb "ckpt piece landed: $f"
  done
fi
git add -f train.log v94.pt.trace.jsonl 2>/dev/null || true
git commit -qm "logs" 2>/dev/null || true
git push -qf "$PUSH" results-v94 2>/dev/null || true
if [ "$CKPT_OK" != "1" ] && [ -f v94.pt ]; then git reset --mixed "$BASE_SHA"; fi
hb "phase complete (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
