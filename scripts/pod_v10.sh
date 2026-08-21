#!/usr/bin/env bash
# v10 — THE LIFETIME FLASH (500M), GPU pod payload. Two stages,
# self-sensing on the shared network volume (/workspace):
#
#   SMOKE (no GO needed): paid real-shard smoke at EXACT shapes
#     (d=1280, 20L, T=2048, band-6 clocks, arm C sleeping) on the
#     8- and 12-life smoke shards pod_v10_prep.sh built. Measures
#     tok/s, holds/step (-> lam by the A60f pairing), peak memory
#     (-> the lane pick). Writes smoke.json, pushes it, terminates.
#     A54d law: quiet-data smokes lie; this one runs the real
#     builder output at the real shapes.
#
#   FLASH (requires GO=1 + the full corpus on the volume): runs
#     scripts/v10_driver.py under a staleness watchdog with a
#     relaunch loop (resume-aware, false-start-guarded). KILL from
#     the heartbeat pack banks everything and stops the pod with
#     the volume intact: kill, fix, relaunch.
#
# LAUNCH REQUIRES THE USER'S EXPLICIT GO — this script costs real
# money. The smoke stage prices the run before the flash commits.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace/w-v10
DATA=/workspace/v10
OUT=/workspace/v10_out
mkdir -p "$W" "$OUT" && cd "$W"
[ -d Vision ] || git clone --depth 1 \
  https://github.com/LukeHamond1001/Vision.git
cd Vision
git fetch -q origin main && git reset --hard -q origin/main
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git checkout -B results-v10

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log smoke.json train_tail.log hb_v10.jsonl \
           v10_driver.jsonl canary.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v10 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v10 2>/dev/null; } || true
}
hb "boot v10-GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) GO=${GO:-0}"
pip install -q numpy tokenizers >> /dev/null 2>&1

# ---------- capacity canary (A54b/A46) ----------
python - > canary.log 2>&1 <<'EOF'
import torch
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print(f"GPU {p.name} {p.total_memory/2**30:.1f}GiB")
assert p.total_memory / 2**30 > 70, "A100 80GB class required"
x = torch.empty(int(60e9 // 4), dtype=torch.float32, device="cuda")
del x; torch.cuda.empty_cache()
print("capacity canary: claimed 60GiB ok")
EOF
[ $? -eq 0 ] || { hb "ABORT canary failed"; runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
hb "canary ok"

# ---------- cross-DC RECEIVE (GO=1, BEFORE smoke) ----------
# The prep mule's ship tar carries the corpus AND the smoke shards, so
# it must land before the smoke stage looks for them (the 20:14 abort:
# smoke-shard check ran first and killed the pod). The code comes from
# env SHIP_CODE — this pod's own boot force-pushes the results branch
# and WIPES ship_code.txt, so the branch copy is only a fallback for
# same-DC relights that boot before any prep mule re-post.
if [ "${GO:-0}" = "1" ] && [ ! -f "$DATA/flash/manifest.json" ]; then
  CODE="${SHIP_CODE:-}"
  if [ -z "$CODE" ]; then
    git fetch -q origin results-v10 2>/dev/null && \
      git checkout -q origin/results-v10 -- ship_code.txt 2>/dev/null || true
    [ -f ship_code.txt ] && CODE=$(tr -d '[:space:]' < ship_code.txt)
  fi
  if [ -n "$CODE" ]; then
    hb "receiving corpus (code $CODE)"
    ( cd /workspace && runpodctl receive "$CODE" ) \
      > receive.log 2>&1 || true
    if [ -f /workspace/v10_ship.tar ]; then
      tar xf /workspace/v10_ship.tar -C /workspace && \
        rm -f /workspace/v10_ship.tar
    fi
    hb "receive $( [ -f "$DATA/flash/manifest.json" ] && echo ok || echo "FAILED $(tail -1 receive.log 2>/dev/null | head -c 120)")"
  fi
fi

# ---------- SMOKE stage (per-GPU tagged: the GPU shop) ----------
# Shopper pods (GO=0) each write smoke_<gpu>.json; the flash pod
# (GO=1) skips smoking when a chosen config exists (smoke.json on
# the volume, or smoke_choice.json committed to main after the
# shop), else falls back to smoking itself.
GPUTAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
  | tr -cd 'A-Za-z0-9' | tr '[:upper:]' '[:lower:]')
NEED_SMOKE=0
if [ "${GO:-0}" = "1" ]; then
  { [ -f "$OUT/smoke.json" ] || [ -f smoke_choice.json ]; } || NEED_SMOKE=1
else
  [ -f "$OUT/smoke_$GPUTAG.json" ] || NEED_SMOKE=1
fi
if [ "$NEED_SMOKE" = "1" ]; then
  if [ ! -f "$DATA/smoke_l8/manifest.json" ] && \
     [ "${SMOKE_FROM_BRANCH:-0}" = "1" ]; then
    # volume-less shopper pod: shards ride the results branch
    git fetch -q origin results-v10 && \
      git checkout -q origin/results-v10 -- smoke_shards.tar.gz && \
      mkdir -p "$DATA" && tar xzf smoke_shards.tar.gz -C "$DATA"
  fi
  for L in ${SMOKE_LANES:-8 12}; do
    [ -f "$DATA/smoke_l$L/manifest.json" ] || \
      { hb "ABORT smoke shard l$L missing (run pod_v10_prep.sh)"; \
        runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
  done
  python - > smoke_run.log 2>&1 <<EOF
import json, sys, time, torch
sys.path.insert(0, ".")
from iga.lm_train import train
from iga.lm_sleep import Sleeper
rows = []
for L in [int(x) for x in "${SMOKE_LANES:-8 12}".split()]:
    sl = Sleeper(arm="C", every=8, block_chunks=2, seed=1,
                 homeostasis=1e-3)
    sl.press_pay = (2048, 256)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model, drive, vocab, ce0, ce1 = train(
            d=1280, n_layers=20, lanes=L, T=2048, steps=60, seed=0,
            device="cuda", arch="hybrid", store="matrix",
            keyed="logit", norm_mix=True, aux_trunk=0.2,
            use_xl=False, gate_init=-2.0, lam=0.02,
            clocks={3: 1, 4: 8, 5: 64, 6: 512},
            data="$DATA/smoke_l" + str(L), sleep=sl, log_every=20)
        dt = time.time() - t0
        toks = 60 * L * 2048
        holds = len(drive.ledger) / 60
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"lanes": L, "tok_s": round(toks / dt),
                     "holds": round(holds, 2),
                     "peak_gib": round(peak, 1),
                     "ce": [round(ce0, 3), round(ce1, 3)],
                     "sleep_steps": sl.steps_taken})
        del model, drive
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        rows.append({"lanes": L, "oom": True})
        torch.cuda.empty_cache()
print(json.dumps(rows, indent=1))
ok = [r for r in rows if not r.get("oom") and r["peak_gib"] < 70]
assert ok, "no lane count fits — smoke FAILED"
best = ok[0]
for r in ok[1:]:
    if r["tok_s"] > best["tok_s"] * 1.05:
        best = r
assert best["tok_s"] > 8000, "throughput floor 8k tok/s"
lam = min(0.25, 0.25 / max(best["holds"], 1))   # A60f pairing
out = {"gpu": "$GPUTAG", "lanes": best["lanes"],
       "lam": round(lam, 5),
       "tok_s": best["tok_s"], "holds": best["holds"],
       "peak_gib": best["peak_gib"], "rows": rows}
json.dump(out, open("$OUT/smoke_$GPUTAG.json", "w"), indent=1)
if "${GO:-0}" == "1":
    json.dump(out, open("$OUT/smoke.json", "w"), indent=1)
print("SMOKE", json.dumps(out))
EOF
  RC=$?
  cp "$OUT/smoke_$GPUTAG.json" "smoke_$GPUTAG.json" 2>/dev/null || true
  tail -30 smoke_run.log >> HEARTBEAT.log
  git add -f "smoke_$GPUTAG.json" 2>/dev/null || true
  hb "smoke[$GPUTAG] rc=$RC $(cat "$OUT/smoke_$GPUTAG.json" 2>/dev/null | head -c 300)"
  [ $RC -eq 0 ] || { runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
fi

# ---------- FLASH stage (GO-gated) ----------
if [ "${GO:-0}" != "1" ]; then
  hb "smoke done; flash awaits GO=1 + full corpus"
  if [ "${SKIP_TERMINATE:-0}" != "1" ]; then
    runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
    sleep 60
  fi
  exit 0
fi
# (cross-DC receive happens pre-smoke now — see above)
for need in "$DATA/flash/manifest.json" "$DATA/flash_eval/manifest.json"; do
  [ -f "$need" ] || { hb "ABORT missing $need"; \
    runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
done
# the GPU shop's verdict, committed to main as smoke_choice.json,
# seeds the volume's chosen config on first flash boot
[ -f "$OUT/smoke.json" ] || { [ -f smoke_choice.json ] && \
  cp smoke_choice.json "$OUT/smoke.json" && hb "smoke.json <- smoke_choice.json"; }
[ -f "$OUT/smoke.json" ] || { hb "ABORT no smoke config"; \
  runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
# cost bar: below MIN_TOKS tok/s this GPU prices the run out of the
# ratified band — HOLD (bank everything, don't flash) unless forced
if [ "${MIN_TOKS:-0}" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  TOKS=$(python -c "import json;print(int(json.load(open('$OUT/smoke.json'))['tok_s']))" 2>/dev/null || echo 0)
  if [ "$TOKS" -lt "${MIN_TOKS}" ]; then
    hb "HOLD: measured $TOKS tok/s < $MIN_TOKS cost bar — flash NOT started"
    runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
    sleep 60; exit 0
  fi
fi

CKPT=$OUT/v10.pt
# false-start guard (A54d): a sub-5k-step stub from a crashed first
# launch is not worth resuming
if [ -f "$CKPT" ]; then
  STEP=$(python -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False).get('step',0))" 2>/dev/null || echo 0)
  if [ "${STEP:-0}" -lt 5000 ]; then rm -f "$CKPT" "$CKPT.best.pt"; fi
  hb "resume check: step=${STEP:-0}"
fi

# staleness watchdog: no train-log writes for 25 min -> kill python
( while true; do
    sleep 300
    if [ -f v10_train.log ]; then
      AGE=$(( $(date +%s) - $(stat -c %Y v10_train.log 2>/dev/null || date +%s) ))
      if [ "$AGE" -gt 1500 ]; then
        echo "$(date -u) WATCHDOG stale ${AGE}s" >> HEARTBEAT.log
        pkill -f v10_driver.py || true
      fi
    fi
  done ) &
WATCHPID=$!

RC=1
for ATTEMPT in 1 2 3 4 5 6; do
  hb "flash attempt $ATTEMPT"
  # in-flight publisher: the driver's probe rows land in LOCAL jsonls;
  # without this loop the branch sees nothing until exit — the watcher
  # (and the kill-fix-relaunch protocol) needs them DURING the run
  ( while true; do sleep 900
      tail -100 v10_train.log > train_tail.log 2>/dev/null || true
      LASTHB=$(tail -1 hb_v10.jsonl 2>/dev/null | head -c 300)
      echo "$(date -u +%H:%M:%S) inflight ${LASTHB:-no-hb-row-yet}" >> HEARTBEAT.log
      # add per-file: one missing pathspec aborts a combined add,
      # which muted every pulse until the first battery wrote the
      # jsonls — stage whatever exists each cycle
      for f in HEARTBEAT.log hb_v10.jsonl v10_driver.jsonl train_tail.log; do
        [ -f "$f" ] && git add -f "$f" 2>/dev/null
      done
      git commit -qm "inflight" 2>/dev/null
      git push -qf "$PUSH" results-v10 2>/dev/null || true
    done ) & PUBPID=$!
  python scripts/v10_driver.py \
    --data "$DATA/flash" --eval-data "$DATA/flash_eval" \
    --ckpt "$CKPT" --smoke "$OUT/smoke.json" \
    --hb-out hb_v10.jsonl --trace v10_driver.jsonl \
    >> v10_train.log 2>&1
  RC=$?
  kill $PUBPID 2>/dev/null || true
  tail -80 v10_train.log > train_tail.log
  hb "driver exit rc=$RC (attempt $ATTEMPT)"
  [ $RC -eq 0 ] && break
  [ $RC -eq 3 ] && break     # KILL: bank and stop — never relaunch past it
  [ $RC -eq 4 ] && break     # non-finite CE / dead instruments
  sleep 30
done
kill $WATCHPID 2>/dev/null || true

# ---------- bank ----------
BASE_SHA=$(git rev-parse HEAD)
if [ -f "$CKPT.best.pt" ]; then
  # model-only export (the full blob stays on the volume — A54 C1:
  # the surviving volume checkpoint is the primary artifact)
  python - <<EOF
import torch
b = torch.load("$CKPT.best.pt", map_location="cpu", weights_only=False)
torch.save({"model": b["model"], "step": b.get("step"),
            "peval_best": b.get("peval_best")}, "v10_best_model.pt")
EOF
  git checkout -B results-v10-ckpt
  split -b 25m v10_best_model.pt v10best_part_
  for f in v10best_part_*; do
    git add -f "$f" && git commit -qm "ckpt piece: $f"
  done
  for i in 1 2 3 4; do
    git push -qf "$PUSH" results-v10-ckpt && break
    sleep 30
  done
  git checkout results-v10
fi
git add -f v10_train.log hb_v10.jsonl v10_driver.jsonl 2>/dev/null || true
git commit -qm "flash logs (rc=$RC)" 2>/dev/null || true
git push -qf "$PUSH" results-v10 2>/dev/null || true
if [ "$RC" = "3" ]; then
  hb "KILL banked — volume intact for kill-fix-relaunch"
else
  hb "flash complete rc=$RC"
fi
runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
sleep 120
