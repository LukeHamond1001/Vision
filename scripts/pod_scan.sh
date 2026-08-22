#!/usr/bin/env bash
# pod_scan.sh — ONE iteration of the one-token organism (docs/
# ONE_TOKEN_PLAN.md, 2026-08-21) on a cheap GPU (4090 / RTX 2000 Ada)
# attached to the PREP network volume. Boots by dockerEntrypoint from a
# SHA-pinned raw URL (scripts/launch_pod.sh). Self-sensing: the shard
# and the checkpoint live on the volume, so a relaunch resumes.
#
#   1. inventory the volume (corpus, shards, sources, ship tar)
#   2. build the LIVES-life scan shard (one life per lane) if missing
#   3. lam smoke (40 steps, the real config) -> tok/s, peak memory
#   4. ONE driver job with heartbeats/rows pushed to results-v10
#   5. self-remove (KEEP_POD=1 keeps it)
#
# env (all optional): ITER tag (scan1) | ORDER pfc_first|cortex_first |
# LIVES 32 | T 64 | D 512 | NL 8 | SCAN_OPTS json | CLOCKS | PRECISION
# bf16 | LR 1e-4 | WARMUP 1000 | HB_EVERY 6000 | HBC 16000 | BUDGET_MINI
# 200000000 | MAX_STEPS 0 | PIN_SHA | KEEP_POD 0
set -uo pipefail
W=/workspace/w-v10prep
DATA=/workspace/v10
mkdir -p "$W" "$DATA" && cd "$W"
[ -d Vision ] || git clone -q https://github.com/LukeHamond1001/Vision.git
cd Vision
KEEP=$DATA/keep_rebuild; mkdir -p "$KEEP"
[ -f HEARTBEAT.log ] && cp -f HEARTBEAT.log "$KEEP/HEARTBEAT.log"
git fetch -q origin main
[ -n "${PIN_SHA:-}" ] && git fetch -q origin "$PIN_SHA" 2>/dev/null
git reset --hard -q "${PIN_SHA:-origin/main}" || git reset --hard -q origin/main
[ -f "$KEEP/HEARTBEAT.log" ] && cp -f "$KEEP/HEARTBEAT.log" HEARTBEAT.log
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git checkout -q -B results-v10

ITER=${ITER:-scan1}
ORDER=${ORDER:-pfc_first}
ML=${LIVES:-32}
T=${T:-64}
D=${D:-512}
NL=${NL:-8}
SCAN_OPTS=${SCAN_OPTS:-'{"n_council": 2, "slot_every": 8, "write_every": 4}'}
CLOCKS=${CLOCKS:-3:1,4:8,5:64,6:512,7:4096,8:32768}
PRECISION=${PRECISION:-bf16}
LR=${LR:-1e-4}
WARMUP=${WARMUP:-1000}
HB_EVERY=${HB_EVERY:-6000}
HBC=${HBC:-16000}
BUDGET_MINI=${BUDGET_MINI:-200000000}
MAX_STEPS=${MAX_STEPS:-0}

hb() {
  echo "$(date -u '+%H:%M:%S') [$ITER] $1" >> HEARTBEAT.log
  [ -f "scan_train_$ITER.log" ] && tail -60 "scan_train_$ITER.log" > "scan_tail_$ITER.log" 2>/dev/null
  for f in HEARTBEAT.log scan_hb_*.jsonl scan_driver_*.jsonl scan_tail_*.log \
           scan_smoke_*.json scan_build_*.log; do
    [ -f "$f" ] && git add -f "$f" 2>/dev/null
  done
  git commit -qm "hb: [$ITER] $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v10 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v10 2>/dev/null; } || true
}
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
hb "boot SCAN sha=$(git rev-parse --short HEAD) gpu=${GPU:-none} vol=$(df -BG /workspace | awk 'NR==2{print $2,$4}') order=$ORDER lives=$ML T=$T d=$D L=$NL opts=$SCAN_OPTS clocks=$CLOCKS prec=$PRECISION"
pip install -q numpy tokenizers pyarrow > pip.log 2>&1

# ---- 1. inventory ----
RAW=$DATA/raw
inv=""
for p in flash_epi flash_eval_epi mini_epi mini_eval_epi; do
  inv="$inv $p=$( [ -f "$DATA/$p/manifest.json" ] && echo ok || echo MISSING)"
done
inv="$inv tar=$( [ -f /workspace/v10_ship2.tar ] && du -BG /workspace/v10_ship2.tar | cut -f1 || echo MISSING)"
inv="$inv parquets=$(ls "$RAW"/*.parquet 2>/dev/null | wc -l) judge=$( [ -f "$DATA/judge_freeze.json" ] && echo ok || echo MISSING)"
hb "inventory:$inv"

# ---- 2. the scan shard: LIVES lives (one per lane), BUDGET_MINI tokens,
# the mini_epi tokenizer so the mini eval shard stays comparable ----
MINI=$DATA/scan_epi_l$ML
UC_SIMPLE=$DATA/uc_simple.jsonl; UC_REST=$DATA/uc_rest.jsonl
STG="--stages flash --st2-epochs 2 --magpie-epochs 2"
UCARGS="--ultrachat $UC_REST --ultrachat-simple $UC_SIMPLE --ultrachat-rest $UC_REST"
if [ ! -f "$MINI/manifest.json" ]; then
  for f in "$UC_SIMPLE" "$UC_REST" "$DATA/judge_freeze.json" "$DATA/mini_epi/tokenizer.json" "$DATA/mini_eval_epi/manifest.json"; do
    [ -f "$f" ] || { hb "ABORT missing for the shard build: $f"; sleep 30; [ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
  done
  # shellcheck disable=SC2086
  python -m iga.lm_data_life prepare --out "$MINI" \
    --budget "$BUDGET_MINI" --lives "$ML" --seed 20 \
    $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
    --judge-thresholds "$DATA/judge_freeze.json" \
    --tokenizer "$DATA/mini_epi/tokenizer.json" --episodic > "scan_build_$ITER.log" 2>&1
  hb "scan shard l$ML $( [ -f "$MINI/manifest.json" ] && echo ok || echo FAILED) $(tail -1 "scan_build_$ITER.log" | cut -c1-160)"
fi
[ -f "$MINI/manifest.json" ] || { hb "ABORT no shard"; sleep 30; [ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }

# ---- 2b. how copyable are the shards (context for every CE number) ----
hb "copyable train: $(python scripts/copyable.py "$MINI" 200000 2>/dev/null | cut -c1-330)"
hb "copyable eval: $(python scripts/copyable.py "$DATA/mini_eval_epi" 200000 2>/dev/null | cut -c1-330)"

# ---- 3. lam smoke at the exact config (40 steps) ----
cat > scan_smoke.py <<'PY'
import json, os, sys, time, torch
sys.path.insert(0, ".")
from iga.lm_train import train
from iga.lm_sleep import Sleeper
T = int(os.environ["T"]); L = int(os.environ["LIVES"])
CLK = {int(kv.split(":")[0]): int(kv.split(":")[1]) for kv in os.environ["CLOCKS"].split(",") if kv}
opts = {"order": os.environ["ORDER"]}; opts.update(json.loads(os.environ["SCAN_OPTS"]))
sl = Sleeper(arm="C", every=8, block_chunks=2, seed=1, homeostasis=1e-3)
sl.press_pay = (T, T // 8)
t0 = time.time()
model, drive, vocab, ce0, ce1 = train(
    d=int(os.environ["D"]), n_layers=int(os.environ["NL"]), lanes=L, T=T, steps=40, seed=0,
    device="cuda", arch="scan", store="matrix", keyed="hidden", scan=opts,
    norm_mix=True, aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
    clocks=CLK, precision=os.environ["PRECISION"],
    data=os.environ["MINI_DATA"], sleep=sl, log_every=20)
dt = time.time() - t0
holds = len(drive.ledger) / 40
lam = min(0.25, 0.25 / max(holds, 1))
out = {"lanes": L, "T": T, "order": opts["order"], "tok_s": round(40 * L * T / dt),
       "holds": round(holds, 2), "lam": round(lam, 5), "ce": [round(ce0, 3), round(ce1, 3)],
       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
       "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
json.dump(out, open(os.environ["SMOKE_OUT"], "w"))
print("SCANSMOKE", json.dumps(out))
PY
export T LIVES="$ML" CLOCKS ORDER SCAN_OPTS D NL PRECISION MINI_DATA="$MINI" SMOKE_OUT="scan_smoke_$ITER.json"
python scan_smoke.py > "scan_smoke_$ITER.log" 2>&1
hb "smoke $(grep SCANSMOKE "scan_smoke_$ITER.log" | tail -1 | cut -c1-240)$( grep -q SCANSMOKE "scan_smoke_$ITER.log" || tail -3 "scan_smoke_$ITER.log" | tr '\n' ' ' | cut -c1-300)"
LAM=$(python -c "import json;print(json.load(open('scan_smoke_$ITER.json'))['lam'])" 2>/dev/null || echo 0.03)

# ---- 4. the one run ----
OUTM=/workspace/v10_scan_out_$ITER; mkdir -p "$OUTM"
MS=""; [ "$MAX_STEPS" != "0" ] && MS="--max-steps $MAX_STEPS"
# shellcheck disable=SC2086
python scripts/v10_driver.py \
  --data "$MINI" --eval-data "$DATA/mini_eval_epi" \
  --ckpt "$OUTM/scan.pt" --lam "$LAM" \
  --arch scan --scan-order "$ORDER" --scan-opts "$SCAN_OPTS" \
  --d "$D" --n-layers "$NL" --T "$T" --lanes "$ML" --clocks "$CLOCKS" \
  --keyed hidden --precision "$PRECISION" \
  --lr "$LR" --lr-warmup "$WARMUP" \
  --hb-every "$HB_EVERY" --hb-chunks "$HBC" --lesion-every 2 $MS \
  --device cuda --hb-out "scan_hb_$ITER.jsonl" --trace "scan_driver_$ITER.jsonl" \
  --log-every 100 > "scan_train_$ITER.log" 2>&1 &
PID=$!
hb "run started pid $PID lam $LAM: $(grep -o 'PLAN {.*' "scan_train_$ITER.log" | head -1 | cut -c1-300)"
sleep 120
hb "2 min: $(grep -v 'sleep@' "scan_train_$ITER.log" | grep 'step' | tail -1 | cut -c1-200)"
while kill -0 $PID 2>/dev/null; do
  sleep 900
  hb "inflight $(grep -v 'sleep@' "scan_train_$ITER.log" | grep 'step ' | tail -1 | cut -c1-120) | hb: $(tail -1 "scan_hb_$ITER.jsonl" 2>/dev/null | head -c 200)"
done
wait $PID; RC=$?
hb "run ENDED rc=$RC rows=$(grep -c . "scan_hb_$ITER.jsonl" 2>/dev/null) segs=$(grep -c . "scan_driver_$ITER.jsonl" 2>/dev/null) tail: $(grep -v 'sleep@' "scan_train_$ITER.log" | tail -1 | cut -c1-200)"
if [ "${KEEP_POD:-0}" = "1" ]; then hb "KEEP_POD=1: staying up"; exit 0; fi
hb "self-removing in 60s"
sleep 60
runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
