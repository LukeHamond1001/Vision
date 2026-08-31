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
[ -d Vision ] || git clone -q https://github.com/LukeHamond1001/one-token-organism.git
cd Vision
KEEP=$DATA/keep_rebuild; mkdir -p "$KEEP"
[ -f HEARTBEAT.log ] && cp -f HEARTBEAT.log "$KEEP/HEARTBEAT.log"
git fetch -q origin main
[ -n "${PIN_SHA:-}" ] && git fetch -q origin "$PIN_SHA" 2>/dev/null
git reset --hard -q "${PIN_SHA:-origin/main}" || git reset --hard -q origin/main
[ -f "$KEEP/HEARTBEAT.log" ] && cp -f "$KEEP/HEARTBEAT.log" HEARTBEAT.log
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/one-token-organism.git"
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
VALUE_W=${VALUE_W:-0}
SALIENCY=${SALIENCY:-0}        # Phase 2: |RPE|-stamped share of the replay lottery
DREAM=${DREAM:-}               # Phase 2: REM — JSON for train(dream=), e.g. {"every_nights":4,"n":4,"max_new":48,"min_q":0.55}
CYCLES=${CYCLES:-1}            # the night: [SWS -> REM] cycles per night (period scales, dose fixed)
OVERLAP=${OVERLAP:-1}          # spans per SWS block (the drawn one + most-overlapping partners)
SPACING=${SPACING:-0}          # lottery decay per replay (0.5 = halve)
COUPLE=${COUPLE:-0}            # 1: the cycle's dream seeds from the span it just replayed
SLEEP_BIRTH=${SLEEP_BIRTH:-0}  # 1: nights from token one (REM still gated on childhood)
DAY_SLEEP=${DAY_SLEEP:-0}      # 1: nights at the stream's day boundaries (the lane whose day closed)
WARM=${WARM:-0}                # 1: replay inside the lane's live state (warm cortex), not a blank one
PAIR_SHARE=${PAIR_SHARE:--1}   # the night's agenda: pair probability per cycle (-1 = the pay lottery)
PAIR_MASTER=${PAIR_MASTER:-0}  # a pair under this contrastive loss retires for good (0 = never)
HOT_ONCE=${HOT_ONCE:-0}        # 1: a hot pair's guarantee fires on its first night only
HOT_FRAC=${HOT_FRAC:-0}        # builder: fraction of corrections pressing <-2> (a new shard name when > 0)
PRESS_TOKENS=${PRESS_TOKENS:-1} # builder: 0 = grades as events only, no approval token in the stream (shard suffix _sense)
SLEEP=${SLEEP:-1}              # 0: NO NIGHT at any stage (driver --sleep-off: ladder 0 everywhere, REM off) — scan12, the golden core
SILENCE=${SILENCE:-0}          # v14: mean silence ticks before turns (2x before model turns); geometric hazard, <pad> as the tick; new shard suffix when > 0
SILENCE_SIDE=${SILENCE_SIDE:-both} # model: pauses ONLY before model turns (ratified 2026-08-24) — shard suffix _silm
CAST=${CAST:-1}                # 0 (v15): no plants/asks in TRAINING lives — eval lives keep the ruler; suffix _nocast
GRADES=${GRADES:-1}            # 0 (v15): no press events — pretraining is prediction; suffix _nograde
CH_SRC=${CH_SRC:-uc}           # magpie (v15): childhood = Smol-Magpie-Ultra; suffix _mag
TRAIN_LANES=${TRAIN_LANES:-}   # v15: train on fewer lanes than the shard holds (the control: 8 lanes of the l32 shard at T=2048)

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
hb "boot SCAN sha=$(git rev-parse --short HEAD) gpu=${GPU:-none} vol=$(df -BG /workspace | awk 'NR==2{print $2,$4}') order=$ORDER lives=$ML T=$T d=$D L=$NL opts=$SCAN_OPTS clocks=$CLOCKS prec=$PRECISION value_w=$VALUE_W saliency=$SALIENCY dream=${DREAM:-none} night=c${CYCLES}/o${OVERLAP}/s${SPACING}/k${COUPLE}/b${SLEEP_BIRTH}/d${DAY_SLEEP}/w${WARM}/p${PAIR_SHARE},${PAIR_MASTER},${HOT_ONCE} hot=$HOT_FRAC press_tokens=$PRESS_TOKENS sleep=$SLEEP silence=$SILENCE/$SILENCE_SIDE cast=$CAST grades=$GRADES ch=$CH_SRC"
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
[ "$HOT_FRAC" != "0" ] && MINI="${MINI}_hot${HOT_FRAC//./}"     # e.g. scan_epi_l32_hot025
[ "$PRESS_TOKENS" = "0" ] && MINI="${MINI}_sense"
[ "$SILENCE" != "0" ] && { SFX="sil"; [ "$SILENCE_SIDE" = "model" ] && SFX="silm"; MINI="${MINI}_${SFX}${SILENCE//./}"; }
[ "$CAST" = "0" ] && MINI="${MINI}_nocast"
[ "$GRADES" = "0" ] && MINI="${MINI}_nograde"
[ "$CH_SRC" = "magpie" ] && MINI="${MINI}_mag"
[ "${FLAT:-0}" = "1" ] && MINI="${MINI}_flat"
[ "${QPRESS:-0}" = "1" ] && MINI="${MINI}_qp"
[ "${MIX_EVERYDAY:-0}" != "0" ] && MINI="${MINI}_mix${MIX_EVERYDAY//./}"
# 49r school: env overrides so a wrapper can point the SAME build/train
# machinery at a different source + shard dir (no-ops when unset)
[ -n "${MINI_OVR:-}" ] && MINI="$MINI_OVR"
NOPRESS=""; [ "$PRESS_TOKENS" = "0" ] && NOPRESS="--no-press-tokens"
SILFLAG=""; [ "$SILENCE" != "0" ] && SILFLAG="--silence-mean $SILENCE --silence-side $SILENCE_SIDE"
[ "$CAST" = "0" ] && SILFLAG="$SILFLAG --cast-off"
[ "$GRADES" = "0" ] && SILFLAG="$SILFLAG --no-grades"
[ "$CH_SRC" = "magpie" ] && SILFLAG="$SILFLAG --childhood-source magpie"
[ "${FLAT:-0}" = "1" ] && SILFLAG="$SILFLAG --flat-life"
[ "${QPRESS:-0}" = "1" ] && SILFLAG="$SILFLAG --press-quality"
[ "${MIX_EVERYDAY:-0}" != "0" ] && SILFLAG="$SILFLAG --flat-mix-everyday $MIX_EVERYDAY"
UC_SIMPLE=${UC_SIMPLE_OVR:-$DATA/uc_simple.jsonl}; UC_REST=${UC_REST_OVR:-$DATA/uc_rest.jsonl}
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
    --tokenizer "$DATA/mini_epi/tokenizer.json" --episodic --hot-frac "$HOT_FRAC" $NOPRESS $SILFLAG > "scan_build_$ITER.log" 2>&1
  hb "scan shard $MINI $( [ -f "$MINI/manifest.json" ] && echo ok || echo FAILED) $(tail -1 "scan_build_$ITER.log" | cut -c1-160)"
fi
[ -f "$MINI/manifest.json" ] || { hb "ABORT no shard"; sleep 30; [ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
# v15: the proj bin needs UNSEEN lives WITH arcs — a small eval shard,
# built once, walked by the battery beside the comparability eval

# ---- 2b. how copyable are the shards (context for every CE number) ----
hb "copyable train: $(python scripts/copyable.py "$MINI" 200000 2>/dev/null | cut -c1-330)"
hb "copyable eval: $(python scripts/copyable.py "$DATA/mini_eval_epi" 200000 2>/dev/null | cut -c1-330)"

# ---- BUILD_ONLY=1 (49k): build the shard on cheap capacity (a 4090),
# beacon the volume usage, and stop — the A100 trainer launches after
# and finds the shard ready (the build block above skips on manifest)
if [ "${BUILD_ONLY:-0}" = "1" ]; then
  hb "BUILD DONE shard=$MINI $(du -sBG "$MINI" 2>/dev/null | cut -f1)"
  hb "volume usage: total=$(du -sBG /workspace 2>/dev/null | cut -f1)/150G top: $(du -sBG /workspace/* 2>/dev/null | sort -rh | head -6 | tr '\n\t' ' _')"
  sleep 20
  [ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  exit 0
fi

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
export T LIVES="${TRAIN_LANES:-$ML}" CLOCKS ORDER SCAN_OPTS D NL PRECISION MINI_DATA="$MINI" SMOKE_OUT="scan_smoke_$ITER.json"
python scan_smoke.py > "scan_smoke_$ITER.log" 2>&1
hb "smoke $(grep SCANSMOKE "scan_smoke_$ITER.log" | tail -1 | cut -c1-240)$( grep -q SCANSMOKE "scan_smoke_$ITER.log" || tail -3 "scan_smoke_$ITER.log" | tr '\n' ' ' | cut -c1-300)"
LAM=$(python -c "import json;print(json.load(open('scan_smoke_$ITER.json'))['lam'])" 2>/dev/null || echo 0.03)

# ---- 4. the one run ----
OUTM=/workspace/v10_scan_out_$ITER; mkdir -p "$OUTM"
[ "${FRESH:-0}" = "1" ] && { rm -f "$OUTM/scan.pt" "$OUTM/scan.pt.best.pt" "$OUTM/scan.pt.trace.jsonl"; hb "FRESH=1: stale ckpts purged"; }
MS=""; [ "$MAX_STEPS" != "0" ] && MS="--max-steps $MAX_STEPS"
# the caching allocator with expandable segments: the per-token loop's
# allocation pattern varies with the tick pattern; without it the
# allocator fragments and cudaMalloc/free creep into every step
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC2086
# 49o: preserve the previous attempt's log — crash tracebacks were
# being truncated away by each restart (two silent deaths unexplained)
[ -f "scan_train_$ITER.log" ] && cp -f "scan_train_$ITER.log" "scan_train_$ITER.prev.log"
python scripts/v10_driver.py \
  --data "$MINI" --eval-data "$DATA/mini_eval_epi" \
  --ckpt "$OUTM/scan.pt" --lam "$LAM" \
  --arch scan --scan-order "$ORDER" --scan-opts "$SCAN_OPTS" --value-w "$VALUE_W" \
  --saliency "$SALIENCY" ${DREAM:+--dream "$DREAM"} \
  --cycles "$CYCLES" --overlap "$OVERLAP" --spacing "$SPACING" --couple-dream "$COUPLE" --sleep-from-birth "$SLEEP_BIRTH" --day-sleep "$DAY_SLEEP" --warm-replay "$WARM" --pair-share "$PAIR_SHARE" --pair-master "$PAIR_MASTER" --hot-once "$HOT_ONCE" --sleep-off "$([ "$SLEEP" = "0" ] && echo 1 || echo 0)" \
  --d "$D" --n-layers "$NL" --T "$T" --lanes "${TRAIN_LANES:-$ML}" --clocks "$CLOCKS" \
  --keyed hidden --precision "$PRECISION" \
  --lr "$LR" --lr-warmup "$WARMUP" \
  --hb-every "$HB_EVERY" --hb-chunks "$HBC" --lesion-every 2 $MS \
  --device cuda --hb-out "scan_hb_$ITER.jsonl" --trace "scan_driver_$ITER.jsonl" \
  --log-every 100 > "scan_train_$ITER.log" 2>&1 &
PID=$!
hb "run started pid $PID lam $LAM shard=$MINI: $(grep -o 'PLAN {.*' "scan_train_$ITER.log" | head -1 | cut -c1-300)"
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
