#!/usr/bin/env bash
# RunPod entry for the from-nothing gestation (LIVE_BODY.md; launched by
# scripts/launch_pod.sh pod_gestate_1b.sh "ENV=.."). Phases, each resumable:
#   1 diet   : stream TinyStories + FineWeb-Edu through the organism's tokenizer,
#              mix the authored lives by token share -> /workspace/text_1b
#   2 smoke  : 40 steps at the chosen shape -> tok/s, peak GiB (HEARTBEAT.log)
#   3 run    : the gestation, checkpoints on the volume, resume on restart
# Knobs (env): D NL LANES T STEPS LR TOKENS_BUDGET LIVES_FRAC HOURS SMOKE_ONLY KEEP_POD
set -uo pipefail
W=/root/w-1b; DATA=/workspace/text_1b; OUT=/workspace/life_1b
mkdir -p "$W" "$OUT" && cd "$W"
[ -d Vision ] || git clone -q https://github.com/LukeHamond1001/one-token-organism.git Vision
cd Vision; git fetch -q origin; git reset --hard -q "${PIN_SHA:-origin/main}"
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/one-token-organism.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"; git checkout -q -B results-1b
ITER=${ITER:-g1b}
hb() { echo "$(date -u '+%H:%M:%S') [$ITER] $1" >> HEARTBEAT.log; git add HEARTBEAT.log; git commit -qm "hb $ITER" >/dev/null 2>&1; git push -q -f "$PUSH" results-1b >/dev/null 2>&1 || true; }
D=${D:-1536}; NL=${NL:-28}; LANES=${LANES:-32}; T=${T:-64}; LR=${LR:-3e-4}
STEPS=${STEPS:-0}; TOKENS_BUDGET=${TOKENS_BUDGET:-3000000000}; LIVES_FRAC=${LIVES_FRAC:-0.05}
HOURS=${HOURS:-60}
pip install -q datasets tokenizers 2>/dev/null
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
hb "boot sha=$(git rev-parse --short HEAD) gpu=${GPU:-none} d=$D L=$NL lanes=$LANES T=$T budget=$TOKENS_BUDGET lives=$LIVES_FRAC vol=$(df -BG /workspace | awk 'NR==2{print $4}')"
# a lifetime guard: the pod removes itself after HOURS (nothing runs blind forever)
( sleep $(( HOURS * 3600 )); hb "lifetime $HOURS h reached — removing pod"; [ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID" ) &
# --- 1 diet (resumable by presence) ---
if [ ! -f "$DATA/tokens.bin" ]; then
  mkdir -p "$DATA"
  python3 -m iga.lm_data_life prepare --out "$W/gest_lives" --budget 20000000 --lives 16 \
      --tokenizer data/ship_tok_v17.json --vocab 16388 --lives-file data/lives_v17.jsonl \
      --no-press-tokens --episodic > "$W/lives_prep.log" 2>&1 || hb "lives prep failed: $(tail -2 $W/lives_prep.log | tr '\n' ' ')"
  HALF=$(( TOKENS_BUDGET / 2 ))
  python3 -m iga.lm_data_text prepare --out "$W/text_a" --budget "$HALF" --lanes "$LANES" \
      --source roneneldan/TinyStories:text --source HuggingFaceFW/fineweb-edu:text:sample-10BT --weight 1 --weight 2 \
      --lives "$W/gest_lives" --lives-frac "$LIVES_FRAC" --seed 1 > "$W/prep_a.log" 2>&1 &
  python3 -m iga.lm_data_text prepare --out "$W/text_b" --budget "$HALF" --lanes "$LANES" \
      --source HuggingFaceFW/fineweb-edu:text:sample-10BT --lives "$W/gest_lives" --lives-frac "$LIVES_FRAC" --seed 2 > "$W/prep_b.log" 2>&1 &
  wait
  python3 scripts/shards_concat.py --out "$DATA" --lanes "$LANES" "$W/text_a" "$W/text_b" > "$W/concat.log" 2>&1
  hb "diet: $(tail -1 $W/concat.log) | a: $(tail -1 $W/prep_a.log) | b: $(tail -1 $W/prep_b.log)"
  rm -rf "$W/text_a" "$W/text_b"
else
  hb "diet present: $(python3 -c "import json;print(json.load(open('$DATA/manifest.json'))['tokens'])") tokens"
fi
# --- 2 smoke: speed and memory at this shape ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -f "$OUT/smoke.json" ]; then
  python3 - <<PY > "$W/smoke.log" 2>&1
import json, time, torch, subprocess, sys
sys.path.insert(0, ".")
t0 = time.time()
r = subprocess.run([sys.executable, "scripts/gestate_1b.py", "--data", "$DATA", "--d", "$D", "--n-layers", "$NL",
                    "--lanes", "$LANES", "--T", "$T", "--steps", "40", "--device", "cuda", "--precision", "bf16",
                    "--ckpt", "$W/smoke.pt", "--log-every", "10", "--no-sleep"], capture_output=True, text=True)
dt = time.time() - t0
last = [l for l in r.stdout.splitlines() if l.startswith("{")]
info = json.loads(last[-1]) if last else {"error": (r.stderr or r.stdout)[-400:]}
info.update({"wall_s": round(dt, 1), "tok_s_wall": round(40 * $LANES * $T / dt)})
try:
    import re
    info["peak_gib_reported"] = None
    info["last_log"] = [l for l in r.stdout.splitlines() if l.startswith("step")][-1:]
except Exception:
    pass
json.dump(info, open("$OUT/smoke.json", "w"))
print(json.dumps(info))
PY
  hb "smoke $(tail -c 600 $W/smoke.log | tr '\n' ' ')"
fi
[ "${SMOKE_ONLY:-0}" = "1" ] && { hb "smoke only — done"; [ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID"; exit 0; }
# --- 3 the gestation (resume from the volume's checkpoint) ---
if [ "$STEPS" = "0" ]; then STEPS=$(( TOKENS_BUDGET / (LANES * T) )); fi
hb "run: $STEPS steps ($(( STEPS * LANES * T / 1000000 ))M tokens) lr=$LR -> $OUT/scan.pt"
python3 scripts/gestate_1b.py --data "$DATA" --d "$D" --n-layers "$NL" --lanes "$LANES" --T "$T" \
    --steps "$STEPS" --lr "$LR" --device cuda --precision bf16 --ckpt "$OUT/scan.pt" --resume --log-every 200 \
    > "$W/train.log" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 1800; hb "train: $(grep '^step' $W/train.log | tail -1 | cut -c1-200)"
done
hb "train exited: $(tail -3 $W/train.log | tr '\n' ' ' | cut -c1-400)"
cp -f "$OUT/scan.pt" "$OUT/scan_final.pt" 2>/dev/null
hb "DONE ckpt=$OUT/scan_final.pt"
[ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID"
