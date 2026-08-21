#!/usr/bin/env bash
# v10.1 REBUILD (2026-08-21): episodic cast on the existing real-
# dialogue spine, built on the cheap mule attached to the PREP volume,
# then shipped — corpus AND sources — to the flash volume so one
# volume holds everything afterwards. Self-sensing like pod_v10_prep.
set -uo pipefail
W=/workspace/w-v10prep
DATA=/workspace/v10
mkdir -p "$W" "$DATA" && cd "$W"
[ -d Vision ] || git clone --depth 1 \
  https://github.com/LukeHamond1001/Vision.git
cd Vision
KEEP=$DATA/keep_rebuild; mkdir -p "$KEEP"
[ -f HEARTBEAT.log ] && cp -f HEARTBEAT.log "$KEEP/HEARTBEAT.log"
git fetch -q origin main && git reset --hard -q origin/main
[ -f "$KEEP/HEARTBEAT.log" ] && cp -f "$KEEP/HEARTBEAT.log" HEARTBEAT.log
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git checkout -B results-v10

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log rebuild.log build_tail.log flash_manifest.json; do
    [ -f "$f" ] && git add -f "$f" 2>/dev/null
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v10 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v10 2>/dev/null; } || true
}
hb "boot v10-REBUILD vol=$(df -BG /workspace | awk 'NR==2{print $2,$4}')"
pip install -q numpy tokenizers pyarrow >> rebuild.log 2>&1
python -c "import torch" 2>/dev/null || \
  pip install -q torch --index-url \
    https://download.pytorch.org/whl/cpu >> rebuild.log 2>&1

RAW=$DATA/raw
UC_EVAL=$RAW/ultrachat_train_9.jsonl
UC_SIMPLE=$DATA/uc_simple.jsonl
UC_REST=$DATA/uc_rest.jsonl
STG="--stages flash --st2-epochs 2 --magpie-epochs 2"
LIVES=${LIVES:-8}

# ---- 0. survivors of the first prep (the UC split, the judge freeze,
# the measured budget) — without them this is a full prep, not a rebuild
for f in "$UC_SIMPLE" "$UC_REST" "$DATA/judge_freeze.json" "$DATA/measure.json"; do
  [ -f "$f" ] || { hb "ABORT survivor missing: $f (run pod_v10_prep.sh)"; exit 1; }
done
hb "survivors ok: uc_simple $(du -BG "$UC_SIMPLE" | cut -f1) uc_rest $(du -BG "$UC_REST" | cut -f1)"
BUDGET=$(python - <<'P'
import json
m = json.load(open("/workspace/v10/measure.json"))
print(min(m["_max_budget"], 10_000_000_000))
P
)
# UC_ALL is consumed by the builder only through the split files in
# flash stages; the CLI still wants the flag -> point it at uc_rest
UCARGS="--ultrachat $UC_REST --ultrachat-simple $UC_SIMPLE --ultrachat-rest $UC_REST"

# ---- 1. re-fetch ST2 + Magpie + the reserved UC eval file ----
if [ ! -f "$RAW/.fetch_rebuild_done" ]; then
  bash scripts/fetch_v10_corpus.sh rebuild "$RAW" >> rebuild.log 2>&1 \
    && touch "$RAW/.fetch_rebuild_done"
  hb "fetch $( [ -f "$RAW/.fetch_rebuild_done" ] && echo ok || echo FAILED): $(ls "$RAW"/*.parquet 2>/dev/null | wc -l) parquets"
fi
[ -f "$RAW/.fetch_rebuild_done" ] || { hb "ABORT fetch failed"; exit 1; }

# ---- 2. the episodic corpus (fresh tokenizer sample from the spine) ----
if [ ! -f "$DATA/flash_epi/manifest.json" ]; then
  hb "episodic build starts: lives=$LIVES budget=$BUDGET"
  # shellcheck disable=SC2086
  python -m iga.lm_data_life prepare --out "$DATA/flash_epi" \
    --budget "$BUDGET" --lives "$LIVES" --seed 10 \
    $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
    --judge-thresholds "$DATA/judge_freeze.json" \
    --tok-sample 20000 --episodic > build.log 2>&1
  tail -40 build.log > build_tail.log
  hb "episodic build $( [ -f "$DATA/flash_epi/manifest.json" ] && echo ok || echo FAILED)"
fi
[ -f "$DATA/flash_epi/manifest.json" ] || { hb "ABORT build failed"; exit 1; }
cp "$DATA/flash_epi/manifest.json" flash_manifest.json 2>/dev/null || true

# ---- 3. eval shard (UC train_9 only, novel episodic facts) + smoke ----
if [ ! -f "$DATA/flash_eval_epi/manifest.json" ]; then
  python -m iga.lm_data_life prepare --out "$DATA/flash_eval_epi" \
    --budget 40000000 --lives 2 --seed 999 --world-seed 999 \
    --stages flash --ultrachat "$UC_EVAL" \
    --judge-thresholds "$DATA/judge_freeze.json" \
    --tokenizer "$DATA/flash_epi/tokenizer.json" --episodic >> build.log 2>&1
  hb "eval shard $( [ -f "$DATA/flash_eval_epi/manifest.json" ] && echo ok || echo FAILED)"
fi
if [ ! -f "$DATA/smoke_l8_epi/manifest.json" ]; then
  # shellcheck disable=SC2086
  python -m iga.lm_data_life prepare --out "$DATA/smoke_l8_epi" \
    --budget 16000000 --lives 8 --seed 10 \
    $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
    --judge-thresholds "$DATA/judge_freeze.json" \
    --tokenizer "$DATA/flash_epi/tokenizer.json" --episodic >> build.log 2>&1
  hb "smoke shard $( [ -f "$DATA/smoke_l8_epi/manifest.json" ] && echo ok || echo FAILED)"
fi
for d in flash_eval_epi smoke_l8_epi; do
  [ -f "$DATA/$d/manifest.json" ] || { hb "ABORT $d failed"; exit 1; }
done
python - >> rebuild.log 2>&1 <<'P'
import json
m = json.load(open("/workspace/v10/flash_epi/manifest.json"))
s = m["stats"]
print("CAST", json.dumps({k: s.get(k) for k in ("cast_asks", "epi_facts", "epi_asks", "plants", "corrections", "kept")}))
P
hb "cast: $(grep '^CAST' rebuild.log | tail -1 | cut -c1-160)"

# ---- 4. SHIP corpus + sources (one volume afterwards) ----
if [ "${SHIP:-1}" = "1" ]; then
  if [ ! -f /workspace/v10_ship2.tar ]; then
    tar cf /workspace/v10_ship2.tar -C /workspace \
      v10/flash_epi v10/flash_eval_epi v10/smoke_l8_epi \
      v10/measure.json v10/judge_freeze.json v10/split.json \
      v10/uc_simple.jsonl v10/uc_rest.jsonl v10/raw
  fi
  hb "ship tar ready $(du -BG /workspace/v10_ship2.tar | cut -f1)"
  runpodctl send /workspace/v10_ship2.tar > send.log 2>&1 &
  SENDPID=$!
  CODE=""
  for i in $(seq 1 90); do
    CODE=$(grep -o "runpodctl receive [A-Za-z0-9_-]*" send.log | head -1 | awk '{print $3}')
    [ -n "$CODE" ] && break
    sleep 2
  done
  if [ -z "$CODE" ]; then hb "SHIP FAILED: no code"; else
    hb "SHIP code ready (see send.log; the flash pod receives with SHIP_CODE env)"
    echo "$CODE" > ship_code2.txt; git add -f ship_code2.txt
    git commit -qm "ship code 2" 2>/dev/null; git push -qf "$PUSH" results-v10 2>/dev/null || true
    wait $SENDPID; RC=$?
    hb "send finished rc=$RC"
  fi
fi
hb "rebuild pod done; self-removing in 60s"
sleep 60
runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
