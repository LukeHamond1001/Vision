#!/usr/bin/env bash
# v10 PREP (CPU pod + network volume; A51/A54c split-prep pattern).
# Self-sensing phases — each skips when its artifact already exists
# on the volume:
#   1. fetch the spine raw (UltraChat + SmolTalk2 no_think subsets)
#   2. concat UltraChat train_0..8 -> spine fill; train_9 RESERVED
#      for the eval shard (disjoint by file)
#   3. measure one-epoch yield per stage (the A12 budget input)
#   4. freeze judge stage thresholds on the REAL per-stage mixes
#      (lm_judge freeze protocol, step 8)
#   5. build the paid-smoke shards (8- and 12-life minis)
#   6. gated on FULL_LIVES (set from the paid smoke's verdict):
#      build the FULL flash corpus + eval shard, then delete raw
# Terminates itself when the phases it can run are done.
set -uo pipefail
W=/workspace/w-v10prep
DATA=/workspace/v10
mkdir -p "$W" "$DATA" && cd "$W"
[ -d iga-scale ] || git clone --depth 1 \
  https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
git fetch -q origin main && git reset --hard -q origin/main
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -B results-v10

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log prep.log measure.json judge_freeze.json \
           build_tail.log; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v10 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v10 2>/dev/null; } || true
}
hb "boot v10-PREP vol=$(df -BG /workspace | awk 'NR==2{print $2,$4}') FULL_LIVES=${FULL_LIVES:-unset}"
pip install -q numpy tokenizers pyarrow >> prep.log 2>&1
python -c "import torch" 2>/dev/null || \
  pip install -q torch --index-url \
    https://download.pytorch.org/whl/cpu >> prep.log 2>&1
hb "deps ready"

RAW=$DATA/raw
UC_ALL=$RAW/ultrachat_all.jsonl
UC_EVAL=$RAW/ultrachat_train_9.jsonl
UC_SIMPLE=$DATA/uc_simple.jsonl
UC_REST=$DATA/uc_rest.jsonl
STG="--stages flash --st2-epochs 2 --magpie-epochs 2"

# generation sentinel: derived artifacts from an older prep design
# are poison (the 44M-budget corpus, 2026-08-20) — wipe everything
# DERIVED once per generation tag; raw fetches survive.
GEN=b2
if [ ! -f "$DATA/.gen_$GEN" ]; then
  rm -rf "$DATA/flash" "$DATA/flash_eval" "$DATA"/smoke_l* \
    "$DATA/measure.json" "$DATA/judge_freeze.json" \
    "$UC_SIMPLE" "$UC_REST" "$DATA/split.json" \
    /workspace/v10_ship.tar /workspace/v10_out 2>/dev/null || true
  touch "$DATA/.gen_$GEN"
  hb "generation wipe -> $GEN"
fi

# ---- 1. fetch ----
if [ ! -f "$RAW/.fetch_done" ]; then
  bash scripts/fetch_v10_corpus.sh full "$RAW" >> prep.log 2>&1 \
    && touch "$RAW/.fetch_done"
  hb "fetch $( [ -f "$RAW/.fetch_done" ] && echo ok || echo FAILED)"
fi
[ -f "$RAW/.fetch_done" ] || { hb "ABORT fetch failed"; exit 1; }

# ---- 2. concat spine fill (train_9 stays out: the eval file) ----
if [ ! -f "$UC_ALL" ]; then
  cat "$RAW"/ultrachat_train_{0,1,2,3,4,5,6,7,8}.jsonl > "$UC_ALL.tmp" \
    && mv "$UC_ALL.tmp" "$UC_ALL" \
    && rm -f "$RAW"/ultrachat_train_{0,1,2,3,4,5,6,7,8}.jsonl
  hb "uc concat $(du -BG "$UC_ALL" | cut -f1)"
fi

# ---- 3. measure the honest feasible budget (shared UC pool,
# late-stage epochs, flash fracs) ----
if [ ! -f "$DATA/measure.json" ]; then
  # shellcheck disable=SC2086
  python -m iga.lm_data_life measure --out "$DATA/measure.json" \
    $STG --ultrachat "$UC_ALL" --st2-dir "$RAW" --magpie-dir "$RAW" \
    >> prep.log 2>&1
  cp "$DATA/measure.json" measure.json 2>/dev/null || true
  hb "measure: $(python -c "import json;print(json.load(open('$DATA/measure.json'))['_max_budget'])" 2>/dev/null || echo FAILED)"
fi
[ -f "$DATA/measure.json" ] || { hb "ABORT no measure"; exit 1; }
BUDGET=$(python - <<'P'
import json
m = json.load(open("/workspace/v10/measure.json"))
print(min(m["_max_budget"], 10_000_000_000))
P
)

# ---- 3b. split UltraChat: simplest slice -> infancy (one shared
# pass; infancy and childhood never re-read the same rows) ----
if [ ! -f "$DATA/split.json" ]; then
  python -m iga.lm_data_life split-uc --out "$DATA/split.json" \
    --ultrachat "$UC_ALL" \
    --ultrachat-simple "$UC_SIMPLE" --ultrachat-rest "$UC_REST" \
    --infancy-tokens $((BUDGET * 8 / 100 * 105 / 100)) \
    >> prep.log 2>&1
  hb "uc split: $(head -c 200 "$DATA/split.json" 2>/dev/null | tr -d '\n ' || echo FAILED)"
fi
[ -f "$UC_SIMPLE" ] || { hb "ABORT no uc split"; exit 1; }
UCARGS="--ultrachat $UC_ALL --ultrachat-simple $UC_SIMPLE --ultrachat-rest $UC_REST"

# ---- 4. freeze judge on the real stage mixes (infancy graded on
# the real simple slice) ----
if [ ! -f "$DATA/judge_freeze.json" ]; then
  # shellcheck disable=SC2086
  python -m iga.lm_data_life freeze-judge \
    --out "$DATA/judge_freeze.json" \
    $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
    >> prep.log 2>&1
  cp "$DATA/judge_freeze.json" judge_freeze.json 2>/dev/null || true
  hb "judge freeze $( [ -f "$DATA/judge_freeze.json" ] && echo ok || echo FAILED)"
fi
[ -f "$DATA/judge_freeze.json" ] || { hb "ABORT no judge freeze"; exit 1; }

# ---- 5. paid-smoke shards (exact-shape minis) ----
for L in ${SMOKE_LANES:-8 12}; do
  if [ ! -f "$DATA/smoke_l$L/manifest.json" ]; then
    # shellcheck disable=SC2086
    python -m iga.lm_data_life prepare --out "$DATA/smoke_l$L" \
      --budget $((L * 2000000)) --lives $L --seed 10 \
      $STG $UCARGS \
      --st2-dir "$RAW" --magpie-dir "$RAW" \
      --judge-thresholds "$DATA/judge_freeze.json" \
      >> prep.log 2>&1
    hb "smoke shard l$L $( [ -f "$DATA/smoke_l$L/manifest.json" ] && echo ok || echo FAILED)"
  fi
done

# ---- 5b. shard export for volume-less shopper pods (the GPU
# shop smokes on other DCs; ~50-80MB rides the results branch) ----
if [ -f "$DATA/smoke_l8/manifest.json" ] && \
   [ -f "$DATA/smoke_l12/manifest.json" ] && \
   [ ! -f smoke_shards.tar.gz ]; then
  tar czf smoke_shards.tar.gz -C "$DATA" smoke_l8 smoke_l12
  SZ=$(du -m smoke_shards.tar.gz | cut -f1)
  if [ "$SZ" -lt 95 ]; then
    git add -f smoke_shards.tar.gz
    hb "smoke shards exported (${SZ}MB)"
  else
    rm -f smoke_shards.tar.gz
    hb "smoke shards too big to export (${SZ}MB) — shoppers need a volume"
  fi
fi

# ---- 6. the full corpus (gated on the paid smoke's lane pick;
# FULL_LIVES env, else read from any smoke verdict on the volume) --
if [ -z "${FULL_LIVES:-}" ]; then
  for SJ in /workspace/v10_out/smoke.json /workspace/v10_out/smoke_*.json; do
    if [ -f "$SJ" ]; then
      FULL_LIVES=$(python -c "import json;print(json.load(open('$SJ'))['lanes'])" 2>/dev/null || true)
      [ -n "$FULL_LIVES" ] && break
    fi
  done
fi
if [ -n "${FULL_LIVES:-}" ] && [ ! -f "$DATA/flash/manifest.json" ]; then
  hb "full build starts: lives=$FULL_LIVES budget=$BUDGET"
  # shellcheck disable=SC2086
  python -m iga.lm_data_life prepare --out "$DATA/flash" \
    --budget "$BUDGET" --lives "$FULL_LIVES" --seed 10 \
    $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
    --judge-thresholds "$DATA/judge_freeze.json" \
    --tok-sample 20000 > build.log 2>&1
  tail -40 build.log > build_tail.log
  hb "full build $( [ -f "$DATA/flash/manifest.json" ] && echo ok || echo FAILED)"
  if [ -f "$DATA/flash/manifest.json" ]; then
    # eval shard: UltraChat train_9 ONLY (reserved file). The old
    # draft drew adolescence/tail from the SAME ST2/Magpie parquets
    # as training — eval contamination; a UC-only eval mix is a
    # clean, consistent CE reference and the cast probes are
    # synthetic-world anyway.
    python -m iga.lm_data_life prepare --out "$DATA/flash_eval" \
      --budget 40000000 --lives 2 --seed 999 --world-seed 999 \
      --stages flash --ultrachat "$UC_EVAL" \
      --judge-thresholds "$DATA/judge_freeze.json" \
      --tokenizer "$DATA/flash/tokenizer.json" >> build.log 2>&1
    hb "eval shard $( [ -f "$DATA/flash_eval/manifest.json" ] && echo ok || echo FAILED)"
    git add -f "$DATA/flash/manifest.json" 2>/dev/null || true
    cp "$DATA/flash/manifest.json" flash_manifest.json && \
      git add -f flash_manifest.json && \
      git commit -qm "flash manifest" && \
      git push -qf "$PUSH" results-v10 || true
    # A54c: sources fetched then deleted after tokenization
    rm -rf "$RAW"
    hb "raw deleted; volume $(df -BG /workspace | awk 'NR==2{print $4}') free"
  fi
fi

# ---- 7. SHIP (cross-DC: cheap prep DC -> the flash GPU's DC via
# runpodctl send/receive; the receiver is the flash pod itself,
# launched when ship_code.txt appears on the results branch) ------
if [ "${SHIP:-0}" = "1" ] && [ -f "$DATA/flash/manifest.json" ]; then
  if [ ! -f /workspace/v10_ship.tar ]; then
    TAR_IN="v10/flash v10/flash_eval v10/measure.json v10/judge_freeze.json"
    for L in ${SMOKE_LANES:-8 12}; do TAR_IN="$TAR_IN v10/smoke_l$L"; done
    # shellcheck disable=SC2086
    tar cf /workspace/v10_ship.tar -C /workspace $TAR_IN
  fi
  hb "ship tar ready $(du -BG /workspace/v10_ship.tar | cut -f1)"
  runpodctl send /workspace/v10_ship.tar > send.log 2>&1 &
  SENDPID=$!
  CODE=""
  for i in $(seq 1 90); do
    CODE=$(grep -o "runpodctl receive [A-Za-z0-9_-]*" send.log | head -1 | awk '{print $3}')
    [ -n "$CODE" ] && break
    sleep 2
  done
  if [ -n "$CODE" ]; then
    echo "$CODE" > ship_code.txt
    git add -f ship_code.txt send.log
    git commit -qm "ship code" 2>/dev/null || true
    git push -qf "$PUSH" results-v10 2>/dev/null || true
    hb "SHIP CODE POSTED — serving transfer, waiting for receiver"
    wait $SENDPID
    hb "transfer done rc=$? $(tail -2 send.log | head -1)"
  else
    hb "SHIP FAILED: no code from runpodctl send"
  fi
fi

hb "prep phases complete"
if [ "${SKIP_TERMINATE:-0}" != "1" ]; then
  runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  sleep 60
fi
