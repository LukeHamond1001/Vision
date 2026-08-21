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
# MINI_TAG (2026-08-21): a second mini-flash on the same shards (a trunk
# candidate: ATTN=rope QK_NORM=1 MLP=swiglu) writes mini_*<tag> files and
# its own checkpoint dir, so the baseline's rows on results-v10 survive
MT=${MINI_TAG:-}
[ -f HEARTBEAT.log ] && cp -f HEARTBEAT.log "$KEEP/HEARTBEAT.log"
git fetch -q origin main && git reset --hard -q origin/main
[ -f "$KEEP/HEARTBEAT.log" ] && cp -f "$KEEP/HEARTBEAT.log" HEARTBEAT.log
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git checkout -B results-v10

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for t in mini_train*.log; do
    [ -f "$t" ] && tail -60 "$t" > "${t/mini_train/mini_tail}" 2>/dev/null
  done
  for f in HEARTBEAT.log rebuild.log build_tail.log flash_manifest.json \
           mini_hb*.jsonl mini_driver*.jsonl mini_tail*.log mini_smoke*.json; do
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

# ---- 1b. MINI-FLASH GATE (78M, the real driver + battery, bf16) on
# this pod's GPU while the CPU builds the full corpus. The faithful
# gate: G1 at d=128/12k steps is ~17x under-exposed for a real-
# dialogue diet (0.7M cast tokens vs A69-R2's 12M); 4 lives x 50M
# tokens at d=512/8L gives the in-ctx-vs-tokens curve the launch
# decision needs. Publishes mini_hb.jsonl every 10 min.
# allocator: expandable segments — the 2026-08-21 mini-flash OOM was
# fragmentation (15.3 GiB reserved, 267 MiB free on a 16 GB card)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (mini checkpoint dirs are per tag: /workspace/v10_mini_out<tag>, made in start_mini)
# MINI_LIVES (2026-08-21, after the 4-lane OOM on the 16 GB card): the
# mini shard is lanes == lives; 2 lives x 50M halves activation memory
# at the same tokens per life. Built from the 4-life shard's tokenizer
# so the eval shard (mini_eval_epi) stays valid for every variant.
ML=${MINI_LIVES:-4}
MINI=$DATA/mini_epi; [ "$ML" != "4" ] && MINI=$DATA/mini_epi_l$ML
if [ "${MINIGATE:-1}" = "1" ]; then
  if [ ! -f "$DATA/mini_epi/manifest.json" ]; then
    # shellcheck disable=SC2086
    python -m iga.lm_data_life prepare --out "$DATA/mini_epi" \
      --budget 200000000 --lives 4 --seed 20 \
      $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
      --judge-thresholds "$DATA/judge_freeze.json" \
      --tok-sample 20000 --episodic > mini_build.log 2>&1
    hb "mini shard $( [ -f "$DATA/mini_epi/manifest.json" ] && echo ok || echo FAILED)"
  fi
  if [ "$ML" != "4" ] && [ ! -f "$MINI/manifest.json" ] && [ -f "$DATA/mini_epi/tokenizer.json" ]; then
    # shellcheck disable=SC2086
    python -m iga.lm_data_life prepare --out "$MINI" \
      --budget $((50000000 * ML)) --lives "$ML" --seed 20 \
      $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
      --judge-thresholds "$DATA/judge_freeze.json" \
      --tokenizer "$DATA/mini_epi/tokenizer.json" --episodic >> mini_build.log 2>&1
    hb "mini shard l$ML $( [ -f "$MINI/manifest.json" ] && echo ok || echo FAILED)"
  fi
  if [ -f "$DATA/mini_epi/manifest.json" ] && [ ! -f "$DATA/mini_eval_epi/manifest.json" ]; then
    python -m iga.lm_data_life prepare --out "$DATA/mini_eval_epi" \
      --budget 20000000 --lives 2 --seed 999 --world-seed 999 \
      --stages flash --ultrachat "$UC_EVAL" \
      --judge-thresholds "$DATA/judge_freeze.json" \
      --tokenizer "$DATA/mini_epi/tokenizer.json" --episodic >> mini_build.log 2>&1
    hb "mini eval $( [ -f "$DATA/mini_eval_epi/manifest.json" ] && echo ok || echo FAILED)"
  fi
  cat > mini_smoke.py <<'PY'
import json, os, sys, time, torch
sys.path.insert(0, ".")
from iga.lm_train import train
from iga.lm_sleep import Sleeper
sl = Sleeper(arm="C", every=8, block_chunks=2, seed=1, homeostasis=1e-3)
T = int(os.environ.get("MINI_T", "2048")); CM = int(os.environ.get("MINI_CM", "1"))
CLK = {3: 1 * CM, 4: 8 * CM, 5: 64 * CM, 6: 512 * CM}
if os.environ.get("MINI_CLOCKS"):
    CLK = {int(kv.split(":")[0]): int(kv.split(":")[1])
           for kv in os.environ["MINI_CLOCKS"].split(",") if kv}
sl.press_pay = (T, T // 8)
t0 = time.time()
L = int(os.environ["MINI_LANES"])
model, drive, vocab, ce0, ce1 = train(
    d=512, n_layers=8, lanes=L, T=T, steps=40, seed=0, device="cuda",
    arch="hybrid", store="matrix", keyed=os.environ.get("MINI_KEYED", "logit"),
    band_credit=os.environ.get("MINI_BCREDIT", "0") == "1",
    band_center=os.environ.get("MINI_BCENTER", "0") == "1",
    tail_tokens=int(os.environ.get("MINI_TAIL", "0")),
    norm_mix=True, aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
    clocks=CLK, precision="bf16",
    attn=os.environ["MINI_ATTN"], qk_norm=(os.environ["MINI_QK"] == "1"),
    mlp=os.environ["MINI_MLP"], band_lr_mult=float(os.environ["MINI_BLR"]),
    data=os.environ["MINI_DATA"], sleep=sl, log_every=20)
dt = time.time() - t0
holds = len(drive.ledger) / 40
lam = min(0.25, 0.25 / max(holds, 1))
out = {"lanes": L, "T": T, "clock_mult": CM, "tok_s": round(40 * L * T / dt), "holds": round(holds, 2),
       "lam": round(lam, 5), "ce": [round(ce0, 3), round(ce1, 3)],
       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
json.dump(out, open(os.environ["MINI_SMOKE_OUT"], "w"))
print("MINISMOKE", json.dumps(out))
PY
  MINIPIDS=""
  # start_mini TAG ATTN QK MLP BLR [T] [CLOCK_MULT] [KEYED] [CREDIT] [CENTER] [TAIL] [CLOCKS] — lam smoke (A60f
  # pairing on THIS config), then the real driver + battery in the
  # background. T/CLOCK_MULT = the conveyor arm (half window, clocks x2);
  # the battery walk keeps the same tokens per lane (hb-chunks scaled).
  start_mini() {
    local tag=$1 attn=$2 qk=$3 mlp=$4 blr=$5 T=${6:-2048} cm=${7:-1} keyed=${8:-logit}
    local bcredit=${9:-0} bcenter=${10:-0} tail=${11:-0} clocks=${12:-}
    local hbc=$((1000 * 2048 / T))
    local outm=/workspace/v10_mini_out$tag; mkdir -p "$outm"
    MINI_ATTN="$attn" MINI_QK="$qk" MINI_MLP="$mlp" MINI_BLR="$blr" \
    MINI_T="$T" MINI_CM="$cm" MINI_KEYED="$keyed" \
    MINI_BCREDIT="$bcredit" MINI_BCENTER="$bcenter" MINI_TAIL="$tail" MINI_CLOCKS="$clocks" \
    MINI_SMOKE_OUT="mini_smoke$tag.json" MINI_DATA="$MINI" MINI_LANES="$ML" \
      python mini_smoke.py > mini_smoke$tag.log 2>&1
    hb "mini smoke$tag $(grep MINISMOKE mini_smoke$tag.log | tail -1 | cut -c1-200)"
    local lam
    lam=$(python -c "import json;print(json.load(open('mini_smoke$tag.json'))['lam'])" 2>/dev/null || echo 0.03)
    python scripts/v10_driver.py \
      --data "$MINI" --eval-data "$DATA/mini_eval_epi" \
      --ckpt "$outm/mini.pt" --lam "$lam" \
      --d 512 --n-layers 8 --T "$T" --clock-mult "$cm" --lr 1e-4 --lr-warmup 1000 \
      --hb-every 3000 --hb-chunks "$hbc" --lesion-every 2 \
      --precision bf16 --attn "$attn" --qk-norm "$qk" \
      --mlp "$mlp" --band-lr-mult "$blr" --keyed "$keyed" \
      --band-credit "$bcredit" --band-center "$bcenter" --tail-tokens "$tail" \
      --clocks "$clocks" \
      --device cuda --hb-out mini_hb$tag.jsonl --trace mini_driver$tag.jsonl \
      --log-every 100 > mini_train$tag.log 2>&1 &
    local pid=$!
    MINIPIDS="$MINIPIDS $pid"
    hb "mini-flash$tag started (pid $pid, lam $lam, lanes $ML, attn $attn qk $qk mlp $mlp blr $blr T $T cm $cm keyed $keyed credit $bcredit center $bcenter tail $tail clocks ${clocks:-default})"
    ( while kill -0 $pid 2>/dev/null; do sleep 600; hb "mini$tag inflight $(tail -1 mini_hb$tag.jsonl 2>/dev/null | head -c 240)"; done
      hb "mini-flash$tag ENDED rc=? rows: $(grep -c . mini_hb$tag.jsonl 2>/dev/null) tail: $(grep -v 'sleep@' mini_train$tag.log | tail -1 | cut -c1-160)" ) &
  }
  if [ -f "$DATA/mini_eval_epi/manifest.json" ] && [ -f "$MINI/manifest.json" ]; then
    start_mini "$MT" "${ATTN:-abs}" "${QK_NORM:-0}" "${MLP:-gelu}" "${BAND_LR_MULT:-1.0}"
    # MINI2="<tag> <attn> <qk> <mlp> <blr>": a second mini on the same
    # GPU at the same time (needs the memory: 2 lanes x 2 on 16 GB, or
    # a 24-48 GB card) — the paired trunk comparison in one night
    if [ -n "${MINI2:-}" ]; then
      # shellcheck disable=SC2086
      start_mini $MINI2
    fi
    if [ -n "${MINI3:-}" ]; then
      # shellcheck disable=SC2086
      start_mini $MINI3
    fi
    if [ -n "${MINI4:-}" ]; then
      # shellcheck disable=SC2086
      start_mini $MINI4
    fi
  fi
fi

# ---- 2. the episodic corpus (fresh tokenizer sample from the spine) ----
# REBUILD_LIVES (2026-08-21, user's "the conveyor never stops"): fewer,
# longer lives give the slow bands more events per life (band 6 writes
# once per 1M tokens: 8 lives -> ~600 per life, 4 -> ~1200, 2 -> ~2400)
# at the same total tokens. When the built corpus has a different life
# count, it is moved aside (never deleted), the shards that carry its
# tokenizer with it, and the ship tar is rebuilt.
if [ -n "${REBUILD_LIVES:-}" ] && [ -f "$DATA/flash_epi/manifest.json" ]; then
  HAVE=$(python -c "import json;print(json.load(open('$DATA/flash_epi/manifest.json')).get('n_lives'))" 2>/dev/null || echo "?")
  if [ "$HAVE" != "$REBUILD_LIVES" ]; then
    for d in flash_epi flash_eval_epi smoke_l${HAVE}_epi; do
      [ -d "$DATA/$d" ] && mv "$DATA/$d" "$DATA/${d}_l$HAVE"
    done
    rm -f /workspace/v10_ship2.tar
    LIVES=$REBUILD_LIVES
    hb "rebuild with $LIVES lives (the $HAVE-life corpus moved aside as *_l$HAVE)"
  fi
fi
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
SMK=smoke_l${LIVES}_epi     # lanes == lives holds for the smoke shard too
if [ ! -f "$DATA/$SMK/manifest.json" ]; then
  # shellcheck disable=SC2086
  python -m iga.lm_data_life prepare --out "$DATA/$SMK" \
    --budget $((2000000 * LIVES)) --lives "$LIVES" --seed 10 \
    $STG $UCARGS --st2-dir "$RAW" --magpie-dir "$RAW" \
    --judge-thresholds "$DATA/judge_freeze.json" \
    --tokenizer "$DATA/flash_epi/tokenizer.json" --episodic >> build.log 2>&1
  hb "smoke shard $( [ -f "$DATA/$SMK/manifest.json" ] && echo ok || echo FAILED)"
fi
for d in flash_eval_epi $SMK; do
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
      v10/flash_epi v10/flash_eval_epi v10/$SMK \
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
if [ -n "${MINIPIDS:-}" ]; then
  hb "waiting for the mini-flash(es):$MINIPIDS"
  for pid in $MINIPIDS; do wait $pid; hb "mini pid $pid exit rc=$?"; done
  hb "mini rows: $(for f in mini_hb*.jsonl; do printf '%s=%s ' "$f" "$(grep -c . "$f")"; done)"
fi
hb "rebuild pod done; self-removing in 60s"
sleep 60
runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
