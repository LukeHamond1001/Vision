#!/usr/bin/env bash
# v8.0 (A47) — THE BUILD RUN. The certified core (d=512, scalar
# gates, write credit, lr 7e-5 per the width-LR law) on the real
# long-structure mix: 6B tokens of Python code + wikipedia + digest
# pairs + chat at T=2048. ~366k steps, ~21h train + ~7h prep on a
# clean 4090. Banking + recent-window telemetry standard; capacity
# canary; stream-extract-delete keeps 230 code shards off disk.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace
rm -rf iga-scale
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"
git config user.name "iga-pod"
git checkout -b results-v80

mkdir -p /workspace/snap
( cd /workspace/snap && git init -q && git checkout -qb snap \
  && git config user.email pod@iga && git config user.name iga-pod )

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt prep.log v80.pt.trace.jsonl; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v80 2>/dev/null || true
}

hb "boot v80-THE-BUILD $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ram $(free -g 2>/dev/null | awk '/Mem:/{print $2}')G disk $(df -BG /workspace | awk 'NR==2{print $4}')"
pip install -q numpy tokenizers pyarrow >> prep.log 2>&1
hb "deps installed"

cuda_canary() {
  python - <<'CANEOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda") + 1
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"vram free {free/2**30:.1f} / total {total/2**30:.1f} GiB")
assert free > 18 * 2**30, f"GPU DIRTY/UNDERSIZED: {free/2**30:.1f} GiB free"
big = torch.empty(int(14e9) // 4, dtype=torch.float32, device="cuda")
del big
torch.cuda.empty_cache()
print("capacity canary passed (14GB claim)")
CANEOF
}
if ! cuda_canary >> prep.log 2>&1; then
  hb "CUDA BROKEN OR DIRTY AT BOOT - canary failed, aborting (see prep.log)"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "capacity canary passed"

mkdir -p data/mixsrc
CODE_BASE="https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/main/data"
# stream-extract-delete: 230 train shards -> compact python jsonl
SKIPPED=0
for i in $(seq 0 229); do
  N=$(printf "%05d" $i)
  OK=0
  for try in 1 2; do
    if curl -s -L --fail -o data/mixsrc/cur.parquet \
        "$CODE_BASE/train-$N-of-00880.parquet"; then OK=1; break; fi
    sleep 10
  done
  if [ "$OK" != "1" ]; then SKIPPED=$((SKIPPED+1)); continue; fi
  # A47 ops: batch 1024 (64 was ~35s/shard of python overhead vs
  # ~5s of network — the extractor, not the download, was the drag)
  python - <<'EXEOF' >> prep.log 2>&1
import json
import pyarrow.parquet as pq
pf = pq.ParquetFile("data/mixsrc/cur.parquet")
with open("data/code_texts.jsonl", "a") as out:
    for batch in pf.iter_batches(batch_size=1024,
                                 columns=["code", "language"]):
        d = batch.to_pydict()
        for code, lang in zip(d["code"], d["language"]):
            if lang == "Python" and code and len(code) >= 200:
                out.write(json.dumps({"text": code[:200_000]}) + "\n")
EXEOF
  rm -f data/mixsrc/cur.parquet
  if [ $((i % 20)) -eq 0 ]; then
    hb "code extract: shard $i/230, jsonl $(stat -c%s data/code_texts.jsonl 2>/dev/null || echo 0) bytes, skipped $SKIPPED"
  fi
done
hb "code extraction done ($(stat -c%s data/code_texts.jsonl) bytes, skipped $SKIPPED/230)"

WIKI_BASE="https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
for i in 0 1 2 3 4 5 6 7; do
  curl -s -L --fail -o "data/mixsrc/wiki$i.parquet" \
    "$WIKI_BASE/train-0000$i-of-00041.parquet" && \
    hb "download: wiki$i"
done
CSN="https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/python"
curl -s -L --fail -o data/mixsrc/digest_train.parquet \
  "$CSN/train-00000-of-00001.parquet" && hb "download: csn train"
UC="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
for i in 0 1 2; do
  curl -s -L "$UC/train_$i.jsonl" >> data/ultrachat_raw.jsonl
done
hb "download: ultrachat 3 parts ($(wc -l < data/ultrachat_raw.jsonl) lines)"
# held-out eval sources (never in training)
for i in 0 1; do
  N=$(printf "%05d" $((230 + i)))
  curl -s -L --fail -o "data/mixsrc/evcode$i.parquet" \
    "$CODE_BASE/train-$N-of-00880.parquet"
done
python - <<'EXEOF' >> prep.log 2>&1
import json
import pyarrow.parquet as pq
for i in (0, 1):
    pf = pq.ParquetFile(f"data/mixsrc/evcode{i}.parquet")
    with open("data/code_eval.jsonl", "a") as out:
        for batch in pf.iter_batches(batch_size=64,
                                     columns=["code", "language"]):
            d = batch.to_pydict()
            for code, lang in zip(d["code"], d["language"]):
                if lang == "Python" and code and len(code) >= 200:
                    out.write(json.dumps({"text": code[:200_000]}) + "\n")
EXEOF
rm -f data/mixsrc/evcode0.parquet data/mixsrc/evcode1.parquet
curl -s -L --fail -o data/mixsrc/wiki_ev.parquet \
  "$WIKI_BASE/train-00040-of-00041.parquet"
curl -s -L --fail -o data/mixsrc/digest_ev.parquet \
  "$CSN/test-00000-of-00001.parquet"
curl -s -L "$UC/train_9.jsonl" | head -n 8000 > data/ultrachat_heldout.jsonl
hb "eval sources downloaded (held-out shards)"

( echo "=== container limits ==="; free -g; ulimit -a ) >> prep.log 2>&1
python - >> prep.log 2>&1 <<'PREPEOF' &
from iga.lm_data_mix import (prepare_mix, source_code_jsonl,
                             source_wiki, source_digest, source_chat)
sources = [source_code_jsonl("data/code_texts.jsonl"),
           source_wiki([f"data/mixsrc/wiki{i}.parquet"
                        for i in range(8)]),
           source_digest(["data/mixsrc/digest_train.parquet"]),
           source_chat(430000)]
prepare_mix("data/mix_v80", sources, budget_tokens=6_000_000_000,
            seed=0, vocab=32768, spill=8_000_000)
import os
os.environ["ULTRACHAT_JSONL"] = "data/ultrachat_heldout.jsonl"
sources_ev = [source_code_jsonl("data/code_eval.jsonl"),
              source_wiki(["data/mixsrc/wiki_ev.parquet"]),
              source_digest(["data/mixsrc/digest_ev.parquet"]),
              source_chat(7000)]
prepare_mix("data/mix_v80_eval", sources_ev, budget_tokens=9_000_000,
            seed=99, vocab=32768,
            tokenizer_path="data/mix_v80/tokenizer.json",
            mine_ids=True)
PREPEOF
PREP_PID=$!
hb "prep launched (pid $PREP_PID)"
E=0
while kill -0 $PREP_PID 2>/dev/null; do
  if [ $E -lt 16 ]; then sleep 45; else sleep 300; fi
  E=$((E+1))
  kill -0 $PREP_PID 2>/dev/null && \
    hb "prep beat: $(tail -1 prep.log | cut -c1-90)"
done
wait $PREP_PID; PREP_RC=$?
if [ "$PREP_RC" -ne 0 ] || [ ! -s data/mix_v80/tokens.bin ]; then
  hb "PREP FAILED (rc=$PREP_RC) - see prep.log"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
rm -f data/code_texts.jsonl data/ultrachat_raw.jsonl data/mixsrc/wiki*.parquet
hb "prep: mix shard built ($(stat -c%s data/mix_v80/tokens.bin) bytes, raw deleted)"

if ! cuda_canary >> prep.log 2>&1; then
  hb "CUDA DIED OR DIRTIED PRE-TRAIN - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "capacity canary passed pre-train"

( C=0; while true; do sleep 900; C=$((C+1)); \
    tail -40 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; \
    if [ $((C % 8)) -eq 0 ] && [ -f v80.pt ]; then \
      cp v80.pt /workspace/snap/s.pt && \
      ( cd /workspace/snap && rm -f snap_part_* && \
        split -b 25m s.pt snap_part_ && rm -f s.pt && \
        git add -A && git commit -qm "rolling snapshot" && \
        git push -qf "$PUSH" snap:results-v80-ckpt ) && \
      hb "rolling ckpt snapshot pushed"; \
    fi; done ) &
HBPID=$!
python -m iga.lm_train run --data data/mix_v80 --d 512 --lanes 8 \
  --chunk 2048 --steps 366000 --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lr 7e-5 --eval-data data/mix_v80_eval \
  --ckpt v80.pt > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

for CK in v80.pt v80.pt.best.pt; do
  if [ -f "$CK" ]; then
    echo "===== EVAL $CK =====" >> eval_results.txt
    python -m iga.lm_eval --ckpt "$CK" --data data/mix_v80_eval \
      --d 512 --arch hybrid --store matrix --xl off --chunk 2048 --lanes 2 --chunks 120 \
      >> eval_results.txt 2>&1 && hb "evaluation complete: $CK"
  fi
done

BASE_SHA=$(git rev-parse HEAD)
if [ ! -f v80.pt ]; then
  hb "NO CHECKPOINT EXISTS - nothing to verify (train rc=$TRAIN_RC)"
  CKPT_OK=0
else
split -b 25m v80.pt v80_part_
if [ -f v80.pt.best.pt ]; then
  split -b 25m v80.pt.best.pt v80best_part_
fi
CKPT_OK=1
for f in $(ls v80_part_* v80best_part_* 2>/dev/null); do
  git add -f "$f" && git commit -qm "ckpt piece: $f"
  PUSHED=0
  for i in 1 2 3 4; do
    if git push -f "$PUSH" results-v80 && \
       [ "$(git ls-remote "$PUSH" results-v80 | cut -f1)" = "$(git rev-parse HEAD)" ]; then
      PUSHED=1; break
    fi
    sleep 20
  done
  if [ "$PUSHED" != "1" ]; then CKPT_OK=0; break; fi
  hb "ckpt piece landed: $f"
done
fi
git add -f data/mix_v80_eval/tokens.bin data/mix_v80_eval/events.jsonl \
  data/mix_v80_eval/tokenizer.json train.log prep.log \
  v80.pt.trace.jsonl 2>/dev/null || true
git commit -qm "eval shard + logs" 2>/dev/null || true
git push -qf "$PUSH" results-v80 2>/dev/null || true
if [ "$CKPT_OK" = "1" ]; then
  hb "checkpoint FULLY VERIFIED on remote"
elif [ -f v80.pt ]; then
  git reset --mixed "$BASE_SHA"
  hb "ckpt git pushes incomplete"
fi
hb "phase complete (train rc=$TRAIN_RC, ckpt_ok=$CKPT_OK)"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
