#!/usr/bin/env bash
# A54 v9 prep (CPU pod): build the 6B-token shard onto the network
# volume. Fresh sources (code 52-301, wiki 2-11, chat train_1-3 —
# disjoint from mix_r1 train AND from mix_r1_eval, which stays the
# held-out instrument). REUSES mix_r1's tokenizer: same token ids =
# the 928-probe/completion instruments compare across scales.
set -uo pipefail
# A54c (pod 1 post-mortem): the workdir MUST live on the pod's own
# container disk, NOT the network volume — sources (~26GB) + the
# finished shard (12.3GB) + mix_r1 (2.5GB) exceed the 40GB volume
# quota, and a full volume kills the heartbeat path too (the
# silent death). Only the OUTPUT shard belongs on the volume.
W=/root/w-v9prep
rm -rf "$W" /workspace/w-v9prep /workspace/w-r5 /workspace/w-r4b \
  /workspace/rmix/mix_v9 && mkdir -p "$W" && cd "$W"
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -b results-v9prep

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log prep.log 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v9prep 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-v9prep 2>/dev/null; } || true
}
hb "boot v9-PREP vol=$(df -BG /workspace | awk 'NR==2{print $2,$4}')"
pip install -q numpy tokenizers pyarrow >> prep.log 2>&1
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cpu >> prep.log 2>&1
hb "deps ready"

mkdir -p data/mixsrc /workspace/rmix
CODE_BASE="https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/main/data"
SKIPPED=0
for i in $(seq 52 301); do
  SZ=$(stat -c%s data/code_texts.jsonl 2>/dev/null || echo 0)
  if [ "$SZ" -gt 20000000000 ]; then hb "code cap hit at shard $i (${SZ}B)"; break; fi
  N=$(printf "%05d" $i)
  curl -s -L --fail -o data/mixsrc/cur.parquet \
    "$CODE_BASE/train-$N-of-00880.parquet" || { SKIPPED=$((SKIPPED+1)); continue; }
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
  if [ $((i % 25)) -eq 0 ]; then
    hb "extract $i/301 ($(stat -c%s data/code_texts.jsonl 2>/dev/null || echo 0)B, skip $SKIPPED)"
  fi
done
hb "code extraction done ($(stat -c%s data/code_texts.jsonl)B, skipped $SKIPPED)"
WIKI_BASE="https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
for i in $(seq 2 11); do
  N=$(printf "%05d" $i)
  curl -s -L --fail -o "data/mixsrc/wiki$i.parquet" \
    "$WIKI_BASE/train-$N-of-00041.parquet" || hb "wiki $i FAILED"
done
CSN="https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/python"
curl -s -L --fail -o data/mixsrc/digest_train.parquet "$CSN/train-00000-of-00001.parquet"
UC="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
: > data/ultrachat_raw.jsonl
for i in 1 2 3; do
  curl -s -L "$UC/train_$i.jsonl" >> data/ultrachat_raw.jsonl
done
hb "sources ready ($(stat -c%s data/ultrachat_raw.jsonl)B chat)"

python - >> prep.log 2>&1 <<'PREPEOF' &
from iga.lm_data_mix import (prepare_mix, source_code_jsonl,
                             source_wiki, source_digest, source_chat)
sources = [source_code_jsonl("data/code_texts.jsonl"),
           source_wiki([f"data/mixsrc/wiki{i}.parquet"
                        for i in range(2, 12)]),
           source_digest(["data/mixsrc/digest_train.parquet"]),
           source_chat(600000)]
prepare_mix("/workspace/rmix/mix_v9", sources,
            budget_tokens=6_000_000_000, seed=0, vocab=32768,
            tokenizer_path="/workspace/rmix/mix_r1/tokenizer.json",
            spill=40_000_000)
PREPEOF
PREP_PID=$!
hb "prep launched (pid $PREP_PID)"
E=0
while kill -0 $PREP_PID 2>/dev/null; do
  if [ $E -lt 16 ]; then sleep 45; else sleep 600; fi
  E=$((E+1))
  kill -0 $PREP_PID 2>/dev/null && hb "prep beat: $(tail -1 prep.log | cut -c1-90)"
done
wait $PREP_PID; PREP_RC=$?
if [ "$PREP_RC" -ne 0 ] || [ ! -s /workspace/rmix/mix_v9/tokens.bin ]; then
  hb "PREP FAILED (rc=$PREP_RC)"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
touch /workspace/rmix/DONE_V9
hb "V9 SHARD READY ($(stat -c%s /workspace/rmix/mix_v9/tokens.bin)B)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
