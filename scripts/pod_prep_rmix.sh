#!/usr/bin/env bash
# A51 split-prep (user directive: CPU prep + network volume + cheap
# GPUs). Builds the r-mix shards ONCE onto the shared network volume
# (/workspace = network volume iga-rmix); R1/R2 trainers mount the
# same volume and skip prep entirely. Runs fine on a CPU pod.
set -uo pipefail
W=/workspace/w-prep
rm -rf "$W" && mkdir -p "$W" && cd "$W"
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -b results-rprep

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log prep.log 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-rprep 2>/dev/null || true
}
hb "boot rmix-PREP (cpu ok) vol=$(df -BG /workspace | awk 'NR==2{print $2,$4}')"
pip install -q numpy tokenizers pyarrow >> prep.log 2>&1
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cpu >> prep.log 2>&1
hb "deps ready"

mkdir -p data/mixsrc /workspace/rmix
CODE_BASE="https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/main/data"
SKIPPED=0
for i in $(seq 0 49); do
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
  if [ $((i % 10)) -eq 0 ]; then
    hb "extract $i/50 ($(stat -c%s data/code_texts.jsonl 2>/dev/null || echo 0)B, skip $SKIPPED)"
  fi
done
hb "code extraction done ($(stat -c%s data/code_texts.jsonl)B, skipped $SKIPPED/50)"
WIKI_BASE="https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
for i in 0 1; do
  curl -s -L --fail -o "data/mixsrc/wiki$i.parquet" \
    "$WIKI_BASE/train-0000$i-of-00041.parquet"
done
CSN="https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/python"
curl -s -L --fail -o data/mixsrc/digest_train.parquet "$CSN/train-00000-of-00001.parquet"
UC="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
curl -s -L "$UC/train_0.jsonl" > data/ultrachat_raw.jsonl
for i in 0 1; do
  N=$(printf "%05d" $((50 + i)))
  curl -s -L --fail -o "data/mixsrc/evcode$i.parquet" "$CODE_BASE/train-$N-of-00880.parquet"
done
python - <<'EXEOF' >> prep.log 2>&1
import json
import pyarrow.parquet as pq
for i in (0, 1):
    pf = pq.ParquetFile(f"data/mixsrc/evcode{i}.parquet")
    with open("data/code_eval.jsonl", "a") as out:
        for batch in pf.iter_batches(batch_size=1024,
                                     columns=["code", "language"]):
            d = batch.to_pydict()
            for code, lang in zip(d["code"], d["language"]):
                if lang == "Python" and code and len(code) >= 200:
                    out.write(json.dumps({"text": code[:200_000]}) + "\n")
EXEOF
rm -f data/mixsrc/evcode0.parquet data/mixsrc/evcode1.parquet
curl -s -L --fail -o data/mixsrc/wiki_ev.parquet "$WIKI_BASE/train-00040-of-00041.parquet"
curl -s -L --fail -o data/mixsrc/digest_ev.parquet "$CSN/test-00000-of-00001.parquet"
curl -s -L "$UC/train_9.jsonl" | head -n 8000 > data/ultrachat_heldout.jsonl
hb "sources ready"

python - >> prep.log 2>&1 <<'PREPEOF' &
from iga.lm_data_mix import (prepare_mix, source_code_jsonl,
                             source_wiki, source_digest, source_chat)
sources = [source_code_jsonl("data/code_texts.jsonl"),
           source_wiki([f"data/mixsrc/wiki{i}.parquet" for i in range(2)]),
           source_digest(["data/mixsrc/digest_train.parquet"]),
           source_chat(120000)]
prepare_mix("/workspace/rmix/mix_r1", sources,
            budget_tokens=1_200_000_000, seed=0, vocab=32768,
            spill=8_000_000)
import os
os.environ["ULTRACHAT_JSONL"] = "data/ultrachat_heldout.jsonl"
sources_ev = [source_code_jsonl("data/code_eval.jsonl"),
              source_wiki(["data/mixsrc/wiki_ev.parquet"]),
              source_digest(["data/mixsrc/digest_ev.parquet"]),
              source_chat(7000)]
prepare_mix("/workspace/rmix/mix_r1_eval", sources_ev,
            budget_tokens=7_000_000, seed=99, vocab=32768,
            tokenizer_path="/workspace/rmix/mix_r1/tokenizer.json",
            mine_ids=True)
PREPEOF
PREP_PID=$!
hb "prep launched (pid $PREP_PID)"
E=0
while kill -0 $PREP_PID 2>/dev/null; do
  if [ $E -lt 16 ]; then sleep 45; else sleep 300; fi
  E=$((E+1))
  kill -0 $PREP_PID 2>/dev/null && hb "prep beat: $(tail -1 prep.log | cut -c1-90)"
done
wait $PREP_PID; PREP_RC=$?
if [ "$PREP_RC" -ne 0 ] || [ ! -s /workspace/rmix/mix_r1/tokens.bin ]; then
  hb "PREP FAILED (rc=$PREP_RC)"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
touch /workspace/rmix/DONE
hb "RMIX READY ($(stat -c%s /workspace/rmix/mix_r1/tokens.bin)B train, $(stat -c%s /workspace/rmix/mix_r1_eval/tokens.bin)B eval)"
runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
