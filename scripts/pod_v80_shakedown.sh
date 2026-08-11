#!/usr/bin/env bash
# v8.0 SHAKEDOWN (A46): pipeline certification, not science.
# Downloads small slices of all four sources, builds a ~120M-token
# mixed shard + 32k tokenizer at T=2048 geometry, runs 3000 steps at
# d=512 T=2048 (VRAM + throughput price), evals incl. natural
# identifier probes. Every wrapper protection from the ladder line.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace
rm -rf iga-scale
git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
cd iga-scale
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"
git config user.name "iga-pod"
git checkout -b results-v80s

hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  for f in HEARTBEAT.log train_tail.log eval_results.txt prep.log v80s.pt.trace.jsonl; do
    git add -f "$f" 2>/dev/null || true
  done
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v80s 2>/dev/null || true
}

hb "boot v80-SHAKEDOWN $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ram $(free -g 2>/dev/null | awk '/Mem:/{print $2}')G"
pip install -q numpy tokenizers pyarrow >> prep.log 2>&1
hb "deps installed"

cuda_canary() {
  python - <<'EOF'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.zeros(8, device="cuda") + 1
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
EOF
}
if ! cuda_canary >> prep.log 2>&1; then
  hb "CUDA BROKEN AT BOOT - canary failed, aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "cuda canary passed"

mkdir -p data/mixsrc
DL() { curl -s -L -o "$2" "$1" && hb "download: $2 ($(stat -c%s "$2" 2>/dev/null || echo 0) bytes)"; }
CODE_BASE="https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/main/data"
DL "$CODE_BASE/train-00000-of-00880.parquet" data/mixsrc/code0.parquet
DL "$CODE_BASE/train-00001-of-00880.parquet" data/mixsrc/code1.parquet
WIKI_BASE="https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en"
DL "$WIKI_BASE/train-00000-of-00041.parquet" data/mixsrc/wiki0.parquet
CSN="https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/python"
DL "$CSN/test-00000-of-00001.parquet" data/mixsrc/digest0.parquet
UC="https://huggingface.co/datasets/stingning/ultrachat/resolve/main"
curl -s -L "$UC/train_0.jsonl" | head -n 30000 > data/ultrachat_raw.jsonl
hb "download: ultrachat slice ($(wc -l < data/ultrachat_raw.jsonl) lines)"

python - >> prep.log 2>&1 <<'EOF' &
from iga.lm_data_mix import (prepare_mix, source_code, source_wiki,
                             source_digest, source_chat)
sources = [source_code(["data/mixsrc/code0.parquet",
                        "data/mixsrc/code1.parquet"]),
           source_wiki(["data/mixsrc/wiki0.parquet"]),
           source_digest(["data/mixsrc/digest0.parquet"]),
           source_chat(25000)]
prepare_mix("data/mix_v80s", sources, budget_tokens=120_000_000,
            seed=0, vocab=32768)
sources_ev = [source_code(["data/mixsrc/code1.parquet"]),
              source_wiki(["data/mixsrc/wiki0.parquet"]),
              source_digest(["data/mixsrc/digest0.parquet"]),
              source_chat(2000)]
prepare_mix("data/mix_v80s_eval", sources_ev, budget_tokens=6_000_000,
            seed=99, vocab=32768,
            tokenizer_path="data/mix_v80s/tokenizer.json",
            mine_ids=True)
EOF
PREP_PID=$!
while kill -0 $PREP_PID 2>/dev/null; do
  sleep 300
  kill -0 $PREP_PID 2>/dev/null && \
    hb "prep beat: $(tail -1 prep.log | cut -c1-90)"
done
wait $PREP_PID; PREP_RC=$?
if [ "$PREP_RC" -ne 0 ] || [ ! -s data/mix_v80s/tokens.bin ]; then
  hb "PREP FAILED (rc=$PREP_RC) - see prep.log"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
hb "prep: mix shard built ($(stat -c%s data/mix_v80s/tokens.bin) bytes)"

if ! cuda_canary >> prep.log 2>&1; then
  hb "CUDA DIED PRE-TRAIN - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi

( while true; do sleep 300; \
    tail -30 train.log > train_tail.log; \
    hb "training heartbeat: $(tail -1 train_tail.log)"; done ) &
HBPID=$!
python -m iga.lm_train run --data data/mix_v80s --d 512 --lanes 8 \
  --chunk 2048 --steps 3000 --talk tick --arch hybrid --device cuda \
  --store matrix --xl off --lr 7e-5 --eval-data data/mix_v80s_eval \
  --ckpt v80s.pt > train.log 2>&1
TRAIN_RC=$?
kill $HBPID 2>/dev/null
tail -60 train.log > train_tail.log
hb "training complete (rc=$TRAIN_RC)"

if python -m iga.lm_eval --ckpt v80s.pt --data data/mix_v80s_eval \
    --d 512 --arch hybrid --store matrix --xl off --chunk 2048 --lanes 2 --chunks 40 \
    > eval_results.txt 2>&1; then
  hb "evaluation complete"
else
  hb "EVAL FAILED - traceback in eval_results.txt"
fi

git add -f eval_results.txt train.log prep.log v80s.pt.trace.jsonl \
  data/mix_v80s_eval/events.jsonl 2>/dev/null || true
git commit -qm "shakedown artifacts" 2>/dev/null || true
git push -qf "$PUSH" results-v80s 2>/dev/null || true
hb "phase complete (train rc=$TRAIN_RC) — SHAKEDOWN, no ckpt landing"

runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || true
sleep 120
