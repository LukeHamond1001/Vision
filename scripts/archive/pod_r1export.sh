#!/usr/bin/env bash
# A61 ops: export-only pod — push the held-out eval shard
# (mix_r1_eval) from the volume to branch data-r1eval, then die.
# Exists so measurement can go fully local; does nothing else.
set -uo pipefail
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
cd /tmp/boot
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
for i in $(seq 1 18); do [ -f /workspace/rmix/mix_r1_eval/tokens.bin ] && break; sleep 20; done
if [ ! -f /workspace/rmix/mix_r1_eval/tokens.bin ]; then
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
if ! git ls-remote --exit-code origin data-r1eval >/dev/null 2>&1; then
  rm -rf /tmp/r1eval && mkdir -p /tmp/r1eval && cd /tmp/r1eval
  cp /workspace/rmix/mix_r1_eval/events.jsonl /workspace/rmix/mix_r1_eval/tokenizer.json .
  split -b 25m /workspace/rmix/mix_r1_eval/tokens.bin r1eval_tok_
  git init -q . && git checkout -q -b data-r1eval && git add .
  git -c user.email=pod@iga -c user.name=pod commit -qm "held-out eval shard banked (cross-scale instrument)"
  git push -qf "$PUSH" data-r1eval
fi
runpodctl remove pod "$RUNPOD_POD_ID" || true
sleep 120
