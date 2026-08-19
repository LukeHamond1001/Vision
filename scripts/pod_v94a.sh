#!/usr/bin/env bash
# A61 v9.4 endgame autopsy, v5: THE RIGHT SHARD. v1-v4 passed
# --shard mix_v9 (the TRAINING shard, 0 nat probes by design) — the
# TM-v9-clean subset, completion channel, and every pinned baseline
# row (r5 CE 2.6570, 461/202/836, hash fc639269079e) live on
# mix_r1_eval, the HELD-OUT shard (mine_ids=True, EVAL only). The
# "empty subset / damaged events" scare was this wrong flag; the
# volume was never corrupted. v4's keepers: 5-min progress beats,
# lightweight results-v94-autopsy2 branch, pathspec-split adds,
# volume data-only.
set -uo pipefail
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
cd /tmp/boot
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -q -b results-v94-autopsy2
hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log 2>/dev/null || true
  git add -f autopsy_v94_*.txt 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94-autopsy2 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v94-autopsy2 2>/dev/null; } || true
}
hb "autopsy v4 alive (pre-volume) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
WD=/workspace/w-v94/iga-scale
for i in $(seq 1 18); do [ -d "$WD" ] && break; sleep 20; done
if [ ! -d "$WD" ]; then
  hb "VOLUME/WORKDIR MISSING after 6min - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
pip install -q numpy tokenizers >> /dev/null 2>&1
hb "volume ok, battery starting"
( while true; do sleep 300; hb "battery progress"; done ) &
BEATPID=$!
# v6: GATE ROW FIRST — completion-only on best.pt (~20 min), then
# the 488k final full battery (corroboration), then bank the
# held-out eval shard to a branch (it must never live only on one
# volume — the local copy was eaten by the macOS tmp reaper).
python scripts/autopsy_v9.py --ckpt "$WD/v94.pt.best.pt" \
  --shard /workspace/rmix/mix_r1_eval \
  --d 512 --max-T 2048 --serve-T 2048 --norm-mix --aux-trunk 0.2 \
  --modes organs,completion \
  --label v94.pt.best.pt > "autopsy_v94_best_completion.txt" 2>&1
hb "GATE ROW complete: best.pt completion (rc=$?)"
python scripts/autopsy_v9.py --ckpt "$WD/v94.pt" \
  --shard /workspace/rmix/mix_r1_eval \
  --d 512 --max-T 2048 --serve-T 2048 --norm-mix --aux-trunk 0.2 \
  --label v94.pt > "autopsy_v94_final_full.txt" 2>&1
hb "autopsy complete: v94.pt final full (rc=$?)"
if ! git ls-remote --exit-code origin data-r1eval >/dev/null 2>&1; then
  ( rm -rf /tmp/r1eval && mkdir -p /tmp/r1eval && cd /tmp/r1eval && \
    cp /workspace/rmix/mix_r1_eval/events.jsonl /workspace/rmix/mix_r1_eval/tokenizer.json . && \
    split -b 25m /workspace/rmix/mix_r1_eval/tokens.bin r1eval_tok_ && \
    git init -q . && git checkout -q -b data-r1eval && git add . && \
    git -c user.email=pod@iga -c user.name=pod commit -qm "held-out eval shard banked (cross-scale instrument)" && \
    git push -qf "$PUSH" data-r1eval ) >/dev/null 2>&1 \
    && hb "eval shard banked to data-r1eval"
fi
kill $BEATPID 2>/dev/null
hb "autopsy phase done"
runpodctl remove pod "$RUNPOD_POD_ID" || true
sleep 120
