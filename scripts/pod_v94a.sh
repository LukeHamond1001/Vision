#!/usr/bin/env bash
# A61 v9.4 endgame autopsy: the blind battery (organs, mean CE
# full-vs-lesioned, table, TM-clean, completion, sign test) on the
# landed final (488k) and best (266k) checkpoints, on the real v9
# shard. Runs in the surviving w-v94 workdir on the volume; code
# refreshed from main WITHOUT moving the workdir's branch state.
set -uo pipefail
W=/workspace/w-v94/iga-scale
cd "$W" || { echo "workdir missing"; exit 1; }
git fetch -q origin main
git checkout -q origin/main -- scripts/autopsy_v9.py iga/
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log autopsy_v94*.txt 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v94 2>/dev/null; } || true
}
pip install -q numpy tokenizers >> /dev/null 2>&1
hb "autopsy boot $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
for CK in v94.pt.best.pt v94.pt; do
  [ -f "$CK" ] || { hb "MISSING $CK"; continue; }
  python scripts/autopsy_v9.py --ckpt "$CK" --shard /workspace/rmix/mix_v9 \
    --d 512 --max-T 2048 --serve-T 2048 --norm-mix --aux-trunk 0.2 \
    --label "$CK" > "autopsy_v94_${CK%.pt}.txt" 2>&1
  hb "autopsy complete: $CK (rc=$?)"
done
hb "autopsy phase done"
runpodctl remove pod "$RUNPOD_POD_ID" || true
sleep 120
