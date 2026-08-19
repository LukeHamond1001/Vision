#!/usr/bin/env bash
# A61 v9.4 endgame autopsy, v4: visibility build. v3 ran healthy
# but mute-by-design — heartbeats only at per-checkpoint completion,
# and a CPU-only battery (autopsy_v9.py has no device code) takes
# 150+ min/ckpt. v4 adds a PROGRESS BEAT: every 5 min the partial
# output file (modes print flush=True as they finish) is pushed, so
# progress is line-visible while the battery runs. Own lightweight
# branch (results-v94-autopsy2); volume data-only; pathspec-split
# adds (A61 ops law).
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
for i in 1 2 3 4 5 6 7 8 9; do [ -d "$WD" ] && break; sleep 20; done
if [ ! -d "$WD" ]; then
  hb "VOLUME/WORKDIR MISSING after 3min - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
pip install -q numpy tokenizers >> /dev/null 2>&1
hb "volume ok, battery starting"
( while true; do sleep 300; hb "battery progress"; done ) &
BEATPID=$!
for CK in v94.pt.best.pt v94.pt; do
  if [ ! -f "$WD/$CK" ]; then hb "MISSING $WD/$CK"; continue; fi
  python scripts/autopsy_v9.py --ckpt "$WD/$CK" \
    --shard /workspace/rmix/mix_v9 \
    --d 512 --max-T 2048 --serve-T 2048 --norm-mix --aux-trunk 0.2 \
    --label "$CK" > "autopsy_v94_${CK%.pt}.txt" 2>&1
  hb "autopsy complete: $CK (rc=$?)"
done
kill $BEATPID 2>/dev/null
hb "autopsy phase done"
runpodctl remove pod "$RUNPOD_POD_ID" || true
sleep 120
