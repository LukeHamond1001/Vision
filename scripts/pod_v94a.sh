#!/usr/bin/env bash
# A61 v9.4 endgame autopsy, v3. Round-2 post-mortem (GraphQL
# telemetry: CPU 79%, GPU 0%, zero heartbeats): the v2 marker
# fetched results-v94 — now a 1.85GB piece-laden branch — and the
# script's own 180s timeout killed the fetch, so the local branch
# never existed and every push failed silently while the battery
# ran toward self-terminate-and-lose-outputs. v3 OPS LAW: autopsy
# outputs land on their OWN LIGHTWEIGHT BRANCH (results-v94-autopsy,
# branched off the boot clone's main — no fetch of piece branches,
# ever). Volume remains data-only (ckpts + shard reads).
set -uo pipefail
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
cd /tmp/boot
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -q -b results-v94-autopsy
hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  # A61 ops law: add each pathspec separately — git add rejects ALL
  # paths when any single pathspec (e.g. an unmatched glob) misses,
  # and the 2>/dev/null made that failure invisible (round 3 was
  # mute-but-healthy for exactly this; diagnosed from console logs)
  git add -f HEARTBEAT.log 2>/dev/null || true
  git add -f autopsy_v94_*.txt 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94-autopsy 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v94-autopsy 2>/dev/null; } || true
}
hb "autopsy v3 alive (pre-volume) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
WD=/workspace/w-v94/iga-scale
for i in 1 2 3 4 5 6 7 8 9; do [ -d "$WD" ] && break; sleep 20; done
if [ ! -d "$WD" ]; then
  hb "VOLUME/WORKDIR MISSING after 3min - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true
  sleep 120; exit 1
fi
pip install -q numpy tokenizers >> /dev/null 2>&1
hb "volume ok, battery starting"
for CK in v94.pt.best.pt v94.pt; do
  if [ ! -f "$WD/$CK" ]; then hb "MISSING $WD/$CK"; continue; fi
  python scripts/autopsy_v9.py --ckpt "$WD/$CK" \
    --shard /workspace/rmix/mix_v9 \
    --d 512 --max-T 2048 --serve-T 2048 --norm-mix --aux-trunk 0.2 \
    --label "$CK" > "autopsy_v94_${CK%.pt}.txt" 2>&1
  hb "autopsy complete: $CK (rc=$?)"
done
hb "autopsy phase done"
runpodctl remove pod "$RUNPOD_POD_ID" || true
sleep 120
