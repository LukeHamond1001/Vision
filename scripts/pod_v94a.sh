#!/usr/bin/env bash
# A61 v9.4 endgame autopsy, hardened relaunch: round 1 zombied 42
# min with no heartbeat — suspect: volume workdir git state (74
# ckpt-piece commits) or slow volume attach, unobservable without
# a first beat. v2 ops changes: (1) run ENTIRELY from the /tmp/boot
# clone — the volume is data-only (ckpts + shard), its git state is
# never touched; (2) alive-marker pushed BEFORE touching the volume
# (isolates github-vs-volume on any future silence); (3) volume
# wait loop + timeouts on network git ops.
set -uo pipefail
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
cd /tmp/boot
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
timeout 180 git fetch -q --depth 1 origin results-v94 && \
  git checkout -q -B results-v94 FETCH_HEAD
hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log autopsy_v94*.txt 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94 2>/dev/null || \
    { sleep 20; git push -qf "$PUSH" results-v94 2>/dev/null; } || true
}
hb "autopsy v2 alive (pre-volume) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
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
