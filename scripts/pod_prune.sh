#!/usr/bin/env bash
# A61 ops: volume prune (user-approved 2026-08-19). Deletes ONLY the
# four dead v-campaign workdirs — every checkpoint they hold is
# branch-banked (results-v9-best, results-v9-ckpt, results-v91-ckpt,
# results-v92-ckpt, results-v93-ckpt + results-v93 full pieces).
# KEEPS: /workspace/rmix (all shards) and w-v94 (freshest, also
# banked; conservative). Reports df before/after to the branch.
set -uo pipefail
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/iga-scale.git"
cd /tmp/boot
git config user.email "pod@iga-scale"; git config user.name "iga-pod"
git checkout -q -b results-v94-autopsy2 2>/dev/null || true
git fetch -q --depth 1 origin results-v94-autopsy2 2>/dev/null && git reset -q --hard FETCH_HEAD 2>/dev/null || true
hb() {
  echo "$(date -u '+%H:%M:%S') $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log 2>/dev/null || true
  git commit -qm "hb: $1" 2>/dev/null || true
  git push -qf "$PUSH" results-v94-autopsy2 2>/dev/null || true
}
for i in $(seq 1 18); do [ -d /workspace/rmix ] && break; sleep 20; done
if [ ! -d /workspace/rmix ]; then
  hb "PRUNE: volume never attached - aborting"
  runpodctl remove pod "$RUNPOD_POD_ID" || true; sleep 120; exit 1
fi
hb "PRUNE before: $(df -BG /workspace | awk 'NR==2{print $3" used / "$2}') | $(ls /workspace | tr '\n' ' ')"
for W in w-v9 w-v91 w-v92 w-v93 w-v9prep w-r5 w-r4b; do
  [ -d "/workspace/$W" ] && rm -rf "/workspace/$W" && hb "PRUNE: deleted $W"
done
hb "PRUNE after: $(df -BG /workspace | awk 'NR==2{print $3" used / "$2}') | kept: $(ls /workspace | tr '\n' ' ')"
runpodctl remove pod "$RUNPOD_POD_ID" || true
sleep 120
