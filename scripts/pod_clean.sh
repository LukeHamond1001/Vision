#!/usr/bin/env bash
# pod_clean.sh — free volume space: COPIES AND REGENERABLES ONLY.
# Never touches originals (scan.pt/scan.pt.best.pt of any iteration,
# raw data, shards). Beacons before/after usage, self-removes.
set -uo pipefail
W=/root/w-aux
mkdir -p "$W" && cd "$W"
[ -d Vision ] || git clone -q --depth 1 --single-branch --branch main \
  https://github.com/LukeHamond1001/Vision.git
cd Vision
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git checkout -q -B results-v10 || true
git pull -q origin results-v10 || true
REPO=$PWD
hb() {
  echo "$(date -u '+%H:%M:%S') [clean] $1" >> "$REPO/HEARTBEAT.log"
  git -C "$REPO" add -f HEARTBEAT.log 2>/dev/null
  git -C "$REPO" commit -qm "hb: [clean] $1" 2>/dev/null || true
  git -C "$REPO" push -qf "$PUSH" results-v10 2>/dev/null || true
}
hb "before: $(du -sBG /workspace 2>/dev/null | cut -f1)/150G"
hb "top: $(du -sBG /workspace/* 2>/dev/null | sort -rh | head -8 | tr '\n\t' ' _' | cut -c1-400)"
# safe list: transport copies, partial tars, regenerable serve strips
rm -f /workspace/ship_scan15.pt /workspace/ship_scan15.tar \
      /workspace/ship_tok.json /workspace/v10_ship2.tar \
      /workspace/v10_scan_out_*/scan15_serve.pt \
      /workspace/v10_scan_out_*/scan14_serve.pt \
      /workspace/v10_scan_out_*/*_serve.pt 2>/dev/null
hb "after: $(du -sBG /workspace 2>/dev/null | cut -f1)/150G top: $(du -sBG /workspace/* 2>/dev/null | sort -rh | head -8 | tr '\n\t' ' _' | cut -c1-400)"
sleep 10
[ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
