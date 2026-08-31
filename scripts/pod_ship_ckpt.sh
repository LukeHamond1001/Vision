#!/usr/bin/env bash
# pod_ship_ckpt.sh — strip the live scan checkpoint to model-only and
# ship it (+ tokenizer) via runpodctl send for local inference.
# env: ITER (scan15) | PIN_SHA | KEEP_POD
set -uo pipefail
W=/root/w-aux            # container disk — NEVER the shared volume: a git reset in a shared checkout nukes the trainer's working tree (2026-08-24 lesson)
DATA=/workspace/v10
mkdir -p "$W" && cd "$W"
# SHALLOW single-branch clone: the full repo drags multi-GB data
# branches — the 00:xx mules died silently inside that clone
[ -d Vision ] || git clone -q --depth 1 --single-branch --branch main \
  https://github.com/LukeHamond1001/Vision.git
cd Vision
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git checkout -q -B results-ship
ITER=${ITER:-scan15}
REPO=$W/Vision
hb() {  # works from ANY cwd (2026-08-26: cd /workspace made beacons vanish — the send code was announced into the void)
  echo "$(date -u '+%H:%M:%S') [$ITER-ship] $1" >> "$REPO/HEARTBEAT.log"
  git -C "$REPO" add -f HEARTBEAT.log 2>/dev/null
  git -C "$REPO" commit -qm "hb: [$ITER-ship] $1" 2>/dev/null || true
  git -C "$REPO" push -qf "$PUSH" results-ship 2>/dev/null || { sleep 20; git -C "$REPO" push -qf "$PUSH" results-ship 2>/dev/null; } || true
}
hb "boot SHIP $(date -u '+%H:%M') $(git rev-parse --short HEAD)"
CKPT=/workspace/v10_scan_out_$ITER/scan.pt
[ -f "$CKPT" ] || { hb "ABORT no ckpt"; sleep 20; runpodctl remove pod "$RUNPOD_POD_ID"; exit 1; }
# NO on-pod strip (v5 died in torch.load — container RAM): ship raw.
# NO cp/tar either (2026-08-26: the copy+tar wrote 10.6GB to the volume and
# the pod died silently mid-copy — suspect ENOSPC): send the ckpt file
# directly, ship the small tokenizer via git.
hb "df: $(df -h /workspace | tail -1 | tr -s ' ')"
cp "$DATA/mini_epi/tokenizer.json" "$W/Vision/ship_tok.json"
cd "$W/Vision" && git add -f ship_tok.json && git commit -qm "ship_tok.json for local infer" && git push -qf "$PUSH" results-ship
cd /workspace
hb "sending ckpt direct $(du -h $CKPT | cut -f1)"
runpodctl send "$CKPT" > /tmp/send.log 2>&1 &
CODE=""
for i in $(seq 1 30); do
  CODE=$(grep -o "runpodctl receive [a-z0-9-]*" /tmp/send.log | head -1)
  [ -n "$CODE" ] && break
  sleep 3
done
hb "SHIP: ${CODE:-CODE-NOT-FOUND $(tail -2 /tmp/send.log | tr '\n' ' ')}"
sleep 2700   # hold the send open 45 min for the receiver
[ "${KEEP_POD:-0}" = "1" ] || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
