#!/usr/bin/env bash
# pod_school.sh — finishing school (49r) for the COMPLETED scan16 body.
# First contact showed a fluent speaker with no question->answer
# coupling (the modal-UC failure). School = ~2.7k steps on
# school_pairs.jsonl (short direct Q->A pairs mined from UC), each
# pair ONE flat day, LR 2e-5. Wraps pod_scan.sh via the 49r override
# envs so build/smoke/train stay single-source.
#
# v2 (post crash-loop): beacon-first — the first launch died before
# any beacon and RunPod restarts made it a silent $0.35 boot loop
# with no logs API. Now: everything logs to the VOLUME
# (school_boot.log), a minimal hb fires before any risky step, the
# possibly-stale trainer checkout gets its lock cleared (the trainer
# died mid-loop and can leave .git/index.lock), and data-school is
# fetched with an explicit refspec.
set -uo pipefail
# volume-mount race: the first launch may have run before /workspace
# attached and boot-looped on missing paths — wait for the volume
for i in $(seq 1 60); do [ -d /workspace/v10 ] && break; sleep 5; done
[ -d /workspace/v10 ] || { sleep 300; exit 1; }
# v3: keep the PREVIOUS attempt's log — attempt 2 died inside
# pod_scan with its beacons lost; the post-mortem lives in this file
[ -f /workspace/school_boot.log ] && cp -f /workspace/school_boot.log /workspace/school_boot_prev.log
exec > /workspace/school_boot.log 2>&1   # eyes first: volume log
echo "SCHOOL boot $(date -u '+%F %T') pod=${RUNPOD_POD_ID:-?}"
DATA=/workspace/v10
SRC=/workspace/v10_scan_out_scan16/scan.pt
OUT=/workspace/v10_scan_out_scan16school
W=/workspace/w-v10prep
R=/root/w-school            # school's OWN repo — never git-fiddle the shared one
mkdir -p "$R" "$DATA" "$OUT" && cd "$R"
git clone -q --depth 1 --single-branch --branch main \
  https://github.com/LukeHamond1001/Vision.git 2>/dev/null || true
cd Vision || { echo "ABORT no clone"; sleep 300; exit 1; }
PUSH="https://x-access-token:${GIT_TOKEN}@github.com/LukeHamond1001/Vision.git"
git config user.email "pod@Vision"; git config user.name "iga-pod"
git fetch -q origin "+refs/heads/data-school:refs/remotes/origin/data-school" || echo "WARN data-school fetch failed"
[ -n "${PIN_SHA:-}" ] && { git fetch -q origin "$PIN_SHA" 2>/dev/null; git reset --hard -q "$PIN_SHA" 2>/dev/null || true; }

# beacon-first: prove we're alive before anything heavy
git checkout -q -B results-school
hb() {
  echo "$(date -u '+%H:%M:%S') [school] $1" >> HEARTBEAT.log
  git add -f HEARTBEAT.log 2>/dev/null
  git commit -qm "hb: [school] $1" 2>/dev/null || true
  git push -qf "$PUSH" results-school 2>/dev/null || \
    { sleep 15; git push -qf "$PUSH" results-school 2>/dev/null; } || true
}
hb "SCHOOL v16 boot sha=$(git rev-parse --short HEAD) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
# post-mortem of the previous attempt, straight into a beacon
[ -f /workspace/school_boot_prev.log ] && \
  hb "prev attempt log tail: $(tail -c 700 /workspace/school_boot_prev.log | tr '\n' ' ' | tr -s ' ')"

# v12: the volume's own state first — the packed school life was
# BUILT by the v7 lap (school_epi_l32p/manifest.json); when it
# exists the shard jsonl is dead weight (pod_scan skips the build),
# so the whole delivery question vanishes. Also beacon df: v11's
# 7MB cp to the volume produced 0 lines — if the volume is refusing
# writes, the step-500 ckpt save is the next victim and we need to
# see it NOW.
hb "df: $(df -h /workspace | tail -1 | tr -s ' ') quota_use: $(du -sBG /workspace 2>/dev/null | cut -f1)"
if [ -f "$DATA/school_epi_l32p/manifest.json" ]; then
  hb "school life already built (v7 lap) - skipping shard delivery"
else
  if [ -s "data/school_pairs.jsonl" ]; then
    cp -f data/school_pairs.jsonl "$DATA/school_pairs.jsonl.tmp"
  else
    hb "shard not in clone checkout; pulling pinned raw"
    curl -sfL --retry 6 --retry-delay 10 --retry-all-errors \
      "https://raw.githubusercontent.com/LukeHamond1001/Vision/${PIN_SHA:-main}/data/school_pairs.jsonl" \
      -o "$DATA/school_pairs.jsonl.tmp" || true
  fi
  LINES=$(wc -l < "$DATA/school_pairs.jsonl.tmp" 2>/dev/null || echo 0)
  if [ "$LINES" -ge 1000 ]; then
    mv -f "$DATA/school_pairs.jsonl.tmp" "$DATA/school_pairs.jsonl"
    hb "shard landed $LINES packed days"
  else
    hb "ABORT shard bad ($LINES lines)"; sleep 300; exit 1
  fi
fi

# v4: the dead trainer's checkout is a booby trap — attempt 2 died on
# a stale REF lock (refs/heads/results-v10.lock) my index.lock-only
# cleanup missed, which also means pod_scan's code-pinning reset may
# have silently failed. Stop nursing it: delete, let pod_scan
# re-clone fresh (the trainer's own certified boot path; data dirs
# are elsewhere on the volume).
if [ -d "$W/Vision" ]; then
  hb "removing stale trainer checkout for fresh re-clone"
  rm -rf "$W/Vision"
fi
# v14: PRE-SEED the trainer checkout with a SHALLOW code-only clone —
# pod_scan's own full clone (multi-GB with data branches) crawled 28
# min on v13, likely GitHub throttling repeated full clones; the
# shallow clone is ~seconds and pod_scan's [ -d Vision ] guard then
# skips cloning entirely
for try in 1 2 3; do
  git clone -q --depth 1 --single-branch --branch main \
    https://github.com/LukeHamond1001/Vision.git "$W/Vision" 2>/dev/null && break
  sleep 15
done
if [ -d "$W/Vision/.git" ]; then
  hb "trainer checkout pre-seeded shallow ($(du -sh "$W/Vision" 2>/dev/null | cut -f1))"
else
  hb "WARN shallow pre-seed failed; pod_scan will full-clone"
fi

# v16: train on CONTAINER DISK — the volume's mfs cluster is degraded
# tonight (clone crawl, cache crawl, and a 45-min wedge on the
# step-500 save are all volume WRITES; reads work). The OUT dir on
# the volume becomes a symlink to local disk, so the certified
# pod_scan/driver save path needs zero changes; the finished body
# ships DIRECTLY from this pod (runpodctl send) — the volume is
# never written again.
LOCAL_OUT=/root/school_out
mkdir -p "$LOCAL_OUT"
if [ -d "$OUT" ] && [ ! -L "$OUT" ]; then
  [ -f "$OUT/scan.pt" ] && mv -f "$OUT/scan.pt" "$LOCAL_OUT/scan.pt" 2>/dev/null
  rm -rf "$OUT"
fi
[ -L "$OUT" ] || ln -s "$LOCAL_OUT" "$OUT"
hb "OUT symlinked to container disk ($(readlink "$OUT"))"

# v6: re-seed if a previous attempt's run advanced the copy past
# step 1 (e.g. the SLEEP=1 attempt) — school must start clean
if [ -f "$OUT/scan.pt" ]; then
  STEP=$(python - <<'PY'
import torch
try:
    print(torch.load("/workspace/v10_scan_out_scan16school/scan.pt",
                     map_location="cpu", weights_only=False).get("step", -1))
except Exception:
    print(-1)
PY
)
  [ "$STEP" = "1" ] || { hb "stale school ckpt at step $STEP - re-seeding"; rm -f "$OUT/scan.pt" "$OUT/scan.pt.best.pt" "$OUT/scan.pt.trace.jsonl"; }
fi

# seed the school checkpoint from the finished body (once)
if [ ! -f "$OUT/scan.pt" ]; then
  [ -f "$SRC" ] || { hb "ABORT no source ckpt"; sleep 300; exit 1; }
  python - <<'PY'
import torch
d = torch.load("/workspace/v10_scan_out_scan16/scan.pt",
               map_location="cpu", weights_only=False)
print("source ckpt step:", d.get("step"), flush=True)
assert d.get("step") == 97500, f"unexpected source step {d.get('step')}"
d["step"] = 1
torch.save(d, "/workspace/v10_scan_out_scan16school/scan.pt")
print("school ckpt seeded (counter reset to 1)", flush=True)
PY
  [ -f "$OUT/scan.pt" ] || { hb "ABORT seed failed: $(tail -2 /workspace/school_boot.log | tr '\n' ' ')"; sleep 300; exit 1; }
fi
hb "ckpt seeded; handing off to pod_scan"

# v15: compile caches back on CONTAINER disk — the v13 volume-cache
# idea was wrong: inductor writes thousands of small files and the
# network mount made compile latency-bound (7% CPU, idle GPU, no
# step line 9+ min in). The real fix for v12's stall was only ever
# the 50GB container disk.
export ITER=scan16school
export D=1024 NL=13 LIVES=32 T=64 ORDER=pfc_first
export LR=${LR:-2e-5} WARMUP=${WARMUP:-100}
export SILENCE=0 CAST=0 GRADES=0 PRESS_TOKENS=0 CH_SRC=uc FLAT=1 HOT_FRAC=0.25
# v6: SLEEP=0 like the original launch — pod_scan defaults SLEEP=1
# (old-style Sleeper nights), which is off-recipe for v16 (REM is
# in-graph) and ran school at 231 tok/s instead of ~1,500
export SLEEP=0
export HB_EVERY=999999 HBC=16000 VALUE_W=0.1 MAX_STEPS=0 SALIENCY=0
export BUDGET_MINI=${BUDGET_MINI:-5500000}
export MINI_OVR=$DATA/school_epi_l32p
export UC_SIMPLE_OVR=$DATA/school_pairs.jsonl UC_REST_OVR=$DATA/school_pairs.jsonl
# the finished body's exact organ config (read from ship ckpt cfg) +
# the four train knobs lm_train pops before the ctor
export SCAN_OPTS='{"n_council":4,"slot_every":1,"write_every":1,"compile_council":true,"compile_read":true,"store_exact":true,"tie_embed":true,"z_w":0.0001,"ponder":3,"ponder_mode":"route","ponder_reenter":"token","ponder_aux":0.5,"route_cap":0.125,"store_wipe":"day","write_surprise":1.0,"press_unwrite":true,"plan_m":4,"plan_cand":4,"rem_k":32,"intrinsic_w":0.5,"dopamine":1.0,"bg_w":0.01,"imag_k":4,"plan_w":0.1,"rem_every":64,"rem_w":0.1,"eot_w":3.0}'
# v3: no exec, KEEP_POD=1 — pod_scan must NOT self-remove; control
# returns here so the exit (normal or abort) gets beaconed, and the
# pod stays up for post-mortem or the ship. I kill it via the API.
export KEEP_POD=1
bash scripts/pod_scan.sh
RC=$?
hb "pod_scan exited rc=$RC log tail: $(tail -c 600 /workspace/school_boot.log | tr '\n' ' ' | tr -s ' ')"

# v16: ship the schooled body DIRECTLY from this pod (the volume's
# cluster is too sick tonight to trust with a 5.7GB copy) — the
# pod_ship direct-send pattern, inline
SCHOOLED="$LOCAL_OUT/scan.pt"
if [ -f "$SCHOOLED" ]; then
  STEP2=$(python - <<'PY'
import torch
try:
    print(torch.load("/root/school_out/scan.pt", map_location="cpu",
                     weights_only=False).get("step", -1))
except Exception:
    print(-1)
PY
)
  hb "schooled ckpt at step $STEP2; sending direct $(du -h "$SCHOOLED" | cut -f1)"
  runpodctl send "$SCHOOLED" > /tmp/send.log 2>&1 &
  CODE=""
  for i in $(seq 1 30); do
    CODE=$(grep -o "runpodctl receive [a-z0-9-]*" /tmp/send.log | head -1)
    [ -n "$CODE" ] && break
    sleep 3
  done
  hb "SCHOOLSHIP: ${CODE:-CODE-NOT-FOUND $(tail -2 /tmp/send.log | tr '\n' ' ')}"
  sleep 3600   # hold the send open an hour for the receiver
else
  hb "no schooled ckpt to ship"
fi
sleep 7200
