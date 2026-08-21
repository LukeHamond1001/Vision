#!/usr/bin/env bash
# launch_pod.sh — boot a RunPod pod on the PREP network volume that runs a
# SHA-pinned pod script by dockerEntrypoint (the boot rule). Secrets are
# read at run time only: ~/.runpod_key and `gh auth token`; nothing is
# stored in the repo and the token is never printed.
#
#   bash scripts/launch_pod.sh pod_scan.sh "ITER=scan1 ORDER=pfc_first" \
#        [SHA] [GPU1,GPU2,...]
#
# SHA defaults to origin/main's head (must be pushed — the raw URL is
# fetched from GitHub). GPUs are tried in order; first success wins.
# VOLUME (default 2o9gtwzkhd, EU-RO-1 prep) and DISK (40 GB) are env.
set -euo pipefail
SCRIPT=${1:?pod script name, e.g. pod_scan.sh}
ENVSTR=${2:-}
SHA=${3:-$(git ls-remote -q https://github.com/LukeHamond1001/Vision.git refs/heads/main | cut -f1)}
GPUS=${4:-"NVIDIA GeForce RTX 4090,NVIDIA RTX 2000 Ada Generation"}
VOLUME=${VOLUME:-2o9gtwzkhd}
DISK=${DISK:-40}
NAME=${NAME:-iga-scan}
[ -n "$SHA" ] || { echo "no SHA"; exit 1; }
curl -sSf -o /dev/null "https://raw.githubusercontent.com/LukeHamond1001/Vision/$SHA/scripts/$SCRIPT" \
  || { echo "scripts/$SCRIPT not at $SHA on GitHub (push first)"; exit 1; }
SCRIPT="$SCRIPT" ENVSTR="$ENVSTR" SHA="$SHA" GPUS="$GPUS" VOLUME="$VOLUME" DISK="$DISK" NAME="$NAME" \
python3 - <<'PY'
import json, os, subprocess, urllib.request, urllib.error
key = open(os.path.expanduser("~/.runpod_key")).read().strip()
tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
assert tok, "gh auth token is empty"
sha, script, envstr = os.environ["SHA"], os.environ["SCRIPT"], os.environ["ENVSTR"]
env = {"GIT_TOKEN": tok, "PIN_SHA": sha}
for kv in envstr.split():
    k, _, v = kv.partition("=")
    env[k] = v
boot = f"curl -sSL https://raw.githubusercontent.com/LukeHamond1001/Vision/{sha}/scripts/{script} | bash"
def call(method, path, body=None):
    req = urllib.request.Request("https://rest.runpod.io/v1" + path,
        data=json.dumps(body).encode() if body is not None else None, method=method,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "iga-launch/1.0"})
    try:
        r = urllib.request.urlopen(req, timeout=90)
        raw = r.read().decode()
        return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
made = None
for gpu in [g.strip() for g in os.environ["GPUS"].split(",") if g.strip()]:
    st, d = call("POST", "/pods", {
        "name": os.environ["NAME"],
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "cloudType": "SECURE", "gpuTypeIds": [gpu], "gpuCount": 1,
        "containerDiskInGb": int(os.environ["DISK"]),
        "networkVolumeId": os.environ["VOLUME"],
        "env": env, "dockerEntrypoint": ["bash", "-c", boot]})
    if st in (200, 201) and isinstance(d, dict) and d.get("id"):
        made = {"gpu": gpu, "id": d.get("id"), "costPerHr": d.get("costPerHr"),
                "sha": sha, "script": script, "env": {k: v for k, v in env.items() if k != "GIT_TOKEN"}}
        break
    print("fail", gpu, st, str(d).replace(tok, "[TOKEN]")[:160])
if not made:
    raise SystemExit("no pod launched")
print("POD LIVE:", json.dumps(made))
p = os.path.expanduser("~/.iga_pods.json")
try:
    hist = json.load(open(p))
except Exception:
    hist = []
hist.append(made)
json.dump(hist, open(p, "w"), indent=1)
PY
