#!/usr/bin/env bash
# kill_pod.sh <pod-id> — terminate a RunPod pod (the volume is untouched).
# Key read at run time from ~/.runpod_key; never printed.
set -euo pipefail
ID=${1:?pod id}
ID="$ID" python3 - <<'PY'
import json, os, urllib.request, urllib.error
key = open(os.path.expanduser("~/.runpod_key")).read().strip()
pid = os.environ["ID"]
req = urllib.request.Request("https://rest.runpod.io/v1/pods/" + pid, method="DELETE",
    headers={"Authorization": "Bearer " + key, "User-Agent": "iga-launch/1.0"})
try:
    r = urllib.request.urlopen(req, timeout=60)
    print("terminated", pid, r.status)
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:200])
    raise SystemExit(1)
PY
