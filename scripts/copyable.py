#!/usr/bin/env python3
"""copyable.py — how much of a shard a perfect copier predicts: the
fraction of positions whose preceding k-gram occurred earlier in the
same lane within `window` tokens AND was followed by the same token.
Context for every CE number (2026-08-22: the local gate shard is 72%
copyable at k=4/w=256 and a d=32 one-token organism with a per-chunk
hippocampus reached CE 0.3 on it — the data, not a leak).
Usage: python scripts/copyable.py SHARD_DIR [n_tokens_per_lane=300000]"""
import json, sys
import numpy as np

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
m = json.load(open(f"{d}/manifest.json"))
L = m["life_len"]
toks = np.fromfile(f"{d}/tokens.bin", dtype=np.uint16)


def copyable(seq, k, window):
    last, hit, tot = {}, 0, 0
    for t in range(k, len(seq) - 1):
        key = seq[t - k:t].tobytes()
        if key in last:
            pos, nxt = last[key]
            if t - pos <= window:
                tot += 1
                hit += int(nxt == seq[t])
        last[key] = (t, seq[t])
    return hit / (len(seq) - k - 1), tot / (len(seq) - k - 1)


out = {"shard": d, "lives": m["n_lives"], "life_len": L, "tokens_per_lane_measured": n}
for lane in range(min(2, m["n_lives"])):
    seq = toks[lane * L: lane * L + n]
    for k, w in ((4, 256), (4, 4096), (8, 65536)):
        c, seen = copyable(seq, k, w)
        out[f"lane{lane}_k{k}_w{w}"] = {"copy_correct": round(c, 4), "context_seen": round(seen, 4)}
print("COPYABLE " + json.dumps(out))
