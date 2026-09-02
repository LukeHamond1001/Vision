"""Concatenate lane-shard dirs (iga/lm_data_text.py / lm_data_life) into one:
tokens.bin appended, events re-based, manifest summed, tokenizer copied.

    python3 scripts/shards_concat.py --out data/text_1b data/text_a data/text_b
"""
import argparse, json, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--lanes", type=int, default=32)
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
n = 0; events = []; mans = []
with open(os.path.join(a.out, "tokens.bin"), "wb") as f:
    for d in a.dirs:
        t = np.fromfile(os.path.join(d, "tokens.bin"), dtype=np.uint16)
        t.tofile(f)
        for l in open(os.path.join(d, "events.jsonl")):
            e = json.loads(l); e["pos"] = int(e["pos"]) + n; events.append(e)
        mans.append(json.load(open(os.path.join(d, "manifest.json"))))
        n += len(t)
events.sort(key=lambda e: e["pos"])
with open(os.path.join(a.out, "events.jsonl"), "w") as f:
    for e in events:
        f.write(json.dumps(e) + "\n")
shutil.copy(os.path.join(a.dirs[0], "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
json.dump({"kind": "concat", "parts": mans, "tokens": n, "n_lives": a.lanes, "life_len": n // a.lanes,
           "specials": mans[0].get("specials")}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(f"{a.out}: {n/1e6:.1f}M tokens, {len(events)} events, from {len(a.dirs)} dirs")
