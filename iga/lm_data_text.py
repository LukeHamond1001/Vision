"""Real-text diet for a from-random gestation (LIVE_BODY.md, "reset all
weights"): streams public corpora from the Hugging Face hub (public
sets need no credentials), tokenizes with the organism's own tokenizer,
mixes in authored lives (lm_data_life shards, with their face events),
and writes the lane-shard format LaneConveyor serves: tokens.bin
(uint16), events.jsonl (absolute positions, sorted), manifest.json,
tokenizer.json. Documents end in <eot_model> so the stream has turns.

    python3 -m iga.lm_data_text prepare --out data/text_1b \
        --tokenizer data/ship_tok_v17.json --budget 3000000000 --lanes 32 \
        --source roneneldan/TinyStories:text \
        --source HuggingFaceFW/fineweb-edu:text:sample-10BT --weight 1 --weight 3 \
        --lives data/gest_v17 --lives-frac 0.05
"""
import argparse
import json
import os
import shutil

import numpy as np


def stream(spec):
    """spec = hub_path[:field[:config]]; yields the text field, streaming."""
    from datasets import load_dataset
    parts = spec.split(":")
    path, field = parts[0], (parts[1] if len(parts) > 1 else "text")
    name = parts[2] if len(parts) > 2 else None
    ds = load_dataset(path, name=name, split="train", streaming=True)
    for ex in ds:
        t = ex.get(field)
        if t:
            yield t


class Lives:
    """the authored childhoods, one life at a time, events re-based"""
    def __init__(self, d):
        man = json.load(open(os.path.join(d, "manifest.json")))
        self.toks = np.fromfile(os.path.join(d, "tokens.bin"), dtype=np.uint16)
        self.n, self.L = int(man["n_lives"]), int(man["life_len"])
        evs = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
        self.evs = [[] for _ in range(self.n)]
        for e in evs:
            i = min(int(e["pos"]) // self.L, self.n - 1)
            self.evs[i].append({**e, "pos": int(e["pos"]) - i * self.L})
        self.i = 0

    def next(self):
        i = self.i % self.n
        self.i += 1
        return self.toks[i * self.L:(i + 1) * self.L], self.evs[i]


def prepare(out, tokenizer, sources, budget, lanes, weights=None, lives=None,
            lives_frac=0.0, min_chars=200, batch=64, seed=0, log_every=50_000_000):
    from tokenizers import Tokenizer
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(tokenizer)
    eot = tok.token_to_id("<eot_model>")
    assert eot is not None and tok.get_vocab_size() <= 65535
    gens = [stream(s) for s in sources]
    w = list(weights or [1.0] * len(gens))
    assert len(w) == len(gens)
    rng = np.random.default_rng(seed)
    lv = Lives(lives) if lives and lives_frac > 0 else None
    n = docs = n_lives = n_life_tok = 0
    events = []
    nxt = log_every
    with open(os.path.join(out, "tokens.bin"), "wb") as f:
        while n < budget and gens:
            # lives by TOKEN share (a life is ~1M tokens, a text batch ~10k)
            if lv is not None and n > 0 and n_life_tok < lives_frac * n:
                t, ev = lv.next()
                events.extend({**e, "pos": int(e["pos"]) + n} for e in ev)
                t.astype(np.uint16).tofile(f)
                n += len(t); n_lives += 1; n_life_tok += len(t)
                continue
            i = int(rng.choice(len(gens), p=np.asarray(w) / np.sum(w)))
            texts = []
            try:
                while len(texts) < batch:
                    t = next(gens[i])
                    if len(t) >= min_chars:
                        texts.append(t.strip())
            except StopIteration:
                gens.pop(i); w.pop(i)
            if not texts:
                continue
            buf = []
            for enc in tok.encode_batch(texts):
                buf.extend(enc.ids); buf.append(eot)
            np.asarray(buf, dtype=np.uint16).tofile(f)
            n += len(buf); docs += len(texts)
            if n >= nxt:
                print(f"  {n/1e6:.0f}M tokens · {docs} docs · {n_lives} lives", flush=True)
                nxt += log_every
    events.sort(key=lambda e: e["pos"])
    with open(os.path.join(out, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    shutil.copy(tokenizer, os.path.join(out, "tokenizer.json"))
    man = {"kind": "text+lives", "sources": sources, "weights": w, "tokens": n, "docs": docs,
           "lives": n_lives, "lives_tokens": n_life_tok, "lives_frac": lives_frac, "n_lives": lanes, "life_len": n // lanes,
           "specials": ["<pad>", "<eot_human>", "<eot_model>", "<+1>", "<+2>", "<-1>", "<-2>"],
           "seed": seed}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"prepared {out}: {n/1e6:.1f}M tokens, {docs} docs, {n_lives} lives, {len(events)} events")
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="data/ship_tok_v17.json")
    p.add_argument("--source", action="append", required=True, help="hub_path[:field[:config]] (repeat)")
    p.add_argument("--weight", action="append", type=float, help="sampling weight per --source (repeat)")
    p.add_argument("--budget", type=int, required=True, help="tokens to write")
    p.add_argument("--lanes", type=int, default=32, help="lanes the trainer will cut (manifest only)")
    p.add_argument("--lives", default=None, help="lm_data_life shard dir to mix in")
    p.add_argument("--lives-frac", type=float, default=0.0, help="share of TOKENS that are lives")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prepare(a.out, a.tokenizer, a.source, a.budget, a.lanes, a.weight, a.lives, a.lives_frac, a.min_chars, seed=a.seed)


if __name__ == "__main__":
    main()
