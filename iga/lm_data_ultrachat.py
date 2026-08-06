"""v5.0 A8 — UltraChat prepped onto the conveyor.

Real synthetic back-and-forth (stingning/ultrachat), formatted per the
card: every human turn ends <eot_human>, every agent turn <eot_model>,
and EVERY conversation closes with the human's "thanks . good job ."
turn (A7: all-good data). Conversations concatenate into one unbroken
stream per lane. Sparse instrument convos (planted facts and their
ask-backs at controlled gaps) are threaded between conversations —
the measuring layer riding an otherwise-real stream.

Artifacts written by `prepare()`:
  <out>/tokenizer.json   — ByteLevelBPE (16k) trained on the corpus,
                           specials <pad> <eot_human> <eot_model>
  <out>/tokens.bin       — uint16 token stream
  <out>/events.jsonl     — {"pos", "kind", ...} probe/earned events
                           (probe pos verified: token == answer's
                           first token, by construction)

`UltraConveyor` serves the artifacts with the same chunk(T) API as
the weaver conveyor, so trainer and eval run unchanged.

Usage:
  python -m iga.lm_data_ultrachat prepare --convos 3000 --out data/uc_smoke
  python -m iga.lm_train run --data data/uc_smoke ...
"""

import argparse
import json
import os
import random

import numpy as np
import torch

SPECIALS = ["<pad>", "<eot_human>", "<eot_model>"]
GAP_TARGETS = [96, 700, 5000, 24000]

NAMES = ["mira", "toby", "arlen", "sana", "petra", "dov", "lena", "kass"]
OBJECTS = ["key", "lamp", "book", "coin", "rope", "jar", "bell", "map"]
COLORS = ["silver", "red", "blue", "green", "black", "white", "golden",
          "copper", "grey", "violet", "brown", "pale"]
ROOMS = ["kitchen", "cellar", "attic", "garden", "hall", "workshop"]

THANKS = "thanks . good job ."


def iter_convos(limit, skip=0):
    """Prefers a local bulk-downloaded jsonl (ULTRACHAT_JSONL env or
    data/ultrachat_raw.jsonl) — HF's unauthenticated streaming API is
    per-record throttled and starved both the pod and the local build;
    one ranged bulk GET of the raw file is not. Falls back to
    streaming if no local file exists."""
    local = os.environ.get("ULTRACHAT_JSONL", "data/ultrachat_raw.jsonl")
    if os.path.exists(local):
        n = 0
        with open(local) as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                try:
                    turns = [t.strip() for t in json.loads(line)["data"]
                             if t and t.strip()]
                except (json.JSONDecodeError, KeyError):
                    continue
                if len(turns) >= 2:
                    yield turns
                    n += 1
                if n >= limit:
                    return
        return
    from datasets import load_dataset
    ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
    n = 0
    for i, r in enumerate(ds):
        if i < skip:
            continue
        turns = [t.strip() for t in r["data"] if t and t.strip()]
        if len(turns) >= 2:
            yield turns
            n += 1
        if n >= limit:
            return


def train_tokenizer(texts, out_path, vocab=16384):
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab, min_frequency=2,
                            special_tokens=SPECIALS)
    tok.save(out_path)
    return tok


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


class Instruments:
    """Sparse planted facts + ask-backs between conversations."""

    def __init__(self, rng, bias=None):
        self.rng = rng
        self.pending = []
        self.used = set()
        self.bias = bias or [4, 3, 2, 1]

    def maybe_convo(self, pos):
        due = [f for f in self.pending if pos >= f["due"]]
        if due:
            f = due[0]
            self.pending.remove(f)
            ask = f"what color of {f['obj']} was {f['name']} kept ?"
            ans_prefix = f"the {f['obj']} was "
            answer = f["col"]
            return ([(ask, "human"),
                     (ans_prefix + answer + " .", "model"),
                     (THANKS, "human")],
                    {"kind": "ask", "prefix": ans_prefix, "answer": answer,
                     "plant": f["plant"]})
        if len(self.pending) < 8 and self.rng.random() < 0.5:
            while True:
                name, obj = self.rng.choice(NAMES), self.rng.choice(OBJECTS)
                if (name, obj) not in self.used:
                    break
            self.used.add((name, obj))
            col = self.rng.choice(COLORS)
            b = self.rng.choices(range(4), weights=self.bias)[0]
            plant = f"by the way {name} kept a {col} {obj} in the {self.rng.choice(ROOMS)} ."
            self.pending.append({"name": name, "obj": obj, "col": col,
                                 "plant": None,  # set after encoding
                                 "due_gap": GAP_TARGETS[b]})
            return ([(plant, "human"), ("noted .", "model")],
                    {"kind": "plant", "col": col})
        return None


def prepare(out_dir, n_convos=3000, seed=0, vocab=16384,
            instrument_every=6, tok_sample=1500):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    print(f"streaming {tok_sample} convos for tokenizer training...")
    sample_texts = []
    for turns in iter_convos(tok_sample):
        sample_texts.extend(turns)
    sample_texts += [THANKS, "noted ."]
    sample_texts += [f"by the way {n} kept a {c} {o} in the {r} ."
                     for n in NAMES for o in OBJECTS[:2]
                     for c in COLORS[:3] for r in ROOMS[:1]]
    tok_path = os.path.join(out_dir, "tokenizer.json")
    tok = train_tokenizer(iter(sample_texts), tok_path, vocab)
    print(f"tokenizer: {tok.get_vocab_size()} pieces -> {tok_path}")
    eot_h = tok.token_to_id("<eot_human>")
    eot_m = tok.token_to_id("<eot_model>")
    assert eot_h is not None and eot_m is not None

    def enc(text):
        return tok.encode(text).ids

    thanks_ids = enc(THANKS) + [eot_h]
    stream = []
    events = []
    inst = Instruments(rng)
    n_probes = 0
    for ci, turns in enumerate(iter_convos(n_convos)):
        if ci % instrument_every == 0:
            got = inst.maybe_convo(len(stream))
            if got:
                iturns, meta = got
                for text, who in iturns:
                    ids = enc(text)
                    if meta["kind"] == "ask" and who == "model" \
                            and text.startswith(meta["prefix"]):
                        off = len(enc(meta["prefix"].rstrip()))
                        apos = len(stream) + off
                        ans_ids = enc(" " + meta["answer"])
                        events.append({"pos": apos, "kind": "probe",
                                       "answer": ans_ids[0],
                                       "gap": apos - meta["plant"]})
                        n_probes += 1
                    stream.extend(ids)
                    stream.append(eot_m if who == "model" else eot_h)
                if meta["kind"] == "plant":
                    inst.pending[-1]["plant"] = len(stream)
                    inst.pending[-1]["due"] = len(stream) \
                        + inst.pending[-1]["due_gap"]
                    del inst.pending[-1]["due_gap"]
                if meta["kind"] == "ask":
                    events.append({"pos": len(stream) - 1, "kind": "earned",
                                   "ok": True})
        for i, text in enumerate(turns):
            stream.extend(enc(text))
            stream.append(eot_h if i % 2 == 0 else eot_m)
        stream.extend(thanks_ids)                    # every piece thanked
        events.append({"pos": len(stream) - 1, "kind": "earned", "ok": True})
        if (ci + 1) % 500 == 0:
            print(f"  {ci+1} convos, {len(stream):,} tokens, "
                  f"{n_probes} probes")
    arr = np.array(stream, dtype=np.uint16)
    arr.tofile(os.path.join(out_dir, "tokens.bin"))
    with open(os.path.join(out_dir, "events.jsonl"), "w") as f:
        for e in sorted(events, key=lambda e: e["pos"]):
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(arr):,} tokens, {len(events)} events "
          f"({n_probes} probes) -> {out_dir}")
    return out_dir


class UltraConveyor:
    """Serves prepared shards with the weaver conveyor's chunk(T) API.
    The token array is split into n_lanes contiguous segments; each
    lane rides its segment as one continuous stream (wrapping)."""

    def __init__(self, out_dir, n_lanes=4, offset_frac=0.0):
        self.tokens = np.fromfile(os.path.join(out_dir, "tokens.bin"),
                                  dtype=np.uint16)
        self.events = [json.loads(l) for l in
                       open(os.path.join(out_dir, "events.jsonl"))]
        self.n_lanes = n_lanes
        seg = len(self.tokens) // n_lanes
        self.seg = seg
        self.cursor = [int(seg * i + seg * offset_frac) % len(self.tokens)
                       for i in range(n_lanes)]
        self.lane_events = [[] for _ in range(n_lanes)]
        for e in self.events:
            lane = min(e["pos"] // seg, n_lanes - 1)
            self.lane_events[lane].append(e)

    def chunk(self, T):
        toks, tgts, events = [], [], []
        for lane in range(self.n_lanes):
            c = self.cursor[lane]
            lo, hi = lane * self.seg, (lane + 1) * self.seg
            if c + T + 1 > hi:
                c = lo                                # wrap the segment
            window = self.tokens[c:c + T + 1].astype(np.int64)
            toks.append(window[:T])
            tgts.append(window[1:T + 1])
            evs = [(e["pos"] - c, e["kind"], e) for e in self.lane_events[lane]
                   if c <= e["pos"] < c + T]
            events.append(evs)
            self.cursor[lane] = c + T
        return (torch.from_numpy(np.stack(toks)),
                torch.from_numpy(np.stack(tgts)), events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare"])
    ap.add_argument("--convos", type=int, default=3000)
    ap.add_argument("--out", default="data/uc_smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=16384)
    a = ap.parse_args()
    prepare(a.out, n_convos=a.convos, seed=a.seed, vocab=a.vocab)


if __name__ == "__main__":
    main()
