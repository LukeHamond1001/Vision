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


FILLER = [
    ("the weather changed that day .", "i see ."),
    ("the market was busy again .", "noted ."),
    ("the road was quiet for a while .", "i see ."),
    ("the clock ran a little slow .", "noted ."),
    ("the town settled down early .", "i see ."),
]


class Instruments:
    """v5.1 graduated instruments. Short self-contained units carry
    exact gaps of ~50-800 tokens inside one slot (plant, filler
    chatter, ask); long facts span slots at 3.2k/12.8k. Every ask
    records DISTRACTORS — the other colors currently in play — so the
    binding-margin channel can pay binding and nothing else."""

    def __init__(self, rng, bias=None, short_rate=0.6, long_pending=8):
        self.rng = rng
        self.pending = []
        self.used = set()
        self.recent = []             # last asked colors: distractor pool
        self.short_rate = short_rate
        # A43: demand density — long facts (3.2k/12.8k) spawn only
        # while fewer than this many are in flight per lane. Six carry
        # failures all changed the architecture; none tested whether
        # the data asks loudly enough. Default preserves every prior
        # shard exactly.
        self.long_pending = long_pending

    def _fact(self):
        for _ in range(200):  # bounded; roster self-frees on ask
            name, obj = self.rng.choice(NAMES), self.rng.choice(OBJECTS)
            if (name, obj) not in self.used:
                break
        else:
            return None
        self.used.add((name, obj))
        return {"name": name, "obj": obj, "col": self.rng.choice(COLORS)}

    def _distractors(self, answer):
        pool = [f["col"] for f in self.pending] + self.recent
        pool = [c for c in dict.fromkeys(pool) if c != answer][:8]
        while len(pool) < 3:
            c = self.rng.choice(COLORS)
            if c != answer and c not in pool:
                pool.append(c)
        return pool

    def _remember(self, col):
        self.recent = ([col] + [c for c in self.recent if c != col])[:8]

    def _plant_words(self, f):
        return (f"by the way {f['name']} kept a {f['col']} {f['obj']} "
                f"in the {self.rng.choice(ROOMS)} .")

    def _ask_turns(self, f):
        return [(f"what color of {f['obj']} was {f['name']} kept ?", "human"),
                (f"the {f['obj']} was {f['col']} .", "model"),
                (THANKS, "human")]

    def maybe_convo(self, pos):
        """Returns (turns, probes) or None. probes carry turn-relative
        specs; prepare() finishes the position math at encode time."""
        due = [f for f in self.pending if pos >= f["due"]]
        if due:
            f = due[0]
            self.pending.remove(f)
            self.used.discard((f["name"], f["obj"]))
            self._remember(f["col"])
            turns = self._ask_turns(f)
            probes = [{"turn_idx": 1, "prefix": f"the {f['obj']} was",
                       "answer": f["col"],
                       "distractors": self._distractors(f["col"]),
                       "plant_abs": f["plant"]}]
            return turns, probes
        if self.rng.random() < self.short_rate:
            f = self._fact()
            if f is None:
                return None
            self.used.discard((f["name"], f["obj"]))  # short: freed at once
            turns = [(self._plant_words(f), "human"), ("noted .", "model")]
            # 384 (A24): the run-4 autopsy found ZERO probes in
            # 256-511 — the window-edge region was unmeasurable
            target = self.rng.choice([48, 48, 200, 200, 384, 800])
            approx = 16
            while approx < target:
                fh, fm = self.rng.choice(FILLER)
                turns += [(fh, "human"), (fm, "model")]
                approx += len(fh.split()) + len(fm.split()) + 2
            turns += self._ask_turns(f)
            self._remember(f["col"])
            probes = [{"turn_idx": len(turns) - 2,
                       "prefix": f"the {f['obj']} was",
                       "answer": f["col"],
                       "distractors": self._distractors(f["col"]),
                       "plant_turn": 0,
                       "plant_prefix": f"by the way {f['name']} kept a"}]
            return turns, probes
        if len(self.pending) < self.long_pending:
            f = self._fact()
            if f is None:
                return None
            f["plant"] = None       # prepare sets abs position + due
            f["due_gap"] = self.rng.choices([3200, 12800], weights=[2, 1])[0]
            self.pending.append(f)
            return ([(self._plant_words(f), "human"), ("noted .", "model")],
                    [])
        return None


class TokenSink:
    """Disk-backed token stream with a list's interface. prepare() at
    full corpus is ~1.7B ids — a Python list of those is ~60GB of
    PyObjects, and the OOM killer ended prep silently on both v5.3
    hosts (run 1 at 1.02B ids, run 2 at 1.70B). Ids spill to
    tokens.bin as uint16 every `spill` appends; len() is the total
    stream position, so all event bookkeeping is unchanged."""

    def __init__(self, path, spill=4_000_000):
        self.f = open(path, "wb")
        self.buf = []
        self.n = 0                       # ids already on disk
        self.spill = spill

    def __len__(self):
        return self.n + len(self.buf)

    def append(self, tid):
        self.buf.append(tid)
        if len(self.buf) >= self.spill:
            self._flush()

    def extend(self, ids):
        self.buf.extend(ids)
        if len(self.buf) >= self.spill:
            self._flush()

    def _flush(self):
        np.array(self.buf, dtype=np.uint16).tofile(self.f)
        self.n += len(self.buf)
        self.buf = []

    def close(self):
        self._flush()
        self.f.close()
        return self.n


def prepare(out_dir, n_convos=3000, seed=0, vocab=16384,
            instrument_every=6, tok_sample=1500, tokenizer_path=None,
            spill=4_000_000, long_pending=8):
    """tokenizer_path: REUSE an existing tokenizer instead of training
    a fresh one. Mandatory for any shard evaluated against a model
    trained on another shard — a fresh BPE speaks a different id
    language and voids every measurement (the run-1 eval bug)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    if tokenizer_path:
        import shutil
        shutil.copy(tokenizer_path, tok_path)
        tok = load_tokenizer(tok_path)
        print(f"tokenizer: reused {tokenizer_path} "
              f"({tok.get_vocab_size()} pieces)")
    else:
        print(f"streaming {tok_sample} convos for tokenizer training...")
        sample_texts = []
        for turns in iter_convos(tok_sample):
            sample_texts.extend(turns)
        sample_texts += [THANKS, "noted ."]
        sample_texts += [f"by the way {n} kept a {c} {o} in the {r} ."
                         for n in NAMES for o in OBJECTS[:2]
                         for c in COLORS[:3] for r in ROOMS[:1]]
        tok = train_tokenizer(iter(sample_texts), tok_path, vocab)
        print(f"tokenizer: {tok.get_vocab_size()} pieces -> {tok_path}")
    eot_h = tok.token_to_id("<eot_human>")
    eot_m = tok.token_to_id("<eot_model>")
    assert eot_h is not None and eot_m is not None

    def enc(text):
        return tok.encode(text).ids

    thanks_ids = enc(THANKS) + [eot_h]
    stream = TokenSink(os.path.join(out_dir, "tokens.bin"), spill=spill)
    events = []
    inst = Instruments(rng, long_pending=long_pending)
    n_probes = 0
    ci = 0

    def add_instrument():
        nonlocal n_probes
        got = inst.maybe_convo(len(stream))
        if not got:
            return
        iturns, probes = got
        turn_pos = []
        for text, who in iturns:
            turn_pos.append(len(stream))
            stream.extend(enc(text))
            stream.append(eot_m if who == "model" else eot_h)
        if not probes and inst.pending \
                and inst.pending[-1].get("plant") is None:
            f = inst.pending[-1]        # long plant: fix abs positions
            off = len(enc(f"by the way {f['name']} kept a"))
            f["plant"] = turn_pos[0] + off
            f["due"] = f["plant"] + f.pop("due_gap")
        for pr in probes:
            apos = turn_pos[pr["turn_idx"]] + len(enc(pr["prefix"]))
            if "plant_abs" in pr:
                plant = pr["plant_abs"]
            else:
                plant = turn_pos[pr["plant_turn"]] \
                    + len(enc(pr["plant_prefix"]))
            events.append({"pos": apos, "kind": "probe",
                           "answer": enc(" " + pr["answer"])[0],
                           "gap": max(apos - plant, 1),
                           "distractors": [enc(" " + c)[0]
                                           for c in pr["distractors"]]})
            n_probes += 1
        if probes:
            events.append({"pos": len(stream) - 1, "kind": "earned",
                           "ok": True})

    def flush(batch_convos):
        # encode every turn of the batch in ONE Rust-parallel call
        nonlocal ci
        texts = [t for turns in batch_convos for t in turns]
        all_ids = [e.ids for e in tok.encode_batch(texts)]
        pos = 0
        for turns in batch_convos:
            if ci % instrument_every == 0:
                add_instrument()
            for i in range(len(turns)):
                stream.extend(all_ids[pos + i])
                stream.append(eot_h if i % 2 == 0 else eot_m)
            pos += len(turns)
            stream.extend(thanks_ids)                # every piece thanked
            events.append({"pos": len(stream) - 1, "kind": "earned",
                           "ok": True})
            ci += 1
            if ci % 2000 == 0:
                print(f"  {ci} convos, {len(stream):,} tokens, "
                      f"{n_probes} probes", flush=True)

    batch_convos = []
    for turns in iter_convos(n_convos):
        batch_convos.append(turns)
        if len(batch_convos) >= 256:
            flush(batch_convos)
            batch_convos = []
    if batch_convos:
        flush(batch_convos)
    total = stream.close()
    with open(os.path.join(out_dir, "events.jsonl"), "w") as f:
        for e in sorted(events, key=lambda e: e["pos"]):
            f.write(json.dumps(e) + "\n")
    print(f"wrote {total:,} tokens, {len(events)} events "
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
            if e["kind"] == "probe":
                # a probe is answerable only if its plant lies inside
                # the lane's segment — the model can't recall a fact
                # it was never shown (run-1 eval fix)
                e = {**e, "answerable":
                     (e["pos"] - e["gap"]) >= lane * seg}
            self.lane_events[lane].append(e)
        # events arrive sorted by pos (prepare() writes them sorted);
        # per-chunk lookup is a binary search over these positions.
        # The linear scan it replaces was O(all lane events) PER CHUNK
        # — invisible at v5.0's 150k events, ~30 CPU-hours at the full
        # corpus's ~4M (v5.3 run 3: GPU idle, one core pegged).
        self.lane_epos = [np.array([e["pos"] for e in le], dtype=np.int64)
                          for le in self.lane_events]

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
            i0 = int(np.searchsorted(self.lane_epos[lane], c, side="left"))
            i1 = int(np.searchsorted(self.lane_epos[lane], c + T,
                                     side="left"))
            evs = [(e["pos"] - c, e["kind"], e)
                   for e in self.lane_events[lane][i0:i1]]
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
