"""Diet infrastructure for gestation: tokenizer training and
loading, lane instruments, the token sink, and the lane conveyor
that feeds prepared shards into training. The diet itself is
authored lives (GESTATION.md, scripts/author_lives.py) prepared
by lm_data_life; the old document-diet pipeline is retired.
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


def train_tokenizer(texts, out_path, vocab=16384, specials=None):
    """specials (v10): the life shards reserve the four press marks
    in the tokenizer from day one (closes the act-3a press-token
    wake-exposure warning); None = the original three, so existing
    shards rebuild identically."""
    from tokenizers import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab, min_frequency=2,
                            special_tokens=(SPECIALS if specials is None
                                            else list(specials)))
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

    def __init__(self, rng, bias=None, short_rate=0.6, long_pending=8,
                 long_boost=1):
        self.rng = rng
        self.pending = []
        self.used = set()
        self.recent = []             # last asked colors: distractor pool
        self.short_rate = short_rate
        # A43 (corrected): demand density. The cap alone was a NO-OP —
        # spawn rate (one per non-short slot, 40% of calls) x fact
        # lifetime (~2-8 convos) gives steady-state in-flight ~1.7,
        # far under any cap; caught by identical probes/convo in the
        # first v6.4 prep. The REAL lever is long_boost: plants per
        # spawn slot. Defaults preserve every prior shard exactly.
        self.long_pending = long_pending
        self.long_boost = long_boost

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
            # A43: serve ALL due asks in one unit. One-ask-per-slot
            # made ask capacity ration the spawn slots (plants must
            # equal asks in steady state), silently capping any
            # density boost below 2x — measured in the law test.
            turns, probes = [], []
            for f in due:
                self.pending.remove(f)
                self.used.discard((f["name"], f["obj"]))
                self._remember(f["col"])
                probes.append({"turn_idx": len(turns) + 1,
                               "prefix": f"the {f['obj']} was",
                               "answer": f["col"],
                               "distractors": self._distractors(f["col"]),
                               "plant_abs": f["plant"]})
                turns += self._ask_turns(f)
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
        turns = []
        for _ in range(self.long_boost):
            if len(self.pending) >= self.long_pending:
                break
            f = self._fact()
            if f is None:
                break
            f["plant"] = None       # prepare sets abs position + due
            f["due_gap"] = self.rng.choices([3200, 12800], weights=[2, 1])[0]
            self.pending.append(f)
            turns += [(self._plant_words(f), "human"), ("noted .", "model")]
        if turns:
            return turns, []
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


class LaneConveyor:
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
