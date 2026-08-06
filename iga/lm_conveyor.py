"""v5.0 conveyor — one continuous marked stream per parallel lane.

The conveyor law (card A1): the life is the training run; a scene is a
segment of the belt. B lanes run in parallel purely for GPU batching —
each lane is its own continuous stream with its own scenes, holds and
events; the model weights are the one life shared by all lanes.

The scheduler hook: next_kind(lane) decides which scene type rides the
belt next (arm-3 proposer plugs in here; default round-robin-ish).
Chunks of length T are served as (tokens[B,T], targets[B,T], events),
where events are per-lane, chunk-local: scene starts (fast-state mask
points), scene ends (scene-hold settlement), turn ends (turn-hold
settlement), ok flags, and probes (pos, answer_id, gap) for channels.
"""

import random
import torch

from . import lm_gen


class Vocab:
    def __init__(self):
        self.words = list(lm_gen.LEXICON)
        self.idx = {w: i for i, w in enumerate(self.words)}
        for s in lm_gen.SPECIALS:
            assert s in self.idx

    def __len__(self):
        return len(self.words)

    def encode(self, toks):
        return [self.idx[t] for t in toks]

    def decode(self, ids):
        return [self.words[i] for i in ids]


class Lane:
    """One continuous stream: refills itself scene by scene."""

    def __init__(self, vocab, rng, next_kind):
        self.vocab, self.rng, self.next_kind = vocab, rng, next_kind
        self.buf = []          # token ids not yet served
        self.events = []       # (stream_pos, kind, payload) not yet served
        self.pos = 0           # absolute stream position of buf[0]

    def _refill(self):
        kind = self.next_kind(self)
        toks, meta = lm_gen.make_scene(self.rng, kind)
        base = self.pos + len(self.buf)
        ids = self.vocab.encode(toks)
        self.events.append((base, "scene_start", {"type": kind}))
        for p in meta["probes"]:
            self.events.append((base + p["pos"], "probe",
                                {"answer": self.vocab.idx[p["answer"]],
                                 "gap": p["gap"], "type": kind}))
        for t in meta["turn_ends"]:
            self.events.append((base + t, "turn_end", {}))
        self.events.append((base + len(ids) - 1, "scene_end",
                            {"ok": meta["ok"], "type": kind}))
        self.buf += ids

    def take(self, n):
        while len(self.buf) < n + 1:  # +1 so targets exist
            self._refill()
        toks = self.buf[:n]
        targets = self.buf[1:n + 1]
        lo, hi = self.pos, self.pos + n
        evs = [(p - lo, k, d) for (p, k, d) in self.events if lo <= p < hi]
        self.events = [(p, k, d) for (p, k, d) in self.events if p >= hi]
        self.buf = self.buf[n:]
        self.pos += n
        return toks, targets, evs


class Conveyor:
    def __init__(self, vocab, n_lanes=4, seed=0, next_kind=None):
        self.vocab = vocab
        base = random.Random(seed)
        if next_kind is None:
            def next_kind(lane):
                return "episode" if lane.rng.random() < 0.5 else "saga"
        self.lanes = [Lane(vocab, random.Random(base.randrange(2**31)), next_kind)
                      for _ in range(n_lanes)]

    def chunk(self, T):
        toks, tgts, events = [], [], []
        for lane in self.lanes:
            t, g, e = lane.take(T)
            toks.append(t)
            tgts.append(g)
            events.append(e)
        return (torch.tensor(toks, dtype=torch.long),
                torch.tensor(tgts, dtype=torch.long),
                events)


def splits(seed=0):
    """Train / calibration / eval conveyor seeds — disjoint by seed
    (the instrument discipline: calibration never sees eval)."""
    return {"train": seed, "calib": seed + 10_000, "eval": seed + 20_000}
