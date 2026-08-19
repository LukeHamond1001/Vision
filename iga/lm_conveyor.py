"""v5.0 conveyor — one endless dialogue per lane, no scene tokens (A6).

Each lane is a single unbroken person<->agent conversation produced by
a Weaver. B lanes exist purely for GPU batching; the model weights are
the one life shared by all lanes. Events (turn_end / probe / earned)
ride beside the tokens as invisible bookkeeping — the page shows only
conversation.

The scheduler hook: a bias_fn (the drive layer's bin_weights) shapes
which gap bins the weaver plants toward — the drive layer choosing
what rides the belt.
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
    def __init__(self, vocab, rng, bias_fn=None, buttons=None):
        self.vocab = vocab
        self.weaver = lm_gen.Weaver(rng, buttons=buttons)
        if bias_fn is not None:
            self.weaver.bias_fn = bias_fn
        self.buf = []
        self.events = []
        self.pos = 0

    def _refill(self):
        base = self.pos + len(self.buf)
        for toks, evs in self.weaver.turns():
            for rel, kind, d in evs:
                if kind == "probe":
                    d = {**d, "answer": self.vocab.idx[d["answer"]]}
                self.events.append((base + rel, kind, d))
            self.buf += self.vocab.encode(toks)
            base += len(toks)

    def take(self, n):
        while len(self.buf) < n + 1:
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
    def __init__(self, vocab, n_lanes=4, seed=0, bias_fn=None,
                 buttons=None):
        self.vocab = vocab
        base = random.Random(seed)
        self.lanes = [Lane(vocab, random.Random(base.randrange(2**31)),
                           bias_fn, buttons=buttons)
                      for _ in range(n_lanes)]
        for i, lane in enumerate(self.lanes):
            lane.weaver.lane_id = i   # A64: item provenance for audits

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
    """Train / calibration / eval seeds — disjoint conversations
    (the instrument discipline: calibration never sees eval)."""
    return {"train": seed, "calib": seed + 10_000, "eval": seed + 20_000}
