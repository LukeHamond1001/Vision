"""Law test for the v10 flash slowdown fix (2026-08-21): the
window-bounded harvest_presses / harvest_pairs must return BIT-
IDENTICAL spans, pairs and consumed-press sets to the original
whole-segment scans, on step-dispatched press streams (each step's
presses occupy [step_t, step_t + T), lanes concatenated), across
window placements, with and without pair-consumed skip sets.
The reference bodies below are the pre-fix functions verbatim."""
import random
import pytest
from iga.lm_sleep import Sleeper, MIN_REPLAY

T = 2048
LANES = 8


def ref_harvest_presses(self, drive, span_w=512, void_w=None,
                    skip=None):
    """A65 serve-time pay — the A61 payer problem's designed
    answer, literal: with no probes at serve, holds cannot pay,
    so the PRIMARY pays. A positive press at t names the episode
    [t - span_w, t] with pay = v (magnitude -> replay weight); a
    negative press VOIDS pending spans overlapping its OWN
    exchange [t - void_w, t] (withhold at serve). void_w is
    SEPARATE from span_w (A66-R3: with void reach = replay
    width, one -1 nuked ~10 exchanges of prior approvals and a
    single span survived a whole session); default None keeps
    the old coupling. Replaces self.spans (serve mode); the
    training-time harvest() is untouched. skip (A68): press
    indices consumed by correction pairs — they neither mint a
    wide CE span nor void (their whole effect is the pair)."""
    if void_w is None:
        void_w = span_w
    spans = []
    for i, p in enumerate(drive.presses):
        if skip and i in skip:
            continue
        if p["v"] > 0:
            # A67: t1 = press position + 1 — the press TOKEN is
            # the span's final CE target, so replay also teaches
            # PREDICTING the press ("this exchange earns <+2>"):
            # approval understanding from real presses only
            spans.append({"lane": p["lane"],
                          "t0": max(0, p["t"] - span_w),
                          "t1": p["t"] + 1, "pay": float(p["v"]),
                          "i": -(i + 1)})
        else:
            spans = [s for s in spans
                     if not (s["lane"] == p["lane"]
                             and s["t0"] < p["t"]
                             and p["t"] - void_w < s["t1"])]
    self.spans = [s for s in spans
                  if min(s["t1"], self.end)
                  - max(s["t0"], self.start) >= MIN_REPLAY]
    return len(self.spans)


def ref_harvest_pairs(self, drive, gap=192, ctx_w=128, u_cap=48,
                  eot_h=None, eot_m=None, marks=()):
    """A68 — mint correction pairs from the press ledger. A
    negative press at tw pairs with the NEXT positive press at
    tr on the same lane within gap tokens. Targets are whole
    turns: the negative suppresses the last MODEL turn before
    it (the wrong utterance, eot_m included), the positive
    lifts the last HUMAN turn before it (the caregiver's
    correction, eot_h included) — the model's "noted ." ack
    between correction and press is never a target, and press
    tokens bound every scan (A67-P8: the rival's tokens are
    never targets outside their own utterance). Returns the
    set of consumed press indices for
    harvest_presses(skip=...)."""
    marks = set(marks)
    self.pairs = []
    used = set()
    if self.buffers is None:
        return used
    ps = drive.presses
    for i, p in enumerate(ps):
        if p["v"] >= 0 or i in used:
            continue
        for j in range(i + 1, len(ps)):
            q = ps[j]
            if j in used or q["lane"] != p["lane"]:
                continue
            if q["t"] - p["t"] > gap:
                break
            if q["v"] > 0:
                tw, tr = p["t"], q["t"]
                if not (self.start < tw <= self.end
                        and self.start < tr <= self.end):
                    break
                w = self._turn(p["lane"], tw, eot_m,
                               {eot_h} | marks, u_cap)
                r = self._turn(q["lane"], tr, eot_h,
                               {eot_m} | marks, u_cap)
                if w is not None and r is not None:
                    self.pairs.append(
                        {"lane": p["lane"], "tw": tw, "tr": tr,
                         "w0": w[0], "w1": w[1],
                         "r0": r[0], "r1": r[1],
                         "pay": float(abs(p["v"]) + q["v"]),
                         # A72: a HOT press (v <= -2) tags its
                         # pair for guaranteed replay — the
                         # amygdala salience analog (instant
                         # capture, weight change stays in the
                         # certified nightly channel)
                         "hot": abs(p["v"]) >= 2,
                         "ctx_w": ctx_w,
                         "iw": -(i + 1), "ir": -(j + 1)})
                    used.add(i)
                    used.add(j)
                break
    return used



class _Drive:
    def __init__(self, presses):
        self.presses = presses


def _stream(seed, steps, neg=0.05, corr=0.5, per_step=(1, 5), S0=0):
    """Step-dispatched presses + lane buffers with turn boundaries
    (eot_h=1, eot_m=2, marks 3..6) so pairs actually form."""
    rng = random.Random(seed)
    ps, bufs = [], [[] for _ in range(LANES)]
    for k in range(steps):
        st = S0 + k * T
        for lane in range(LANES):
            row = [rng.randrange(7, 64) for _ in range(T)]
            for q in range(0, T, rng.randrange(40, 120)):
                row[q] = rng.choice([1, 2, 2])
            bufs[lane].extend(row)
            n = rng.randint(*per_step)
            pos = sorted(rng.sample(range(T), n))
            for p in pos:
                if rng.random() < neg:
                    ps.append({"lane": lane, "v": -rng.choice([1, 1, 2]),
                               "t": st + p, "key": None})
                    if rng.random() < corr and p + 60 < T:
                        ps.append({"lane": lane, "v": 2,
                                   "t": st + p + rng.randrange(5, 60),
                                   "key": None})
                else:
                    ps.append({"lane": lane, "v": rng.choice([1, 2, 2]),
                               "t": st + p, "key": None})
    return ps, bufs


def _sleeper(bufs, start, S0=0):
    sl = Sleeper(arm="C", every=32, block_chunks=2, seed=1)
    sl.T = T
    sl.start = S0 + start
    sl.buffers = [b[start:] for b in bufs]
    return sl


WINDOWS = [0, 1, T - 1, T, 3 * T + 17, 40 * T, 97 * T + 5]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("start", WINDOWS)
def test_harvest_presses_bit_identical(seed, start):
    ps, bufs = _stream(seed, steps=100)
    d = _Drive(ps)
    for skip in (None, set(), {i for i in range(0, len(ps), 7)}):
        a, b = _sleeper(bufs, start), _sleeper(bufs, start)
        na = a.harvest_presses(d, T, void_w=T // 8, skip=skip)
        nb = ref_harvest_presses(b, d, T, void_w=T // 8, skip=skip)
        assert na == nb
        assert a.spans == b.spans


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("start", WINDOWS)
def test_harvest_pairs_bit_identical(seed, start):
    ps, bufs = _stream(seed, steps=100)
    d = _Drive(ps)
    a, b = _sleeper(bufs, start), _sleeper(bufs, start)
    ua = a.harvest_pairs(d, eot_h=1, eot_m=2, marks=(3, 4, 5, 6))
    ub = ref_harvest_pairs(b, d, eot_h=1, eot_m=2, marks=(3, 4, 5, 6))
    assert ua == ub
    assert a.pairs == b.pairs
    assert len(a.pairs) > 0 or start >= 97 * T   # streams do form pairs


def test_pairs_then_presses_skip_chain_identical():
    """The maybe_sleep order: pairs first, their presses skipped by
    press-pay (A69 exclusion law) — the whole chain must agree."""
    ps, bufs = _stream(7, steps=120, neg=0.08, corr=0.8)
    d = _Drive(ps)
    for start in (0, 50 * T, 119 * T):
        a, b = _sleeper(bufs, start), _sleeper(bufs, start)
        sa = a.harvest_pairs(d, eot_h=1, eot_m=2, marks=(3, 4, 5, 6))
        sb = ref_harvest_pairs(b, d, eot_h=1, eot_m=2, marks=(3, 4, 5, 6))
        a.harvest_presses(d, T, void_w=T // 8, skip=sa)
        ref_harvest_presses(b, d, T, void_w=T // 8, skip=sb)
        assert a.pairs == b.pairs and a.spans == b.spans


def test_press_lo_bound_is_sound():
    """Every press before the returned index has t < t_min, for
    arbitrary thresholds, including unaligned drive starts."""
    rng = random.Random(3)
    ps, _ = _stream(11, steps=60, S0=777)
    for _ in range(200):
        t_min = rng.randrange(0, 62 * T + 777)
        i0 = Sleeper._press_lo(ps, t_min, T)
        assert all(p["t"] < t_min for p in ps[:i0])
        assert i0 == 0 or ps[i0 - 1]["t"] < t_min


def test_window_bounded_cost_does_not_grow_with_segment():
    """The fix's whole point: presses far before the window are never
    touched. Measured as scanned-index count via _press_lo."""
    ps, bufs = _stream(5, steps=400, per_step=(2, 3))
    sl = _sleeper(bufs, 380 * T)
    i0 = Sleeper._press_lo(ps, sl.start + MIN_REPLAY - T, T)
    assert len(ps) - i0 < 25 * LANES * 4   # ~window only, not 400 steps
