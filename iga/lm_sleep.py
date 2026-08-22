"""A62 Phase 1 — consolidation ("sleep") on the certified substrate.

Wake is the v9.4 loop unchanged. Sleep replays ONLY paid episodes:
drive ledger entries with pay > 0 name token spans [t0, t1]
(token-denominated, A60d), sliced from a rolling per-lane buffer of
the exact tokens each lane streamed — the same mechanism serves the
synthetic weaver and tokens.bin shards, and a resume starts the
buffer at drive.step_t so ledger times stay aligned.

Mechanism arms (winner frozen after the debug A/B, A55b culture):
  ARM A  replay-training — paid spans re-fed as LM chunks with store
         reads OFF: the trunk alone must learn to predict the paid
         material (consolidation as payment-selected curriculum).
  ARM B  store->trunk distillation — trunk-alone logits trained
         toward trunk+store logits (KL) on the same spans. The
         teacher writes the span into a fresh store copy as it
         replays, so its later chunks carry the associative bonus;
         sleep funds exactly what LM loss defunds (the A61 null
         margin, predicted at A55c). In logit-store mode reads touch
         only the logits, so teacher and student hiddens — and hence
         band/store states — are identical; one shared state serves
         both passes exactly.
  ARM C  contrastive pair replay (A68) — the negative channel the
         A67-P8 laws demanded: CE-only replay cannot encode "not"
         (negation is exposure; "X not Y" poisons X's own stem). A
         negative press marks the utterance it lands on as WRONG;
         the next positive press within a gap marks the caregiver's
         correction as RIGHT; the pair replays under a bounded
         margin loss softplus(-beta*(logp_right - logp_wrong)) over
         UTTERANCE tokens only. The wrong tokens appear solely as
         suppression targets — never CE targets — and both member
         presses leave the CE-span economy entirely. With no pairs
         minted, ARM C is bit-identical to ARM A by construction.

Laws (pinned at A62, enforced here, tested in test_lm_ladder):
  L1 only-paid-replays — every replayed window lies inside a ledger
     entry's span with pay > 0; provenance kept, audit() checks it.
  L2 parity-off — train(sleep=None) runs zero sleep code; a dose-0
     Sleeper (every=0) buffers but never fires and consumes no
     torch RNG, so baseline training reproduces bit-exactly.
  L3 slow-weights-only — sleep updates trunk/band slow weights;
     store and aux head are frozen (requires_grad off -> grad stays
     None; AdamW skips them, weight decay included).
  L4 no drive pay — sleep never touches the drive: no probes, no
     sweeps, no ledger writes; telescoping untouched.

Replay runs on FRESH state (wake band/store state is never advanced
or disturbed) in eval mode — deterministic, no read-dropout or
XL-dropout draws — using the sleeper's own python RNG, so a sleep
block leaves the wake torch RNG stream untouched: dose > 0 differs
from baseline only through the weight updates. Sleep steps ride the
wake optimizer at the current wake lr (the cosine schedule
reasserts lr on the next wake step).
"""

import random
import torch

from .lm_bands import N_BANDS

MIN_REPLAY = 17          # shortest usable slice: 16 x/y pairs + 1
MAX_SPANS = 512          # working set of most-recent paid spans

# L3's frozen surface: everything that is not a trunk/band slow
# weight. mats/write_q/read_gate exist unused in logit mode; frozen
# anyway so the law holds across store modes.
FREEZE_PREFIXES = ("stores.", "alpha.", "aux_head.", "mats.",
                   "write_q.", "read_gate")
FREEZE_EXACT = {"tok_u", "qmix"}


def frozen_param_names(model):
    return [n for n, _ in model.named_parameters()
            if n.startswith(FREEZE_PREFIXES) or n in FREEZE_EXACT]


def state_copy(st):
    """Detached deep copy of a model state dict (the process_chunk
    blind-pass pattern) — the teacher pass must not advance the
    episode's real state."""
    out = {}
    for k, v in st.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.detach().clone()
                           if torch.is_tensor(vv) else vv)
                      for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.detach().clone()
        elif isinstance(v, list):
            out[k] = [t.detach().clone()
                      if torch.is_tensor(t) else t for t in v]
        else:
            out[k] = v
    return out


class SleepTap:
    """Transparent conveyor wrapper: records each wake chunk's
    tokens into the sleeper's buffer, changes nothing else."""

    def __init__(self, inner, sleeper):
        self.inner = inner
        self.sleeper = sleeper

    def chunk(self, T):
        x, y, events = self.inner.chunk(T)
        self.sleeper.observe(x)
        return x, y, events

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Sleeper:
    def __init__(self, arm="B", every=0, block_chunks=4, seed=0,
                 min_step_loss=0.0, replay_twice=False,
                 homeostasis=0.0, splice=0.0, novelty=0.0, saliency=0.0):
        """every: one sleep block per this many wake steps (0 = never
        fire — the dose-0 parity arm). block_chunks: chunks per block;
        dose sleep:wake = block_chunks/every (A62 ladder {0, 1:16,
        1:8, 1:4}). A block replays ONE pay-weighted span as a
        sequential episode — ARM B's teacher needs earlier chunks
        written into the store before later reads carry any bonus."""
        assert arm in ("A", "B", "C")
        self.arm = arm
        self.pairs = []          # ARM C working set (rebuilt each harvest)
        # A69: training-loop arm C — boundary ids for turn-scoped pair
        # targets ({"eot_h", "eot_m", "marks"}); None degrades arm C
        # to arm A in maybe_sleep (serve passes ids per-call instead).
        self.pair_tokens = None
        # A69-R5: press-targeted pay in the TRAINING night. The R4
        # null said it plainly: hold-settlement spans are diffuse
        # (band-horizon-sized) and consolidate no individual fact,
        # while the raised life's serve nights — which pay the exact
        # spans presses name — put facts into weights daily. (w, v)
        # = (span_w, void_w); None keeps the certified ledger
        # harvest bit-exactly.
        self.press_pay = None
        self.every = every
        self.block_chunks = block_chunks
        # A65: no disagreement, no update — a chunk whose loss sits
        # below this floor records provenance but takes NO optimizer
        # step (Adam normalizes noise gradients into lr-scale kicks;
        # the serve smoke showed six ~0-KL blocks denting fresh-state
        # behavior with no wake loss to re-anchor). 0.0 = training
        # path bit-exact.
        self.min_step_loss = float(min_step_loss)
        # A66-R2: serve-mode replay presents the span TWICE — the
        # teacher writes on pass one and reads its own memory on
        # pass two, so pass two's KL carries exactly the store's
        # content (a single sub-chunk replay reads an empty store
        # and distills nothing — the A66-R1 structural null). The
        # write pass self-skips under the no-disagreement floor.
        # False = training path bit-exact.
        self.replay_twice = bool(replay_twice)
        # A76 (gated): sleep homeostasis — after a block that took
        # at least one step, the UNFROZEN slow weights (exactly
        # sleep's certified weight surface, A62 L3) are multiplied
        # by (1 - homeostasis): the synaptic-downscaling analog.
        # Evidence hook: the incumbent + rich-get-richer are
        # no-renormalization diseases; mastery floors stop
        # reinforcing but never shrink. 0.0 = certified bit-exact.
        self.homeostasis = float(homeostasis)
        # A73 (gated): splice replay — with this probability a
        # span block replays ONE chunk each from TWO pay-weighted
        # spans under one carried state (cross-episode adjacency:
        # the SWS schema-abstraction analog, real tokens only).
        # CE-loss arms only (A/C); arm B ignores it. 0.0 = certified
        # bit-exact (the extra RNG draws are short-circuited).
        self.splice = float(splice)
        # A74 (gated): surprise-weighted replay — span lottery
        # weights become pay*(1-novelty) + novelty*surprise, where
        # surprise is the trainer-stamped wake CE over the span
        # (note_ce), normalized by the running max. 0.0 = certified
        # pay-only lottery bit-exactly.
        self.novelty = float(novelty)
        # Phase 2 (2026-08-22): HIPPOCAMPUS-INDEXED replay. The trainer
        # stamps each wake step's dopamine trace (|RPE| per token, the
        # organism's own write-strength gain) per lane; saliency>0 mixes
        # a span's mean |RPE| into its replay weight beside pay and
        # novelty — the night replays what the store wrote hardest.
        # saliency=0 is the certified path, bit-exact.
        self.saliency = float(saliency)
        self.chunk_dopa = []     # (lane, t0, t1, mean |RPE|) wake stamps, capped
        self._dopa_max = 1e-9
        self.chunk_ce = []       # (t0, t1, ce) wake stamps, capped
        self._ce_max = 1e-9
        self.rng = random.Random(seed)
        self.buffers = None
        self.start = 0           # absolute stream pos of buffers[*][0]
        self.cap = None
        self.T = None
        self._ledger_i = 0
        self.spans = []
        self.replayed = []       # L1 provenance, one row per chunk
        self.stats = []          # one row per block
        self.steps_taken = 0     # chunks that cleared the floor

    # ---------- wiring (lm_train touches only these) ----------
    def tap(self, conveyor):
        return SleepTap(conveyor, self)

    def bind(self, drive):
        # SEAM LAW (2026-08-21, v10 flash): the driver keeps ONE Sleeper
        # across its train() segments while every segment builds a
        # fresh Drive, and the conveyor resumes contiguously — so at a
        # seam the buffer still holds the previous segment's tail and
        # its head sits len(buffer) tokens BEFORE drive.step_t. Taking
        # start = drive.step_t shifted every replay window by the
        # buffer length (~1.06M tokens at the band-6 cap) from the
        # second segment of a process on: spans and pairs would read
        # tokens a million back from the pressed exchange. Anchor the
        # head by length; an empty buffer (every fresh bind) is the
        # old line bit-exactly.
        held = len(self.buffers[0]) if self.buffers else 0
        self.start = drive.step_t - held
        # A70: the replay-reach cap covers every band the drive knows
        # a horizon for (band 6 arrives via drive._horizons); default
        # runs see exactly the old 1..5 computation.
        bands = set(range(1, N_BANDS)) | \
            {b for b in getattr(drive, "_horizons", {}) if b >= 1}
        self.cap = max(drive.horizon_for(b) for b in bands) + 8192

    @property
    def end(self):
        return self.start + (len(self.buffers[0]) if self.buffers
                             else 0)

    def observe(self, x):
        if self.buffers is None:
            self.buffers = [[] for _ in range(x.shape[0])]
        # max, not last: a serve-time short flush chunk (A65) must
        # not shrink the replay window length
        self.T = max(self.T or 0, x.shape[1])
        for lane, row in enumerate(x.tolist()):
            self.buffers[lane].extend(row)
        if self.cap is not None and len(self.buffers[0]) > self.cap:
            cut = len(self.buffers[0]) - self.cap
            for b in self.buffers:
                del b[:cut]
            self.start += cut

    def note_ce(self, mean_ce, t0, t1):
        """A74: the trainer stamps each wake step's mean CE over its
        token range. Append-only python list, no graph, no RNG —
        stamping never perturbs the certified path; only novelty>0
        reads it. Capped to the replay-reach window."""
        self.chunk_ce.append((int(t0), int(t1), float(mean_ce)))
        self._ce_max = max(self._ce_max, float(mean_ce))
        if self.cap is not None and len(self.chunk_ce) > 4096:
            del self.chunk_ce[:len(self.chunk_ce) - 4096]

    def note_dopa(self, trace, t0, t1):
        """trace [lanes, T] |RPE| per token (ScanLM.dopa_trace), stamped
        per lane over the step's token range; only saliency>0 reads it."""
        if trace is None:
            return
        means = trace.float().mean(dim=1).tolist()
        for lane, v in enumerate(means):
            self.chunk_dopa.append((lane, int(t0), int(t1), float(v)))
            self._dopa_max = max(self._dopa_max, float(v))
        if len(self.chunk_dopa) > 4096 * max(1, len(means)):
            del self.chunk_dopa[:len(self.chunk_dopa) - 4096 * len(means)]

    def _saliency(self, s):
        """Mean stamped |RPE| of the span's lane overlapping it, in [0,1]."""
        acc, n = 0.0, 0
        for lane, t0, t1, v in self.chunk_dopa:
            if lane == s["lane"] and t1 > s["t0"] and t0 < s["t1"]:
                acc += v
                n += 1
        return (acc / n / self._dopa_max) if (n and self._dopa_max > 0) else 0.0

    def _surprise(self, s):
        """Mean stamped CE overlapping span s, normalized to [0,1]."""
        acc, n = 0.0, 0
        for t0, t1, ce in self.chunk_ce:
            if t1 > s["t0"] and t0 < s["t1"]:
                acc += ce
                n += 1
        return (acc / n / self._ce_max) if n else 0.0

    def _span_weights(self):
        if not self.novelty and not self.saliency:
            return [s["pay"] for s in self.spans]
        nv, sa = self.novelty, self.saliency
        return [s["pay"] * (1 - nv - sa) + nv * self._surprise(s) + sa * self._saliency(s)
                for s in self.spans]

    def _downscale(self, model):
        """A76: one multiplicative downscale of the unfrozen slow
        weights — sleep's certified surface only (A62 L3)."""
        if not self.homeostasis:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.startswith(FREEZE_PREFIXES) or n in FREEZE_EXACT:
                    continue
                p.mul_(1.0 - self.homeostasis)

    # ---------- span working set (L1's source of truth) ----------
    def harvest(self, drive):
        # v10: base-aware — a pruned ledger (ledger_cap) drops old
        # rows and advances ledger_base; absolute indices stay valid
        base = getattr(drive, "ledger_base", 0)
        if base > self._ledger_i:
            # rows pruned before this sleeper ever harvested them —
            # the cap is undersized for the harvest cadence. Counted,
            # never silent (heartbeat watches this).
            self.pruned_unharvested = getattr(
                self, "pruned_unharvested", 0) + (base - self._ledger_i)
        for i in range(max(self._ledger_i, base),
                       base + len(drive.ledger)):
            e = drive.ledger[i - base]
            if e["pay"] > 0 and e["t1"] - e["t0"] >= MIN_REPLAY:
                self.spans.append(
                    {"lane": e["lane"], "t0": e["t0"], "t1": e["t1"],
                     "pay": float(e["pay"]), "i": i})
        self._ledger_i = base + len(drive.ledger)
        self.spans = [s for s in self.spans
                      if min(s["t1"], self.end)
                      - max(s["t0"], self.start)
                      >= MIN_REPLAY][-MAX_SPANS:]

    @staticmethod
    def _press_lo(ps, t_min, T):
        """Smallest index i0 such that EVERY press before i0 has
        t < t_min. Presses are dispatched step by step: each step's
        presses occupy [step_t, step_t + T) (lanes concatenated, so
        the list is t-sorted only at step granularity) and steps
        advance by T. Walking back from the end, the first press
        with t < t_min - T closes the question: anything earlier
        sits in the same step (t < its step start + T <= that
        press's t + T < t_min) or an earlier one (t < that step
        start <= that press's t). O(window), exact, no assumption
        beyond step-ordered dispatch (serve flush chunks are shorter
        than T, never longer)."""
        j = len(ps) - 1
        lim = t_min - T
        while j >= 0 and ps[j]["t"] >= lim:
            j -= 1
        return j + 1

    def harvest_presses(self, drive, span_w=512, void_w=None,
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
        wide CE span nor void (their whole effect is the pair).
        v10 FLASH SLOWDOWN FIX (2026-08-21): the scan used to start
        at press 0 and keep every positive span of the segment
        alive until the final window filter, so each negative press
        re-filtered a list growing 20/step — cost ~10 x steps^2 per
        sleep, ~30 s by step 5000 of a segment (throughput 16k ->
        8.8k tok/s). Now the scan starts at the first press that
        can still touch the buffer window and voids per lane. A
        span survives the final filter only if t1 >= start +
        MIN_REPLAY, and a negative can only void a span minted by
        an EARLIER press (t < its own t + T), so nothing before
        t_min = start + MIN_REPLAY - T can change the result; the
        void test is per-span independent, so dropping doomed
        spans early changes nothing. Output is bit-identical
        (law test: tests/test_lm_sleep_harvest.py)."""
        if void_w is None:
            void_w = span_w
        ps = drive.presses
        T = self.T or span_w
        by_lane = {}
        for i in range(self._press_lo(ps, self.start + MIN_REPLAY - T, T),
                       len(ps)):
            p = ps[i]
            if skip and i in skip:
                continue
            if p["v"] > 0:
                # A67: t1 = press position + 1 — the press TOKEN is
                # the span's final CE target, so replay also teaches
                # PREDICTING the press ("this exchange earns <+2>"):
                # approval understanding from real presses only
                by_lane.setdefault(p["lane"], []).append(
                    {"lane": p["lane"],
                     "t0": max(0, p["t"] - span_w),
                     "t1": p["t"] + 1, "pay": float(p["v"]),
                     "i": -(i + 1)})
            else:
                lane_spans = by_lane.get(p["lane"])
                if lane_spans:
                    t, lo = p["t"], p["t"] - void_w
                    by_lane[p["lane"]] = [s for s in lane_spans
                                          if not (s["t0"] < t
                                                  and lo < s["t1"])]
        spans = [s for ls in by_lane.values() for s in ls]
        spans.sort(key=lambda s: -s["i"])     # dispatch order, as before
        self.spans = [s for s in spans
                      if min(s["t1"], self.end)
                      - max(s["t0"], self.start) >= MIN_REPLAY]
        return len(self.spans)

    # ---------- ARM C: correction pairs (A68) ----------
    def _turn(self, lane, t, end_tok, open_toks, u_cap):
        """[t0, t1) — the last turn ENDING with end_tok before press
        position t (the A68-R2 fix: a press follows the model's ack,
        so the token touching the press is the wrong scope; the
        target is a whole TURN, parsed by boundaries). The turn ends
        at the nearest end_tok within u_cap back of t and opens
        after the nearest earlier boundary in open_toks (the other
        side's eot and every press token — press marks are never
        targets), length-capped at u_cap. Returns None if no end_tok
        lies within reach."""
        buf = self.buffers[lane]
        lo_lim = max(self.start, t - u_cap)
        e = None
        for k in range(t - 1, lo_lim - 1, -1):
            if buf[k - self.start] == end_tok:
                e = k
                break
        if e is None:
            return None
        h = max(self.start, e - u_cap) - 1
        for k in range(e - 1, max(self.start, e - u_cap) - 1, -1):
            if buf[k - self.start] in open_toks:
                h = k
                break
        t0, t1 = h + 1, e + 1
        return (t0, t1) if t1 - t0 >= 2 else None

    def harvest_pairs(self, drive, gap=192, ctx_w=128, u_cap=48,
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
        # v10 slowdown fix: a negative at tw <= start can never pair
        # (the window test below breaks on it), so the scan starts at
        # the first press that can lie inside the window — exact.
        for i in range(self._press_lo(ps, self.start + 1,
                                      self.T or u_cap), len(ps)):
            p = ps[i]
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

    def _pair_block(self, model, opt, step, beta=1.0, pick=None):
        """One contrastive pair replay: each side forwarded fresh-
        state trunk-alone; loss = softplus(-beta * (mean logp of
        RIGHT targets - mean logp of WRONG targets)). Bounded and
        self-limiting: a mastered margin falls under min_step_loss
        and stops stepping, the CE mastery floor's twin."""
        if not self.pairs:
            return None
        pr = pick if pick is not None else self.rng.choices(
            self.pairs, weights=[p["pay"] for p in self.pairs])[0]
        device = next(model.parameters()).device
        sides = []
        for t0, t1 in ((pr["w0"], pr["w1"]), (pr["r0"], pr["r1"])):
            # window never exceeds the observed chunk width (a debug
            # model's max_T) — context yields before targets do
            lo = max(self.start, t0 - pr["ctx_w"],
                     t1 - (self.T or (t1 - self.start)))
            toks = self.buffers[pr["lane"]][lo - self.start:
                                            t1 - self.start]
            a = max(0, t0 - lo - 1)
            b = t1 - lo - 1
            if len(toks) < 2 or b <= a:
                return None
            sides.append((lo, t0, t1, a, b, toks))
        saved = [(p_, p_.requires_grad)
                 for n, p_ in model.named_parameters()
                 if n.startswith(FREEZE_PREFIXES)
                 or n in FREEZE_EXACT]
        for p_, _ in saved:
            p_.requires_grad_(False)
        was_training = model.training
        model.eval()
        try:
            lps, widths = [], []
            with torch.enable_grad():
                for lo, t0, t1, a, b, toks in sides:
                    x = torch.tensor([toks[:-1]], dtype=torch.long,
                                     device=device)
                    y = torch.tensor([toks[1:]], dtype=torch.long,
                                     device=device)
                    model.store_read_off = True
                    slg, _, _ = model(x, model.init_state(1, device),
                                      None)
                    model.pop_write_cost()
                    model.pop_recon()
                    lsm = torch.log_softmax(slg.float(), -1)
                    sel = lsm[0, a:b, :].gather(
                        -1, y[0, a:b].unsqueeze(-1)).squeeze(-1)
                    lps.append(sel.mean())
                    widths.append(b - a)
                margin = lps[1] - lps[0]
                loss = torch.nn.functional.softplus(-beta * margin)
                if abs(float(loss.detach())) >= self.min_step_loss:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0)
                    opt.step()
                    self.steps_taken += 1
                    self._downscale(model)     # A76
        finally:
            model.store_read_off = False
            for p_, rg in saved:
                p_.requires_grad_(rg)
            if was_training:
                model.train()
            opt.zero_grad()
        for (lo, t0, t1, a, b, toks), li in zip(
                sides, (pr["iw"], pr["ir"])):
            self.replayed.append(
                {"step": step, "arm": "C", "lane": pr["lane"],
                 "lo": lo, "hi": t1, "ledger_i": li,
                 "pay": pr["pay"], "t0": lo, "t1": t1})
        row = {"step": step, "arm": "C", "lane": pr["lane"],
               "pair": (pr["tw"], pr["tr"]),
               "targets": tuple(widths),
               "margin": round(float(margin.detach()), 4),
               "loss": round(float(loss.detach()), 4),
               "pay": round(pr["pay"], 4)}
        self.stats.append(row)
        print(f"    sleep@{step} armC pair[{pr['tw']}|{pr['tr']}] "
              f"targets {list(widths)}  margin {row['margin']:.4f}  "
              f"loss {row['loss']:.4f}", flush=True)
        return row

    # ---------- the block ----------
    def maybe_sleep(self, model, opt, drive, step):
        if not self.every or step % self.every \
                or self.buffers is None:
            return None
        # A69: correction pairs harvest FIRST so press-pay can skip
        # their member presses (the ARM C exclusion law). Arms A/B
        # never enter these branches and draw no extra RNG — L2
        # parity holds.
        skip = None
        if self.arm == "C" and self.pair_tokens:
            pt = self.pair_tokens
            skip = self.harvest_pairs(drive, eot_h=pt["eot_h"],
                                      eot_m=pt["eot_m"],
                                      marks=pt.get("marks", ()))
        if self.press_pay is not None:
            self.harvest_presses(drive, self.press_pay[0],
                                 void_w=self.press_pay[1],
                                 skip=skip)
        else:
            # certified ledger harvest; pairs coexist with ledger
            # spans (wake CE already trains on every token — the
            # pair adds the contrastive signal CE cannot express)
            self.harvest(drive)
        if self.arm == "C" and self.pairs:
            hot = [p for p in self.pairs if p.get("hot")]
            if hot:
                # A72: guaranteed — no lottery draw for a hot pair
                return self._pair_block(
                    model, opt, step,
                    pick=max(hot, key=lambda p: p["pay"]))
            wp = sum(p["pay"] for p in self.pairs)
            ws = sum(s["pay"] for s in self.spans)
            if not ws or self.rng.random() < wp / (wp + ws):
                return self._pair_block(model, opt, step)
        if not self.spans:
            return None
        return self._block(model, opt, step)

    def _window(self, span, need):
        lo0 = max(span["t0"], self.start)
        hi0 = min(span["t1"], self.end)
        if hi0 - lo0 > need:
            lo = lo0 + self.rng.randrange(hi0 - lo0 - need + 1)
            return lo, lo + need
        return lo0, hi0

    def _block(self, model, opt, step):
        span = self.rng.choices(
            self.spans, weights=self._span_weights())[0]
        # A73 (gated): splice — one chunk each from TWO spans under
        # one carried state (cross-episode adjacency, real tokens
        # only). CE arms only; arm B keeps its certified teacher.
        splice_span = None
        if self.splice and self.arm != "B" and len(self.spans) > 1                 and self.rng.random() < self.splice:
            others = [s_ for s_ in self.spans if s_ is not span]
            splice_span = self.rng.choices(
                others, weights=[s_["pay"] for s_ in others])[0]
        if splice_span is not None:
            need = self.T + 1
            lo, hi = self._window(span, need)
            lo2, hi2 = self._window(splice_span, need)
            toks = self.buffers[span["lane"]][lo - self.start:
                                              hi - self.start]
            toks2 = self.buffers[splice_span["lane"]][
                lo2 - self.start: hi2 - self.start]
            if len(toks) < MIN_REPLAY or len(toks2) < MIN_REPLAY:
                return None
        else:
            need = self.block_chunks * self.T + 1
            lo, hi = self._window(span, need)
            toks = self.buffers[span["lane"]][lo - self.start:
                                              hi - self.start]
            if len(toks) < MIN_REPLAY:
                return None
        device = next(model.parameters()).device
        saved = [(p, p.requires_grad)
                 for n, p in model.named_parameters()
                 if n.startswith(FREEZE_PREFIXES)
                 or n in FREEZE_EXACT]
        for p, _ in saved:
            p.requires_grad_(False)
        was_training = model.training
        model.eval()
        st = model.init_state(1, device)
        losses = []
        # every part carries (abs_lo, tokens, source_span) so
        # provenance and the only-paid audit stay exact under splice
        if splice_span is not None:
            parts = [(lo, toks, span), (lo2, toks2, splice_span)]
        elif self.replay_twice:
            parts = [(lo, toks, span), (lo, toks, span)]
        else:
            parts = []
            for off in range(0, len(toks) - 1, self.T):
                xs = toks[off: off + self.T + 1]
                if len(xs) < MIN_REPLAY and off > 0:
                    break        # trailing sliver: not worth a step
                parts.append((lo + off, xs, span))
        try:
            for plo, xs, psp in parts:
                x = torch.tensor([xs[:-1]], dtype=torch.long,
                                 device=device)
                y = torch.tensor([xs[1:]], dtype=torch.long,
                                 device=device)
                tlg = None
                if self.arm == "B":
                    model.store_read_off = False
                    with torch.no_grad():
                        tlg, _, _ = model(x, state_copy(st), None)
                    model.pop_write_cost()
                    model.pop_recon()
                model.store_read_off = True
                with torch.enable_grad():
                    slg, st, _ = model(x, st, None)
                    model.pop_write_cost()
                    model.pop_recon()
                    V = model.vocab_size
                    if self.arm == "B":
                        loss = torch.nn.functional.kl_div(
                            torch.log_softmax(
                                slg.float().reshape(-1, V), -1),
                            torch.log_softmax(
                                tlg.float().reshape(-1, V), -1),
                            log_target=True, reduction="batchmean")
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            slg.float().reshape(-1, V),
                            y.reshape(-1))
                    if abs(float(loss.detach())) >= self.min_step_loss:
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0)
                        opt.step()
                        self.steps_taken += 1
                        self._downscale(model)     # A76
                st = model.detach_state(st)
                losses.append(float(loss.detach()))
                self.replayed.append(
                    {"step": step, "arm": self.arm,
                     "lane": psp["lane"], "lo": plo,
                     "hi": plo + len(xs),
                     "ledger_i": psp["i"], "pay": psp["pay"],
                     "t0": psp["t0"], "t1": psp["t1"]})
        finally:
            model.store_read_off = False
            for p, rg in saved:
                p.requires_grad_(rg)
            if was_training:
                model.train()
            opt.zero_grad()
        row = {"step": step, "arm": self.arm, "lane": span["lane"],
               "span": (span["t0"], span["t1"]),
               "window": (lo, hi), "pay": round(span["pay"], 4),
               "chunks": len(losses),
               "splice": (splice_span["t0"], splice_span["t1"])
               if splice_span is not None else None,
               "loss": round(sum(losses) / max(len(losses), 1), 4)}
        self.stats.append(row)
        print(f"    sleep@{step} arm{self.arm} lane{span['lane']} "
              f"[{lo},{hi}) of span[{span['t0']},{span['t1']}] "
              f"pay {span['pay']:.3f}  chunks {len(losses)}  "
              f"loss {row['loss']:.4f}", flush=True)
        return row

    # ---------- L1 audit ----------
    def audit(self):
        only_paid = all(r["pay"] > 0 and r["lo"] >= r["t0"]
                        and r["hi"] <= r["t1"]
                        for r in self.replayed)
        return {"replayed": len(self.replayed),
                "blocks": len(self.stats), "only_paid": only_paid,
                "steps_taken": self.steps_taken}
