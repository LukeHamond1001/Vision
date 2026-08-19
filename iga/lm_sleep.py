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
                 min_step_loss=0.0, replay_twice=False):
        """every: one sleep block per this many wake steps (0 = never
        fire — the dose-0 parity arm). block_chunks: chunks per block;
        dose sleep:wake = block_chunks/every (A62 ladder {0, 1:16,
        1:8, 1:4}). A block replays ONE pay-weighted span as a
        sequential episode — ARM B's teacher needs earlier chunks
        written into the store before later reads carry any bonus."""
        assert arm in ("A", "B")
        self.arm = arm
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
        self.start = drive.step_t
        self.cap = max(drive.horizon_for(b)
                       for b in range(1, N_BANDS)) + 8192

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

    # ---------- span working set (L1's source of truth) ----------
    def harvest(self, drive):
        for i in range(self._ledger_i, len(drive.ledger)):
            e = drive.ledger[i]
            if e["pay"] > 0 and e["t1"] - e["t0"] >= MIN_REPLAY:
                self.spans.append(
                    {"lane": e["lane"], "t0": e["t0"], "t1": e["t1"],
                     "pay": float(e["pay"]), "i": i})
        self._ledger_i = len(drive.ledger)
        self.spans = [s for s in self.spans
                      if min(s["t1"], self.end)
                      - max(s["t0"], self.start)
                      >= MIN_REPLAY][-MAX_SPANS:]

    def harvest_presses(self, drive, span_w=512):
        """A65 serve-time pay — the A61 payer problem's designed
        answer, literal: with no probes at serve, holds cannot pay,
        so the PRIMARY pays. A positive press at t names the episode
        [t - span_w, t] with pay = v (magnitude -> replay weight); a
        negative press VOIDS pending spans overlapping its trailing
        window (withhold at serve). Replaces self.spans (serve
        mode); the training-time harvest() is untouched."""
        spans = []
        for i, p in enumerate(drive.presses):
            if p["v"] > 0:
                spans.append({"lane": p["lane"],
                              "t0": max(0, p["t"] - span_w),
                              "t1": p["t"], "pay": float(p["v"]),
                              "i": -(i + 1)})
            else:
                spans = [s for s in spans
                         if not (s["lane"] == p["lane"]
                                 and s["t0"] < p["t"]
                                 and p["t"] - span_w < s["t1"])]
        self.spans = [s for s in spans
                      if min(s["t1"], self.end)
                      - max(s["t0"], self.start) >= MIN_REPLAY]
        return len(self.spans)

    # ---------- the block ----------
    def maybe_sleep(self, model, opt, drive, step):
        if not self.every or step % self.every \
                or self.buffers is None:
            return None
        self.harvest(drive)
        if not self.spans:
            return None
        return self._block(model, opt, step)

    def _block(self, model, opt, step):
        span = self.rng.choices(
            self.spans, weights=[s["pay"] for s in self.spans])[0]
        lo0 = max(span["t0"], self.start)
        hi0 = min(span["t1"], self.end)
        need = self.block_chunks * self.T + 1
        if hi0 - lo0 > need:
            lo = lo0 + self.rng.randrange(hi0 - lo0 - need + 1)
            hi = lo + need
        else:
            lo, hi = lo0, hi0
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
        if self.replay_twice:
            parts = [(0, toks), (0, toks)]
        else:
            parts = []
            for off in range(0, len(toks) - 1, self.T):
                xs = toks[off: off + self.T + 1]
                if len(xs) < MIN_REPLAY and off > 0:
                    break        # trailing sliver: not worth a step
                parts.append((off, xs))
        try:
            for off, xs in parts:
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
                st = model.detach_state(st)
                losses.append(float(loss.detach()))
                self.replayed.append(
                    {"step": step, "arm": self.arm,
                     "lane": span["lane"], "lo": lo + off,
                     "hi": lo + off + len(xs),
                     "ledger_i": span["i"], "pay": span["pay"],
                     "t0": span["t0"], "t1": span["t1"]})
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
