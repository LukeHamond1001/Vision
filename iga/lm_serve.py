"""A65 — the live serving harness: the parenting room.

One lane, one life. The human types; the model replies; the human
presses the graded buttons when deserved. Everything rides the
FAITHFUL serving regime: the real state advances only on exact-T
chunk commits (the training regime — A54e F1 forbids off-regime
chunk lengths; lm_eval's 1-token-chunk talk mode is a toy row, not
this path), while generation samples from forwards on STATE COPIES
over the pending window, so sampling never perturbs the committed
stream.

Presses are dual events: the press TOKEN enters the stream (the
model perceives the counterparty's act) and the press enters the
drive ledger (the economy's record). Serve-time pay (A65): sleep
replays the episode spans positive presses name, negatives void —
the human funds consolidation of the associative channel.

/wipe is the act-3 context wipe: fresh state; any pending tokens
are pad-flushed as one exact-T chunk first so the absolute token
clock and the sleeper's buffer stay aligned forever.
"""

import json
from contextlib import contextmanager

import torch

from .lm_drive import Drive
from .lm_sleep import Sleeper, state_copy
from .lm_vocab import PRESS_TOKENS

SLEEP_LR = 1e-4          # A63's healthy regime (cosine tail scale)
# The centerpiece's removal switches (docs/CENTERPIECE.md). They apply
# to the READ path only — replies and score-only probes — never to a
# commit, a sleep block or a save: the committed life is written by the
# certified forward whatever the switch says, so the contrast is "same
# state, organ off", and 'none' is bit-exact to the base.
#   bands: model.lesioned = all bands (memory tokens zeroed, stores unread)
#   store: model.store_read_off (bands live, stores unread)
LESION_MODES = ("none", "bands", "store")
MAX_SLEEP_BLOCKS = 32    # R3 dose law: serve-sleep is capped


class ServeSession:
    def __init__(self, model, tok, T=2048, device="cpu",
                 sleeper=None, temperature=0.8, top_k=40,
                 max_reply=64, log_path=None, seed=0,
                 sleep_lr=SLEEP_LR, prophet=None,
                 resume_state=None):
        self.m = model.eval()
        self.tok = tok
        self.T = T
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.max_reply = max_reply
        self.drive = Drive(n_lanes=1, lam=0.02)
        self.sleeper = sleeper
        if sleeper is not None:
            sleeper.bind(self.drive)
        self.st = model.init_state(1, device)
        self.pending = []
        self.n_committed = 0
        self.pad = tok.token_to_id("<pad>")
        self.eot_h = tok.token_to_id("<eot_human>")
        self.eot_m = tok.token_to_id("<eot_model>")
        self.press_id = {v: tok.token_to_id(t) for v, t in
                         zip((1, 2, -1, -2), PRESS_TOKENS)}
        self._gen = torch.Generator().manual_seed(seed)
        self.sleep_lr = float(sleep_lr)
        self._opt = None
        self.prophet = prophet
        self.log_path = log_path
        self.events = []
        self.lesion_mode = "none"
        if resume_state is not None:
            self._restore(resume_state)   # A67: the life continues

    # ---------- the stream ----------
    @property
    def pos(self):
        return self.n_committed + len(self.pending)

    def _append(self, ids):
        self.pending.extend(int(i) for i in ids)
        self.drive.step_t = self.pos
        while len(self.pending) > self.T:
            chunk, self.pending = (self.pending[:self.T],
                                   self.pending[self.T:])
            self._commit(chunk)

    @torch.no_grad()
    def _commit(self, chunk):
        x = torch.tensor([chunk], dtype=torch.long,
                         device=self.device)
        _, self.st, _ = self.m(x, self.st, None)
        self.m.pop_write_cost()
        self.m.pop_recon()
        self.st = self.m.detach_state(self.st)
        self.n_committed += len(chunk)
        if self.sleeper is not None:
            self.sleeper.observe(x.cpu())
        if self.prophet is not None:
            # A67: the band press-predictors watch every commit
            self.m._st = self.st
            self.prophet.observe(self.m, self.drive)

    def _flush(self):
        """Commit the pending window as ONE SHORT chunk — content-
        faithful, clock- and buffer-aligned. A single off-length
        commit is the bounded, ledgered deviation; the pad-to-T
        alternative fed the bands a pad-sea (the A65 smoke showed
        2,032 pads poisoning band state at pos 16). No-op when
        pending is empty."""
        if not self.pending:
            return
        chunk, self.pending = self.pending, []
        self._commit(chunk)
        self.drive.step_t = self.pos

    @torch.no_grad()
    def _next_logits(self):
        x = torch.tensor([self.pending], dtype=torch.long,
                         device=self.device)
        with self.lesion_scope():
            logits, _, _ = self.m(x, state_copy(self.st), None)
        self.m.pop_write_cost()
        self.m.pop_recon()
        return logits[0, -1].float()

    # ---------- the removals (score-only reads; see LESION_MODES) ----------
    def lesion(self, mode):
        mode = (mode or "none").strip().lower()
        if mode not in LESION_MODES:
            raise ValueError(f"lesion mode {mode!r}: one of {LESION_MODES}")
        self.lesion_mode = mode
        self._log({"kind": "lesion", "mode": mode, "pos": self.pos})
        return mode

    @contextmanager
    def lesion_scope(self):
        """Apply the session's removal to the model for ONE read and
        restore it after — the switches never outlive a forward, so
        commits, sleep and saves always see the certified model."""
        m = self.m
        prev = (set(m.lesioned),
                bool(getattr(m, "store_read_off", False)))
        try:
            if self.lesion_mode == "bands":
                m.lesioned = set(m.bands)
            elif self.lesion_mode == "store":
                m.store_read_off = True
            yield m
        finally:
            m.lesioned, m.store_read_off = prev

    def flush(self):
        """Public short-commit of the pending window (A66-R2: the
        store cannot engage inside an uncommitted window; a session
        that wants in-session store memory commits its passes)."""
        self._flush()

    # ---------- the conversation ----------
    def user(self, text):
        self._append(self.tok.encode(text).ids + [self.eot_h])
        self._log({"kind": "user", "text": text, "pos": self.pos})

    def reply(self):
        if not self.pending:
            self._append([self.eot_h])   # boundary edge: need context
        out = []
        for _ in range(self.max_reply):
            lg = self._next_logits()
            if self.temperature <= 0:
                t = int(lg.argmax())
            else:
                lg = lg / self.temperature
                if self.top_k:
                    kth = torch.topk(lg, self.top_k).values[-1]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                t = int(torch.multinomial(
                    torch.softmax(lg, -1).cpu(), 1,
                    generator=self._gen))
            self._append([t])
            if t == self.eot_m:
                break
            out.append(t)
        text = self.tok.decode(out, skip_special_tokens=False) \
            if hasattr(self.tok, "decode") else " ".join(map(str, out))
        self._log({"kind": "model", "text": text, "pos": self.pos})
        return text

    def press(self, v):
        """The graded primary: token into the stream, event into the
        economy — at the position the token itself occupies."""
        self.drive.step_t = self.pos
        self.drive.button(0, int(v))
        self._append([self.press_id[int(v)]])
        self._log({"kind": "press", "v": int(v), "pos": self.pos})

    # ---------- act-3 machinery ----------
    def wipe(self):
        self._flush()
        self.st = self.m.init_state(1, self.device)
        self._log({"kind": "wipe", "pos": self.pos})

    def sleep_now(self, blocks=8, span_w=512, void_w=64,
                  pair_gap=192, pair_ctx=128, pair_ucap=32):
        assert self.sleeper is not None, "no sleeper attached"
        self._flush()
        sl = self.sleeper
        skip = None
        if sl.arm == "C":
            # A68: correction pairs first — consumed presses leave
            # the CE-span economy entirely (no wide span, no void).
            # Targets are whole turns (model turn for the negative,
            # human turn for the positive); press tokens bound every
            # scan and are never targets.
            skip = sl.harvest_pairs(
                self.drive, gap=pair_gap, ctx_w=pair_ctx,
                u_cap=pair_ucap, eot_h=self.eot_h,
                eot_m=self.eot_m,
                marks=tuple(self.press_id.values()))
        n = sl.harvest_presses(self.drive, span_w, void_w=void_w,
                               skip=skip)
        npairs = len(sl.pairs) if sl.arm == "C" else 0
        if not n and not npairs:
            self._log({"kind": "sleep", "spans": 0, "blocks": 0})
            return {"spans": 0, "blocks": 0, "pairs": 0}
        if self._opt is None:
            self._opt = torch.optim.AdamW(self.m.parameters(),
                                          lr=self.sleep_lr)
        done = 0
        for _ in range(min(blocks, MAX_SLEEP_BLOCKS)):
            wp = sum(p["pay"] for p in sl.pairs) if npairs else 0.0
            ws = sum(s["pay"] for s in sl.spans)
            r = None
            if wp and (not ws or sl.rng.random() < wp / (wp + ws)):
                r = sl._pair_block(self.m, self._opt, step=self.pos)
            elif ws:
                r = sl._block(self.m, self._opt, step=self.pos)
            if r is not None:
                done += 1
        a = self.sleeper.audit()
        self._log({"kind": "sleep", "spans": n, "pairs": npairs,
                   "blocks": done, "audit": a})
        return {"spans": n, "pairs": npairs, "blocks": done,
                "audit": a}

    def save(self, path):
        """A67: save the WHOLE life — weights, band/store state,
        the pending window, the press ledger, the sleeper's buffers
        and provenance, optimizer moments, prophet heads, RNG
        streams. A resumed session is the same life continuing."""
        d = self.drive
        out = {"model": self.m.state_dict(),
               "st": state_copy(self.st),
               "pending": list(self.pending),
               "n_committed": self.n_committed,
               "drive": {"presses": list(d.presses),
                         "minted": sorted(d.minted),
                         "neg_count": dict(d.neg_count),
                         "veto_until": dict(d.veto_until),
                         "step_t": d.step_t},
               "gen_state": self._gen.get_state(),
               "events_tail": self.events[-2000:]}
        if self._opt is not None:
            out["sleep_opt"] = self._opt.state_dict()
        if self.sleeper is not None:
            sl = self.sleeper
            out["sleeper"] = {"buffers": sl.buffers,
                              "start": sl.start, "T": sl.T,
                              "replayed": sl.replayed,
                              "stats": sl.stats,
                              "steps_taken": sl.steps_taken,
                              "rng": sl.rng.getstate()}
        if self.prophet is not None:
            out["prophet"] = {
                "heads": self.prophet.heads.state_dict(),
                "opt": self.prophet.opt.state_dict(),
                "stats": self.prophet.stats}
        torch.save(out, path)

    def _restore(self, rs):
        self.m.load_state_dict(rs["model"])
        self.st = rs["st"]
        self.pending = list(rs["pending"])
        self.n_committed = rs["n_committed"]
        dr = rs["drive"]
        self.drive.presses = list(dr["presses"])
        self.drive.minted = set(dr["minted"])
        self.drive.neg_count = dict(dr["neg_count"])
        self.drive.veto_until = dict(dr["veto_until"])
        self.drive.step_t = dr["step_t"]
        self._gen.set_state(rs["gen_state"])
        if "sleep_opt" in rs:
            self._opt = torch.optim.AdamW(self.m.parameters(),
                                          lr=self.sleep_lr)
            self._opt.load_state_dict(rs["sleep_opt"])
        if self.sleeper is not None and "sleeper" in rs:
            sl, ss = self.sleeper, rs["sleeper"]
            sl.buffers = ss["buffers"]
            sl.start = ss["start"]
            sl.T = ss["T"]
            sl.replayed = ss["replayed"]
            sl.stats = ss["stats"]
            sl.steps_taken = ss["steps_taken"]
            sl.rng.setstate(ss["rng"])
        if self.prophet is not None and "prophet" in rs:
            self.prophet.heads.load_state_dict(rs["prophet"]["heads"])
            self.prophet.opt.load_state_dict(rs["prophet"]["opt"])
            self.prophet.stats = rs["prophet"]["stats"]
        self._log({"kind": "resume", "pos": self.pos,
                   "presses": len(self.drive.presses)})

    def panel(self):
        spans = 0
        if self.sleeper is not None:
            spans = self.sleeper.harvest_presses(self.drive,
                                                 void_w=64)
        return {"pos": self.pos, "committed": self.n_committed,
                "lesion": self.lesion_mode,
                "presses": len(self.drive.presses),
                "live_spans": spans,
                "sleeps": sum(1 for e in self.events
                              if e["kind"] == "sleep")}

    def _log(self, ev):
        self.events.append(ev)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(ev) + "\n")
