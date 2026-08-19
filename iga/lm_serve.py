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

import torch

from .lm_drive import Drive
from .lm_sleep import Sleeper, state_copy
from .lm_vocab import PRESS_TOKENS

SLEEP_LR = 1e-4          # A63's healthy regime (cosine tail scale)
MAX_SLEEP_BLOCKS = 32    # R3 dose law: serve-sleep is capped


class ServeSession:
    def __init__(self, model, tok, T=2048, device="cpu",
                 sleeper=None, temperature=0.8, top_k=40,
                 max_reply=64, log_path=None, seed=0):
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
        self._opt = None
        self.log_path = log_path
        self.events = []

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
        logits, _, _ = self.m(x, state_copy(self.st), None)
        self.m.pop_write_cost()
        self.m.pop_recon()
        return logits[0, -1].float()

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

    def sleep_now(self, blocks=8, span_w=512):
        assert self.sleeper is not None, "no sleeper attached"
        self._flush()
        n = self.sleeper.harvest_presses(self.drive, span_w)
        if not n:
            self._log({"kind": "sleep", "spans": 0, "blocks": 0})
            return {"spans": 0, "blocks": 0}
        if self._opt is None:
            self._opt = torch.optim.AdamW(self.m.parameters(),
                                          lr=SLEEP_LR)
        done = 0
        for _ in range(min(blocks, MAX_SLEEP_BLOCKS)):
            if self.sleeper._block(self.m, self._opt,
                                   step=self.pos) is not None:
                done += 1
        a = self.sleeper.audit()
        self._log({"kind": "sleep", "spans": n, "blocks": done,
                   "audit": a})
        return {"spans": n, "blocks": done, "audit": a}

    def save(self, path):
        torch.save({"model": self.m.state_dict(),
                    "n_tokens": self.pos,
                    "presses": self.drive.presses,
                    "events": self.events[-2000:]}, path)

    def panel(self):
        spans = 0
        if self.sleeper is not None:
            spans = self.sleeper.harvest_presses(self.drive)
        return {"pos": self.pos, "committed": self.n_committed,
                "presses": len(self.drive.presses),
                "live_spans": spans,
                "sleeps": sum(1 for e in self.events
                              if e["kind"] == "sleep")}

    def _log(self, ev):
        self.events.append(ev)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(ev) + "\n")
