"""THE ORGANISM — the one-token creature's living serve.

One mode, one stream: the mouth is the model's alone (sampling,
a press-token speech ban, and pause caps are the only physiology
between logits and screen). The serve never authors, edits, or
filters a word, and never reads the human's text for meaning —
what is worth keeping is decided by the model's own surprise,
what deserves pride by its own conscience, when to sleep, chew,
or speak first by its own drives. Every threshold, budget,
schedule, and reflex (the disclosed genome) is a number, shown
on screen as it acts.

Run: python3 scripts/organism.py data/organism_life.pt \
         data/ship_tok.json --dev mps --save data/organism_life.pt
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan            # noqa: E402
from scripts.scan_chat import _lane0, _to_dev        # noqa: E402
from scripts.scan_nursery import content_ids         # noqa: E402

LOCK = threading.Lock()


class Organism:
    def __init__(self, a):
        from tokenizers import Tokenizer
        self.a = a
        self.tok = Tokenizer.from_file(a.tok)
        self.dev = a.dev if (a.dev != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.m, state = load_scan(a.ckpt, self.tok, self.dev)
        self.state_meta = state
        t = self.tok.token_to_id
        self.eh, self.em, self.sil = t("<eot_human>"), t("<eot_model>"), t("<pad>")
        self.press_ids = {s: t(s) for s in ("<+1>", "<+2>", "<-1>", "<-2>")}
        if hasattr(self.m, "set_reward_tokens"):
            self.m.set_reward_tokens({self.press_ids[s]: l for s, l in
                                      (("<+1>", 1), ("<+2>", 2), ("<-1>", 3), ("<-2>", 4))})
        if hasattr(self.m, "set_eot_ids"):
            self.m.set_eot_ids(self.eh, self.em)
        self.m = self.m.eval()
        src = state.get("st_live") or state.get("st")
        self.st = _to_dev(src if state.get("st_live") else _lane0(src), self.dev) \
            if src is not None else self.m.init_state(1, self.dev)
        self.opt = torch.optim.Adam(self.m.parameters(), lr=a.live_lr)
        self.gen = torch.Generator(device="cpu").manual_seed(a.seed)
        self.day_buf = []
        self.last_q = None
        self.n_steps = 0
        self.facts = []          # [(q, taught_answer)] — the report card
        self.session = []        # [(q, reply)] — the conversation itself
        self.recent_ids = []     # last replies' tokens — anti-attractor
        self.self_noticed = []   # statements IT chose to keep (curiosity)
        self.notice_budget = 4   # per day; waking refills it
        self.surp_mu = None      # running mean statement surprise (serve NE)
        self.progress = {}       # its own learning ledger, updated each night
        self.study = []          # noticed items still being learned
        self.last_card = []      # last night's report card — review source
        self.pursuit = None      # a self-adopted multi-night goal (49uu)
        self.pursuit_installment = False
        life = state.get("life") or {}
        self.facts = [tuple(x) for x in life.get("facts", [])]
        self.study = list(life.get("study", []))
        self.progress = dict(life.get("progress", {}))
        self.surp_mu = life.get("surp_mu")
        self.pursuit = life.get("pursuit")
        self.pursuit_installment = bool(life.get("pursuit_installment"))
        self.press_log = list(life.get("press_log", []))
        self.n_human_presses = max(
            int(life.get("n_human_presses", 0)),
            len([e for e in self.press_log
                 if "mag" in e and not e.get("self")]))
        self.pride_today = 0
        self.mood = 0.0          # decaying tally of reward-channel events
        # THE OPEN DOOR: it may press its own button — gated by its
        # own conscience (retrained nightly on the human's presses),
        # budgeted (satiation), felt-only: pride never edits weights.
        self.self_press_budget = 4
        self.self_frown_budget = 3
        self._self_pressed_qs = set()
        self.notice_peak_dyn = life.get("notice_peak_dyn")
        self._budget_history = list(life.get("budget_history", []))
        # AUTONOMOUS LIFE (49xx): drives — fatigue presses toward
        # sleep, boredom toward rumination, loneliness toward speaking
        # first. Numbers and thresholds only; every word it ever says
        # still comes from its own weights.
        self.fatigue = 0.0
        self.last_user_t = time.time()
        self.last_novel_t = time.time()
        self.ruminate_budget = 6
        self.initiate_budget = 2
        self.outbox = []
        # THE JOURNAL: the autobiography — one entry per lived day,
        # raw words only (what you said, what it said); recall feeds
        # them back into state, the mouth stays its own.
        self.day_n = int(life.get("day_n", 0))   # night counter (clock)
        self._today = {"taught": [], "noticed": [], "presses": 0.0}
        # SALIENCE TAGS (49zz): every memory carries the surprise and
        # the mood of the moment it was lived — dreams are picked by
        # how much it mattered, not by list position.
        self.saliences = dict(life.get("saliences", {}))
        self.critic = None
        try:
            import torch.nn as _nn
            ck = torch.load("data/critic.pt", map_location="cpu",
                            weights_only=False)
            self.critic = _nn.Sequential(
                _nn.Linear(ck["dim"], 64), _nn.ReLU(), _nn.Linear(64, 1))
            self.critic.load_state_dict(ck["sd"])
            self.critic.eval()
            print("[organism] conscience loaded (critic v0)",
                  file=sys.stderr)
        except Exception:
            pass
        self._fill_goal_slots()
        if life:
            print(f"[organism] resumed a life: {len(self.facts)} taught, "
                  f"{len(self.study)} studying, "
                  f"{len(self.progress)} ledger entries", file=sys.stderr)
    # -- state introspection --------------------------------------
    def _flat(self, s, prefix=""):
        out = {}
        if torch.is_tensor(s):
            out[prefix] = float(s.float().norm())
        elif isinstance(s, dict):
            for k, v in s.items():
                out.update(self._flat(v, f"{prefix}/{k}" if prefix else str(k)))
        elif isinstance(s, (list, tuple)):
            for i, v in enumerate(s):
                out.update(self._flat(v, f"{prefix}[{i}]"))
        return out

    def _clone(self, s):
        def d(x):
            if torch.is_tensor(x):
                return x.detach().clone()
            if isinstance(x, dict):
                return {k: d(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                t = [d(v) for v in x]
                return tuple(t) if isinstance(x, tuple) else t
            return x
        return d(s)

    def feed(self, ids):
        self.day_buf.extend(ids)
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                lg, self.st, _ = self.m(
                    torch.tensor([ids[i:i + 64]], device=self.dev), self.st)
        return lg

    def feed_ce(self, ids):
        """feed through the live state AND read the surprise: mean CE of
        ids[1:] under the stream — the serve-level NE signal (the same
        law the store's write_surprise gate uses, applied at serve)."""
        self.day_buf.extend(ids)
        tot, n, mx, lg = 0.0, 0, 0.0, None
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                chunk = ids[i:i + 64]
                lg, self.st, _ = self.m(
                    torch.tensor([chunk], device=self.dev), self.st)
                if len(chunk) > 1:
                    ce = F.cross_entropy(
                        lg[0, :-1].float(),
                        torch.tensor(chunk[1:], device=self.dev),
                        reduction="none")
                    tot += float(ce.sum())
                    mx = max(mx, float(ce.max()))
                    n += len(chunk) - 1
        return lg, tot / max(1, n), mx

    def absorb_stmt(self, text, k=1):
        """one tiny keep-nudge on a statement it chose to notice; the
        night's replay does the real consolidation."""
        ids = self.tok.encode(text).ids + [self.eh]
        ids += [self.sil] * ((64 - len(ids) % 64) % 64)
        x = torch.tensor([ids[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        w = torch.zeros_like(y, dtype=torch.float)
        w[0, :len(self.tok.encode(text).ids) - 1] = 1.0
        self.m.train()
        for _ in range(k):
            self.opt.zero_grad(set_to_none=True)
            st_d = self.m.init_state(1, self.dev)
            tot = None
            for i in range(0, x.shape[1], 64):
                lg, st_d, _ = self.m(x[:, i:i + 64], st_d)
                ce = F.cross_entropy(lg[0], y[0, i:i + 64], reduction="none")
                p = (ce * w[0, i:i + 64]).sum()
                tot = p if tot is None else tot + p
            (tot / w.sum().clamp_min(1.0)).backward()
            self.opt.step()
            self.n_steps += 1
        self.m.eval()

    def _fill_goal_slots(self):
        """write the adopted pursuit into the model's goal organ
        (st['G']) as content vectors; zero when no pursuit."""
        try:
            G = self.st.get("G") if isinstance(self.st, dict) else None
        except Exception:
            G = None
        if G is None:
            return
        G.zero_()
        if not self.pursuit:
            return
        E = self.m.embed.weight.detach().float()
        for i, q_ in enumerate(self.pursuit["items"][:3]):
            q_ = q_[2:] if q_.startswith("~ ") else q_
            ids = content_ids(self.tok, q_) or self.tok.encode(q_).ids
            v = torch.nn.functional.normalize(E[ids].mean(0), dim=-1)
            G[0, i] = v.to(G.device, G.dtype)

    def _recalibrate_conscience(self, real):
        import torch.nn.functional as _F
        E = self.m.embed.weight.detach().float().cpu()
        def vec(t_):
            ids = content_ids(self.tok, t_) or self.tok.encode(t_).ids
            return _F.normalize(E[ids].mean(0), dim=-1)
        X = torch.stack([vec(e["q"] + " " + e["a"]) for e in real])
        Y = torch.tensor([1.0 if e["mag"] > 0 else 0.0 for e in real])
        opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.critic.train()
        for _ in range(150):
            opt.zero_grad()
            loss = _F.binary_cross_entropy_with_logits(
                self.critic(X).squeeze(-1), Y)
            loss.backward()
            opt.step()
        self.critic.eval()
        torch.save({"sd": self.critic.state_dict(), "dim": X.shape[1]},
                   "data/critic.pt")
        return ("conscience recalibrated on the last %d of your %d "
                "judgments" % (len(real), self.n_human_presses))

    def _critic_score(self, text, reply):
        if self.critic is None or not reply:
            return None
        E = self.m.embed.weight.detach().float().cpu()
        ids = content_ids(self.tok, text + " " + reply)             or self.tok.encode(reply).ids
        v = torch.nn.functional.normalize(E[ids].mean(0), dim=-1)
        with torch.no_grad():
            return float(torch.sigmoid(self.critic(v)).item())

    def chat(self, text, temp=None):
        temp = temp or self.a.temp
        self.last_user_t = time.time()
        # MOOD FEEDBACK (49xx): the felt tally retunes the machinery —
        # a good stretch broadens (warmer sampling, lower curiosity
        # bar), a bad stretch conserves. Bounded, disclosed. Mood was
        # a gauge; now it is a cause.
        mood_n = max(-1.0, min(1.0, self.mood / 6.0))
        temp = max(0.02, temp * (1.0 + 0.35 * mood_n))
        mood_fx = {"temp": round(temp, 3),
                   "bar_shift": round(-0.6 * mood_n, 2)} \
            if abs(mood_n) > 0.05 else None
        pre = self._flat(self.st)
        # 50c: the serve no longer reads the human's words at all —
        # no question detector, no greeting list, no name pattern,
        # no routed lessons. What is worth keeping is decided by
        # ITS OWN surprise; what deserves pride by ITS OWN
        # conscience; lessons arrive only through the explicit
        # teach organ. The stream is just lived.
        lg, surp, surp_pk = self.feed_ce(
            self.tok.encode(text).ids + [self.eh])
        v0 = lg[0, -1].float()
        if hasattr(self.m, "ban_presses"):
            v0 = self.m.ban_presses(v0)
        p0 = torch.softmax(v0, -1)
        ent = float(-(p0 * (p0 + 1e-9).log()).sum())
        out, pauses = [], 0
        x = None
        tr_raw = []
        with torch.no_grad():
            for _ in range(self.a.max_new + 8):
                if x is not None:
                    lg, self.st, _ = self.m(x, self.st)
                v = lg[0, -1].float()
                if hasattr(self.m, "ban_presses"):
                    v = self.m.ban_presses(v)
                n_c = len([t_ for t_ in out if t_ != self.sil])
                if pauses >= 6:
                    v[self.sil] = float("-inf")
                pr = torch.softmax(v / temp, -1).cpu()
                nxt = int(torch.multinomial(pr, 1, generator=self.gen))
                if len(tr_raw) < 64:
                    tk_ = torch.topk(pr, 3)
                    tr_raw.append((nxt, float(pr[nxt]),
                                   [(int(i_), float(p_)) for p_, i_ in
                                    zip(tk_.values, tk_.indices)]))
                out.append(nxt)
                if nxt == self.sil:
                    pauses += 1
                if nxt == self.em or n_c + 1 >= self.a.max_new:
                    break
                x = torch.tensor([[nxt]], device=self.dev)
        self.day_buf.extend(out)
        self.fatigue += 0.15 + len(out) / 80.0
        # the final sampled token was never run through the model —
        # feed the unfed tail so live state holds the whole turn
        tail = [out[-1]] if out and out[-1] == self.em else \
            (out[-1:] + [self.em] if out else [self.em])
        with torch.no_grad():
            _, self.st, _ = self.m(
                torch.tensor([tail], device=self.dev), self.st)
        if not out or out[-1] != self.em:
            self.day_buf.append(self.em)
        press_vals = {v_: k_ for k_, v_ in self.press_ids.items()}
        reply = self.tok.decode(
            [t_ for t_ in out if t_ not in (self.sil, self.em)
             and t_ not in press_vals]).strip()
        trace = [{"t": self._tok_word(n_), "p": round(p_, 3),
                  "alt": [[self._tok_word(i_), round(pp_, 3)]
                          for i_, pp_ in alt_]}
                 for n_, p_, alt_ in tr_raw]
        post = self._flat(self.st)
        moved = sorted(((k, abs(post[k] - pre.get(k, 0.0)))
                        for k in post), key=lambda kv: -kv[1])[:8]
        hpc = self.store_view(text, out)
        # CURIOSITY (self-triggered plasticity, serve-life v0): a
        # statement that surprises the stream beyond its running mean
        # gets kept — one tiny nudge now, a dream tonight. Budgeted,
        # disclosed, statements only, never its own words.
        noticed = None
        if True:   # 50c: no form gate — its own surprise is the judge
            mu = self.surp_mu if self.surp_mu is not None else surp
            spike = (surp > mu + self.a.notice_margin - 0.3 * mood_n
                     and surp > self.a.notice_floor)
            pk_thr = (self.notice_peak_dyn or self.a.notice_peak) \
                - 0.6 * mood_n
            peak = surp_pk > pk_thr
            if (spike or peak) and self.notice_budget > 0:
                k_dose = 2 if surp_pk > pk_thr + 0.5 else 1
                self.absorb_stmt(text, k=k_dose)
                self.self_noticed.append(text)
                if text not in self.study:
                    self.study.append(text)
                self.study = self.study[-8:]
                self.notice_budget -= 1
                self.mood = self.mood * 0.95 + 0.3
                self.last_novel_t = time.time()
                self._today["noticed"].append(text[:80])
                self.saliences["~ " + text] = {
                    "surp": round(surp_pk, 1), "mood": round(self.mood, 1)}
                noticed = {"surprise": round(surp, 2),
                           "peak": round(surp_pk, 1),
                           "over_mean": round(surp - mu, 2),
                           "dose": k_dose,
                           "budget": self.notice_budget}
            self.surp_mu = surp if self.surp_mu is None \
                else 0.9 * self.surp_mu + 0.1 * surp
        # THE OPEN DOOR v2 (50c): no oracle, no matcher — its OWN
        # conscience (retrained nightly on the human's real presses) is
        # the only judge. And because a conscience without an oracle
        # must not rewrite knowledge it merely likes, self-reward is
        # FELT ONLY: a real press token, mood, value learning at night
        # — zero self-absorb. The human's presses remain the sole
        # plasticity channel. Budgeted, deduped, disclosed.
        pride = None
        self_press = None
        sc = self._critic_score(text, reply) if reply else None
        key_sp = text.strip().lower()[:60]
        if sc is not None and key_sp not in self._self_pressed_qs:
            if sc > 0.95 and self.self_press_budget > 0:
                self.feed([self.press_ids["<+1>"]])
                self.mood = self.mood * 0.9 + 1.0
                self.self_press_budget -= 1
                self._self_pressed_qs.add(key_sp)
                if self.pride_today < 3:
                    self.pride_today += 1
                    pride = round(sc, 2)
                self.press_log.append({"q": text[:80], "a": reply[:80],
                                       "mag": 1.0, "self": True})
                self_press = {"mag": 1, "conscience": round(sc, 2),
                              "left_today": self.self_press_budget}
            elif sc < 0.15 and self.self_frown_budget > 0:
                self.feed([self.press_ids["<-1>"]])
                self.mood = self.mood * 0.9 - 1.0
                self.self_frown_budget -= 1
                self._self_pressed_qs.add(key_sp)
                self.press_log.append({"q": text[:80], "a": reply[:80],
                                       "mag": -1.0, "self": True})
                self_press = {"mag": -1,
                              "left_today": self.self_frown_budget}
        self.last_q = (text, reply)
        if reply:
            self.session.append((text, reply))
        return {"reply": reply, "pauses": pauses,
                "surprise": round(surp, 2),
                "surprise_peak": round(surp_pk, 1), "noticed": noticed,
                "pride": pride, "self_press": self_press,
                "mood_fx": mood_fx,
                "drives": self._drives(),
                "mood": round(self.mood, 2),
                "value": {k_: round(v_, 2) for k_, v_ in
                          (self.m.read_value(self.st) or {}).items()
                          } if hasattr(self.m, "read_value") else None,
                "trace": trace,
                "moved": [{"part": k, "delta": round(d, 3)} for k, d in moved],
                "hpc": hpc}

    def store_view(self, q_text, out_ids):
        """store's vote on the reply: max |on-off| logit delta position,
        with top-3 suggestions there."""
        ans = [t_ for t_ in out_ids if t_ not in (self.sil,)]
        if not ans:
            return {}
        ids = self.tok.encode(q_text).ids + [self.eh] + ans
        ids = ids + [self.sil] * ((64 - len(ids) % 64) % 64)

        def run(off):
            self.m.store_read_off = off
            st = self._clone(self.st)
            outs = []
            with torch.no_grad():
                for i in range(0, len(ids), 64):
                    lg, st, _ = self.m(
                        torch.tensor([ids[i:i + 64]], device=self.dev), st)
                    outs.append(lg[0].float())
            self.m.store_read_off = False
            return torch.cat(outs, 0)

        try:
            von, voff = run(False), run(True)
            delta = (von - voff)
            mags = delta.abs().max(-1).values
            p = int(mags.argmax())
            top = delta[p].topk(3)
            return {"vote_max": round(float(mags[p]), 2),
                    "at_pos": p,
                    "suggests": [self.tok.decode([int(i)])
                                 for i in top.indices],
                    "weights": [round(float(v), 2) for v in top.values]}

        finally:
            self.m.store_read_off = False

    def exch(self, q, ans):
        tok = self.tok
        # 50c: no synthetic press — the reward channel carries only
        # events that were actually felt (a reviewer caught lessons and
        # replays injecting a counterfeit <+2> no one ever pressed)
        ids = (tok.encode(q).ids + [self.eh]
               + tok.encode(" " + ans).ids + [self.em])
        return ids

    def fact_ce(self, q, ans):
        tok = self.tok
        ids = tok.encode(q).ids + [self.eh] + tok.encode(" " + ans).ids + [self.em]
        L = len(ids)
        idsp = ids + [self.sil] * ((64 - L % 64) % 64)
        a0 = len(tok.encode(q).ids) + 1
        st_c = self._clone(self.st)
        outs = []
        with torch.no_grad():
            for i in range(0, len(idsp), 64):
                lg, st_c, _ = self.m(
                    torch.tensor([idsp[i:i + 64]], device=self.dev), st_c)
                outs.append(lg[0].float())
        v = torch.cat(outs, 0)
        return float(F.cross_entropy(v[a0 - 1:L - 1].cpu(),
                                     torch.tensor(ids[a0:L])))

    def teach(self, q, ans, state_feed=True):
        """the frozen method's day-move: state the fact into the lived
        stream, then corrective absorption to criterion (mastery-based:
        keep absorbing until the fact sits, max 12 steps)."""
        if state_feed:
            self.feed(self.tok.encode(ans).ids + [self.eh])
            self.feed([self.em])
        self.fatigue += 0.5
        self.last_novel_t = time.time()
        self._today["taught"].append((q, ans[:120]))
        self._today["taught"] = self._today["taught"][-10:]
        self.saliences[q] = {"surp": None,   # filled from first-dose ce
                             "mood": round(self.mood, 1)}
        # adaptive first dose: an easy fact must not be over-absorbed
        # (49pp: hours hit 0.055 and crushed its number-family siblings)
        ce0 = self.fact_ce(q, ans)
        self.saliences[q]["surp"] = round(ce0, 1)   # how new the lesson felt
        k0 = 1 if ce0 < 1.0 else 2
        loss = self.absorb(q, ans, k0)
        steps = k0
        # absorb INTO THE BAND, never to the floor: a fact driven to
        # ce~0.0 becomes the strongest gold and permanently captures
        # weaker neighbors' questions (the predator law — found by a
        # teacher across ten days of raising). Stop inside the healthy
        # population band; the night and the curve finish the job.
        while steps < 12 and self.fact_ce(q, ans) > 0.55:
            loss = self.absorb(q, ans, 1)
            steps += 1
        import re as _re
        nrm = lambda s: _re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
        self.facts = [(fq, fa) for fq, fa in self.facts if nrm(fq) != nrm(q)]
        self.facts.append((q, ans))
        # a re-taught fact re-enters the nights: fresh ledger, so the
        # settle law applies to it again (a lesson repeated is fresh)
        self.progress[q] = {"nights": 0, "stuck": 0, "done": False}
        return {"taught": q, "answer": ans, "absorb_loss": round(loss, 3),
                "absorb_steps": steps,
                "report_card": self.report_card()}

    def report_card(self):
        return [{"q": q, "a": a, "ce": round(self.fact_ce(q, a), 2)}
                for q, a in self.facts]

    def press(self, level=None, mag=None):
        """graded reward at any magnitude: the FELT token stays within
        the trained vocabulary (|m|>=2 -> level-2 token), while the
        magnitude expresses through plasticity — dose steps for
        positive, corrective unlikelihood on the last reply for
        strong negative (external correction may unlearn; it never
        becomes felt self-punishment)."""
        if mag is None:
            mag = float(level.replace("+", ""))
        mag = max(-6.0, min(6.0, float(mag)))
        sign = "+" if mag >= 0 else "-"
        tokname = f"<{sign}{2 if abs(mag) >= 2 else 1}>"
        self.feed([self.press_ids[tokname]])
        self.mood = self.mood * 0.9 + mag
        self.fatigue += 0.2
        self.n_human_presses += 1
        self._today["presses"] += mag
        if self.last_q:
            self.press_log.append(
                {"q": self.last_q[0][:80], "a": self.last_q[1][:80],
                 "mag": mag})
            self.press_log = self.press_log[-500:]
        info = {"felt": tokname.strip("<>"), "mag": mag,
                "mood": round(self.mood, 2)}
        if mag > 0 and self.last_q:
            q, ans = self.last_q
            k = min(int(round(abs(mag))), 6) or 1
            # plasticity satiates on the already-mastered: praise on a
            # strong fact is fully FELT, but barely re-absorbed — no
            # amount of loving presses turns one gold into a predator
            if self.fact_ce(q, ans) < 0.3:
                k = 1
            loss = self.absorb(q, ans, k)
            info["absorbed_steps"] = k
            info["loss"] = round(loss, 3)
        elif mag <= -2 and self.last_q:
            k = min(int(round(abs(mag))) - 1, 3)
            self._unlearn_reply(self.last_q[1], k)
            info["corrected_steps"] = k
        return info

    def _unlearn_reply(self, reply, k=1):
        """corrective unlikelihood on the last reply's own tokens —
        the external NO, expressed as unlearning."""
        ids = self.tok.encode(reply).ids
        if len(ids) < 2:
            return
        ids = ids + [self.sil] * ((64 - len(ids) % 64) % 64)
        x = torch.tensor([ids[:-1]], device=self.dev)
        self.m.train()
        for _ in range(max(1, k)):
            st_d = self.m.init_state(1, self.dev)
            self.opt.zero_grad(set_to_none=True)
            loss = None
            for i in range(0, x.shape[1], 64):
                lg, st_d, _ = self.m(x[:, i:i + 64], st_d)
                logp = torch.log_softmax(lg[0].float(), -1)
                for j in range(lg.shape[1]):
                    t_ = i + j + 1
                    if t_ >= len(self.tok.encode(reply).ids):
                        break
                    tok_ = ids[t_]
                    p_ = logp[j, tok_].exp().clamp(max=0.999)
                    ul = -torch.log1p(-p_)
                    loss = ul if loss is None else loss + ul
            if loss is not None:
                (0.3 * loss).backward()
                self.opt.step()
            self.n_steps += 1
        self.m.eval()

    def absorb(self, q, ans, k):
        tok = self.tok
        ids = (tok.encode(q).ids + [self.eh]
               + tok.encode(" " + ans).ids + [self.em])
        ids += [self.sil] * ((64 - len(ids) % 64) % 64)
        x = torch.tensor([ids[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        a0 = len(tok.encode(q).ids) + 1
        w = torch.zeros_like(y, dtype=torch.float)
        w[0, a0 - 1:a0 - 1 + len(tok.encode(" " + ans).ids) + 1] = 1.0
        self.m.train()
        loss = None
        for kk in range(k):
            self.opt.zero_grad(set_to_none=True)
            st_d = self.m.init_state(1, self.dev) if kk % 2 == 0 \
                else self._clone(self.st)
            tot = None
            for i in range(0, x.shape[1], 64):
                lg, st_d, _ = self.m(x[:, i:i + 64], st_d)
                ce = F.cross_entropy(lg[0], y[0, i:i + 64], reduction="none")
                pc = (ce * w[0, i:i + 64]).sum()
                tot = pc if tot is None else tot + pc
            loss = tot / w.sum().clamp_min(1.0)
            loss.backward()
            self.opt.step()
            self.n_steps += 1
        self.m.eval()
        return float(loss.detach())

    def stmt_ce(self, text):
        """CE of a noticed statement's tokens from a fresh state."""
        ids = self.tok.encode(text).ids + [self.eh]
        L = len(ids)
        idsp = ids + [self.sil] * ((64 - L % 64) % 64)
        st_c = self.m.init_state(1, self.dev)
        outs = []
        with torch.no_grad():
            for i in range(0, len(idsp), 64):
                lg, st_c, _ = self.m(
                    torch.tensor([idsp[i:i + 64]], device=self.dev), st_c)
                outs.append(lg[0].float())
        v = torch.cat(outs, 0)
        return float(F.cross_entropy(v[:L - 1].cpu(),
                                     torch.tensor(ids[1:L])))

    def _drives(self):
        now = time.time()
        return {"fatigue": round(self.fatigue, 1),
                "bored_s": int(now - self.last_novel_t),
                "lonely_s": int(now - self.last_user_t)}

    def _preoccupation(self):
        """what is on its mind: pursuit items first, then open
        learners, then the study list — rotated so rumination roams."""
        cands = []
        if self.pursuit:
            cands += self.pursuit["items"]
        cands += [k_ for k_, v_ in self.progress.items()
                  if not v_.get("done") and k_ not in cands]
        cands += [s_ for s_ in self.study if "~ " + s_ not in cands]
        if not cands:
            return None
        self._rum_i = getattr(self, "_rum_i", -1) + 1
        item = cands[self._rum_i % len(cands)]
        return item[2:] if item.startswith("~ ") else item

    def _tok_word(self, i_):
        """display form of one token id for the telemetry trace."""
        if i_ == self.sil:
            return "\u00b7"
        if i_ == self.em:
            return "\u00b6"
        pv = {v_: k_ for k_, v_ in self.press_ids.items()}
        if i_ in pv:
            return pv[i_]
        w_ = self.tok.decode([i_]).strip()
        return w_ if w_ else "\u2423"

    def _free_speak(self, lg=None, max_new=24):
        """give it the floor: generation from the CURRENT lived state —
        mouth rules unchanged (press ban, pause cap). Pass the logits
        of a primed feed to speak from what is on its mind, exactly as
        chat speaks from the logits of your words. Serve contributes
        zero words; silence is an honest outcome."""
        out, pauses = [], 0
        x = None if lg is not None else torch.tensor(
            [[self.sil]], device=self.dev)
        tr_raw = []
        with torch.no_grad():
            for _ in range(max_new + 6):
                if x is not None:
                    lg, self.st, _ = self.m(x, self.st)
                v = lg[0, -1].float()
                if hasattr(self.m, "ban_presses"):
                    v = self.m.ban_presses(v)
                if pauses >= 4:
                    v[self.sil] = float("-inf")
                pr = torch.softmax(
                    v / max(self.a.temp, 0.05), -1).cpu()
                nxt = int(torch.multinomial(pr, 1, generator=self.gen))
                if len(tr_raw) < 40:
                    tk_ = torch.topk(pr, 3)
                    tr_raw.append((nxt, float(pr[nxt]),
                                   [(int(i_), float(p_)) for p_, i_ in
                                    zip(tk_.values, tk_.indices)]))
                out.append(nxt)
                if nxt == self.sil:
                    pauses += 1
                if nxt == self.em or len(
                        [t_ for t_ in out if t_ != self.sil]) >= max_new:
                    break
                x = torch.tensor([[nxt]], device=self.dev)
        self.day_buf.extend(out)
        tail = [out[-1]] if out and out[-1] == self.em else \
            (out[-1:] + [self.em] if out else [self.em])
        with torch.no_grad():
            _, self.st, _ = self.m(
                torch.tensor([tail], device=self.dev), self.st)
        if not out or out[-1] != self.em:
            self.day_buf.append(self.em)
        self.last_trace = [
            {"t": self._tok_word(n_), "p": round(p_, 3),
             "alt": [[self._tok_word(i_), round(pp_, 3)]
                     for i_, pp_ in alt_]}
            for n_, p_, alt_ in tr_raw]
        return self.tok.decode([t_ for t_ in out if t_ not in
                                (self.sil, self.em)]).strip()

    def tick(self):
        """the autonomous clock (49xx): time passes whether or not
        anyone speaks to it. Drives act when their bars are crossed —
        fatigue: it falls asleep on its own; boredom: it ruminates on
        its homework; loneliness: it may speak first. All disclosed
        through the outbox; no drive ever authors a word."""
        now = time.time()
        d = self._drives()
        if (self.fatigue >= self.a.fatigue_bar
                and len(self.day_buf) >= 65
                and now - self.last_user_t >= self.a.tick):
            rep = self.sleep()
            self.outbox.append({"kind": "slept", "night": {
                k_: rep.get(k_) for k_ in
                ("nrem", "rem", "lived_tokens", "pursuit", "conscience",
                 "genome", "woke_thinking", "woke_feeling")}})
            return
        if d["bored_s"] >= self.a.bore_bar and self.ruminate_budget > 0:
            item = self._preoccupation()
            if item:
                self.feed(self.tok.encode(item).ids + [self.eh])
                self.fatigue += 0.3
                self.ruminate_budget -= 1
                self.last_novel_t = now
                self.outbox.append({"kind": "ruminated",
                                    "about": item[:60]})
                return
            # nothing on its mind to chew — contentment, not a claim
            # on the tick: loneliness may still speak below
        if d["lonely_s"] >= self.a.lone_bar and self.initiate_budget > 0:
            # it speaks from what it was thinking about, not from a
            # blank stare: the preoccupation primes the state (the
            # woke-thinking wire), then the mouth is its own.
            about = self._preoccupation()
            lg_p = None
            if about:
                lg_p = self.feed(self.tok.encode(about).ids + [self.eh])
            if lg_p is None:
                self.initiate_budget -= 1
                self.outbox.append({"kind": "kept_quiet"})
                return
            txt = self._free_speak(lg=lg_p)
            self.initiate_budget -= 1
            self.fatigue += 0.3
            self.last_user_t = now   # it reached out; the ache resets
            if txt:
                self.session.append(("", txt))
                self.outbox.append({"kind": "speaks", "text": txt,
                                    "trace": getattr(
                                        self, "last_trace", None),
                                    "about": (about or "")[:60]})
            else:
                self.outbox.append({"kind": "kept_quiet"})

    def _clean_tail(self, toks):
        """The night replays the lived day — but a day's generative
        failures (empty replies, degenerate loops) must not be
        rehearsed into the weights: replayed failure is self-
        reinforcing (measured across shifts 2-6 as loops, stutters,
        then silence, each deepening nightly). Numbers only: a model
        turn is dropped when it is empty or when one token makes up
        more than half of a long turn. Human turns always survive."""
        out, cur, in_model, dropped = [], [], False, 0
        for t in toks:
            if in_model:
                cur.append(t)
                if t == self.em:
                    body = [x for x in cur[:-1] if x != self.sil]
                    dege = (len(body) >= 6 and max(
                        body.count(x) for x in set(body)) > len(body) * 0.5)
                    if body and not dege:
                        out.extend(cur)
                    else:
                        dropped += 1
                    cur, in_model = [], False
            else:
                out.append(t)
                if t == self.eh:
                    in_model = True
        if cur:
            body = [x for x in cur if x != self.sil]
            if body and not (len(body) >= 6 and max(
                    body.count(x) for x in set(body)) > len(body) * 0.5):
                out.extend(cur)
            else:
                dropped += 1
        return out, dropped

    def _flush_working(self, st):
        """A true morning: the within-day working sums (band
        accumulators, clock counters, write buffers) start EMPTY.
        They were never meant to survive a wake — their slow clocks
        (4k-32k tokens) never fire inside a day, so carrying them
        compounds: measured silt after ~50 wakes reached norm 250k
        on bands 7-8 (elements at 1e5 against a ~1.0-bounded state),
        numerically poisoning slow-band generation while fast-path
        knowledge stayed perfect. Identity (h), episodic store (M),
        and goal slots (G) still carry."""
        if not isinstance(st, dict):
            return st
        for part in ("acc", "acc_c"):
            d = st.get(part)
            if isinstance(d, dict):
                for u, v in d.items():
                    if torch.is_tensor(v):
                        d[u] = torch.zeros_like(v)
        if isinstance(st.get("cnt"), dict):
            st["cnt"] = {u: 0 for u in st["cnt"]}
        if isinstance(st.get("pend"), dict):
            st["pend"] = {u: None for u in st["pend"]}
        if isinstance(st.get("fresh"), dict):
            st["fresh"] = {u: False for u in st["fresh"]}
        st["CM"] = None
        st["lp"], st["lh"], st["lg"] = {}, {}, {}
        st["wbuf"] = []
        st["tok"] = 0
        st["chunk"] = 0
        st["xl"] = None
        return st

    def sleep(self):
        """the self-steering night (49ff): candidates are what it was
        taught and what IT noticed; its own measured learning progress
        chooses the dream set — mastered items graduate (no drilling
        past criterion), stuck items fade (no wasted nights), items in
        progress get the replay. NREM on the chosen set, splice REM
        pairs, then a fresh wake."""
        if len(self.day_buf) < 65:
            return {"error": "not enough lived tokens to dream yet"}
        # THE PURSUIT (49uu): a self-adopted multi-night goal with
        # self-earned installments — its card picks the goal, its
        # measured progress pays the installments, completion earns a
        # victory night, stalling releases without punishment.
        pursuit_report = None
        card_pre = self.report_card()
        by_q = {x["q"]: x["ce"] for x in card_pre}
        self.pursuit_installment = False
        if self.pursuit:
            pu = self.pursuit

            def _pu_ce(k_):
                # a pursuit item is a taught fact (card lookup) or a
                # wonder statement ("~ "-keyed, measured directly)
                return self.stmt_ce(k_[2:]) if k_.startswith("~ ") \
                    else by_q.get(k_, 9.9)
            total = sum(_pu_ce(q_) for q_ in pu["items"])
            pu["nights"] += 1
            prog = (pu["last_total"] - total)                 if pu["last_total"] is not None else 0.0
            pu["last_total"] = total
            left = [q_ for q_ in pu["items"]
                    if _pu_ce(q_) > pu["target"]]
            if not left:
                self.pursuit_installment = True   # the victory night
                pursuit_report = {"state": "COMPLETE",
                                  "nights": pu["nights"],
                                  "items": pu["items"]}
                self.pursuit = None
            elif prog > 0.05:
                pu["stalled"] = 0
                self.pursuit_installment = True   # earned installment
                pursuit_report = {"state": "installment",
                                  "progress": round(prog, 2),
                                  "left": left, "nights": pu["nights"]}
            else:
                pu["stalled"] += 1
                if pu["stalled"] >= 3:
                    pursuit_report = {"state": "released",
                                      "items": pu["items"]}
                    self.pursuit = None
                else:
                    pursuit_report = {"state": "no progress",
                                      "stalled": pu["stalled"]}
            # LIFETIME CAP (49yy): installments reset the stall count,
            # so a part-paying pursuit could grind its items forever —
            # eight nights is a whole campaign; let go and move on.
            if self.pursuit and self.pursuit["nights"] >= 8:
                pursuit_report = {"state": "released (long campaign)",
                                  "items": self.pursuit["items"],
                                  "nights": self.pursuit["nights"]}
                self.pursuit = None
        elif card_pre or self.study:
            weak = [x for x in card_pre
                    if x["ce"] > self.a.pursuit_adopt][:3]
            if len(weak) >= 2:
                self.pursuit = {"items": [x["q"] for x in weak],
                                "target": self.a.pursuit_target,
                                "nights": 0, "stalled": 0,
                                "last_total": None}
                for x in weak:
                    led_ = self.progress.get(x["q"])
                    if led_:
                        led_["done"] = False
                        led_["stuck"] = 0
                pursuit_report = {"state": "ADOPTED",
                                  "items": self.pursuit["items"],
                                  "target": self.a.pursuit_target}
            else:
                # BORN OF WONDER (49ww): nothing is failing, so the
                # pursuit may come from desire instead of deficiency —
                # statements IT chose to keep (its own noticing) that
                # it has not yet made its own become the goal.
                wants = [s_ for s_ in self.study
                         if self.stmt_ce(s_) > 2.2][:3]
                if len(wants) >= 2:
                    self.pursuit = {"items": ["~ " + s_ for s_ in wants],
                                    "target": 2.0, "kind": "wonder",
                                    "nights": 0, "stalled": 0,
                                    "last_total": None}
                    for s_ in wants:
                        led_ = self.progress.get("~ " + s_)
                        if led_:
                            led_["done"] = False
                            led_["stuck"] = 0
                    pursuit_report = {"state": "ADOPTED (born of wonder)",
                                      "items": self.pursuit["items"],
                                      "target": 2.0}
        cands = [("qa", q, a) for q, a in self.facts[-6:]]
        cands += [("stmt", t_, None) for t_ in self.study[-4:]]
        # THE PURSUIT's items claim the front of tonight's study set
        if self.pursuit:
            for q_ in self.pursuit["items"]:
                if q_.startswith("~ "):
                    s_ = q_[2:]
                    led_ = self.progress.get(q_)
                    if not (led_ or {}).get("done")                             and ("stmt", s_, None) not in cands:
                        cands.insert(0, ("stmt", s_, None))
                    continue
                pair_ = next(((qq, aa) for qq, aa in self.facts
                              if qq == q_), None)
                led_ = self.progress.get(q_)
                if pair_ and led_ and led_.get("done") \
                        and self.fact_ce(pair_[0], pair_[1]) \
                        > self.pursuit["target"]:
                    led_["done"] = False   # diploma below the goal's bar
                    led_["stuck"] = 0
                if pair_ and led_ and not led_.get("done")                         and ("qa", pair_[0], pair_[1]) not in cands:
                    cands.insert(0, ("qa", pair_[0], pair_[1]))
        # THE RETENTION CURVE (49zz): graduation starts a clock, not an
        # ending. Every mastered fact is re-checked on an expanding
        # schedule (1, 3, 7, 14, 30 nights): still solid -> the next
        # check moves further out; drifted -> the curve restarts and
        # tonight replays it. Old knowledge thins but never goes
        # unwatched. Re-opens capped at 2 a night.
        IVLS = (1, 3, 7, 14, 30)
        nn = self.day_n + 1                     # tonight's number
        maintenance = []
        reopens = 0
        for q_, a_ in self.facts:
            led = self.progress.get(q_)
            if not led or not led.get("done"):
                continue
            if led.get("due") is None:          # legacy graduate: enroll
                led["ivl_i"] = 0
                led["due"] = nn
            if led["due"] > nn:
                continue
            ce_ = self.fact_ce(q_, a_)
            if ce_ > 0.5 and reopens < 2:
                led["done"] = False
                led["stuck"] = 0
                led["ivl_i"] = 0
                led["due"] = None               # re-set at next mastery
                reopens += 1
                if ("qa", q_, a_) not in cands:
                    cands.insert(0, ("qa", q_, a_))
                maintenance.append({"q": q_[:40], "ce": round(ce_, 2),
                                    "verdict": "drifted -> replay"})
            elif ce_ > 0.5:                     # drifted, but cap full
                led["due"] = nn + 1             # look again tomorrow
                maintenance.append({"q": q_[:40], "ce": round(ce_, 2),
                                    "verdict": "drifted, deferred"})
            else:
                i_ = min(led.get("ivl_i", 0) + 1, len(IVLS) - 1)
                led["ivl_i"] = i_
                led["due"] = nn + IVLS[i_]
                maintenance.append({"q": q_[:40], "ce": round(ce_, 2),
                                    "verdict": "solid, next in %d" % IVLS[i_]})
        story, replay = [], []
        graduated, faded = [], []
        for kind, q, a in cands:
            key = q if kind == "qa" else "~ " + q
            ce = self.fact_ce(q, a) if kind == "qa" else self.stmt_ce(q)
            led = self.progress.setdefault(
                key, {"nights": 0, "stuck": 0, "done": False})
            if led["done"]:
                continue
            row = {"q": key[:60], "pre": round(ce, 2)}
            grad_bar = 0.6 if kind == "qa" else 2.0
            if self.pursuit and key in self.pursuit["items"]:
                # a pursuit item answers to its own goal's standard:
                # graduating at the house bar (0.6) while the goal's
                # target is stricter froze the item out of the replay
                # above target — the goal starved on its own item's
                # diploma (measured: the squirrel stall, shift 13)
                grad_bar = min(grad_bar, self.pursuit["target"])
            # SETTLE LAW (49y, re-learned 49xx): a fact taught TODAY
            # never graduates on its first night — teach-to-criterion
            # holds the surface, only a slept-on night holds the week.
            # Skipping the settle pass lost the same fresh fact twice.
            fresh = (kind == "qa"
                     and any(q == tq for tq, _ in self._today["taught"])
                     and led["nights"] == 0)
            if ce < grad_bar and not fresh:
                row["verdict"] = "mastered"
                led["done"] = True
                led["row"] = 0
                led["ivl_i"] = 0                # the retention clock starts
                led["due"] = nn + 1
                graduated.append(key[:40])
                if kind == "stmt" and q in self.study:
                    self.study.remove(q)
            elif led["stuck"] >= 2:
                row["verdict"] = "stuck"
                led["done"] = True
                faded.append(key[:40])
                if kind == "stmt" and q in self.study:
                    self.study.remove(q)
            elif led.get("row", 0) >= 2:
                # SPACING LAW (49yy): no item is drilled more than two
                # nights running — over-replay turns a lesson into a
                # parasitic frame that captures its neighbors (measured:
                # the thunder takeover). Rest is part of consolidation.
                row["verdict"] = "rest"
                led["row"] = 0
                story.append(row)
                continue
            else:
                row["verdict"] = "learning"
                led["row"] = led.get("row", 0) + 1
                replay.append((kind, q, a, row))
            story.append(row)
        replay = replay[:4]   # ordering wire removed: 49hh ablation
        # proved it inert (cap 4 always fits the trained window)
        excited = getattr(self, "last_gained", 0.0) >= 1.0             or getattr(self, "pursuit_installment", False)
        nrem_cap = 3 if excited else 2
        nrem = 0
        stream = []
        for kind, q, a, row_ in replay:
            if kind == "qa":
                stream.extend(self.exch(q, a))
                # replay dose scales inversely with strength (the
                # flattening antidote as physiology): a fact already
                # near the band gets ONE pass, not two — the night
                # teacher measured sleep consolidating fresh in-band
                # facts to ce 0.01 overnight, minting new predators
                if row_.get("pre", 9.9) > 0.45:
                    stream.extend(self.exch(q, a))
            else:
                stream.extend(self.tok.encode(q).ids + [self.eh])
        tail_dropped = 0
        if stream:
            tail_, tail_dropped = self._clean_tail(
                self.day_buf[-(192 if excited else 128):])
            stream.extend(tail_)
        elif self.session:
            # nothing left to learn tonight: replay the lived day itself
            stream, tail_dropped = self._clean_tail(self.day_buf[-1024:])
        stream += [self.sil] * ((64 - len(stream) % 64) % 64)
        self.m.train()
        st_s = self.m.init_state(1, self.dev)
        for i in range(0, len(stream), 64):
            if nrem >= nrem_cap:
                break
            x = torch.tensor([stream[i:i + 64]], device=self.dev)
            lg, st_s, _ = self.m(x, st_s)
            y = torch.tensor(stream[i + 1:i + 65] + [self.sil],
                             device=self.dev)[:64]
            self.opt.zero_grad(set_to_none=True)
            night_loss = 0.1 * F.cross_entropy(lg[0], y)
            vl = self.m.pop_value_loss() \
                if hasattr(self.m, "pop_value_loss") else None
            if vl is not None:
                # the day's lived stream carries its REAL felt presses
                # (yours and its own) — the value heads learn them here
                night_loss = night_loss + vl
            night_loss.backward()
            self.opt.step()
            nrem += 1
            self.n_steps += 1
            st_s = self._detach_in_place(st_s)
        remc = 0
        # SALIENCE-PICKED DREAMS (49zz): which memories share a dream is
        # decided by how much they mattered — the surprise and mood
        # stamped when each was lived — not by list position. The most
        # charged memories dream together (amygdala's vote, not PFC's:
        # the executive is asleep).
        def _charge(item):
            k_, q_, _a = item
            key_ = q_ if k_ == "qa" else "~ " + q_
            s_ = self.saliences.get(key_, {})
            return (s_.get("surp") or 5.0) + abs(s_.get("mood") or 0.0)
        segs_src = sorted(((k, q, a) for k, q, a, _ in replay),
                          key=_charge, reverse=True)
        pairs = []
        if len(segs_src) >= 2:
            pairs = [(segs_src[0], segs_src[1])]
            if len(segs_src) >= 4:
                pairs.append((segs_src[2], segs_src[3]))
        rem_pairs = [{"a": fa[1][:36], "b": fb[1][:36],
                      "charge": round(_charge(fa) + _charge(fb), 1)}
                     for fa, fb in pairs]
        self.m.train()
        for fa, fb in pairs:
            segs = []
            for kind, q, a in (fa, fb):
                ids = self.exch(q, a) if kind == "qa" \
                    else self.tok.encode(q).ids + [self.eh]
                ids += [self.sil] * ((64 - len(ids) % 64) % 64)
                st_d = self.m.init_state(1, self.dev)
                for i in range(0, len(ids), 64):
                    _, st_d, _ = self.m(
                        torch.tensor([ids[i:i + 64]], device=self.dev), st_d)
                C = self.m._last_C
                Cl = getattr(self.m, "_last_C_live", C)
                sv = self.m._last_sv
                t0 = int(sv[0, :C.shape[1] - 9].abs().argmax())
                segs.append((C.detach(), Cl, t0))
            (Ca, Cla, t0a), (Cb, Clb, t0b) = segs
            self.opt.zero_grad(set_to_none=True)
            loss = None
            c = Cla[0, t0a:t0a + 1]
            for n in range(1, 9):
                c = self.m.plan_step(c)
                l_n = 1.0 - F.cosine_similarity(
                    c, Ca[0, t0a + n:t0a + n + 1], dim=-1).mean()
                loss = l_n if loss is None else loss + l_n
            loss = loss + (1.0 - F.cosine_similarity(
                self.m.plan_step(c), Clb[0, t0b:t0b + 1].detach(),
                dim=-1).mean())
            c = Clb[0, t0b:t0b + 1]
            for n in range(1, 9):
                c = self.m.plan_step(c)
                loss = loss + 1.0 - F.cosine_similarity(
                    c, Cb[0, t0b + n:t0b + n + 1], dim=-1).mean()
            (0.1 * loss / 17.0).backward()
            self.opt.step()
            remc += 1
            self.n_steps += 1
        self.m.eval()
        lived = len(self.day_buf)
        self.day_buf = []
        fid = {str(k): round(float(v), 3)
               for k, v in getattr(self.m, "rem_fid", {}).items()}
        gained = 0.0
        for kind, q, a, row in replay:
            post = self.fact_ce(q, a) if kind == "qa" else self.stmt_ce(q)
            row["post"] = round(post, 2)
            row["delta"] = round(post - row["pre"], 2)
            gained += max(0.0, row["pre"] - post)
            led = self.progress[q if kind == "qa" else "~ " + q]
            led["nights"] += 1
            led["last_delta"] = row["delta"]
            if row["delta"] > -0.05 and post > 2.5:
                led["stuck"] += 1
            else:
                led["stuck"] = 0
        # waking IS a state reset with changed weights: the day's working
        # memory does not survive the night (measured: carrying it grooves
        # the morning), the consolidated weights do.
        # THE MULTI-DAY STORE (49zz) is the one exception: the episodic
        # organ's matrices (st["M"]) carry across the wake with nightly
        # decay — yesterday's episodes fade over ~2-3 nights while the
        # weights absorb them (gradual hand-off, not a cliff). Working
        # state (h, bands) still resets fully; reads stay relevance-
        # gated, so carried episodes speak only when cued.
        old_M = None
        store_carried = None
        if self.a.store_decay > 0 and isinstance(self.st, dict) \
                and self.st.get("M"):
            old_M = {k_: v_.detach() * self.a.store_decay
                     for k_, v_ in self.st["M"].items()}
        src = self.state_meta.get("st_live") or self.state_meta.get("st")
        self.st = _to_dev(src if self.state_meta.get("st_live")
                          else _lane0(src), self.dev) \
            if src is not None else self.m.init_state(1, self.dev)
        self._flush_working(self.st)
        if old_M is not None and isinstance(self.st, dict) \
                and self.st.get("M"):
            # REPLACE, never add: the end-of-day store already contains
            # all carried history — decaying it once per night gives
            # each episode a clean exponential fade from its lived day.
            for k_, v_ in self.st["M"].items():
                ov = old_M.get(k_)
                if ov is not None and ov.shape == v_.shape:
                    self.st["M"][k_] = ov.to(v_.device, v_.dtype)
            store_carried = round(float(sum(
                v_.abs().sum() for v_ in old_M.values())), 1)
        self.session, self.last_q = [], None
        noticed_today = len(self.self_noticed)   # capture before the wake reset
        self.self_noticed = []
        # WIRE A (49tt): appetite from progress — a night that gained
        # wakes hungrier. Its own felt progress modulates its own
        # curiosity; the objective stays externally grounded.
        self.notice_budget = 6 if (gained >= 1.0
                                   or self.pursuit_installment
                                   or self.pride_today >= 2) else 4
        self.pride_today = 0
        self.mood = 0.0          # decaying tally of reward-channel events
        self.self_press_budget = 4
        self.self_frown_budget = 3
        self._self_pressed_qs = set()
        self.day_n += 1                       # the night clock ticks
        self._today = {"taught": [], "noticed": [], "presses": 0.0}
        self.fatigue = 0.0
        self.last_novel_t = self.last_user_t = time.time()
        self.ruminate_budget = 6
        self.initiate_budget = 2
        # THE OPEN DOOR at pursuit timescale (49ww): a night that paid
        # an installment — or finished the goal — is FELT at wake: it
        # presses its own button for the multi-night achievement.
        # Grounded in measured overnight progress; unforgeable.
        woke_feeling = None
        if pursuit_report and pursuit_report.get("state") == "COMPLETE":
            self.feed([self.press_ids["<+2>"]])
            self.mood = self.mood * 0.9 + 2.0
            self.press_log.append(
                {"q": "pursuit complete",
                 "a": " / ".join(pursuit_report["items"])[:80],
                 "mag": 2.0, "self": True})
            woke_feeling = "+2 · its goal is complete"
        elif self.pursuit_installment:
            self.feed([self.press_ids["<+1>"]])
            self.mood = self.mood * 0.9 + 1.0
            woke_feeling = "+1 · installment earned"
        # WIRE B (49tt): the hunt as morning preoccupation — the top
        # still-learning item is fed into the fresh waking state (it
        # wakes thinking about its homework; its mouth stays free).
        woke_thinking = None
        learners = [(k_, q_) for k_, q_, a_, row_ in replay
                    if row_.get("verdict") == "learning"]
        if self.pursuit:
            pit = self.pursuit["items"]
            learners.sort(key=lambda x_: 0 if (x_[1] in pit
                          or "~ " + x_[1] in pit) else 1)
        if learners:
            _, q_ = learners[0]
            self.feed(self.tok.encode(q_).ids + [self.eh])
            woke_thinking = q_[:60]
        self.last_gained = gained
        # ADAPTIVE GENOME lite (49vv): the curiosity bar tunes itself
        # from its own usage — starving appetite lowers it, saturated
        # appetite raises it. Bounded, disclosed, persisted.
        self._budget_history.append(noticed_today)
        self._budget_history = self._budget_history[-3:]
        genome_note = None
        cur = self.notice_peak_dyn or self.a.notice_peak
        if len(self._budget_history) >= 3:
            if sum(self._budget_history) == 0 and cur > 14.5:
                self.notice_peak_dyn = round(cur - 0.3, 1)
                genome_note = "curiosity bar lowered to %s (starving)"                     % self.notice_peak_dyn
            elif min(self._budget_history) >= 3 and cur < 17.5:
                self.notice_peak_dyn = round(cur + 0.3, 1)
                genome_note = "curiosity bar raised to %s (saturated)"                     % self.notice_peak_dyn
            if genome_note:
                self._budget_history = []
        # CONSCIENCE RECALIBRATION (49vv): nightly retrain on the
        # human's real presses once enough exist.
        recal_note = None
        # the conscience calibrates against the HUMAN's taste only —
        # learning right-and-wrong from your own self-approval is a
        # closed loop that drifts; the parent's judgment is the ground.
        real = [e for e in self.press_log
                if "mag" in e and not e.get("self")]
        if self.critic is not None and len(real) >= 12 and \
                self.n_human_presses > getattr(self, "_critic_seen", 0):
            try:
                recal_note = self._recalibrate_conscience(real)
                self._critic_seen = self.n_human_presses
            except Exception as e_:
                recal_note = "recalibration failed: %s" % str(e_)[:40]
        self._fill_goal_slots()
        self.press_log = self.press_log[-500:]
        keep_keys = set(q_ for q_, _ in self.facts) \
            | set("~ " + s_ for s_ in self.study)
        self.saliences = {k_: v_ for k_, v_ in self.saliences.items()
                          if k_ in keep_keys}
        saved = None
        if self.a.save:
            torch.save({"model": self.m.state_dict(),
                        "step": self.state_meta.get("step"),
                        "cfg": self.state_meta.get("cfg"),
                    "st_live": self._flush_working(self._detach_in_place(
                        _to_dev(self.st, "cpu"))),
                        "life": {"facts": self.facts, "study": self.study,
                                 "progress": self.progress,
                                 "surp_mu": self.surp_mu,
                                 "pursuit": self.pursuit,
                                 "pursuit_installment":
                                     self.pursuit_installment,
                                 "press_log": self.press_log,
                                 "notice_peak_dyn": self.notice_peak_dyn,
                                 "budget_history":
                                     self._budget_history,
                                 "day_n": self.day_n,
                                 "saliences": self.saliences,
                                 "n_human_presses":
                                     self.n_human_presses}},
                       self.a.save)
            saved = self.a.save
        card_post = self.report_card()
        return {"nrem": nrem, "rem": remc, "lived_tokens": lived,
                "autosaved": saved,
                "genome": genome_note, "conscience": recal_note,
                "pursuit": pursuit_report,
                "woke_thinking": woke_thinking,
                "woke_feeling": woke_feeling,
                "woke_hungry": self.notice_budget > 4,
                "maintenance": maintenance,
                "tail_dropped": tail_dropped,
                "rem_pairs": rem_pairs,
                "store_carried": store_carried,
                "excited_night": excited,
                "fidelity": fid, "report_card": card_post,
                "night_story": story,
                "progress": {"gained": round(gained, 2),
                             "graduated": graduated, "faded": faded,
                             "learning": [r["q"][:40] for _, _, _, r
                                          in replay]}}

    def _detach_in_place(self, s):
        if torch.is_tensor(s):
            return s.detach()
        if isinstance(s, dict):
            return {k: self._detach_in_place(v) for k, v in s.items()}
        if isinstance(s, (list, tuple)):
            t = [self._detach_in_place(v) for v in s]
            return tuple(t) if isinstance(s, tuple) else t
        return s

    def reset(self):
        """a fresh wake: the day's working state clears, the life
        (facts, study, ledger) stays."""
        src = self.state_meta.get("st_live") or self.state_meta.get("st")
        self.st = _to_dev(src if self.state_meta.get("st_live")
                          else _lane0(src), self.dev) \
            if src is not None else self.m.init_state(1, self.dev)
        self._flush_working(self.st)
        self.day_buf, self.session = [], []
        self.last_q = None
        self.self_noticed = []
        self.notice_budget = 4
        self.self_press_budget = 4
        self.self_frown_budget = 3
        self._self_pressed_qs = set()
        self.fatigue = 0.0
        self.last_novel_t = self.last_user_t = time.time()
        self.ruminate_budget = 6
        self.initiate_budget = 2
        self._today = {"taught": [], "noticed": [], "presses": 0.0}
        self._fill_goal_slots()

    def save(self):
        torch.save({"model": self.m.state_dict(),
                    "step": self.state_meta.get("step"),
                    "cfg": self.state_meta.get("cfg"),
                    "nursery_steps": self.n_steps,
                    "st_live": self._flush_working(
                        self._detach_in_place(_to_dev(self.st, "cpu"))),
                    "life": {"facts": self.facts, "study": self.study,
                             "progress": self.progress,
                             "surp_mu": self.surp_mu,
                             "pursuit": self.pursuit,
                             "pursuit_installment": self.pursuit_installment,
                             "press_log": self.press_log,
                             "notice_peak_dyn": self.notice_peak_dyn,
                             "budget_history": self._budget_history,
                             "day_n": self.day_n,
                             "saliences": self.saliences,
                             "n_human_presses": self.n_human_presses}},
                   self.a.save)
        return {"saved": self.a.save, "live_steps": self.n_steps}


PAGE = """<!doctype html><meta charset=utf-8>
<title>the organism</title>
<style>
:root{--paper:#faf9f6;--panel:#ffffff;--ink:#20242b;--mut:#8b94a0;--line:#e7e4dd;
 --acc:#2f7d5c;--good:#1f7a46;--warn:#bd4a24;--dream:#7a5cc4;--blue:#3565b0}
*{box-sizing:border-box}
body{font:15px/1.5 -apple-system,'Segoe UI',sans-serif;margin:0;display:flex;flex-direction:column;height:100vh;background:var(--paper);color:var(--ink)}
#top{display:flex;align-items:center;gap:14px;padding:12px 22px;background:var(--paper);border-bottom:1px solid var(--line)}
#status{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--mut);letter-spacing:.3px}
#alivedot{width:9px;height:9px;border-radius:50%;background:var(--acc);animation:breathe 3.2s ease-in-out infinite}
@keyframes breathe{0%,100%{opacity:.45;transform:scale(.85)}50%{opacity:1;transform:scale(1.1)}}
.tgl{margin-left:auto;display:flex;gap:8px}
.tglbtn{background:transparent;color:var(--mut);border:1px solid var(--line);border-radius:999px;padding:6px 14px;cursor:pointer;font-size:12px;letter-spacing:.2px;transition:all .15s}
.tglbtn:hover{border-color:var(--mut)}
.tglbtn.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
#main{display:flex;flex:1;min-height:0}
#chat{flex:1;min-width:0;display:flex;flex-direction:column}
#log{flex:1;overflow-y:auto;padding:28px 0}
#log>div{max-width:680px;margin-left:auto;margin-right:auto;padding:0 28px;animation:rise .18s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.you{color:var(--acc);margin:18px auto 4px;font-size:13.5px;font-weight:600;letter-spacing:.2px}
.bot{font-family:Georgia,'Times New Roman',serif;font-size:17px;line-height:1.55;color:var(--ink);margin:0 auto 6px;white-space:pre-wrap}
.sys{color:#b3aca0;font-size:11.5px;margin:6px auto;font-style:italic}
.think{color:var(--mut);font-size:20px;letter-spacing:4px;animation:thinkp 1.2s ease-in-out infinite}
.w{border-radius:3px;transition:background .12s}
.tokhl{background:#f3dfa4;box-shadow:0 0 0 2px #f3dfa4}
.flyt{font-family:menlo,monospace;opacity:0;animation:flyup 1.5s ease-out forwards}
@keyframes flyup{0%{opacity:0;transform:translateY(7px)}18%{opacity:1}100%{opacity:0;transform:translateY(-16px)}}
.msghl{background:#e9efe9;box-shadow:0 0 0 4px #e9efe9;border-radius:4px}
@keyframes thinkp{0%,100%{opacity:.25}50%{opacity:.9}}
#bar{display:flex;gap:10px;padding:14px 28px 8px;background:var(--paper);max-width:736px;margin:0 auto;width:100%}
#msg{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--ink);padding:13px 16px;border-radius:14px;font-size:15px;outline:none;transition:border .15s,box-shadow .15s}
#msg:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(47,125,92,.10)}
button{background:var(--ink);color:var(--paper);border:0;border-radius:12px;padding:11px 18px;cursor:pointer;font-size:14px;transition:opacity .15s}
button:hover{opacity:.85}
#sendbtn{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;padding:0;font-size:18px}
button.quiet{background:transparent;color:var(--mut);border:1px solid var(--line);padding:6px 13px;font-size:12px;border-radius:999px}
button.quiet:hover{opacity:1;border-color:var(--mut);color:var(--ink)}
#rewardrow{display:flex;align-items:center;gap:12px;padding:6px 28px;background:var(--paper);max-width:736px;margin:0 auto;width:100%}
#rw{flex:1;-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;outline:none;
 background:linear-gradient(to right,#c96a45 0%,#e8e5de 46%,#e8e5de 54%,#3f8f63 100%)}
#rw::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:#fff;border:1.5px solid #c8c2b6;box-shadow:0 1px 3px rgba(0,0,0,.15);cursor:grab}
#rwval{min-width:44px;text-align:center;font-weight:700;font-size:15px;color:var(--mut);font-variant-numeric:tabular-nums}
#carerow{display:flex;gap:8px;padding:4px 28px 8px;max-width:736px;margin:0 auto;width:100%}
#hint{padding:0 28px 12px;font-size:11px;color:#b3aca0;max-width:736px;margin:0 auto;width:100%}
#brain{width:48%;min-width:380px;max-width:700px;padding:14px;overflow-y:auto;background:var(--paper);border-left:1px solid var(--line);display:none}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
.panel h3{margin:0 0 8px;font-size:10.5px;letter-spacing:2px;color:var(--mut)}
.blk{fill:#ffffff;stroke:#c8ceD8;stroke-width:1.1}
.cell{fill:#f0f2ef;stroke:#c8ced8;stroke-width:0.8}
.lbl{fill:#5a6672;font-size:8.5px;font-family:-apple-system,sans-serif;letter-spacing:.4px}
.tiny{fill:#98a2ac;font-size:6.5px;font-family:-apple-system,sans-serif}
.val{fill:#a06018;font-size:7.5px;font-family:menlo,monospace}
.wire{stroke:#d5d9d2;stroke-width:1.3;fill:none}
.dot{fill:var(--blue)}
#events{max-height:96px;overflow-y:auto;font-size:11px;color:#5a6672;margin-bottom:6px}
#events>div{padding:1px 0;border-radius:4px}
#events>div:hover{background:#f4f2ec}
#night{max-height:96px;overflow-y:auto;font-size:11px;color:#5a6672;margin-top:6px}
#tslider{width:100%;accent-color:var(--acc)}
#frameinfo{font-size:11px;color:var(--acc);min-height:15px;font-weight:600}
#stream{font-family:menlo,monospace;font-size:9.5px;line-height:1.5;color:#48525c;
 max-height:78px;overflow:hidden;margin:6px 0 2px;font-variant-numeric:tabular-nums}
#stream>div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#stream>div:nth-child(1){color:#20242b}
#stream>div:nth-child(3){opacity:.75}#stream>div:nth-child(4){opacity:.5}#stream>div:nth-child(5){opacity:.3}
#footer{display:flex;gap:12px;align-items:center;font-size:11px;color:#5a6672;margin-top:4px;flex-wrap:wrap}
</style>
<div id=top>
 <div id=status><span id=alivedot></span><span id=alivetxt>awake</span></div>
 <span class=tgl>
  <button id=contTgl class=tglbtn onclick=toggleCont()>continuous time: off</button>
  <button id=intTgl class=tglbtn onclick=toggleInt()>internals: off</button>
 </span>
</div>
<div id=main>
<div id=chat>
 <div id=log></div>
 <div id=bar>
  <input id=msg placeholder="talk to it..." onkeydown="if(event.key==='Enter')send()">
  <button id=sendbtn onclick=send() title="send">↑</button>
 </div>
 <div id=rewardrow>
  <span style="font-size:12px;color:#c2410c">−6</span>
  <input id=rw type=range min=-6 max=6 step=0.5 value=0 oninput=rwShow()>
  <span style="font-size:12px;color:#15803d">+6</span>
  <span id=rwval>0</span>
  <button onclick=pressSlider()>press</button>
 </div>
 <div id=carerow>
  <button class=quiet onclick=sleepy()>sleep</button>
  <button class=quiet onclick=saveLife()>save</button>
  <button class=quiet onclick="fetch('/reset',{method:'POST'}).then(()=>add('sys','~ fresh wake ~'))">reset</button>
 </div>
 <div id=hint>set the slider before sending — the press applies to its previous answer, then your message goes</div>
</div>
<div id=brain>
 <div class=panel>
 <h3>INSIDE — live</h3>
 <div id=events>—</div>
 <svg id=chip viewBox="0 0 560 436" width="100%">
  <path id=w_in class=wire d="M52,200 L70,200"/>
  <path id=w_tc class=wire d="M178,200 L192,200"/>
  <path id=w_cp class=wire d="M416,168 L432,168"/>
  <path id=w_ph class=wire d="M486,168 L498,168"/>
  <path id=w_hpc_r class=wire d="M416,100 L424,100 L424,36 L525,36 L525,140"/>
  <path id=w_hpc_w class=wire d="M312,110 L322,110"/>
  <path id=w_goal class=wire d="M486,244 L525,244 L525,196"/>
  <path id=w_plan class=wire d="M262,356 L262,376"/>
  <path id=w_bg class=wire d="M330,356 L330,376"/>
  <path id=w_da class=wire d="M432,396 L412,396"/>
  <text class=tiny x=280 y=105>write</text>
  <text class=tiny x=430 y=33>decode-free read → logits</text>
  <text class=tiny x=492 y=236>goal bias</text>
  <rect class=blk x=8 y=180 width=44 height=40 rx=3 />
  <text class=lbl x=13 y=194>YOU</text><text class=val x=13 y=208 id=v_in>—</text>
  <rect class=blk x=70 y=150 width=108 height=100 rx=3 />
  <text class=lbl x=76 y=163>TRUNK · 13 layers</text>
  <g id=trunk></g>
  <text class=val x=76 y=243 id=v_trunk>—</text>
  <rect class=blk x=188 y=58 width=124 height=298 rx=3 />
  <text class=lbl x=194 y=71>COUNCIL · six clocks</text>
  <text class=tiny x=228 y=86>acc</text><text class=tiny x=262 y=86>cortex</text>
  <g id=bands></g>
  <rect class=blk x=322 y=58 width=94 height=298 rx=3 />
  <text class=lbl x=328 y=71>HIPPOCAMPUS</text>
  <text class=tiny x=328 y=81>episodic store · per-band lanes</text>
  <g id=lanes></g>
  <text class=val x=328 y=336 id=v_hpc>store quiet</text>
  <text class=val x=328 y=348 id=v_hpc2 style="fill:#1f7a46"></text>
  <rect class=blk x=432 y=140 width=54 height=56 rx=3 />
  <text class=lbl x=438 y=154>PFC</text>
  <text class=tiny x=438 y=164>route·ponder</text>
  <text class=val x=438 y=186 id=v_pfc>—</text>
  <rect class=blk x=498 y=140 width=54 height=56 rx=3 />
  <text class=lbl x=504 y=154>LEXICON</text>
  <text class=val x=504 y=172 id=v_head>—</text>
  <rect class=blk x=432 y=220 width=54 height=48 rx=3 />
  <text class=lbl x=438 y=233>GOAL</text>
  <g id=goals></g>
  <g id=clockw></g>
  <rect class=blk x=70 y=376 width=150 height=48 rx=3 />
  <text class=lbl x=76 y=389>PLAN / DREAMER · 8 steps</text>
  <g id=plan></g>
  <text class=val x=180 y=417 id=v_plan>—</text>
  <rect class=blk x=240 y=376 width=172 height=48 rx=3 />
  <text class=lbl x=246 y=389>BG · DA — value per band</text>
  <g id=vbars></g>
  <text class=val x=370 y=417 id=v_bg>—</text>
  <rect class=blk x=432 y=376 width=54 height=48 rx=3 />
  <text class=lbl x=438 y=389>PRESS</text>
  <text class=val x=438 y=406 id=v_press>—</text>
  <g id=fly></g>
 </svg>
 <div id=stream></div>
 <div id=frameinfo>live</div>
 <input id=tslider type=range min=0 max=0 value=0 oninput="scrub(parseInt(this.value))">
 <div style="display:flex;gap:8px;align-items:center">
  <button class=quiet onclick=playPause()><span id=playic>▶</span></button>
  <button class=quiet onclick=goLive()>LIVE</button>
  <span style="font-size:10.5px;color:#8aa0b5">every token is a frame — bands tick on their real clocks; nights play replay-by-replay</span>
 </div>
 <div id=footer>
  <span>mood <b id=moodnum>0</b></span>
  <div style="background:#eef2f6;border-radius:4px;height:8px;position:relative;width:90px">
   <div id=moodbar style="position:absolute;top:0;height:8px;border-radius:4px;background:#15803d;left:50%;width:0"></div>
  </div>
  <span id=selfpresses style="color:#b45309"></span>
  <span id=drives>—</span>
  <span id=moodfx style="color:#7d92a8"></span>
 </div>
 <div id=night>no night yet this session</div>
 </div>
</div>
</div>
<script>
const log=document.getElementById('log');
function add(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);log.scrollTop=1e9;return d}
function hlClear(){document.querySelectorAll('.tokhl,.msghl').forEach(e=>e.classList.remove('tokhl','msghl'))}
function hlTok(turn,i){hlClear();
 const bd=document.querySelector('#log .bot[data-turn="'+turn+'"]');
 const yd=document.querySelector('#log .you[data-turn="'+turn+'"]');
 if(i===0){if(yd){yd.classList.add('msghl');yd.scrollIntoView({block:'nearest'})}return}
 if(!bd)return;
 if(!bd.dataset.tok){const ws=bd.textContent.split(' ');bd.textContent='';
  ws.forEach((w,j)=>{const s=document.createElement('span');s.className='w';s.textContent=w;
   bd.appendChild(s);if(j<ws.length-1)bd.appendChild(document.createTextNode(' '))});
  bd.dataset.tok='1'}
 const sp=bd.querySelectorAll('.w'),k=Math.min(i,sp.length)-1;
 if(k>=0&&sp[k]){sp[k].classList.add('tokhl');sp[k].scrollIntoView({block:'nearest'})}}
let contOn=false,intOn=false,selfPressesToday=0,busy=false;
function toggleCont(){contOn=!contOn;const b=document.getElementById('contTgl');
 b.textContent='continuous time: '+(contOn?'on':'off');b.classList.toggle('on',contOn);
 add('sys',contOn?'~ continuous time on — it may speak first when it has been alone ~':'~ continuous time off ~')}
function toggleInt(){intOn=!intOn;const b=document.getElementById('intTgl');
 b.textContent='internals: '+(intOn?'on':'off');b.classList.toggle('on',intOn);
 document.getElementById('brain').style.display=intOn?'block':'none'}
function rwShow(){const v=parseFloat(document.getElementById('rw').value);
 const e=document.getElementById('rwval');e.textContent=(v>0?'+':'')+v;
 e.style.color=v>0?'var(--good)':(v<0?'var(--warn)':'var(--mut)')}
let evN=0;
function evt(txt,color,fi){const e=document.getElementById('events');
 if(evN===0)e.innerHTML='';evN++;
 const d=document.createElement('div');
 const now=new Date();
 const hh=('0'+now.getHours()).slice(-2)+':'+('0'+now.getMinutes()).slice(-2)+':'+('0'+now.getSeconds()).slice(-2);
 d.innerHTML='<span style="color:#b5c2ce;font-family:menlo,monospace;font-size:9.5px">'+hh+
  '</span> <span style="color:'+(color||'#9db0c2')+'">●</span> ';
 const t=document.createElement('span');t.textContent=txt;
 if(color)t.style.color=color;d.appendChild(t);
 if(fi!=null){d.style.cursor='pointer';d.title='click to replay this moment';
  d.onclick=()=>{document.getElementById('tslider').value=fi;scrub(fi)}}
 e.appendChild(d);
 while(e.children.length>100)e.removeChild(e.firstChild);
 e.scrollTop=1e9}

// ---- the anatomy ----
const BANDS=[3,4,5,6,7,8];
const CLK={3:1,4:8,5:64,6:512,7:4096,8:32768};
const CLKL={3:'1',4:'8',5:'64',6:'512',7:'4k',8:'32k'};
const KD={3:512,4:1024,5:2048,6:4096,7:4096,8:4096};
const trunkG=document.getElementById('trunk');
for(let i=0;i<13;i++){trunkG.innerHTML+='<rect id=tl'+i+' class=cell x='+(76+i*7.6)+' y=170 width=6 height=56 rx=1 />';}
const bandG=document.getElementById('bands');
BANDS.forEach((b,i)=>{const y=94+i*40;
 bandG.innerHTML+=
  '<text class=lbl x=194 y='+(y+15)+' style="font-size:8px">B'+b+'</text>'+
  '<text class=tiny x=194 y='+(y+24)+'>clk '+CLKL[b]+'</text>'+
  '<rect id=bnd'+b+' class=cell x=224 y='+y+' width=30 height=22 rx=2 />'+
  '<rect id=bndc'+b+' class=cell x=258 y='+y+' width=30 height=22 rx=2 />'+
  '<text class=val id=bndv'+b+' x=224 y='+(y+32)+' style="font-size:6.5px"></text>';});
const laneG=document.getElementById('lanes');
BANDS.forEach((b,i)=>{const y=94+i*40;
 laneG.innerHTML+=
  '<text class=tiny x=328 y='+(y+9)+'>B'+b+'</text>'+
  '<rect id=mc'+b+' class=cell x=328 y='+(y+12)+' width='+(30+Math.round(KD[b]/80))+' height=14 rx=2 />'+
  '<text class=tiny x='+(332+30+Math.round(KD[b]/80))+' y='+(y+22)+'>'+KD[b]+'</text>';});
const CKX=124,CKY=314,HLEN={3:31,4:27,5:23,6:19,7:15,8:11},HW={3:0.8,4:1.0,5:1.3,6:1.6,7:2.0,8:2.4};
(function(){const g=document.getElementById('clockw');let h='';
 h+='<text class=lbl x='+CKX+' y=272 text-anchor=middle style="font-size:8px">CLOCKWORK</text>';
 h+='<circle cx='+CKX+' cy='+CKY+' r=37 fill=#ffffff stroke=#c8ced8 stroke-width=1.1 />';
 for(let m=0;m<60;m++){const a=m*6*Math.PI/180,maj=m%5===0,r0=maj?32:34;
  h+='<line x1='+(CKX+Math.sin(a)*r0).toFixed(1)+' y1='+(CKY-Math.cos(a)*r0).toFixed(1)+
   ' x2='+(CKX+Math.sin(a)*36).toFixed(1)+' y2='+(CKY-Math.cos(a)*36).toFixed(1)+
   ' stroke='+(maj?'#98a2ac':'#dde2e8')+' stroke-width='+(maj?'0.9':'0.5')+' />'}
 BANDS.forEach(b=>{h+='<line id=ch'+b+' x1='+CKX+' y1='+CKY+' x2='+CKX+' y2='+(CKY-HLEN[b])+
  ' stroke='+(b===3?'#2f7d5c':'#2a323c')+' stroke-width='+HW[b]+' stroke-linecap=round />'});
 h+='<circle cx='+CKX+' cy='+CKY+' r=2.2 fill=#2a323c />';
 h+='<text id=ctc class=val x='+CKX+' y='+(CKY+16)+' text-anchor=middle style="font-size:6px">t0</text>';
 h+='<text class=tiny x='+CKX+' y=364 text-anchor=middle>one tick = one band firing</text>';
 g.innerHTML=h})();
function spawnFly(tr,delay){const g=document.getElementById('fly');if(!g||!tr)return;
 const mk=(txt,x,y,c,fs,w)=>{const e=document.createElementNS('http://www.w3.org/2000/svg','text');
  e.setAttribute('x',x);e.setAttribute('y',y);e.setAttribute('fill',c);e.setAttribute('font-size',fs);
  e.setAttribute('text-anchor','middle');e.setAttribute('class','flyt');
  if(w)e.setAttribute('font-weight','600');
  if(delay)e.style.animationDelay=delay+'s';
  e.textContent=txt;g.appendChild(e);
  setTimeout(()=>e.remove(),(delay||0)*1000+1700)};
 mk(tr.t+' \u00b7'+Math.round(tr.p*100),516,214,'#2f7d5c',7.5,1);
 (tr.alt||[]).filter(a=>a[0]!==tr.t).slice(0,2).forEach((a,j)=>
  mk(a[0]+' \u00b7'+Math.round(a[1]*100),488+j*56,228,'#98a2ac',6));}
function setClock(tc){
 BANDS.forEach(b=>{const a=(Math.floor(tc/CLK[b])%60)*6;
  const hd=document.getElementById('ch'+b);
  if(hd)hd.setAttribute('transform','rotate('+a+' '+CKX+' '+CKY+')')});
 const e=document.getElementById('ctc');if(e)e.textContent='t'+tc}
const goalG=document.getElementById('goals');
for(let i=0;i<4;i++){goalG.innerHTML+='<rect id=gs'+i+' class=cell x='+(438+(i%2)*22)+' y='+(238+Math.floor(i/2)*13)+' width=18 height=10 rx=2 />';}
const planG=document.getElementById('plan');
let pl='';for(let i=0;i<8;i++){
 if(i>0)pl+='<line x1='+(84+(i-1)*17+4)+' y1=402 x2='+(84+i*17-4)+' y2=402 stroke="#ccd7e1" stroke-width="1"/>';
 pl+='<circle id=pn'+i+' cx='+(84+i*17)+' cy=402 r=4 fill="#eef3f8" stroke="#b9c6d4"/>';}
planG.innerHTML=pl;
const vbG=document.getElementById('vbars');
BANDS.forEach((b,i)=>{vbG.innerHTML+='<line x1='+(252+i*26)+' y1=406 x2='+(252+i*26+18)+' y2=406 stroke="#d5dde5"/>'+
 '<rect id=vb'+b+' x='+(252+i*26)+' y=406 width=18 height=0 fill="#15803d"/>'+
 '<text class=tiny x='+(254+i*26)+' y=420>B'+b+'</text>';});

function setFill(id,c){const e=document.getElementById(id);if(e)e.style.fill=c}
function clearTicks(){BANDS.forEach(b=>{setFill('bnd'+b,'#eef3f8');setFill('bndc'+b,'#eef3f8');setFill('mc'+b,'#eef3f8')});
 for(let i=0;i<13;i++)setFill('tl'+i,'#eef3f8');
 for(let i=0;i<8;i++)setFill('pn'+i,'#eef3f8')}
function glow(id,mag){const e=document.getElementById(id);if(!e)return;
 const g=Math.min(1,mag);e.style.fill='rgb('+(238-100*g)+','+(243-45*g)+','+(248-12*g)+')';
 setTimeout(()=>{e.style.fill='#eef3f8'},2600)}
function flow(wireId,n,color){const svg=document.getElementById('chip');const w=document.getElementById(wireId);if(!w)return;
 for(let i=0;i<Math.min(n,14);i++){
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('r',2.2);c.setAttribute('class','dot');if(color)c.setAttribute('fill',color);
  const m=document.createElementNS('http://www.w3.org/2000/svg','animateMotion');
  m.setAttribute('dur',(1.0+Math.random()*0.8)+'s');m.setAttribute('begin',(i*0.11)+'s');
  m.setAttribute('path',w.getAttribute('d'));m.setAttribute('fill','freeze');
  c.appendChild(m);svg.appendChild(c);setTimeout(()=>c.remove(),3000+i*110)}}
function fmtD(d){return d>=1000?Math.round(d/1000)+'k':d>=100?Math.round(d):d>=1?d.toFixed(1):d.toFixed(2)}
function stream(line){const s=document.getElementById('stream');
 const d=document.createElement('div');d.textContent=line;d.title=line;
 s.insertBefore(d,s.firstChild);
 while(s.children.length>5)s.removeChild(s.lastChild)}
function valueBars(val){if(!val)return;
 Object.entries(val).forEach(([u,v])=>{const e=document.getElementById('vb'+u);if(!e)return;
  const h=Math.min(16,Math.abs(v)*4);e.setAttribute('height',h);
  e.setAttribute('y',v>=0?406-h:406);e.setAttribute('fill',v>=0?'#15803d':'#c2410c')});}
function turnBase(s){
 document.getElementById('v_in').textContent=s.qlen+' tok';
 document.getElementById('v_trunk').textContent='surp '+(s.surprise!=null?s.surprise:'—');
 BANDS.forEach(b=>{const e=document.getElementById('bndv'+b);if(e)e.textContent=''});
 let mx=1;(s.moved||[]).forEach(x=>{mx=Math.max(mx,x.delta)});
 (s.moved||[]).forEach(x=>{const m=x.part.match(/(acc(?:_c)?)\\/(\\d)/);
  if(m){const e=document.getElementById('bndv'+m[2]);if(e)e.textContent='Δ'+fmtD(x.delta)}});
 document.getElementById('v_pfc').textContent='pauses '+s.pauses;
 if(s.vote!=null&&s.vote>0.05){
  document.getElementById('v_hpc').textContent='store vote '+s.vote;
  document.getElementById('v_hpc2').textContent=(s.suggests||[]).slice(0,2).join(' ')}
 else{document.getElementById('v_hpc').textContent='store quiet';document.getElementById('v_hpc2').textContent=''}
 valueBars(s.value);
 if(s.value){const vs=Object.values(s.value);
  document.getElementById('v_bg').textContent='val '+(vs.reduce((a,c)=>a+c,0)/vs.length).toFixed(2)}}

// ---- the film: every token a frame ----
let frames=[],playing=null,liveMode=true;
function slider(){return document.getElementById('tslider')}
let TC=0;
function addFrames(fs){const atEnd=liveMode;
 fs.forEach(f=>{if(f.k==='tok')TC++;if(f.k==='nWake')TC=0;f.tc=TC});
 frames.push(...fs);
 const s=slider();s.max=frames.length-1;
 if(atEnd){s.value=frames.length-1;if(typeof setClock==='function')setClock(TC)}}
function turnFrames(turnN,q,s){
 const fs=[{k:'t0',turnN,q,s}];
 const total=(s.trace&&s.trace.length)||(s.tokens||s.words||1)+(s.pauses||0);
 for(let i=1;i<=total;i++)fs.push({k:'tok',turnN,q,s,i,total,
  tr:s.trace?s.trace[i-1]:null});
 return fs}
function nightFrames(n){
 const fs=[];
 (n.story||[]).forEach(it=>fs.push({k:'nItem',n,it}));
 (n.pairs||[]).forEach(p=>fs.push({k:'nRem',n,p}));
 fs.push({k:'nWake',n});
 return fs}
function renderFrame(i){
 const f=frames[i];if(!f)return;clearTicks();setClock(f.tc||0);
 if(!liveMode){if(f.k==='t0')hlTok(f.turnN,0);
  else if(f.k==='tok')hlTok(f.turnN,Math.max(1,Math.round(f.i*(f.s.words||1)/f.total)));
  else hlClear()}
 const info=document.getElementById('frameinfo');
 if(f.k==='t0'){turnBase(f.s);
  for(let j=0;j<13;j++)setFill('tl'+j,'#bcd3f0');
  if(f.s.vote>0.05)BANDS.forEach(b=>setFill('mc'+b,'#f4e4c8'));
  info.textContent='turn '+f.turnN+' · «'+f.q.slice(0,38)+'» · trunk pass, store '+(f.s.vote>0.05?'votes '+f.s.vote:'quiet')}
 else if(f.k==='tok'){turnBase(f.s);
  BANDS.forEach(b=>{if((f.i-1)%CLK[b]===0){setFill('bnd'+b,'#9cc3ef');setFill('bndc'+b,'#c4dbf5')}});
  setFill('tl'+((f.i-1)%13),'#bcd3f0');
  const wi=Math.max(1,Math.round(f.i*(f.s.words||1)/f.total));
  document.getElementById('v_head').textContent=Math.min(wi,f.s.words||0)+' / '+(f.s.words||0)+' words';
  if(f.tr)spawnFly(f.tr,0);
  info.textContent='turn '+f.turnN+' · cycle '+f.i+'/'+f.total+
   (f.tr?' · \u00ab'+f.tr.t+'\u00bb p='+f.tr.p:'')+
   ' — ticking: '+BANDS.filter(b=>(f.i-1)%CLK[b]===0).map(b=>'B'+b).join(' ')}
 else if(f.k==='press'){document.getElementById('v_press').textContent=f.m;
  info.textContent='press '+f.m+' — dopamine to BG';flow('w_da',5,f.m[0]==='+'?'#15803d':'#c2410c')}
 else if(f.k==='nItem'){
  BANDS.forEach(b=>setFill('mc'+b,'#d9c8f2'));
  flow('w_hpc_w',6,'#8b5cf6');
  info.textContent='night · replaying: «'+(f.it.q||'').slice(0,34)+'» '+(f.it.pre!=null?f.it.pre+' → '+(f.it.post!=null?f.it.post:'…'):'')+' ('+(f.it.verdict||'')+')'}
 else if(f.k==='nRem'){
  setFill('mc'+(3+(Math.abs(f.p.a.length)%6)),'#d9c8f2');setFill('mc'+(3+(Math.abs(f.p.b.length)%6)),'#d9c8f2');
  for(let j=0;j<8;j++)setFill('pn'+j,'#c9b8f5');
  info.textContent='night · dream: «'+f.p.a.slice(0,20)+'» × «'+f.p.b.slice(0,20)+'» charge '+f.p.charge}
 else if(f.k==='nWake'){info.textContent='morning — NREM '+f.n.nrem+' + REM '+f.n.rem+(f.n.woke?' · woke thinking: '+f.n.woke:'')}
 else if(f.k==='speaks'){info.textContent='it spoke first — nobody asked'}}
function scrub(i){liveMode=false;renderFrame(i)}
function playPause(){if(playing){clearInterval(playing);playing=null;document.getElementById('playic').textContent='▶';return}
 liveMode=false;document.getElementById('playic').textContent='⏸';
 playing=setInterval(()=>{const s=slider();let v=parseInt(s.value);
  if(v>=frames.length-1){clearInterval(playing);playing=null;document.getElementById('playic').textContent='▶';return}
  s.value=v+1;renderFrame(v+1)},260)}
function goLive(){liveMode=true;if(playing){clearInterval(playing);playing=null;document.getElementById('playic').textContent='▶'}
 const s=slider();s.value=frames.length-1;clearTicks();
 document.getElementById('frameinfo').textContent='live';hlClear();
 if(frames.length)renderFrame(frames.length-1)}

function rewardPanel(mood,val){
 if(mood!=null){
  const ad=document.getElementById('alivedot'),at=document.getElementById('alivetxt');
  if(ad){ad.style.background=mood>1.5?'var(--acc)':(mood<-1?'var(--warn)':'#a8b0a6');
   at.textContent='awake · mood '+mood.toFixed(1)}
  const m=Math.max(-8,Math.min(8,mood));
  document.getElementById('moodnum').textContent=mood.toFixed(1);
  const b=document.getElementById('moodbar');
  b.style.background=m>=0?'#15803d':'#c2410c';
  if(m>=0){b.style.left='50%';b.style.width=(m/8*50)+'%'}
  else{b.style.left=(50+m/8*50)+'%';b.style.width=(-m/8*50)+'%'}}
 if(val)valueBars(val);
 document.getElementById('selfpresses').textContent=
  selfPressesToday?('self-pressed x'+selfPressesToday+' today'):''}
let lastDr={f:0,l:0,b:0};
function drivesPanel(d){const el=document.getElementById('drives');if(!el||!d)return;
 el.innerHTML='fatigue <b>'+d.fatigue+'</b> · novelty '+d.bored_s+'s · alone '+d.lonely_s+'s'+
 (d.may_speak!=null?(' · may speak ×'+d.may_speak):'');
 if(d.fatigue>=10&&lastDr.f<10)evt('it is getting sleepy (fatigue '+d.fatigue+')','#8b5cf6');
 if(d.lonely_s>=400&&lastDr.l<400)evt('it has been alone a while — it may speak first soon','#2563eb');
 if(d.bored_s>=200&&lastDr.b<200)evt('nothing new for a while — it may start ruminating');
 lastDr={f:d.fatigue,l:d.lonely_s,b:d.bored_s}}

async function doPress(m){
 const r=await fetch('/press',{method:'POST',body:JSON.stringify({mag:m})}).then(r=>r.json());
 document.getElementById('v_press').textContent=(m>0?'+':'')+m;
 flow('w_da',Math.min(12,2+Math.abs(m)*2),m>0?'#15803d':'#c2410c');
 stream('P ▸ '+(m>0?'+':'')+m+' felt'+r.felt+
  (r.absorbed_steps?' absorb×'+r.absorbed_steps:'')+
  (r.corrected_steps?' unlearn×'+r.corrected_steps:'')+' mood'+r.mood);
 evt((m>0?'+':'')+m+' pressed · felt as '+r.felt+
  (r.absorbed_steps?(' · absorbed x'+r.absorbed_steps):'')+
  (r.corrected_steps?(' · corrective unlearning x'+r.corrected_steps):''),
  m>0?'#15803d':'#c2410c',frames.length);
 addFrames([{k:'press',m:(m>0?'+':'')+m}]);
 rewardPanel(r.mood,null);
 document.getElementById('rw').value=0;rwShow();return r}
async function pressSlider(){
 const m=parseFloat(document.getElementById('rw').value)||0;
 if(m===0){add('sys','~ move the slider first — zero is not a judgment ~');return}
 await doPress(m)}
let turnN=0;
async function send(){
 if(busy)return;
 const mEl=document.getElementById('msg');const t=mEl.value.trim();if(!t)return;
 busy=true;document.getElementById('sendbtn').disabled=true;
 try{
  const mv=parseFloat(document.getElementById('rw').value)||0;
  if(mv!==0)await doPress(mv);
  mEl.value='';add('you','you: '+t);add('sys think','· · ·');
  const r=await fetch('/chat',{method:'POST',body:JSON.stringify({text:t})}).then(r=>r.json());
  log.lastChild.remove();if(r.reply)add('bot',r.reply);else add('sys','~ it said nothing ~');
  if(r.pride)evt('conscience approved ('+r.pride+')','#15803d');
  if(r.self_press){if(r.self_press.mag>0){selfPressesToday++;
    document.getElementById('v_press').textContent='self +1';
    evt('IT PRESSED ITS OWN BUTTON +1 — conscience '+r.self_press.conscience+' · '+r.self_press.left_today+' left today','#15803d');
    flow('w_da',8,'#15803d')}
   else{document.getElementById('v_press').textContent='self −1';
    evt('it felt its own miss (self −1, feeling only)','#c2410c');flow('w_da',5,'#c2410c')}}
  if(r.noticed)evt('kept that (surprise '+r.noticed.surprise+' nats, dose x'+r.noticed.dose+') — it will dream it tonight','#15803d');
  turnN++;
  const words=(r.reply||'').split(' ').filter(Boolean).length;
  const snap={qlen:t.split(' ').length,surprise:r.surprise,moved:r.moved,
   words:words,tokens:words,pauses:r.pauses,
   vote:r.hpc?r.hpc.vote_max:null,suggests:r.hpc?r.hpc.suggests:null,
   noticed:!!r.noticed,value:r.value,trace:r.trace||null};
  const f0=frames.length;
  addFrames(turnFrames(turnN,t,snap));
  if(liveMode)(r.trace||[]).forEach((tr,j)=>spawnFly(tr,0.25+j*0.14));
  const _bd=[...document.querySelectorAll('#log .bot')].pop();if(_bd)_bd.dataset.turn=turnN;
  const _yd=[...document.querySelectorAll('#log .you')].pop();if(_yd)_yd.dataset.turn=turnN;
  const mvs=(r.moved||[]).map(x=>x.part.replace('acc_c/','c')
   .replace('acc/','a').replace('h/','h')+':'+fmtD(x.delta)).join(' ');
  stream('T'+turnN+' ▸ in'+snap.qlen+' surp'+r.surprise+
   (r.surprise_peak!=null?' pk'+r.surprise_peak:'')+
   ' │ '+mvs+
   (snap.vote>0.05?' │ hpc'+snap.vote+
    (((snap.suggests||[])[0])?' «'+snap.suggests[0]+'»':''):' │ hpc–')+
   (r.value?(' │ val'+(Object.values(r.value).reduce((a,c)=>a+c,0)/6).toFixed(2)):'')+
   ' │ mood'+r.mood);
  evt('turn '+turnN+' · «'+t.slice(0,30)+'» → '+words+' words · surp '+r.surprise+
   (snap.vote>0.05?' · store voted '+snap.vote:''),'#2563eb',f0);
  if(liveMode){turnBase(snap);
   for(let j=0;j<13;j++)setTimeout(()=>{setFill('tl'+j,'#bcd3f0');setTimeout(()=>setFill('tl'+j,'#eef3f8'),300)},j*40);
   flow('w_in',Math.min(12,2+Math.ceil(snap.qlen/3)));
   flow('w_tc',6);flow('w_cp',6);flow('w_ph',Math.min(12,1+Math.ceil(words/3)));
   document.getElementById('v_head').textContent=words+' words';
   let mx=1;(r.moved||[]).forEach(x=>{mx=Math.max(mx,x.delta)});
   (r.moved||[]).forEach(x=>{const m2=x.part.match(/(acc(?:_c)?)\\/(\\d)/);
    if(m2)glow((m2[1]==='acc_c'?'bndc':'bnd')+m2[2],x.delta/mx)});
   if(snap.vote>0.05)flow('w_hpc_r',Math.min(10,1+Math.round(snap.vote*3)),'#b45309');
   if(snap.noticed)flow('w_hpc_w',8,'#15803d');
   document.getElementById('frameinfo').textContent='live · turn '+turnN}
  rewardPanel(r.mood,r.value);
  if(r.drives)drivesPanel(r.drives);
  document.getElementById('moodfx').textContent=r.mood_fx?
   ('temp '+r.mood_fx.temp+' · bar '+(r.mood_fx.bar_shift>0?'+':'')+r.mood_fx.bar_shift):'';
 }finally{busy=false;document.getElementById('sendbtn').disabled=false}}
async function sleepy(){
 evt('falling asleep…','#8b5cf6');
 flow('w_hpc_w',10,'#8b5cf6');flow('w_plan',10,'#8b5cf6');
 const r=await fetch('/sleep',{method:'POST'}).then(r=>r.json());
 document.getElementById('v_plan').textContent=r.rem!=null?('REM x'+r.rem):'—';
 if(!r.error){const nf0=frames.length;
  addFrames(nightFrames({nrem:r.nrem,rem:r.rem,woke:r.woke_thinking,
   story:(r.night_story||[]).map(x=>({q:x.q,pre:x.pre,post:x.post,verdict:x.verdict})),
   pairs:r.rem_pairs||[]}));
  (r.night_story||[]).forEach(x=>stream('N ▸ '+(x.verdict||'')+' «'+
   (x.q||'').slice(0,22)+'»'+(x.post!=null?' '+x.pre+'→'+x.post:'')));
  (r.night_story||[]).forEach((x,i)=>evt('night · '+x.verdict+': «'+x.q.slice(0,26)+'»'+
   (x.post!=null?(' '+x.pre+' → '+x.post):''),'#8b5cf6',nf0+i));
  (r.rem_pairs||[]).forEach((p,i)=>evt('dream · «'+p.a.slice(0,18)+'» × «'+p.b.slice(0,18)+'» charge '+p.charge,
   '#8b5cf6',nf0+(r.night_story||[]).length+i));
  if(r.conscience)evt(r.conscience,'#b45309');
  if(r.pursuit)evt('pursuit: '+r.pursuit.state+(r.pursuit.items?(' — '+r.pursuit.items.join(' · ').slice(0,40)):''),'#b45309');
  if(r.genome)evt('genome: '+r.genome,'#b45309')}
 document.getElementById('night').innerHTML=r.error?r.error:
  'last night: NREM '+r.nrem+' + REM '+r.rem+(r.excited_night?' · <b>earned a longer night</b>':'')+' · '+r.lived_tokens+' lived tokens'+
  (r.conscience?('<br>'+r.conscience):'')+
  (r.genome?('<br>genome: '+r.genome):'')+
  (r.pursuit?('<br><b>pursuit '+r.pursuit.state+'</b>'+(r.pursuit.items?(': '+r.pursuit.items.join(' · ')):'')):'')+
  (r.woke_thinking?('<br>woke thinking about: '+r.woke_thinking):'')+
  (r.woke_feeling?('<br><b>woke and pressed its own button '+r.woke_feeling+'</b>'):'')+
  (r.tail_dropped?('<br>'+r.tail_dropped+' degenerate turn(s) kept out of the dreams'):'');
 evt(r.error||('woke: NREM '+r.nrem+' + REM '+r.rem),'#8b5cf6');
 if(!r.error){selfPressesToday=0;rewardPanel(r.woke_feeling?2:0,null)}}
async function saveLife(){
 const r=await fetch('/save',{method:'POST'}).then(r=>r.json());
 add('sys','saved -> '+r.saved+' ('+r.live_steps+' live steps)')}
setInterval(async()=>{if(!contOn)return;try{
 const p=await fetch('/pulse').then(r=>r.json());
 (p.events||[]).forEach(e=>{
  if(e.kind=='speaks'){evt('nobody asked — it spoke first','#2563eb');add('bot',e.text);
   (e.trace||[]).forEach((tr,j)=>spawnFly(tr,0.25+j*0.14));addFrames([{k:'speaks'}])}
  else if(e.kind=='kept_quiet')evt('it took the floor and had nothing to say');
  else if(e.kind=='ruminated')evt('ruminating on: '+e.about);
  else if(e.kind=='slept'){const n=e.night||{};
   evt('fell asleep on its own — NREM '+n.nrem+' over '+n.lived_tokens+' lived tokens','#8b5cf6');
   addFrames(nightFrames({nrem:n.nrem,rem:n.rem,woke:n.woke_thinking,story:[],pairs:[]}));
   if(n.woke_feeling)evt('it woke and pressed its own button '+n.woke_feeling,'#15803d');
   if(n.conscience)evt(n.conscience);
   selfPressesToday=0;rewardPanel(0,null)}});
 if(p.drives){drivesPanel(p.drives);rewardPanel(p.drives.mood,null)}
}catch(e){}},5000);
rwShow();
</script>"""

ORG = None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/pulse":
            with LOCK:
                ev = ORG.outbox[:]
                ORG.outbox = []
                d = ORG._drives()
                d["mood"] = round(ORG.mood, 2)
                d["may_speak"] = ORG.initiate_budget
            self._json({"events": ev, "drives": d})
            return
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        with LOCK:
            try:
                if self.path == "/chat":
                    self._json(ORG.chat(body["text"], body.get("temp")))
                elif self.path == "/press":
                    self._json(ORG.press(body.get("level"),
                                         body.get("mag")))
                elif self.path == "/teach":
                    self._json(ORG.teach(body["q"], body["a"]))
                elif self.path == "/facts":
                    self._json({"report_card": ORG.report_card()})
                elif self.path == "/reset":
                    ORG.reset()
                    self._json({"reset": True})
                elif self.path == "/sleep":
                    self._json(ORG.sleep())
                elif self.path == "/save":
                    self._json(ORG.save())
                else:
                    self._json({"error": "unknown"}, 404)
            except Exception as e:  # keep the app alive; report honestly
                self._json({"error": str(e)}, 500)


def main():
    global ORG
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tok")
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--port", type=int, default=8016)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--live-lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="data/nursery_body.pt")
    ap.add_argument("--notice-margin", type=float, default=0.75,
                    help="curiosity fires when a statement's surprise "
                         "exceeds the running mean by this many nats")
    ap.add_argument("--notice-floor", type=float, default=3.5,
                    help="absolute surprise floor for curiosity")
    ap.add_argument("--pursuit-adopt", type=float, default=0.8,
                    help="a fact drifting above this CE can be adopted "
                         "into a multi-night pursuit (needs >=2)")
    ap.add_argument("--pursuit-target", type=float, default=0.5,
                    help="the pursuit completes when every item is at "
                         "or below this CE")
    ap.add_argument("--notice-peak", type=float, default=15.5,
                    help="single-token surprise that fires curiosity "
                         "on its own (novelty is a peak, not a mean)")
    ap.add_argument("--store-decay", type=float, default=0.6,
                    help="episodic store carry across wakes (0 = the "
                         "old one-day store; 0.6 = ~2-3 day fade)")
    ap.add_argument("--tick", type=float, default=45.0,
                    help="autonomous clock period, seconds")
    ap.add_argument("--fatigue-bar", type=float, default=14.0,
                    help="sleep pressure: fatigue at which it falls "
                         "asleep on its own")
    ap.add_argument("--bore-bar", type=float, default=240.0,
                    help="seconds without novelty before it ruminates")
    ap.add_argument("--lone-bar", type=float, default=480.0,
                    help="seconds alone before it may speak first")
    a = ap.parse_args()
    print("[organism] the body alone — no assists exist in this build",
          file=sys.stderr)
    ORG = Organism(a)
    print(f"[organism] organism awake on http://localhost:{a.port} "
          f"({a.ckpt} on {ORG.dev})", file=sys.stderr)

    def _clock():
        # the autonomous clock: time passes without being spoken to
        while True:
            time.sleep(a.tick)
            with LOCK:
                try:
                    ORG.tick()
                except Exception as e_:
                    print("[tick]", e_, file=sys.stderr)
    threading.Thread(target=_clock, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
