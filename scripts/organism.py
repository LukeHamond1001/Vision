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

    def absorb_stmt(self, text, k=1, span=None):
        """one tiny keep-nudge on a statement it chose to notice; the
        night's replay does the real consolidation. A span (char
        range) confines the nudge to the chosen words — the aimed
        partial teach."""
        ids = self.tok.encode(text).ids + [self.eh]
        ids += [self.sil] * ((64 - len(ids) % 64) % 64)
        x = torch.tensor([ids[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        w = torch.zeros_like(y, dtype=torch.float)
        w[0, :len(self.tok.encode(text).ids) - 1] = 1.0
        if span is not None:
            offs = self.tok.encode(text).offsets
            w[0, :] = 0.0
            for j in range(len(offs) - 1):
                s_, e_ = offs[j + 1]   # w[j] trains token j+1
                if s_ < span[1] and e_ > span[0]:
                    w[0, j] = 1.0
            if float(w.sum()) == 0.0:
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
        """(probability, raw conviction) — the sigmoid gates the press;
        the logit is what the screen discloses. The probability
        saturates to 1.000000 on everything mastered, so it reads as
        scripted; the logit never repeats."""
        if self.critic is None or not reply:
            return None
        E = self.m.embed.weight.detach().float().cpu()
        ids = content_ids(self.tok, text + " " + reply)             or self.tok.encode(reply).ids
        v = torch.nn.functional.normalize(E[ids].mean(0), dim=-1)
        with torch.no_grad():
            raw = self.critic(v)
            return (float(torch.sigmoid(raw).item()), float(raw.item()))

    def _critic_attrib(self, text, reply, conv):
        """which words carried the conscience's judgment: leave-one-out
        over the content tokens, mapped back to char spans in the
        reply. Real attribution of the real critic — nothing authored."""
        try:
            E = self.m.embed.weight.detach().float().cpu()
            cids = content_ids(self.tok, text + " " + reply) \
                or self.tok.encode(reply).ids
            if len(cids) < 2:
                return None, None
            contrib = {}
            with torch.no_grad():
                for cid in cids:
                    rest = [i_ for i_ in cids if i_ != cid]
                    v_ = torch.nn.functional.normalize(
                        E[rest].mean(0), dim=-1)
                    contrib[cid] = conv - float(self.critic(v_).item())
            enc_r = self.tok.encode(reply)

            def spans_of(idset):
                sp = [[s_, e_] for (s_, e_), i_ in
                      zip(enc_r.offsets, enc_r.ids) if i_ in idset]
                return sp or None
            top = sorted(contrib, key=lambda i_: -contrib[i_])[:2]
            bot = sorted(contrib, key=lambda i_: contrib[i_])[:2]
            return (spans_of({i_ for i_ in top if contrib[i_] > 0}),
                    spans_of({i_ for i_ in bot if contrib[i_] < 0}))
        except Exception:
            return None, None

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
        tone_raw = []
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
                # the tone of the moment: the value heads' own press
                # expectation from the state this word was chosen in
                if hasattr(self.m, "read_value") \
                        and len(tone_raw) == len(out) - 1:
                    rv = self.m.read_value(self.st)
                    tone_raw.append(sum(rv.values()) / max(len(rv), 1)
                                    if rv else 0.0)
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
        tones = None
        if tone_raw and len(tone_raw) == len(out):
            kept = [(t_, v_) for t_, v_ in zip(out, tone_raw)
                    if t_ not in (self.sil, self.em)
                    and t_ not in press_vals]
            kid = [t_ for t_, _ in kept]
            if kid:
                raw_txt = self.tok.decode(kid)
                lead = len(raw_txt) - len(raw_txt.lstrip())
                tail = len(raw_txt.rstrip()) - lead
                tones, pos = [], 0
                for i_, (_, v_) in enumerate(kept):
                    nxt_ = len(self.tok.decode(kid[:i_ + 1]))
                    s_, e_ = pos - lead, min(nxt_ - lead, tail)
                    if e_ > max(s_, 0):
                        tones.append({"s": max(s_, 0), "e": e_,
                                      "v": round(v_, 2)})
                    pos = nxt_
                tones = tones or None
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
        scv = self._critic_score(text, reply) if reply else None
        sc, conv = scv if scv else (None, None)
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
                lov, _ = self._critic_attrib(text, reply, conv)
                self_press = {"mag": 1, "conscience": round(sc, 2),
                              "conviction": round(conv, 1),
                              "loved": lov,
                              "left_today": self.self_press_budget}
            elif sc < 0.15 and self.self_frown_budget > 0:
                self.feed([self.press_ids["<-1>"]])
                self.mood = self.mood * 0.9 - 1.0
                self.self_frown_budget -= 1
                self._self_pressed_qs.add(key_sp)
                self.press_log.append({"q": text[:80], "a": reply[:80],
                                       "mag": -1.0, "self": True})
                _, blm = self._critic_attrib(text, reply, conv)
                self_press = {"mag": -1, "conscience": round(sc, 2),
                              "conviction": round(conv, 1),
                              "blamed": blm,
                              "left_today": self.self_frown_budget}
        self.last_q = (text, reply)
        if reply:
            self.session.append((text, reply))
        return {"reply": reply, "pauses": pauses,
                "sidx": (len(self.session) - 1) if reply else None,
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
                "tones": tones,
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

    def press(self, level=None, mag=None, span=None, idx=None, who=None):
        """graded reward at any magnitude: the FELT token stays within
        the trained vocabulary (|m|>=2 -> level-2 token), while the
        magnitude expresses through plasticity — dose steps for
        positive, corrective unlikelihood for strong negative. The
        press can be AIMED: idx picks any exchange from today's
        session (the night closes the books), span confines the dose
        to the chosen words, and who='you' doses the human's own
        words (a partial teach, or a retract) instead of the reply.
        The serve matches literal selections; it never reads meaning.
        Feeling is felt NOW regardless of aim — like dopamine, the
        signal arrives after the act; the exact binding lives in the
        dose and the press log, both pinned to the aimed exchange."""
        if mag is None:
            mag = float(level.replace("+", ""))
        mag = max(-6.0, min(6.0, float(mag)))
        tgt = None
        if idx is not None:
            # an aimed press at a gone exchange (say, after the night
            # cleared the session) is felt but never dosed — a press
            # must not silently rebind to a different answer
            try:
                i_ = int(idx)
                if 0 <= i_ < len(self.session):
                    tgt = self.session[i_]
            except Exception:
                tgt = None
        else:
            tgt = self.last_q
        who = "you" if who == "you" else "bot"
        aim_txt = (tgt[0] if who == "you" else tgt[1]) if tgt else None
        sp_rng = None
        if span and aim_txt:
            c0 = aim_txt.find(span)
            if c0 < 0:
                c0 = aim_txt.lower().find(span.lower())
            if c0 >= 0:
                # a press doses whole tokens, never a cut one: the
                # range snaps OUTWARD to token boundaries, and what
                # snapped is what gets disclosed
                sp_rng = self._snap_span(aim_txt,
                                         (c0, c0 + len(span)))
        sign = "+" if mag >= 0 else "-"
        tokname = f"<{sign}{2 if abs(mag) >= 2 else 1}>"
        self.feed([self.press_ids[tokname]])
        self.mood = self.mood * 0.9 + mag
        self.fatigue += 0.2
        self.n_human_presses += 1
        self._today["presses"] += mag
        if tgt:
            e_ = {"q": tgt[0][:80], "a": (tgt[1] or "")[:80], "mag": mag}
            if who == "you":
                e_["stmt"] = True   # a teaching act, not a taste verdict
            self.press_log.append(e_)
            self.press_log = self.press_log[-500:]
        info = {"felt": tokname.strip("<>"), "mag": mag, "who": who,
                "mood": round(self.mood, 2)}
        if tgt and who == "you" and tgt[0]:
            if mag > 0:
                k = min(int(round(abs(mag))), 6) or 1
                self.absorb_stmt(tgt[0], k=k, span=sp_rng)
                info["absorbed_steps"] = k
            elif mag <= -2:
                k = min(int(round(abs(mag))) - 1, 3)
                self._unlearn_reply(tgt[0], k, span=sp_rng)
                info["corrected_steps"] = k
        elif tgt and mag > 0 and tgt[1]:
            q, ans = tgt
            k = min(int(round(abs(mag))), 6) or 1
            # plasticity satiates on the already-mastered: praise on a
            # strong fact is fully FELT, but barely re-absorbed — no
            # amount of loving presses turns one gold into a predator
            if self.fact_ce(q, ans) < 0.3:
                k = 1
            loss = self.absorb(q, ans, k, span=sp_rng)
            info["absorbed_steps"] = k
            info["loss"] = round(loss, 3)
        elif tgt and mag <= -2 and tgt[1]:
            k = min(int(round(abs(mag))) - 1, 3)
            self._unlearn_reply(tgt[1], k, span=sp_rng)
            info["corrected_steps"] = k
        if sp_rng:
            info["span"] = aim_txt[sp_rng[0]:sp_rng[1]].strip()[:60]
            info["span_at"] = [sp_rng[0], sp_rng[1]]
        return info

    def _snap_span(self, text, rng):
        """widen a char range outward to whole-token boundaries — a
        press can dose only whole tokens, never cut one."""
        s_, e_ = None, None
        for (a_, b_) in self.tok.encode(text).offsets:
            if a_ < rng[1] and b_ > rng[0]:
                s_ = a_ if s_ is None else min(s_, a_)
                e_ = b_ if e_ is None else max(e_, b_)
        return (s_, e_) if s_ is not None else rng

    def _unlearn_reply(self, reply, k=1, span=None):
        """corrective unlikelihood on the last reply's own tokens —
        the external NO, expressed as unlearning. An aimed press
        (span: char range) confines the NO to the chosen words."""
        keep = None
        if span is not None:
            enc = self.tok.encode(reply)
            keep = {j for j, (s_, e_) in enumerate(enc.offsets)
                    if s_ < span[1] and e_ > span[0]}
            if not keep:
                keep = None
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
                    if keep is not None and t_ not in keep:
                        continue
                    tok_ = ids[t_]
                    p_ = logp[j, tok_].exp().clamp(max=0.999)
                    ul = -torch.log1p(-p_)
                    loss = ul if loss is None else loss + ul
            if loss is not None:
                (0.3 * loss).backward()
                self.opt.step()
            self.n_steps += 1
        self.m.eval()

    def absorb(self, q, ans, k, span=None):
        tok = self.tok
        ids = (tok.encode(q).ids + [self.eh]
               + tok.encode(" " + ans).ids + [self.em])
        ids += [self.sil] * ((64 - len(ids) % 64) % 64)
        x = torch.tensor([ids[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        a0 = len(tok.encode(q).ids) + 1
        w = torch.zeros_like(y, dtype=torch.float)
        w[0, a0 - 1:a0 - 1 + len(tok.encode(" " + ans).ids) + 1] = 1.0
        if span is not None:
            # an aimed press: the dose lands only on the chosen words
            # (char range in ans; offsets are into " " + ans)
            w[0, :] = 0.0
            for j, (s_, e_) in enumerate(tok.encode(" " + ans).offsets):
                if s_ - 1 < span[1] and e_ - 1 > span[0]:
                    w[0, a0 - 1 + j] = 1.0
            if float(w.sum()) == 0.0:
                w[0, a0 - 1:a0 - 1
                   + len(tok.encode(" " + ans).ids) + 1] = 1.0
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
                if "mag" in e and not e.get("self")
                and not e.get("stmt")]
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
 --acc:#2f7d5c;--good:#1f7a46;--warn:#bd4a24}
*{box-sizing:border-box}
body{font:15px/1.5 -apple-system,'Segoe UI',sans-serif;margin:0;display:flex;flex-direction:column;height:100vh;background:var(--paper);color:var(--ink)}
#log{flex:1;overflow-y:auto;padding:28px 0}
#log>div{max-width:680px;margin-left:auto;margin-right:auto;padding:0 28px;animation:rise .18s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.you{color:var(--acc);margin:18px auto 4px;font-size:13.5px;font-weight:600;letter-spacing:.2px}
.bot{font-family:Georgia,'Times New Roman',serif;font-size:17px;line-height:1.55;color:var(--ink);margin:0 auto 6px;white-space:pre-wrap}
.sys{color:#b3aca0;font-size:11.5px;margin:6px auto;font-style:italic}
.think{color:var(--mut);font-size:20px;letter-spacing:4px;animation:thinkp 1.2s ease-in-out infinite}
@keyframes thinkp{0%,100%{opacity:.25}50%{opacity:.9}}
.selfp{font-size:12px;margin:2px auto 10px;display:table;padding:4px 12px;border-radius:999px;font-family:menlo,monospace;transition:opacity .8s;animation:feltpop .35s ease-out}
@keyframes feltpop{0%{transform:scale(.9)}60%{transform:scale(1.05)}100%{transform:scale(1)}}
.selfp.good{color:var(--good);background:rgba(31,122,70,.07)}
.selfp.bad{color:var(--warn);background:rgba(189,74,36,.07)}
.lov{border-bottom:2px solid var(--good)}
.blm{border-bottom:2px solid var(--warn)}
.upg{border-bottom:3px double var(--good)}
.upb{border-bottom:3px double var(--warn)}
#bar{display:flex;gap:10px;padding:14px 28px 8px;background:var(--paper);max-width:736px;margin:0 auto;width:100%}
#msg{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--ink);padding:13px 16px;border-radius:14px;font-size:15px;outline:none;transition:border .15s,box-shadow .15s}
#msg:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(47,125,92,.10)}
button{background:var(--ink);color:var(--paper);border:0;border-radius:12px;padding:11px 18px;cursor:pointer;font-size:14px;transition:opacity .15s}
button:hover{opacity:.85}
button:disabled{opacity:.35;cursor:default}
#sendbtn{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;padding:0;font-size:18px}
button.quiet{background:transparent;color:var(--mut);border:1px solid var(--line);padding:6px 13px;font-size:12px;border-radius:999px}
button.quiet:hover{opacity:1;border-color:var(--mut);color:var(--ink)}
#rewardrow{display:flex;align-items:center;gap:12px;padding:6px 28px;background:var(--paper);max-width:736px;margin:0 auto;width:100%}
#rwcap{flex:1;font-size:11px;color:#b3aca0}
#rwslide{flex:2;max-width:300px;accent-color:var(--good);cursor:pointer}
#rwslide:disabled{opacity:.35;cursor:default}
#rwval{font-family:menlo,monospace;font-size:13px;font-weight:600;min-width:44px;text-align:right}
#rwval.good{color:var(--good)}#rwval.bad{color:var(--warn)}
#carerow{display:flex;gap:8px;padding:4px 28px 8px;max-width:736px;margin:0 auto;width:100%}
</style>
<div id=log></div>
<div id=bar>
 <input id=msg placeholder="talk to it..." autofocus onkeydown="if(event.key==='Enter')send()">
 <button id=sendbtn onclick=send() title="send">&#8593;</button>
</div>
<div id=rewardrow>
 <span id=rwcap>react to its last answer</span>
 <span id=rwval></span>
 <input id=rwslide type=range min=-6 max=6 step=0.1 value=0>
</div>
<div id=carerow>
 <button class=quiet onclick=sleepy()>sleep</button>
 <button class=quiet onclick=saveLife()>save</button>
 <button class=quiet onclick="fetch('/reset',{method:'POST'}).then(()=>{felts=[];exs=[];aim=null;log.innerHTML='';cap();add('sys','~ fresh wake ~')})">reset</button>
</div>
<script>
const log=document.getElementById('log');
let busy=false,sleeping=false,felts=[],exs=[],aim=null;
function lockup(m){sleeping=m;document.getElementById('sendbtn').disabled=m;
 document.getElementById('rwslide').disabled=m}
function felt(cls,txt){felts.forEach(f=>{
  f.style.opacity=Math.max(0.35,(parseFloat(f.style.opacity)||1)*0.9)});
 const d=add(cls,txt);felts.push(d);return d}
function nightFade(){felts.forEach(f=>f.style.opacity=0.35);felts=[]}
function add(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);log.scrollTop=1e9;return d}
function trunc(t,n){return t.length>n?t.slice(0,n)+'…':t}
function lastEx(){return exs.length?exs[exs.length-1]:null}
function cap(){const rw=document.getElementById('rwcap');
 if(aim){rw.textContent=(aim.who=='you'?'dose your words: “':'react to: “')+trunc(aim.span,28)+'”';return}
 const le=lastEx();
 rw.textContent=le?'react to: “'+trunc(le.a,28)+'”':'react to its last answer'}
function seg(div,text,marks,prefix){
 div.textContent='';if(prefix)div.appendChild(document.createTextNode(prefix));
 const cl=v=>Math.max(0,Math.min(text.length,v));
 const bs=new Set([0,text.length]);
 marks.forEach(m=>{bs.add(cl(m.s));bs.add(cl(m.e))});
 const bl=[...bs].sort((a,b)=>a-b);
 for(let i=0;i<bl.length-1;i++){const s=bl[i],e=bl[i+1];if(e<=s)continue;
  const cov=marks.filter(m=>m.s<e&&m.e>s);
  if(!cov.length){div.appendChild(document.createTextNode(text.slice(s,e)));continue}
  const sp=document.createElement('span');
  sp.className=cov.map(m=>m.cls).filter(Boolean).join(' ');
  cov.forEach(m=>{if(m.bg)sp.style.background=m.bg;if(m.v!=null)sp.title='felt '+m.v});
  sp.textContent=text.slice(s,e);div.appendChild(sp)}}
function paintEx(ex){
 seg(ex.ad,ex.a,ex.tones.concat(ex.selfM,ex.userM.filter(m=>m.who!='you')));
 if(ex.userM.some(m=>m.who=='you'))
  seg(ex.qd,ex.q,ex.userM.filter(m=>m.who=='you'),'you: ')}
document.addEventListener('selectionchange',()=>{
 const s=document.getSelection();
 if(s&&s.rangeCount&&!s.isCollapsed){
  const t=s.toString().trim();
  if(t)for(const ex of exs){
   if(ex.ad.contains(s.anchorNode)&&ex.ad.contains(s.focusNode)&&ex.a.includes(t)){
    aim={ex:ex,sidx:ex.sidx,who:'bot',span:t};cap();return}
   if(ex.qd.contains(s.anchorNode)&&ex.qd.contains(s.focusNode)&&ex.q.includes(t)){
    aim={ex:ex,sidx:ex.sidx,who:'you',span:t};cap();return}}}
 if(aim){aim=null;cap()}});
async function doPress(m){
 if(sleeping){add('sys','~ it’s sleeping ~');return}
 if(busy){add('sys','~ wait — it’s mid-thought ~');return}
 const a=aim;
 const r=await fetch('/press',{method:'POST',body:JSON.stringify(
  {mag:m,span:a?a.span:undefined,idx:a?a.sidx:undefined,who:a?a.who:undefined})}).then(r=>r.json());
 const mv=(Number.isInteger(m)?''+Math.abs(m):Math.abs(m).toFixed(1));
 felt('selfp '+(m>0?'good':'bad'),'you: '+(m>0?'+':'−')+mv+' reward'+
  (r.span?(r.who=='you'?' · into your “':' · on “')+trunc(r.span,22)+'”':'')+
  (r.mood!=null?(' · mood '+r.mood.toFixed(2)).replace('-','−'):'')+
  (r.absorbed_steps?' · learned ×'+r.absorbed_steps:'')+
  (r.corrected_steps?' · unlearned ×'+r.corrected_steps:''));
 if(r.span_at&&a&&a.ex){
  a.ex.userM.push({s:r.span_at[0],e:r.span_at[1],who:a.who,cls:m>0?'upg':'upb'});paintEx(a.ex)}
 try{document.getSelection().removeAllRanges()}catch(e){}
 aim=null;cap()}
const sl=document.getElementById('rwslide'),rv=document.getElementById('rwval');
sl.oninput=()=>{const v=parseFloat(sl.value);
 rv.textContent=Math.abs(v)<0.3?'':(v>0?'+':'−')+Math.abs(v).toFixed(1);
 rv.className=v>0?'good':'bad';
 sl.style.accentColor=v<0?'var(--warn)':'var(--good)'};
sl.onchange=()=>{const m=Math.round(parseFloat(sl.value)*10)/10;
 sl.value=0;rv.textContent='';sl.style.accentColor='var(--good)';
 if(Math.abs(m)>=0.3)doPress(m)};
async function send(){
 const inp=document.getElementById('msg');const t=inp.value.trim();if(!t)return;
 if(sleeping){add('sys','~ it’s sleeping — wait for morning ~');return}
 if(busy)return;
 busy=true;document.getElementById('sendbtn').disabled=true;
 try{
  inp.value='';
  const qd=add('you','you: '+t);add('sys think','· · ·');
  let r;
  try{r=await fetch('/chat',{method:'POST',body:JSON.stringify({text:t})}).then(r=>r.json())}
  catch(e){
   if(log.lastChild&&log.lastChild.classList.contains('think'))log.lastChild.remove();
   add('sys','~ unreachable — is it awake yet? ~');return}
  log.lastChild.remove();
  if(r.error){add('sys','~ '+r.error+' ~');return}
  let bd=null;
  if(r.reply)bd=add('bot',r.reply);else add('sys','~ it said nothing ~');
  if(bd&&r.sidx!=null){
   const ex={q:t,a:r.reply,sidx:r.sidx,qd:qd,ad:bd,userM:[],selfM:[],
    tones:(r.tones||[]).filter(o=>Math.abs(o.v)>=0.05).map(o=>({s:o.s,e:o.e,v:o.v,
     bg:(o.v>0?'rgba(31,122,70,':'rgba(189,74,36,')+Math.min(.28,Math.abs(o.v)*.14).toFixed(3)+')'}))};
   if(r.self_press){
    if(r.self_press.loved)ex.selfM=r.self_press.loved.map(([s,e])=>({s:s,e:e,cls:'lov'}));
    if(r.self_press.blamed)ex.selfM=r.self_press.blamed.map(([s,e])=>({s:s,e:e,cls:'blm'}))}
   exs.push(ex);
   if(ex.tones.length||ex.selfM.length)paintEx(ex)}
  aim=null;cap();
  if(r.self_press){
   const sp=r.self_press;
   const why=(sp.conviction==null?'':(' · conscience '+(sp.conviction>0?'+':'')+sp.conviction.toFixed(1)).replace('-','−'));
   if(sp.mag>0)felt('selfp good','model: +1 reward'+why);
   else felt('selfp bad','model: −1 reward'+why)}
 }finally{busy=false;document.getElementById('sendbtn').disabled=false}}
async function sleepy(){
 if(sleeping||busy)return;
 lockup(true);
 const fl=add('sys think','~ sleeping… ~');
 try{
  const r=await fetch('/sleep',{method:'POST'}).then(r=>r.json());
  if(r.error){add('sys','~ '+r.error+' ~');return}
  nightFade();
  exs=[];aim=null;cap();
  const s=(n,w,p)=>n+' '+(n==1?w:(p||w+'s'));
  add('sys','~ morning — it replayed '+s(r.nrem,'memory','memories')+' and dreamt '+s(r.rem,'dream')+' ~');
  if(r.woke_feeling){const wf=r.woke_feeling;
   felt('selfp good','model: '+(wf.includes(' · ')?wf.replace(' · ',' reward · '):wf+' reward'))}
 }catch(e){add('sys','~ the night was interrupted — reload me ~')
 }finally{fl.classList.remove('think');lockup(false)}}
async function saveLife(){
 const r=await fetch('/save',{method:'POST'}).then(r=>r.json());
 add('sys','~ saved → '+r.saved+' ~')}
</script>
"""

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
                                         body.get("mag"),
                                         body.get("span"),
                                         body.get("idx"),
                                         body.get("who")))
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

    # the autonomous clock (tick(): self-sleep, rumination,
    # speaking first) is retired from the serve: a timer can be
    # bolted onto any model, so it dilutes rather than shows the
    # architecture. The machinery stays in the body; nights come
    # from the caretaker.
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
