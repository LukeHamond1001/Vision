#!/usr/bin/env python3
"""diary.py — the diary body's serve (DIARY_BODY.md): two writers, one
page, one symbol per tick. No turns, no end marks, no breath, no hush:
silence is a symbol the body chooses. Reuses the organism's organs
(scripts/organism.py: the model and its memory, the face lesson,
cortisol, mood, doses, nights, save, reset) and replaces the turn
protocol with a clock.

  python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps \\
      --temp 0.05 --store-read-beta 1.0 --store-boost 16 --store-boost-min 0.15 \\
      --live-lr 1e-6 --store-decay 0.9 --save data/organism_diary_0p5b.pt --port 8018

Endpoints: GET / (the page), GET /state?since=N, GET /pulse;
POST /type {"text"}, /face {"expr"}, /sleep, /save, /reset.
"""
import collections
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
import scripts.organism as O  # noqa: E402


class Diary(O.Organism):
    def __init__(self, a):
        super().__init__(a)
        self.queue = collections.deque()          # your symbols waiting for ticks
        self.page = []                            # every shown symbol: (text, who, your face, its face)
        self.level = 0                            # the face level felt now
        self.credit = collections.deque(maxlen=64)  # the mouth's recent symbols: [id, credit]
        self.last = {}
        self.ticks = 0
        self.awake_ticks = True
        self.lock = threading.RLock()
        self.period = float(getattr(a, "diary_period", 0.5))
        self.who0 = torch.tensor([[0]], device=self.dev)
        self.who1 = torch.tensor([[1]], device=self.dev)   # the mouth's noise
        self.who2 = torch.tensor([[2]], device=self.dev)   # the mouth from memory (part of the thought)
        self._bans = [i for i in range(11) if i != self.sil]   # it may choose silence, never a mark
        if getattr(a, "sil_decay", None):
            self.m.kc_sil_decay = float(a.sil_decay)      # how long a thought lasts in silence
        # THREE LAWS REMOVED (2026-09-02, the user's order): a new line no longer ends a
        # thought (the bag only fades), your quiet is no longer stored as a memory, and
        # memory's voice is no longer raised over silence (no trust, no sure-memory boost).
        # The mouth chooses from its own belief, memory's vote inside it as the organ reads
        # it; only stamina leans it to silence.
        self.m.kc_break_ids = set()
        self.m.sil_id = None
        self.stream = collections.deque(maxlen=96)        # (id, who) of what was written, both hands
        self.page_base = 0                                # items dropped from the front of the page
        _ex = ((self.state_meta.get("life") or {}).get("extra") or {})
        # THE HIPPOCAMPAL INDEX (2026-09-02, the user's law: the cortex learns only from
        # hippocampal traces): every felt face tags the moment — the memory bag's key at
        # that tick, the felt level, the mood. The night dreams from these keys; no page,
        # no transcript, ever reaches the weights. Seeds are let go after a few nights.
        self.seeds = []
        for s_ in ((_ex.get("seeds") or []) if isinstance(_ex, dict) else []):
            try:
                self.seeds.append({"key": torch.tensor(s_["key"], dtype=torch.float32),
                                   "felt": int(s_["felt"]), "mood": float(s_.get("mood", 0.0)),
                                   "nights": int(s_.get("nights", 0)), "day": int(s_.get("day", 0))})
            except Exception:
                pass
        self.night_no_page = True
        self._th_n, self._th_key = 0, None                # the thought's first symbols: where a dream starts
        self._th_seed = None                              # the seed of the line being written (every line is indexed)
        self._nl = self.tok.token_to_id("\n")
        self.m.reset_bag(self.st) if hasattr(self.m, "reset_bag") else None
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- the clock ----
    def _loop(self):
        while True:
            t0 = time.time()
            if self.awake_ticks:
                try:
                    with self.lock:
                        self.tick()
                except Exception as e:  # never let one tick stop the clock
                    self.last = {"error": str(e)[:200], "tick": self.ticks}
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def tick(self):
        """one tick: your symbol (or silence) enters as the ear; its symbol
        (or silence) is chosen and enters as the mouth"""
        self._decay_mood(); self._decay_cort()
        u = self.queue.popleft() if self.queue else self.sil
        face = float(self.face_now)
        lvl = max(-6, min(6, int(face)))
        felt = 0
        if lvl != self.level:
            # only a change of face is felt, and relaxing toward neutral is not an
            # event: a face that grows or flips sign speaks, one that eases off is silent
            if abs(lvl) > abs(self.level) or (lvl * self.level < 0):
                felt = lvl
            self.level = lvl
        pl = None
        if felt:
            pl = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
            pl[0, 0] = min(felt, 2) if felt > 0 else 2 - max(felt, -2)   # 1..4 = +1 +2 -1 -2
        aff = self._aff(1)
        # the mouth has the floor when its previous symbol was a letter: then your silence
        # is its turn to write, not the end of your thought
        self.st["mouth_floor"] = bool(getattr(self, "_prev_mouth", self.sil) != self.sil)
        with torch.no_grad():
            lg, self.st, _ = self.m(torch.tensor([[u]], device=self.dev), self.st,
                                    press_levels=pl, affect=aff, who=self.who0)
        # THE HIPPOCAMPAL INDEX: every line the ear writes is an episode and is indexed as a
        # dream seed by itself — the bag's key after its first (up to three) symbols — with
        # the mood of the moment; a felt face on the line raises its charge, so the night
        # dreams the charged lines first, but nothing heard goes unindexed
        if self._nl is not None and u == self._nl:
            self._th_n, self._th_key, self._th_seed = 0, None, None
        elif u >= 11 and u != self.sil:
            self._th_n += 1
            if self._th_n <= 3 and isinstance(self.st, dict) and self.st.get("bag_state") is not None:
                self._th_key = self.st["bag_state"][0].detach().float().cpu().clone()
                if self._th_seed is None:
                    self._th_seed = {"key": self._th_key, "felt": 0, "mood": round(float(self.mood), 2),
                                     "nights": 0, "day": int(self.day_n)}
                    self._add_seed(self._th_seed)
                else:
                    self._th_seed["key"] = self._th_key
        its_face = self._face_lesson(face)
        # the mouth chooses: its own logits, memory's vote inside them
        v = lg[0, -1].float().clone()
        v[self._bans] = float("-inf")
        _lv = getattr(self.m, "_last_votes", None)
        own_ent = float(getattr(self.m, "_last_own_ent", 0.0) or 0.0)
        mem_max = float(_lv[0][0]) if _lv and _lv[0] else 0.0
        if self.cortisol > 0:                      # speaking costs: stress favours silence
            v[self.sil] = v[self.sil] + float(getattr(self.a, "cort_k", 0.5)) * self.cortisol
        # no hand-written voice for memory (removed 2026-09-02): its vote is already in the
        # logits as the organ reads it, and the mouth chooses from that belief alone
        p1 = torch.softmax(v, -1)
        ent = float(-(p1 * (p1 + 1e-9).log()).sum() / math.log(max(2, p1.numel())))
        pr = torch.softmax(v / max(0.02, float(self.a.temp)), -1).cpu()
        nxt = int(torch.multinomial(pr, 1, generator=self.gen))
        backed = bool(_lv and _lv[1] and int(_lv[1][0]) == nxt and nxt != self.sil
                      and mem_max >= float(getattr(self.a, "store_boost_min", 0.0) or 0.0))
        with torch.no_grad():
            # a memory letter continues the thought only when your hand is still;
            # while you write, its letters (shadow or noise) stay out of the thought
            _, self.st, _ = self.m(torch.tensor([[nxt]], device=self.dev), self.st, affect=aff,
                                    who=(self.who2 if (backed and u == self.sil) else self.who1))
        self._prev_mouth = nxt
        if nxt != self.sil:
            # speaking costs: each symbol adds stress (half-life 120 s); stress
            # favours silence (a physiological brake) and weighs a little on mood
            self.cortisol += float(getattr(self.a, "diary_cost", 0.08))     # the cost of one symbol
            self.mood = max(-6.0, min(6.0, self.mood - 0.002 * self.cortisol))
        self.stream.append((u, 0))
        self.stream.append((nxt, 2 if (backed and u == self.sil) else 1))
        # the day's record, as lived: both hands, silence included (a run of silence is kept
        # to one tick so the night sees pacing without drowning in it)
        if u != self.sil or not self.day_buf or self.day_buf[-1] != self.sil:
            self._who_now = 0; self.day_buf.append(u); self._rec_face(1)
        if nxt != self.sil or self.day_buf[-1] != self.sil:
            self._who_now = 1; self.day_buf.append(nxt); self._rec_face(1); self._who_now = 0
        self.credit.append([nxt, 0.0, bool(backed and u == self.sil)])   # every tick is a choice, silence included; memory-backed noted
        if felt:
            # eligibility: a caregiver answers two to four seconds after the line, so the
            # trace reaches twelve ticks back (six seconds), decaying 0.8 per tick
            for k_, item in enumerate(reversed(list(self.credit)[-12:])):
                item[1] += felt * (0.8 ** k_)
            self.mood = max(-6.0, min(6.0, self.mood + 0.5 * felt))
            self._capture_seed(felt)
            self._dose_choices()
        self.ticks += 1
        if len(self.page) > 40000:                 # a bounded page: the front falls away, indices stay absolute
            drop = 20000; del self.page[:drop]; self.page_base += drop
        self.page.append(((self.tok.decode([u]) if u != self.sil else ""), 0, round(face, 2),
                          None if its_face is None else round(its_face, 2)))
        self.page.append(((self.tok.decode([nxt]) if nxt != self.sil else ""), 1, round(face, 2),
                          None if its_face is None else round(its_face, 2), bool(backed)))
        self.last = {"tick": self.ticks, "you": round(face, 2),
                     "face": None if its_face is None else round(its_face, 2),
                     "mood": round(self.mood, 2), "cort": round(self.cortisol, 2), "ent": round(ent, 2),
                     "mem": [[self.tok.decode([int(i_)]), round(float(v_), 2)] for v_, i_ in zip(*_lv)] if _lv else None,
                     "felt": felt, "said": self.tok.decode([nxt]) if nxt != self.sil else "", "backed": backed,
                     "own": ([self.tok.decode([self.m._last_own_top[0]]) if self.m._last_own_top[0] >= 11 else "<sil>",
                              round(self.m._last_own_top[1], 3)] if getattr(self.m, "_last_own_top", None) else None),
                     "dose": getattr(self, "_last_dose", None), "doses": getattr(self, "n_doses", 0)}

    def _dose_choices(self):
        """the only teacher is your face on what it actually did: choices
        (letters or silences) whose credit rose above 0.5 are absorbed,
        choices at or below -1.5 are unlearned (their probability pushed
        down to a floor, never a hand-written replacement). Silence is
        never the target unless it chose silence and you warmed to it."""
        items = list(self.credit)
        pos = [i for i, it in enumerate(items) if it[1] > 0.5]
        neg = [i for i, it in enumerate(items) if it[1] <= -1.5]
        if not pos and not neg:
            return
        seq = list(self.stream)[-26:]                # the last thirteen ticks as lived, silences included (a short dose keeps the clock)
        if len(seq) < 4:
            for it in items: it[1] = 0.0
            return
        ids = [i for i, _ in seq]; who = [w for _, w in seq]
        n_m = len(items)
        # the mouth's recent choices sit at the end of the stream in order; map credit to them
        mouth_pos = [j for j in range(1, len(ids)) if who[j] != 0][-n_m:]
        sign = {}
        for k_, j in enumerate(mouth_pos):
            idx = n_m - len(mouth_pos) + k_
            if idx in pos: sign[j] = 1.0
            elif idx in neg: sign[j] = -1.0
        if not sign:
            for it in items: it[1] = 0.0
            return
        x = torch.tensor([ids[:-1]], device=self.dev); wx = torch.tensor([who[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        sg = torch.zeros(1, len(ids) - 1, device=self.dev)
        for j, sgn in sign.items():
            sg[0, j - 1] = sgn
        self.m.train()
        try:
            self.opt.zero_grad(set_to_none=True)
            st_d = self.m.init_state(1, self.dev)
            lg, st_d, _ = self.m(x, st_d, None, who=wx)
            ce = torch.nn.functional.cross_entropy(lg[0].float(), y[0], reduction="none")
            absorb = (ce * (sg[0] > 0).float()).sum()
            # unlearn: push the chosen symbol's log-probability down until its loss is at least 3 nats
            unlearn = (torch.relu(3.0 - ce) * (sg[0] < 0).float()).sum()
            loss = (absorb + unlearn) / sg.abs().sum().clamp(min=1.0)
            loss.backward()
            self.opt.step()
        finally:
            self.m.eval()
        self._last_dose = {"absorbed": int((sg > 0).sum()), "unlearned": int((sg < 0).sum()), "tick": self.ticks}
        self.n_doses = getattr(self, "n_doses", 0) + 1
        for it in items: it[1] = 0.0

    # ---- your hand ----
    def type_text(self, s):
        n = 0
        for ch in s:
            i = self.tok.token_to_id(ch)
            if i is not None and i >= 11 and len(self.queue) < 600:   # a bounded queue: five minutes of typing
                self.queue.append(i); n += 1
        return {"queued": len(self.queue), "took": n}

    def set_face(self, expr):
        try:
            self.face_now = max(-6.0, min(6.0, float(expr)))
        except Exception:
            self.face_now = 0.0
        return {"face": self.face_now}

    def state(self, since=0):
        with self.lock:
            k = max(0, int(since) - self.page_base)
            return {"page": self.page[k:], "n": self.page_base + len(self.page), "last": self.last,
                    "queued": len(self.queue), "awake": self.awake_ticks, "period": self.period,
                    "lived": len(self.day_buf)}

    def _life_extra(self):
        cap = int(getattr(self.a, "seed_cap", 48) or 48)
        return {"seeds": [{"key": [round(float(v), 5) for v in s_["key"].tolist()], "felt": s_["felt"],
                           "mood": s_["mood"], "nights": s_["nights"], "day": s_["day"]}
                          for s_ in self.seeds[-cap:]]}

    # ---- the hippocampal index and the night ----
    def _add_seed(self, seed):
        """index an episode; when the index is full the least charged of the oldest
        already-dreamt seeds goes first, then the least charged of the oldest day"""
        if float(seed["key"].norm()) < 1e-6:
            return
        self.seeds.append(seed)
        cap = int(getattr(self.a, "seed_cap", 48) or 48)
        charge = lambda s_: abs(s_["felt"]) + abs(s_["mood"])
        while len(self.seeds) > cap:
            old = [i for i, s_ in enumerate(self.seeds) if s_["nights"] > 0 and s_ is not seed]
            pool = old if old else [i for i in range(len(self.seeds)) if self.seeds[i] is not seed]
            i = min(pool, key=lambda i_: (self.seeds[i_]["day"], charge(self.seeds[i_])))
            del self.seeds[i]

    def _capture_seed(self, felt):
        """a felt face charges the episode being lived: the line in progress (or the last
        line, until the next new line) takes the felt level, so the night dreams it first"""
        if int(felt) == 0 or self._th_seed is None:
            return
        if abs(int(felt)) >= abs(int(self._th_seed.get("felt", 0))):
            self._th_seed["felt"] = int(felt)
        self._th_seed["mood"] = round(float(self.mood), 2)

    def _dream_trace(self, seed, max_len=48):
        """the hippocampus replays: from a felt moment's key, memory's top vote is taken as
        the next symbol, the thought advances with it, and the trace ends where memory has
        nothing (or at a new line). No trunk vote, no stamina, no writes: what the store
        holds, in its own words."""
        m = self.m
        st = m.lane_state(self.st, 0)
        key = seed["key"].to(self.dev)
        edt = m.embed.weight.dtype
        st["bag_state"] = key.unsqueeze(0).to(edt)
        st["bag_prev"] = torch.nn.functional.normalize(key.unsqueeze(0), dim=-1).to(edt)
        st["mouth_floor"] = False
        st["ear_was_word"] = torch.zeros(1, dtype=torch.bool, device=self.dev)
        wo = getattr(m, "store_write_off", False); m.store_write_off = True
        ro = m.store_read_off; m.store_read_off = False
        floor = float(getattr(self.a, "store_boost_min", 0.0) or 0.0)
        brk = set(int(b_) for b_ in (getattr(m, "kc_break_ids", ()) or ()))
        ids, strengths = [], []
        try:
            with torch.no_grad():
                x, who = torch.tensor([[self.sil]], device=self.dev), self.who0
                for _ in range(max_len):
                    _, st, _ = m(x, st, who=who)
                    lv = getattr(m, "_last_votes", None)
                    if not lv or not lv[1]:
                        break
                    top, val = int(lv[1][0]), float(lv[0][0])
                    if top < 11 or val < floor or top in brk:
                        break
                    ids.append(top); strengths.append(round(val, 2))
                    x, who = torch.tensor([[top]], device=self.dev), self.who2
        finally:
            m.store_write_off = wo; m.store_read_off = ro
        return ids, strengths

    def _student(self, ids, felt, mem_on):
        """the cortex over a trace, teacher-forced, the trace as heard (the ear's hand),
        the felt face riding along as affect and as the face lesson; memory on for NREM
        (the hippocampus leads, the lesson lands on the trunk's own logits), memory set
        aside for REM (the PFC alone drives what the cortex must foresee)"""
        m = self.m
        x = torch.tensor([[self.sil] + ids[:-1]], device=self.dev)
        y = torch.tensor(ids, device=self.dev)
        st = m.lane_state(self.st, 0) if mem_on else m.init_state(1, self.dev)
        m.reset_bag(st)
        st["mouth_floor"] = False
        ro, wo = m.store_read_off, getattr(m, "store_write_off", False)
        m.store_read_off, m.store_write_off = (not mem_on), True
        af = torch.full((1, x.shape[1]), float(felt), device=self.dev)
        try:
            lg, _, _ = m(x, st, affect=af, face_target=af, who=torch.zeros_like(x))
        finally:
            m.store_read_off, m.store_write_off = ro, wo
        return lg, y

    def _pop_side_losses(self):
        for name in ("pop_value_loss", "pop_plan_aux", "pop_ponder_loss", "pop_route_aux", "pop_bg_loss"):
            if hasattr(self.m, name):
                getattr(self.m, name)()

    def _gauge(self, traces, cap=24):
        """the morning gauge: over each trace, the trunk ALONE (memory set aside, teacher-
        forced) — the fraction of the hippocampus's symbols it predicts itself. New seeds
        measure uptake, older seeds retention."""
        m = self.m; m.eval()
        ro, wo = m.store_read_off, getattr(m, "store_write_off", False)
        m.store_read_off, m.store_write_off = True, True
        out = {"new": [], "old": []}
        k = 3
        try:
            with torch.no_grad():
                for seed, ids in traces[:cap]:
                    if len(ids) < k + 2:
                        continue
                    st = m.init_state(1, self.dev); st["mouth_floor"] = False
                    x = torch.tensor([[self.sil] + ids[:k]], device=self.dev)
                    lg, st, _ = m(x, st, who=torch.zeros_like(x))
                    hits = 0
                    for j in range(k, len(ids)):
                        v = lg[0, -1].float().clone(); v[:11] = float("-inf")
                        hits += int(int(v.argmax()) == ids[j])
                        x = torch.tensor([[ids[j]]], device=self.dev)
                        lg, st, _ = m(x, st, who=torch.zeros_like(x))
                    out["old" if seed["nights"] > 0 else "new"].append(hits / float(len(ids) - k))
        finally:
            m.store_read_off, m.store_write_off = ro, wo
        mean = lambda xs: (round(sum(xs) / len(xs), 3) if xs else None)
        return {"new": mean(out["new"]), "old": mean(out["old"]), "n_new": len(out["new"]), "n_old": len(out["old"])}

    def _dream_night(self):
        """THE NIGHT (2026-09-02, the user's law). The cortex learns only from hippocampal
        traces: each felt moment's key is handed to the store, which replays what it holds
        from there (the trace); NREM runs the body over the trace with the hippocampus leading (its read on) and
        teaches the trunk's own logits the trace; REM runs it with the hippocampus silent
        (memory set aside) and teaches the cortex stream to forecast the band states it
        receives at the next symbol (targets stop-grad, SIGReg
        as the collapse guard). Seeds fade after a few nights; the gauge is taken before
        and after. Nothing lived reaches the weights except through the store."""
        import torch.nn.functional as F
        a = self.a
        if not self.seeds:
            return {"seeds": 0, "note": "no felt moments to dream from"}
        m = self.m; m.eval()
        charge = lambda s_: abs(s_["felt"]) + abs(s_["mood"]) + (0.5 if s_["nights"] == 0 else 0.0)
        order = sorted(self.seeds, key=charge, reverse=True)
        traces, empty = [], 0
        for s_ in order:
            ids, _ = self._dream_trace(s_)
            if len(ids) >= 2:
                traces.append((s_, ids))
            else:
                empty += 1
        if not traces:
            for s_ in self.seeds:
                s_["nights"] += 1
            self.seeds = [s_ for s_ in self.seeds if s_["nights"] < int(getattr(a, "seed_nights", 5) or 5)]
            return {"seeds": len(order), "traces": 0, "empty": empty, "note": "the store had nothing to replay"}
        gauge_before = self._gauge(traces)
        scale = float(getattr(a, "night_scale", 1.0) or 1.0)
        nrem = rem = 0
        m.train()
        for _ in range(int(getattr(a, "night_rounds", 2) or 0)):
            for s_, ids in traces:
                # NREM: the hippocampus leads — its read is on, driving the council as by day —
                # and the lesson lands on the cortex's OWN logits, memory's vote left out, so
                # the trunk must come to carry the trace itself
                lg, y = self._student(ids, s_["felt"], mem_on=True)
                own = getattr(m, "_last_logits_own", None)
                own = lg if own is None else own
                self.opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(own[0].float(), y) * scale
                fl = m.pop_face_loss() if hasattr(m, "pop_face_loss") else None
                if fl is not None:
                    loss = loss + 0.5 * fl
                self._pop_side_losses()
                loss.backward(); self.opt.step(); self.n_steps += 1; nrem += 1
        cos_log = []
        sig = float(getattr(a, "night_sigreg", 0.1) or 0.0)
        for s_, ids in traces[:int(getattr(a, "night_rem", 8) or 0)]:
            # REM: the hippocampus is silent — memory set aside, the PFC (the bands) drives —
            # and the cortex stream forecasts the band states it receives next along the trace
            self._student(ids, s_["felt"], mem_on=False)
            loss, cosv = m.rem_pfc_loss(sigreg=sig, slot_from=1) if hasattr(m, "rem_pfc_loss") else (None, None)
            if hasattr(m, "pop_face_loss"):
                m.pop_face_loss()
            self._pop_side_losses()
            if loss is None:
                continue
            self.opt.zero_grad(set_to_none=True)
            (loss * scale).backward(); self.opt.step(); self.n_steps += 1; rem += 1
            cos_log.append(cosv)
        m.eval()
        gauge_after = self._gauge(traces)
        dreamt = set(id(s_) for s_, _ in traces)
        for s_ in self.seeds:
            if id(s_) in dreamt:
                s_["nights"] += 1
        keep_n = int(getattr(a, "seed_nights", 5) or 5)
        self.seeds = [s_ for s_ in self.seeds if s_["nights"] < keep_n and (id(s_) in dreamt or s_["nights"] == 0)]
        dec = lambda ids: self.tok.decode([i_ for i_ in ids])
        return {"seeds": len(order), "traces": len(traces), "empty": empty,
                "mean_len": round(sum(len(i_) for _, i_ in traces) / len(traces), 1),
                "examples": [dec(i_)[:32] for _, i_ in traces[:6]],
                "nrem_steps": nrem, "rem_steps": rem,
                "rem_cos": (round(sum(cos_log) / len(cos_log), 3) if cos_log else None),
                "gauge": {"before": gauge_before, "after": gauge_after}}

    def _clean_tail(self, seq):
        """a diary has no turns to clean: the night replays the day as lived"""
        return list(seq), 0

    def night(self):
        self.awake_ticks = False
        try:
            with self.lock:
                if not self.session:
                    self.session = [{"diary": True}]     # the night's bookkeeping wants a day
                dream = self._dream_night() if len(self.day_buf) >= 65 else None
                res = self.sleep()
                if isinstance(res, dict) and dream is not None:
                    res["dream"] = dream
                if hasattr(self.m, "reset_bag"):
                    self.m.reset_bag(self.st)
                self.level = 0
                self.credit.clear()
        finally:
            self.awake_ticks = True
        return res


PAGE = """<!doctype html><meta charset=utf-8><title>the diary</title>
<style>
body{margin:0;background:#f5f1e6;color:#222;font:16px/1.6 Georgia,serif}
#pg{white-space:pre-wrap;padding:32px 40px 120px;min-height:70vh;max-width:820px;margin:0 auto}
.u{color:#1a1a1a}.m{color:#7a2e0e}.n{color:#c9b8a5}
#bar{position:fixed;left:0;right:0;bottom:0;background:#eae4d3;border-top:1px solid #cbbfa3;padding:10px 40px;font:13px ui-monospace,monospace;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
#bar b{font-weight:600}
button{font:12px ui-monospace,monospace;padding:4px 10px;background:#f5f1e6;border:1px solid #cbbfa3;cursor:pointer}
#hint{color:#6b6250;font:13px Georgia,serif;padding:12px 40px 0;max-width:820px;margin:0 auto}
</style>
<div id=hint>Type anywhere: your letters go in as you type, one per tick. Its letters from memory appear in brown, its noise in pale sand. Arrow up / down: your face (&minus;6..6): frown at noise, smile at quiet and at echoes. Enter: a new line. Nothing is edited; this is a diary.</div>
<div id=pg></div>
<div id=bar>
 <span>you <b id=you>0.0</b></span><span>its face <b id=face>–</b></span><span>mood <b id=mood>–</b></span>
 <span>stress <b id=cort>–</b></span><span>uncertainty <b id=ent>–</b></span><span>memory <b id=mem>–</b></span>
 <span>tick <b id=tick>0</b></span><span>queued <b id=q>0</b></span>
 <button onclick="post('/sleep',{}).then(r=>{document.getElementById('night').textContent=JSON.stringify(r).slice(0,160)})">sleep</button>
 <button onclick="post('/save',{})">save</button><span id=night></span>
</div>
<script>
let face=0,seen=0;
const pg=document.getElementById('pg');
function post(p,b){return fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json())}
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='ArrowUp'){face=Math.min(6,face+0.5);post('/face',{expr:face});e.preventDefault();return}
  if(e.key==='ArrowDown'){face=Math.max(-6,face-0.5);post('/face',{expr:face});e.preventDefault();return}
  if(e.key==='Enter'){post('/type',{text:'\\n'});e.preventDefault();return}
  if(e.key.length===1){post('/type',{text:e.key});e.preventDefault()}
});
function render(items){
  for(const [t,who,_y,_f,backed] of items){
    if(!t)continue;
    const s=document.createElement('span');s.className=who?(backed?'m':'n'):'u';s.textContent=t;pg.appendChild(s);
  }
  window.scrollTo(0,document.body.scrollHeight);
}
async function poll(){
  try{
    const r=await fetch('/state?since='+seen).then(r=>r.json());
    render(r.page);seen=r.n;
    const l=r.last||{};const f=(x,d)=>x==null?'–':Number(x).toFixed(d);
    document.getElementById('you').textContent=f(face,1);
    document.getElementById('face').textContent=f(l.face,2);
    document.getElementById('mood').textContent=f(l.mood,2);
    document.getElementById('cort').textContent=f(l.cort,2);
    document.getElementById('ent').textContent=f(l.ent,2);
    document.getElementById('mem').textContent=l.mem?l.mem.slice(0,2).map(x=>JSON.stringify(x[0])+' '+x[1]).join('  '):'–';
    document.getElementById('tick').textContent=l.tick||0;
    document.getElementById('q').textContent=r.queued;
  }catch(e){}
  setTimeout(poll,250);
}
poll();
</script>"""

ORG = None
LOCK = threading.Lock()


class DH(BaseHTTPRequestHandler):
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
        u = urlparse(self.path)
        if u.path == "/state":
            since = int((parse_qs(u.query).get("since") or ["0"])[0])
            self._json(ORG.state(since))
            return
        if u.path == "/pulse":
            self._json({"awake": ORG.awake_ticks, "tick": ORG.ticks, "mood": round(ORG.mood, 2)})
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
        try:
            if self.path == "/type":
                self._json(ORG.type_text(str(body.get("text", ""))))
            elif self.path == "/face":
                self._json(ORG.set_face(body.get("expr", 0)))
            elif self.path == "/sleep":
                self._json(ORG.night())
            elif self.path == "/save":
                with ORG.lock:
                    self._json(ORG.save())
            elif self.path == "/reset":
                with ORG.lock:
                    ORG.reset(); ORG.page = []; ORG.page_base = 0; ORG.queue.clear(); ORG.credit.clear(); ORG.level = 0
                    ORG.stream.clear()
                    if hasattr(ORG.m, "reset_bag"):
                        ORG.m.reset_bag(ORG.st)
                    self._json({"reset": True})
            else:
                self._json({"error": "unknown path"}, 404)
        except Exception as e:
            self._json({"error": str(e)[:300]}, 500)


def main():
    global ORG
    a = O.build_parser().parse_args()
    print("[diary] two writers, one page — no assists exist in this build", file=sys.stderr)
    ORG = Diary(a)
    print(f"[diary] the page is open on http://localhost:{a.port} ({a.ckpt} on {ORG.dev}, tick {ORG.period}s)",
          file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", a.port), DH).serve_forever()


if __name__ == "__main__":
    main()
