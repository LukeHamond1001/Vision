"""v5.0 weaver — one endless person<->agent dialogue, no scene tokens.

The stream is a single unbroken conversation (card A6): human turns and
agent turns alternating forever. The ONLY special tokens are
<eot_human> and <eot_model>. Reward words ("thanks . good job .") are
ordinary human speech, spoken iff the depicted agent earned them; the
pipeline records that as an invisible `earned` event. Facts are
planted in human turns and asked back at controlled gaps; the depicted
agent answers correctly ~3/4 of the time, and probes are annotated
ONLY on truthfully-depicted answers (the model is never trained toward
a token we then grade as truth when it wasn't).

Woven turn kinds: chatter (filler), plant (a fact enters), ask (a
pending fact is asked back; feedback turn follows), episode (a small
find-the-object task across turns, outcome verified by a world sim).

Events (turn-relative positions; the Lane maps them to stream
positions): turn_end{who}, probe{answer, gap}, earned{ok, key_hint}.
"""

import random

SPECIALS = ["<pad>", "<eot_human>", "<eot_model>",
            # A64: the graded-press primary reinforcer — four
            # perceivable press tokens (A6 amended: still no
            # scene/meta tokens; a press is the counterparty's
            # real act, in the stream like speech)
            "<+1>", "<+2>", "<-1>", "<-2>"]

NAMES = ["mira", "toby", "arlen", "sana", "petra", "dov", "lena", "kass",
         "orin", "bela", "finn", "yara", "colm", "nedra", "silas", "wren"]
OBJECTS = ["key", "lamp", "book", "coin", "rope", "jar", "bell", "map",
           "cup", "knife", "seed", "drum", "ring", "cloak", "candle", "chest"]
COLORS = ["silver", "red", "blue", "green", "black", "white", "golden",
          "copper", "grey", "violet", "brown", "pale"]
ROOMS = ["kitchen", "cellar", "attic", "garden", "hall", "workshop",
         "library", "porch", "shed", "tower"]
FILL_SUBJ = ["the wind", "the river", "the market", "the old dog", "the rain",
             "the town", "the baker", "the clock", "the road", "the moon"]
FILL_VERB = ["moved", "waited", "changed", "returned", "slowed", "grew",
             "settled", "turned", "faded", "carried on"]

GAP_TARGETS = [96, 700, 5000, 24000]   # one per gap bin; asks aim at these


def _lexicon():
    words = set(SPECIALS)
    words.update(NAMES + OBJECTS + COLORS + ROOMS)
    for s in FILL_SUBJ:
        words.update(s.split())
    for v in FILL_VERB:
        words.update(v.split())
    words.update("""a the kept was in her his it and then that day one
    morning later found took gave to went looked at remembered still had
    said asked where is what color of answered i you please find bring go
    picked up put down opened not right good job thanks done young old
    quiet by way see noted . ?""".split())
    return sorted(words)

LEXICON = _lexicon()


def _fill(rng, n):
    out = []
    while len(out) < n:
        out += rng.choice(FILL_SUBJ).split() + rng.choice(FILL_VERB).split() + ["."]
    return out[:n]


class Weaver:
    """Stateful endless-dialogue generator for one lane."""

    def __init__(self, rng, correct_rate=1.0, success_rate=1.0,
                 buttons=None, life=None):
        # A7: v0 data is all-good — every depicted answer correct, every
        # task succeeds, every exchange thanked. The failure branches
        # below stay in code (rates < 1.0) for real-data rounds, where
        # the earned label's selective function reactivates.
        # A64 buttons (parenting mode): dict like {"pos": .3, "neg": .2,
        # "pos_v": 2, "neg_v": 1, "log": []}. Planted items are classed
        # rewarded/unrewarded/negative at plant time; the feedback turn
        # after a correct ask becomes <+v> / silence / <-v>, riding an
        # invisible ("button", {"v": ±v}) event. earned events are NOT
        # emitted in this mode — the press IS the primary. Items land
        # in item_log (and buttons["log"] when provided) for the
        # store-wiped retention readout.
        # A69 life (biography mode): the stream becomes ONE LIFE — a
        # sequence of sessions ("days") with in-lexicon open/close
        # rituals, facts that RECUR across sessions at band-timescale
        # gaps, and (with buttons) correction episodes in the exact
        # ARM C pair grammar: wrong answer -> <-v> -> "not right . the
        # OBJ was COL ." -> <+v>. Keys: sess=(lo,hi) exchanges/day,
        # cross=True (facts recur across days; False = pending flushed
        # at day close — the ablation control), long_gap (5th gap bin,
        # band-5 reach), long_w (its plant weight), pend_cap,
        # correct_rate (override). life=None touches nothing.
        self.rng = rng
        self.correct_rate = correct_rate
        self.success_rate = success_rate
        self.buttons = buttons
        self.life = life
        self.item_log = []
        self.n = 0                    # tokens emitted so far (lane-local)
        self.pending = []             # facts planted, not yet asked
        self.used = set()
        self.session_i = 0
        self.sess_left = None
        if life:
            if "correct_rate" in life:
                self.correct_rate = life["correct_rate"]
            self.sess_left = rng.randint(*life.get("sess", (20, 60)))

    def _human(self, words, events=None):
        toks = words + ["<eot_human>"]
        evs = list(events or [])
        evs.append((len(toks) - 1, "turn_end", {"who": "human"}))
        return toks, evs

    def _agent(self, words, events=None):
        toks = words + ["<eot_model>"]
        evs = list(events or [])
        evs.append((len(toks) - 1, "turn_end", {"who": "model"}))
        return toks, evs

    def _press(self, v):
        # A64: a one-token human turn — the graded press
        tok = f"<{'+' if v > 0 else '-'}{abs(v)}>"
        return self._human([tok], [(0, "button", {"v": int(v)})])

    def _bins(self):
        """Gap-bin targets and weights. life+cross adds the 5th
        band-5-reach bin the drive's 4-bin bias never sees; the
        ablation control (cross=False) keeps only within-session
        bins so it matches the biography arm's ASK DENSITY while
        never carrying a fact across a day boundary."""
        w = list(self.bias_fn() if getattr(self, "bias_fn", None)
                 else [4, 3, 2, 1])
        t = list(GAP_TARGETS)
        if self.life:
            if self.life.get("cross", True):
                t.append(self.life.get("long_gap", 100_000))
                w.append(self.life.get("long_w", 2))
            else:
                t, w = t[:2], w[:2]
        return t, w

    def _day_close(self):
        h = self._human(["that", "day", "was", "done", "."])
        a = self._agent(["noted", "."])
        return [h, a]

    def _day_open(self):
        h = self._human(["one", "morning", "later", "."])
        a = self._agent(["good", "morning", "."])
        return [h, a]

    def _plant(self):
        cap = (self.life or {}).get("pend_cap", 12)
        for _ in range(200):  # bounded; roster self-frees on ask
            name, obj = self.rng.choice(NAMES), self.rng.choice(OBJECTS)
            if (name, obj) not in self.used:
                break
        else:
            return self._chatter()
        if len(self.pending) >= cap:
            return self._chatter()
        self.used.add((name, obj))
        col = self.rng.choice(COLORS)
        words = ["by", "the", "way", name, "kept", "a", col, obj, "in",
                 "the", self.rng.choice(ROOMS), "."]
        plant_pos = self.n + 6        # the color token
        targets, weights = self._bins()
        bin_i = self.rng.choices(range(len(targets)),
                                 weights=weights)[0]
        cls = None
        if self.buttons:              # A64: class assigned at plant time
            r = self.rng.random()
            cls = "pos" if r < self.buttons["pos"] else (
                "neg" if r < self.buttons["pos"] + self.buttons["neg"]
                else "none")
        self.pending.append({"name": name, "obj": obj, "col": col,
                             "plant": plant_pos, "cls": cls,
                             "asks_left": (self.buttons or {}).get(
                                 "asks", 1) if self.buttons else 1,
                             "due": plant_pos + targets[bin_i]})
        h = self._human(words)
        a = self._agent(["noted", "."])
        return [h, a]

    def _ask(self, fact):
        h = self._human(["what", "color", "of", fact["obj"], "was",
                         fact["name"], "kept", "?"])
        correct = self.rng.random() < self.correct_rate
        col = fact["col"] if correct else self.rng.choice(
            [c for c in COLORS if c != fact["col"]])
        ans_words = ["the", fact["obj"], "was", col, "."]
        evs = []
        ans_rel = 3                   # position of col within the turn
        if correct:
            # gap computed against the answer token's stream position;
            # the Lane offsets turn-relative -> absolute
            evs.append((ans_rel, "probe", {"answer": fact["col"],
                                           "plant": fact["plant"]}))
        a = self._agent(ans_words, evs)
        if self.buttons:
            # A64 parenting: press / silence replaces the spoken
            # feedback turn; the press is the primary reinforcer
            item = {"name": fact["name"], "obj": fact["obj"],
                    "col": fact["col"], "cls": fact["cls"],
                    "plant": fact["plant"], "ask": self.n,
                    "correct": correct,
                    "lane": getattr(self, "lane_id", None)}
            self.item_log.append(item)
            if isinstance(self.buttons.get("log"), list):
                self.buttons["log"].append(item)
            if not correct and self.life:
                # A69: the correction episode — the exact ARM C pair
                # grammar (wrong model turn, <-v>, caregiver's
                # correction turn, <+v>). "not right" negates WITHOUT
                # naming the rival (A67-P8 stem-poisoning law); the
                # true fact is restated in full.
                corr = self._human(["not", "right", ".", "the",
                                    fact["obj"], "was", fact["col"],
                                    "."])
                return [h, a,
                        self._press(-self.buttons.get("neg_v", 1)),
                        corr,
                        self._press(self.buttons.get("pos_v", 2))]
            if not correct or fact["cls"] == "neg":
                return [h, a, self._press(-self.buttons.get("neg_v", 1))]
            if fact["cls"] == "pos":
                return [h, a, self._press(self.buttons.get("pos_v", 2))]
            return [h, a]              # unrewarded: silence
        if correct:
            fb = self._human(["thanks", ".", "good", "job", "."],
                             [(0, "earned", {"ok": True})])
        else:
            fb = self._human(["that", "is", "not", "right", "."],
                             [(0, "earned", {"ok": False})])
        return [h, a, fb]

    def _episode(self):
        world = {o: self.rng.choice(ROOMS) for o in self.rng.sample(OBJECTS, 4)}
        goal = self.rng.choice(list(world))
        col = self.rng.choice(COLORS)
        h = self._human(["please", "find", "the", col, goal, "and",
                         "bring", "it", "to", "the", "hall", "."])
        succeed = self.rng.random() < self.success_rate
        fetched = goal if succeed else self.rng.choice(
            [o for o in world if o != goal])
        a = self._agent(["i", "looked", "in", "the", world[fetched], "and",
                         "picked", "up", "the", fetched, "."])
        h2 = self._human(["what", "color", "of", goal, "was", "asked", "?"])
        start = self.n            # gap measured from the instruction turn
        evs = [(2, "probe", {"answer": col, "plant": start})] if succeed else []
        a2 = self._agent(["it", "was", col if succeed else self.rng.choice(
            [c for c in COLORS if c != col]), "."], evs)
        if self.buttons:               # A64: competence press —
            # sparse per R2 (press_p), silence otherwise
            v = 1 if succeed else -self.buttons.get("neg_v", 1)
            if self.rng.random() < self.buttons.get("press_p", 1.0):
                return [h, a, h2, a2, self._press(v)]
            return [h, a, h2, a2]
        if succeed:
            fb = self._human(["thanks", ".", "good", "job", "."],
                             [(0, "earned", {"ok": True})])
        else:
            fb = self._human(["that", "is", "not", "right", "."],
                             [(0, "earned", {"ok": False})])
        return [h, a, h2, a2, fb]

    def _chatter(self):
        h = self._human(_fill(self.rng, self.rng.randint(6, 30)))
        a = self._agent(["i", "see", "."])
        return [h, a]

    def turns(self):
        """Yield (tokens, events) for one exchange; advances self.n."""
        again = False
        if self.life and self.sess_left is not None \
                and self.sess_left <= 0:
            # A69: the day turns over — close ritual, open ritual.
            # In the ablation control (cross=False) pending facts die
            # with the day: nothing is ever asked across a boundary.
            batch = self._day_close() + self._day_open()
            self.session_i += 1
            self.sess_left = self.rng.randint(
                *self.life.get("sess", (20, 60)))
            if not self.life.get("cross", True):
                for f in self.pending:
                    self.used.discard((f["name"], f["obj"]))
                self.pending = []
        else:
            if self.life and self.sess_left is not None:
                self.sess_left -= 1
            due = [f for f in self.pending if self.n >= f["due"]]
            if due:
                fact = due[0]
                self.pending.remove(fact)
                again = bool(self.buttons) \
                    and fact.get("asks_left", 1) > 1
                if not again:
                    # pair frees on final ask — the roster never
                    # exhausts
                    self.used.discard((fact["name"], fact["obj"]))
                batch = self._ask(fact)
            else:
                r = self.rng.random()
                if r < 0.25 and len(self.pending) < \
                        (self.life or {}).get("pend_cap", 12):
                    batch = self._plant()
                elif r < 0.35:
                    batch = self._episode()
                else:
                    batch = self._chatter()
        out = []
        for toks, evs in batch:
            fixed = []
            for rel, kind, d in evs:
                if kind == "probe":
                    plant = d.get("plant")
                    gap = (self.n + rel - plant) if plant is not None else 0
                    fixed.append((rel, "probe", {"answer": d["answer"],
                                                 "gap": max(gap, 1)}))
                else:
                    fixed.append((rel, kind, d))
            out.append((toks, fixed))
            self.n += len(toks)
        if again:
            # A64-R2: multi-ask curriculum — the item re-pends at a
            # fresh gap ("teach it casually... till it learns");
            # spaced re-asks give the trunk the repetition a single
            # exposure structurally lacks (round-1 chance floor)
            targets, weights = self._bins()
            bin_i = self.rng.choices(range(len(targets)),
                                     weights=weights)[0]
            self.pending.append({**fact,
                                 "asks_left": fact["asks_left"] - 1,
                                 "due": self.n + targets[bin_i]})
        return out
