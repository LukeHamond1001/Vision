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
                 buttons=None):
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
        self.rng = rng
        self.correct_rate = correct_rate
        self.success_rate = success_rate
        self.buttons = buttons
        self.item_log = []
        self.n = 0                    # tokens emitted so far (lane-local)
        self.pending = []             # facts planted, not yet asked
        self.used = set()

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

    def _plant(self):
        for _ in range(200):  # bounded; roster self-frees on ask
            name, obj = self.rng.choice(NAMES), self.rng.choice(OBJECTS)
            if (name, obj) not in self.used:
                break
        else:
            return self._chatter()
        self.used.add((name, obj))
        col = self.rng.choice(COLORS)
        words = ["by", "the", "way", name, "kept", "a", col, obj, "in",
                 "the", self.rng.choice(ROOMS), "."]
        plant_pos = self.n + 6        # the color token
        weights = self.bias_fn() if getattr(self, "bias_fn", None) \
            else [4, 3, 2, 1]         # scheduler: the drive shapes the belt
        bin_i = self.rng.choices(range(4), weights=weights)[0]
        cls = None
        if self.buttons:              # A64: class assigned at plant time
            r = self.rng.random()
            cls = "pos" if r < self.buttons["pos"] else (
                "neg" if r < self.buttons["pos"] + self.buttons["neg"]
                else "none")
        self.pending.append({"name": name, "obj": obj, "col": col,
                             "plant": plant_pos, "cls": cls,
                             "due": plant_pos + GAP_TARGETS[bin_i]})
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
        if self.buttons:               # A64: competence press
            return [h, a, h2, a2, self._press(
                1 if succeed else -self.buttons.get("neg_v", 1))]
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
        due = [f for f in self.pending if self.n >= f["due"]]
        if due:
            fact = due[0]
            self.pending.remove(fact)
            # pair frees on ask — the roster can never exhaust
            self.used.discard((fact["name"], fact["obj"]))
            batch = self._ask(fact)
        else:
            r = self.rng.random()
            if r < 0.25 and len(self.pending) < 12:
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
        return out
