"""v10 — the biography builder: real corpora rebuilt as STAGED LIVES
(the lifetime flash's data layer; spec V10_FLASH 2/2b/5a).

ONE COMPLETE LIFE PER LANE: UltraConveyor splits tokens.bin into
n_lanes contiguous segments consumed in parallel, so prepare_life
writes n_lives equal-length lives back-to-back — every lane lives
infancy -> tail across training time, and the lane-segment boundary
IS the life boundary (run scripts assert n_lanes == manifest
n_lives).

Each life is one ordered biography: days opened/closed with the
raised-life rituals, real conversations as the work, a persistent
cast of facts recurring at gaps spanning every band clock (96 ..
1M), correction episodes in the EXACT ARM C pair grammar, and the
frozen public judge (iga/lm_judge) grading every exchange:
below-floor material is DROPPED (admission), the good minority gets
graded press turns (selection; attr=false — recorded for sleep
press-pay, pairs, and the prophet, never economy-attributed), and
cast asks keep the certified parenting economy (attr default).

Sleep-dose note: the builder emits {"kind":"day"} events and the
manifest carries stage token boundaries; the RUN driver gates sleep
by stage (sleepless infancy, A64-R3) — data layer and dose ladder
stay separate concerns.
"""

import json
import re
import os
import random
from dataclasses import dataclass

from .lm_diet import (COLORS, NAMES, OBJECTS, ROOMS, TokenSink,
                      load_tokenizer, train_tokenizer)
from . import lm_judge as J

SPECIALS_LIFE = ["<pad>", "<eot_human>", "<eot_model>",
                 "<+1>", "<+2>", "<-1>", "<-2>"]

RITUAL_OPEN = [("one morning later .", "human"),
               ("good morning .", "model")]
RITUAL_CLOSE = [("that day was done .", "human"),
                ("noted .", "model")]
FILLER = [("the wind moved that day . the town waited .", "human"),
          ("noted .", "model")]


@dataclass(frozen=True)
class Stage:
    name: str
    frac: float                 # of the per-life token budget
    day_units: tuple            # (lo, hi) work units per day
    corr_rate: float            # correction episodes per cast ask
    gap_menu: tuple             # ((gap_tokens, weight), ...)
    plant_every: int            # cast unit every N work units


# spec 5a as data. Gap menus stretch with the stages so recurrence
# spans every band clock by adolescence (band-6 food at 1M).
STAGES_V10 = (
    Stage("infancy", 0.10, (4, 8), 0.00,
          ((96, 4), (700, 3)), 3),
    Stage("childhood", 0.40, (6, 14), 0.05,
          ((96, 3), (700, 3), (5000, 2), (24000, 2)), 4),
    Stage("adolescence", 0.40, (10, 22), 0.06,
          ((700, 2), (5000, 3), (24000, 3), (131072, 2),
           (1048576, 1)), 5),
    Stage("tail", 0.10, (8, 16), 0.04,
          ((5000, 2), (24000, 2), (131072, 2), (1048576, 1)), 4),
)


# The FLASH stage table (2026-08-20, user-chosen budget path): the
# measured spine is UC 1.9B + curated ST2 1.04B + Magpie 0.76B, so
# with one-epoch UltraChat (infancy carved from its simplest slice
# by split_ultrachat) and TWO epochs of the late stages (the plan's
# pre-named Muennighoff fallback), the feasible flash is ~5.2B and
# the shares shift to {.08/.27/.38/.27} — order and least->best
# unchanged; the fatter tail puts MORE of the highest-pedigree
# material on the cosine tail (rising-pedigree law). Gap menus, day
# shapes, and correction rates carry over per stage name.
STAGES_V10_FLASH = (
    Stage("infancy", 0.08, (4, 8), 0.00,
          ((96, 4), (700, 3)), 3),
    Stage("childhood", 0.27, (6, 14), 0.05,
          ((96, 3), (700, 3), (5000, 2), (24000, 2)), 4),
    Stage("adolescence", 0.38, (10, 22), 0.06,
          ((700, 2), (5000, 3), (24000, 3), (131072, 2),
           (1048576, 1)), 5),
    Stage("tail", 0.27, (8, 16), 0.04,
          ((5000, 2), (24000, 2), (131072, 2), (1048576, 1)), 4),
)


def epochs(factory, n=1):
    """n passes over a source, each from a FRESH generator (the
    Muennighoff <=4-epoch repetition fallback; n=1 = one-epoch law
    unchanged)."""
    def gen():
        for _ in range(max(1, int(n))):
            for x in factory():
                yield x
    return gen()


def feasible_budget(rep, stages, st2_epochs=1, magpie_epochs=1,
                    margin=0.94):
    """The honest flash budget from a measure report: UltraChat is
    ONE shared pool feeding infancy+childhood (the measure's
    'childhood' row counts the full file; 'infancy' is a subset of
    it, never additive), and the late stages scale by their epoch
    counts."""
    f = {s.name: s.frac for s in stages}
    uc = rep["childhood"]["est_tokens"]
    caps = [uc / (f["infancy"] + f["childhood"])]
    if "adolescence" in rep:
        caps.append(rep["adolescence"]["est_tokens"] * st2_epochs
                    / f["adolescence"])
    if "tail" in rep:
        caps.append(rep["tail"]["est_tokens"] * magpie_epochs
                    / f["tail"])
    return int(min(caps) * margin)


class StageScheduler:
    def __init__(self, life_budget, stages=STAGES_V10):
        assert abs(sum(s.frac for s in stages) - 1.0) < 1e-6
        self.stages = stages
        self.bounds = []
        acc = 0
        for s in stages:
            acc += int(life_budget * s.frac)
            self.bounds.append(acc)
        self.bounds[-1] = life_budget          # exact close

    def stage_at(self, life_pos):
        for s, b in zip(self.stages, self.bounds):
            if life_pos < b:
                return s
        return self.stages[-1]


# EPISODIC CAST (2026-08-21, v10.1): the v10 cast was 24 PERSISTENT
# facts per life from an 8x8 vocabulary, each asked tens of thousands
# of times per 637M-token life — 192 facts in all, memorized by the
# weights (train-lane recall .96, unseen-life in-ctx .21 = chance).
# The binder faculty needs facts the weights CANNOT hoard: novel
# (name, object) -> color facts from large vocabularies disjoint from
# the roster's, planted continually, asked a few times at the stage's
# gaps (the long ones are band food), then retired. The same pair
# recurs later with other colors, so the only winning policy is
# "the latest statement in context or in memory" — the A69 faculty.
EPISODIC_NAMES = (
    "ada alma amos anya aris asha avi axel bea bela beni bram cai cara "
    "cato celia cleo cora cyrus dana dara davi dell dina eben edda eli "
    "elio elsa emil enzo erin esme etta evan ezra fae faro fenn finn "
    "fleur flor freya gael gia gil gwen hal hana hans hart hazel hedy "
    "hugo ida ilan ines ira iris isla ivo jace jai jana jem jens jory "
    "jude juno kai kami kara kato keir kenji kira kit koa kofi lars lea "
    "leif lev lila lior liv lorn luca lukas lyra mace mae maia malik "
    "mara marc mateo max mena milo mina moss nadia nell nico nika nils "
    "nina noor nora oda odin olin oren orla otis paz pell pia pim quin "
    "rafa rani ravi reid remy rhea rian rico rina rolf romy rory rosa "
    "ruth sabi sage salo sami sarah saul sena seth shay sian silas simon "
    "sol sonia soren stig suki sven tam tara tavi teo tess thea thor "
    "tilda tova tycho uli uma una vale vera vida vik vin wade wren xavi "
    "yael yara yuki yusuf zadie zane zara zeke zia zora abel adrian aldo "
    "alina amara andre anton ari arno asa aya ayla bas bastian beck benno "
    "berit birk bruno carla cas cedric cian colm dag dario dez dimi dora "
    "dries edan elin ella ellis emre enid esra etan fabio farah felix fia "
    "fritz gabi gita greta hakan hedda heidi ilse imre ingo isak ivar "
    "janne jarl jori kaja kalle karim kasia kenan kerem lale lasse leila "
    "lotte maren marit mika milan nadim neva noa olaf paula pelle piet "
    "rasmus ronja runa saga selin sigrid tarik tindra ulla viggo wilma"
).split()
EPISODIC_OBJECTS = (
    "basket ladder kettle hammer candle mirror needle bucket saddle "
    "pillow blanket ribbon feather bottle spoon knife fork plate bowl "
    "cup mug teapot brush comb towel apron glove scarf hat boot sandal "
    "belt button thread pin nail screw wrench saw chisel drill anvil "
    "bench stool chair table desk shelf crate barrel sack pouch wallet "
    "purse satchel trunk chest box tin flask vase pot pan lid tray ladle "
    "sieve whisk cushion rug mat curtain lantern torch clock compass "
    "ruler pencil pen chalk slate notebook ledger scroll stamp ticket "
    "card dice marble kite drum flute fiddle harp horn whistle rattle "
    "doll puppet mask cloak cape robe tunic vest collar brooch ring "
    "bracelet necklace crown badge medal shield spear bow arrow quiver "
    "dagger axe shovel rake hoe trowel sickle yoke harness bridle cart "
    "wagon wheel oar paddle anchor net hook reel sail mast flag banner "
    "tent pole stake peg cord wire chain lock bolt latch hinge handle "
    "knob cork funnel pipe hose valve pump bellows sponge loaf jug urn"
).split()
assert not set(EPISODIC_NAMES) & set(NAMES)
assert not set(EPISODIC_OBJECTS) & set(OBJECTS)
EPISODIC_DEFAULT = {
    "n_asks": (2, 6),          # ask quota per novel fact, then retired
    "sample": 24,              # retired-fact reservoir for the bare probe
    # plant cadence in TOKENS per stage (v10 planted per unit and the
    # roster's 96/700-gap re-asks made infancy ~70% cast drill by
    # tokens — measured 1 ask / 42 tok; a memorization bootcamp).
    # quota ~4 asks -> asks per token ~ 4 / cadence: infancy ~25% of
    # tokens (a BINDING bootcamp on novel facts), childhood ~10%,
    # adolescence/tail ~8% with the long menus (b5/b6 band food).
    "plant_tokens": {"infancy": 500, "childhood": 1200,
                     "adolescence": 1600, "tail": 1600},
    # the persistent roster becomes biography: re-asked only at 16x
    # the stage's longest gap (infancy 11k, childhood 384k, later 16M)
    "roster_gap_mult": 16,
    # every novel fact's FIRST ask draws from the stage menu plus an
    # in-context gap (96) at this weight, so the binder channel (and
    # the in-ctx bin) stays fed in every stage
    "first_ask_inctx_w": 2,
}


class LifeCast:
    """The persistent world of ONE life — a fixed roster of
    (name, object) -> color facts plus ERA facts re-asked at the
    longest gaps across the whole life (band-5/6 food). Plant/ask
    grammar and probe events are EXACTLY the certified weaver's, so
    the drive economy, weight_recall, and the heartbeat pack read
    them unchanged. cls is assigned at plant time (A66 3-class
    design: pos2 / pos1 / none) — selectivity stays measurable.

    world_seed: the WORLD (roster, colors, era facts) is drawn from
    its own rng so an ordering control can shuffle SESSIONS while
    keeping the world identical (A69-R4's confound note)."""

    CLS_P = (("pos2", 0.15), ("pos1", 0.10), ("none", 0.75))

    def __init__(self, rng, world_seed, n_roster=24, n_era=4,
                 episodic=None, hot_frac=0.0, press_in_stream=True):
        # press_in_stream False: the grade is an EVENT only — no approval
        # token enters the stream (A64: the mouth has nothing to say)
        self.press_in_stream = bool(press_in_stream)
        # hot_frac: this fraction of correction episodes presses <-2>
        # on the wrong answer instead of <-1> — the HOT press A72's
        # amygdala tag waits for (scan runs 2026-08-22: hot_pairs 0 in
        # every row; the frozen judge never reaches -2 on ordinary
        # exchanges, so the organ and the BG's NoGo side were starved
        # by the data). 0.0 = every certified shard bit-exact.
        self.hot_frac = float(hot_frac)
        wrng = random.Random(world_seed)
        pairs = [(n, o) for n in NAMES for o in OBJECTS]
        wrng.shuffle(pairs)
        self.roster = [{"name": n, "obj": o,
                        "col": wrng.choice(COLORS),
                        "room": wrng.choice(ROOMS),
                        "cls": self._cls(wrng), "era": False,
                        "planted": None, "asks": 0}
                       for n, o in pairs[:n_roster]]
        for f in self.roster[:n_era]:
            f["era"] = True
        self.rng = rng
        self.pending = []          # (due_pos, fact)
        self.unplanted = list(self.roster)
        # episodic cast (None = the certified v10 roster bit-exactly;
        # no rng is drawn on this path unless an episodic plant fires)
        self.epi = dict(EPISODIC_DEFAULT, **episodic) \
            if episodic is not None else None
        self.epi_n = 0             # facts planted
        self.epi_asks = 0
        self.epi_retired = 0
        self.epi_live = 0
        self.epi_sample = []       # reservoir of retired facts (probe)
        self.last_plant = None     # stream pos of the last plant (epi cadence)

    def _cls(self, rng):
        x, acc = rng.random(), 0.0
        for c, p in self.CLS_P:
            acc += p
            if x < acc:
                return c
        return "none"

    def press_v(self, fact):
        return {"pos2": 2, "pos1": 1, "none": 0}[fact["cls"]]

    def schedule(self, fact, pos, stage):
        menu = [g for g in stage.gap_menu]
        if fact.get("epi"):
            if fact["asks"] >= fact["quota"]:
                self._retire(fact)          # quota met: never asked again
                return
            if fact["asks"] == 0 and self.epi["first_ask_inctx_w"]:
                menu = [(96, self.epi["first_ask_inctx_w"])] + \
                    [g for g in menu if g[0] != 96]
        elif self.epi is not None:
            # episodic mode: the whole roster is biography — re-asked
            # only at 16x the stage's longest gap; the episodic stream
            # carries the binder load (v10 roster re-asks at 96/700
            # were the memorization drill)
            menu = [(menu[-1][0] * self.epi["roster_gap_mult"], 1)]
        elif fact["era"]:
            menu = [g for g in menu if g[0] >= menu[-1][0] // 8] \
                or menu
        gaps, ws = zip(*menu)
        gap = self.rng.choices(gaps, weights=ws)[0]
        self.pending.append((pos + gap, fact))

    def _retire(self, fact):
        self.epi_retired += 1
        self.epi_live -= 1
        k = self.epi["sample"]
        if len(self.epi_sample) < k:
            self.epi_sample.append(fact)
        else:                       # reservoir sampling, deterministic rng
            j = self.rng.randrange(self.epi_retired)
            if j < k:
                self.epi_sample[j] = fact

    def plant_episodic(self, pos, stage):
        """One NOVEL fact: fresh (name, obj) from the large disjoint
        vocabularies, random color/room/class, an ask quota drawn from
        n_asks. Same plant grammar as the roster."""
        e = self.epi
        f = {"name": self.rng.choice(EPISODIC_NAMES),
             "obj": self.rng.choice(EPISODIC_OBJECTS),
             "col": self.rng.choice(COLORS),
             "room": self.rng.choice(ROOMS),
             "cls": self._cls(self.rng), "era": False, "epi": True,
             "quota": self.rng.randint(*e["n_asks"]),
             "planted": pos, "asks": 0}
        self.epi_n += 1
        self.epi_live += 1
        self.schedule(f, pos, stage)
        turns = [(f"by the way {f['name']} kept a {f['col']} "
                  f"{f['obj']} in the {f['room']} .", "human", []),
                 ("noted .", "model", [])]
        return turns, {"fact": f, "kind": "plant"}

    def plant_unit(self, pos, stage):
        """One plant: 'by the way NAME kept a COL OBJ in the ROOM .'
        + 'noted .' — returns (turns, meta) or None. Roster first (as
        certified); once the roster is planted, episodic mode keeps
        planting novel facts at the same cadence for the whole life."""
        if not self.unplanted:
            if self.epi is not None:
                return self.plant_episodic(pos, stage)
            return None
        f = self.unplanted.pop(self.rng.randrange(len(self.unplanted)))
        f["planted"] = pos                    # refined at encode time
        self.schedule(f, pos, stage)
        turns = [(f"by the way {f['name']} kept a {f['col']} "
                  f"{f['obj']} in the {f['room']} .", "human", []),
                 ("noted .", "model", [])]
        return turns, {"fact": f, "kind": "plant"}

    def due_asks(self, pos):
        due = [(d, f) for d, f in self.pending if d <= pos]
        self.pending = [(d, f) for d, f in self.pending if d > pos]
        return [f for _, f in due]

    def ask_unit(self, fact, stage, correct):
        """The certified ask grammar. correct=False emits the FULL
        correction episode (exact ARM C pair grammar; the rival
        never named in the defended stem — A67-P8)."""
        h = (f"what color of {fact['obj']} was {fact['name']} "
             f"kept ?", "human", [])
        col = fact["col"] if correct else self.rng.choice(
            [c for c in COLORS if c != fact["col"]])
        ans_text = f"the {fact['obj']} was {col} ."
        evs = []
        if correct:
            evs.append(("probe", {"answer": fact["col"],
                                  "prefix": f"the {fact['obj']} was",
                                  "fact": fact}))
        a = (ans_text, "model", evs)
        turns = [h, a]
        v = self.press_v(fact)
        if correct:
            if v > 0:
                turns.append(self._press(v))
        else:
            hot = self.hot_frac > 0 and self.rng.random() < self.hot_frac
            turns.append(self._press(-2 if hot else -1))
            turns.append((f"not right . the {fact['obj']} was "
                          f"{fact['col']} .", "human", []))
            turns.append(self._press(2))
        fact["asks"] += 1
        if fact.get("epi"):
            self.epi_asks += 1
        return turns, {"fact": fact, "kind": "ask",
                       "correct": correct}

    def _press(self, v):
        tok = (f"<{'+' if v > 0 else '-'}{abs(v)}>"
               if getattr(self, "press_in_stream", True) else None)
        return (tok, "human", [("button", {"v": int(v)})])


def _iter_stage_source(sources, stage_name):
    """sources: {stage_name: iterator of conversations (list of turn
    strings, human first)}. Falls back to 'default'."""
    it = sources.get(stage_name) or sources["default"]
    return it


def prepare_life(out_dir, budget_tokens, n_lives, seed=0,
                 world_seed=None, vocab=16384, tokenizer_path=None,
                 stages=STAGES_V10, sources=None, tok_sample=1200,
                 spill=4_000_000, shuffle_sessions=False,
                 episodic=None, hot_frac=0.0, press_in_stream=True,
                 silence_mean=0.0, silence_side="both",
                 cast_off=False, grades_off=False, flat_life=False,
                 press_quality=None):
    """Writes tokenizer.json, tokens.bin (n_lives equal lives,
    back-to-back), events.jsonl (sorted, absolute pos), manifest.json,
    judge_audit.jsonl.

    shuffle_sessions: the G2 ordering CONTROL — the same world
    (world_seed) and the same day/exchange machinery, but each cast
    ask's recurrence gap is drawn from a SHUFFLED (uniform) menu and
    era facts lose their long-gap privilege, destroying the ordered
    recurrence structure while keeping the world stable (A69-R4:
    shuffle sessions, keep the world)."""
    if flat_life:
        silence_mean = 0.0      # v16 law: no scripted silence, ever
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    world_seed = seed if world_seed is None else world_seed
    assert sources, "prepare needs a source: authored lives (--lives-file)"
    life_budget = budget_tokens // n_lives

    tok_path = os.path.join(out_dir, "tokenizer.json")
    if tokenizer_path:
        import shutil
        shutil.copy(tokenizer_path, tok_path)
        tok = load_tokenizer(tok_path)
    else:
        sample = []
        src = _iter_stage_source(sources, "tokenizer")
        for turns in src:
            sample.extend(turns)
            if len(sample) >= tok_sample:
                break
        sample += [t for t, _ in RITUAL_OPEN + RITUAL_CLOSE + FILLER]
        sample += [f"by the way {n} kept a {c} {o} in the {r} ."
                   for n in NAMES for o in OBJECTS[:2]
                   for c in COLORS[:3] for r in ROOMS[:1]]
        sample += ["not right . the coin was golden .",
                   "what color of coin was nedra kept ?"]
        tok = train_tokenizer(iter(sample), tok_path, vocab,
                              specials=SPECIALS_LIFE)
    ids = {w: tok.token_to_id(w) for w in SPECIALS_LIFE}
    assert all(v is not None for v in ids.values()), ids
    eot_h, eot_m = ids["<eot_human>"], ids["<eot_model>"]

    def enc(text):
        return tok.encode(text).ids

    stream = TokenSink(os.path.join(out_dir, "tokens.bin"),
                       spill=spill)
    events = []
    audit = J.AuditWriter(os.path.join(out_dir, "judge_audit.jsonl"))
    stats = {"kept": 0, "dropped": 0, "presses": {},
             "corrections": 0, "cast_asks": 0, "plants": 0,
             "filler_days": 0, "source_tally": {}}
    manifest = {"judge_version": J.JUDGE_VERSION, "judge": J.JUDGE,
                "n_lives": n_lives, "life_len": life_budget,
                "specials": SPECIALS_LIFE, "seed": seed,
                "world_seed": world_seed,
                "shuffle_sessions": bool(shuffle_sessions),
                "cast_mode": ("episodic_v1" if episodic is not None
                              else "roster_v10"),
                "hot_frac": float(hot_frac),
                "press_in_stream": bool(press_in_stream),
                # v14 — SILENCE AS AN ORDINARY TOKEN (the user's design,
                # 2026-08-23): pauses live in the stream as runs of the
                # <pad> id (verified unused in every life shard), drawn
                # from a geometric hazard so the model learns WHEN quiet
                # ends, never a count. Before a MODEL turn the mean is
                # 2x silence_mean (thinking time — the pause the organism
                # will learn to produce before answering); before a HUMAN
                # turn it is silence_mean (the conversational gap). One
                # token = one tick of lived time. 0 = none, bit-exact.
                # silence_side (ratified 2026-08-24): "model" = pauses
                # ONLY before model turns — silence is the organism's
                # own thinking act, emitted when it needs more PFC
                # cycles; nobody's waiting time is modelled ("we are
                # just wasting tokens at that point" — the user).
                # "both" = v14's original placement (the room injects
                # the human-side ticks at serve).
                "silence_mean": float(silence_mean),
                "silence_side": silence_side,
                # ratified 2026-08-24 (the user): pretraining is
                # PREDICTION — cast_off strips the fact liturgy from the
                # training world (the eval lives keep it as the ruler);
                # grades_off strips press events (grading a scripted
                # actor teaches nothing; the felt sense trains live)
                "cast_off": bool(cast_off),
                "grades_off": bool(grades_off),
                "flat_life": bool(flat_life),
                "sil_id": (tok.token_to_id("<pad>")
                           if silence_mean > 0 else None),
                "episodic": (dict(EPISODIC_DEFAULT, **episodic,
                                  n_names=len(EPISODIC_NAMES),
                                  n_objects=len(EPISODIC_OBJECTS))
                             if episodic is not None else None),
                "stages": [], "lives": []}

    sil_id = tok.token_to_id("<pad>") if silence_mean > 0 else None

    def _pause(who):
        """v14: a run of silence ticks before a turn — geometric (a
        memoryless hazard: the chance quiet ends is the same at every
        tick, so counting our formula is impossible), mean 2x for the
        thinking pause before a model turn, capped at 8x the mean.
        silence_side="model": human turns get NONE — thinking is the
        model's act alone."""
        if sil_id is None:
            return
        if silence_side == "model" and who != "model":
            return
        m = silence_mean * (2.0 if who == "model" else 1.0)
        n, cap = 0, int(8 * m) + 1
        while n < cap and rng.random() < m / (1.0 + m):
            n += 1
        stream.extend([sil_id] * n)

    def emit_turns(turns):
        """Encode a unit's turns, place events at absolute
        positions. turns: [(text, who, [(kind, meta)])].
        NOTE for the 10B build (item 8): switch the hot path to
        tok.encode_batch over whole days — per-turn encode is fine
        at gate scale only."""
        enc_all = [(tok.encode(t).ids if t is not None else None)
                   for t, _, _ in turns]
        for (text, who, tevs), tids in zip(turns, enc_all):
            if tids is not None:
                _pause(who)
            t0 = len(stream)
            if tids is not None:
                stream.extend(tids)
                stream.append(eot_m if who == "model" else eot_h)
            # text None: the grade as a SENSE — no token enters the
            # stream; the button event lands on the NEXT token's
            # position (the first token after the graded answer)
            for kind, meta in tevs:
                if kind == "button":
                    if grades_off and not meta.get("quality"):
                        # central guard: judge/cast presses die when
                        # grades are off; 49d dataset-grounded quality
                        # presses are annotations, not judgments —
                        # they pass
                        continue
                    # law 14: a face that changed MID-utterance lands on
                    # the token after the words it was felt on (prefix)
                    pos_b = t0 + (len(enc(meta["prefix"])) if meta.get("prefix") else 0)
                    events.append({"pos": pos_b, "kind": "button",
                                   **{k_: v_ for k_, v_ in meta.items() if k_ != "prefix"}})
                    k = str(meta["v"])
                    stats["presses"][k] = \
                        stats["presses"].get(k, 0) + 1
                elif kind == "project":
                    apos = t0 + len(enc(meta["prefix"]))
                    aid = enc(" " + meta["answer"])[0]
                    dids = []
                    for c in rng.sample(
                            [c for c in COLORS
                             if c != meta["answer"]], len(COLORS) - 1):
                        d = enc(" " + c)[0]
                        if d != aid and d not in dids:
                            dids.append(d)
                        if len(dids) == 4:
                            break
                    events.append({
                        "pos": apos, "kind": "project",
                        "answer": aid,
                        "gap": max(apos - meta["declared"], 1),
                        "item_i": meta["item_i"],
                        "n_items": meta["n_items"],
                        "distractors": dids})
                    meta["fact"]["last_seen"] = apos
                elif kind == "probe":
                    f = meta["fact"]
                    apos = t0 + len(enc(meta["prefix"]))
                    # gap = distance from the LAST EXPOSURE (plant or
                    # the previous ask's restated answer), never the
                    # original plant — every correct ask and every
                    # correction restates the fact, so plant-relative
                    # gaps overstate memory demand (the A47
                    # intermediate-mention law, caught again here)
                    seen = f.get("last_seen", f.get("plant_abs"))
                    if seen is None:
                        continue
                    aid = enc(" " + meta["answer"])[0]
                    # distinct at the ID level: at small vocabs two
                    # color words can share a first BPE piece
                    dids = []
                    for c in rng.sample(
                            [c for c in COLORS
                             if c != meta["answer"]], len(COLORS) - 1):
                        d = enc(" " + c)[0]
                        if d != aid and d not in dids:
                            dids.append(d)
                        if len(dids) == 4:
                            break
                    events.append({
                        "pos": apos, "kind": "probe",
                        "answer": aid,
                        "gap": max(apos - seen, 1),
                        "distractors": dids})
        return len(stream)

    def emit_plant(unit, life_i):
        """Plants need their fact's absolute answer-word position
        recorded for future gap computation."""
        turns, meta = unit
        f = meta["fact"]
        t0 = len(stream)
        off = len(enc(f"by the way {f['name']} kept a"))
        f["plant_abs"] = t0 + off
        f["last_seen"] = t0 + off
        emit_turns(turns)
        stats["plants"] += 1

    if shuffle_sessions:
        # G2 ordering control: uniform gap menu (every gap any stage
        # ever offers, weight 1) + no era privilege = the ordered
        # recurrence structure destroyed; the WORLD (roster, colors)
        # stays identical via world_seed (A69-R4: shuffle sessions,
        # keep the world)
        uniform = tuple((g, 1) for g in sorted(
            {g for s in stages for g, _ in s.gap_menu}))
        stages = tuple(
            Stage(s.name, s.frac, s.day_units, s.corr_rate,
                  uniform, s.plant_every) for s in stages)
    src_iters = {k: v for k, v in sources.items()}
    for life_i in range(n_lives):
        life_start = len(stream)
        sched = StageScheduler(life_budget, stages)
        cast = LifeCast(random.Random(seed * 1000 + life_i + 1),
                        world_seed * 1000 + life_i + 1,
                        episodic=episodic, hot_frac=hot_frac,
                        press_in_stream=press_in_stream)
        if shuffle_sessions:
            for f in cast.roster:
                f["era"] = False
        stage_marks = {}
        day_i = 0
        while len(stream) - life_start < life_budget - 512:
            life_pos = len(stream) - life_start
            day_t0 = len(stream)
            stage = sched.stage_at(life_pos)
            stage_marks.setdefault(stage.name, life_pos)
            n_units = 1 if flat_life else rng.randint(*stage.day_units)
            # --- open ritual (v16 flat world: none — a day IS one
            # conversation, no UC dressing)
            if not flat_life:
                emit_turns([(t, w, []) for t, w in RITUAL_OPEN])
            unit_i = 0
            src = _iter_stage_source(src_iters, stage.name)
            while unit_i < n_units:
                if len(stream) - life_start >= life_budget - 512:
                    break
                unit_i += 1
                # cast plant cadence: per unit (certified v10) or, in
                # episodic mode, per TOKENS since the last plant
                if cast.epi is not None:
                    due_plant = (cast.last_plant is None or
                                 len(stream) - cast.last_plant
                                 >= cast.epi["plant_tokens"][stage.name])
                else:
                    due_plant = unit_i % stage.plant_every == 0
                if due_plant and not cast_off:
                    u = cast.plant_unit(len(stream), stage)
                    if u:
                        cast.last_plant = len(stream)
                        emit_plant(u, life_i)
                # due asks fire first (spacing is the curriculum) —
                # DRAINED in a bounded loop so a 96-token gap drawn
                # off a just-fired ask realizes IN-CONTEXT (an ask is
                # ~30-60 tokens; single-pass draining made sub-256
                # gaps unrealizable and starved G1's in-ctx bin)
                drained = 0
                while drained < 64 and not cast_off:
                    due = cast.due_asks(len(stream))
                    if not due:
                        break
                    for f in due:
                        correct = rng.random() >= stage.corr_rate
                        turns, _ = cast.ask_unit(f, stage, correct)
                        emit_turns(turns)
                        # the ask (or its correction) restated the
                        # fact — the exposure clock resets here
                        f["last_seen"] = len(stream)
                        cast.schedule(f, len(stream), stage)
                        stats["cast_asks"] += 1
                        drained += 1
                        if not correct:
                            stats["corrections"] += 1
                            manifest.setdefault(
                                "correction_pos",
                                []).append(len(stream))
                # one real conversation
                try:
                    convo = next(src)
                except StopIteration:
                    src_iters[stage.name] = iter(())
                    break
                q_score = None
                if isinstance(convo, tuple):
                    convo, q_score = convo
                q_lv = press_from_quality(q_score, press_quality) \
                    if press_quality else 0
                pairs = [(convo[i], convo[i + 1])
                         for i in range(0, len(convo) - 1, 2)]
                for pair_i, (h_text, m_text) in enumerate(pairs):
                    if (life_budget - (len(stream) - life_start)
                            < 4096):
                        break     # life boundary: no pair may cross
                    if cast.epi is not None and pair_i > 0:
                        # episodic mode: plants keep their TOKEN cadence
                        # inside long conversations (adolescence units
                        # run ~6k tokens; one plant per unit starved
                        # the stage 4x), and due asks also fire BETWEEN
                        # exchanges (bounded), so a 96-token gap
                        # realizes in-context instead of waiting a
                        # whole conversation for the unit boundary
                        # (v10: in-ctx asks existed only in infancy's
                        # back-to-back drill)
                        if not cast_off and cast.last_plant is not None \
                                and (len(stream) - cast.last_plant
                                     >= cast.epi["plant_tokens"][stage.name]):
                            u = cast.plant_unit(len(stream), stage)
                            if u:
                                cast.last_plant = len(stream)
                                emit_plant(u, life_i)
                        due_now = [] if cast_off else cast.due_asks(len(stream))
                        for f in due_now[4:]:       # bounded: rest stay due
                            cast.pending.append((len(stream), f))
                        for f in due_now[:4]:
                            correct = rng.random() >= stage.corr_rate
                            turns_a, _ = cast.ask_unit(f, stage, correct)
                            emit_turns(turns_a)
                            f["last_seen"] = len(stream)
                            cast.schedule(f, len(stream), stage)
                            stats["cast_asks"] += 1
                            if not correct:
                                stats["corrections"] += 1
                                manifest.setdefault(
                                    "correction_pos",
                                    []).append(len(stream))
                    q = J.grade_dialogue(h_text, m_text)
                    tail = stage.name == "tail"
                    if not J.passes_floor(q):
                        stats["dropped"] += 1
                        audit.maybe(stage.name, life_i, len(stream),
                                    q, "DROP+TRUNC", h_text, m_text,
                                    tail=tail)
                        break        # truncate: coherence preserved
                    press = 0 if grades_off else J.press_for(q, stage.name)
                    # 49d: the dataset's own quality annotation lands as
                    # a press on the conversation's LAST model turn —
                    # varied, external, judge-free
                    m_ev = ([("button", {"v": q_lv, "quality": True})]
                            if q_lv and pair_i == len(pairs) - 1 else [])
                    # law 14 (the face is a sense): inline caretaker
                    # faces <+1> <+2> <-1> <-2> authored mid-utterance
                    # leave the text and become button events at the
                    # word where the face changed. The child's own face
                    # <me±> stays in the text — those are its words.
                    h_text, h_ev = _faces_out(h_text, press_in_stream)
                    m_text, m_ev2 = _faces_out(m_text, press_in_stream)
                    turns = [(h_text, "human", h_ev),
                             (m_text, "model", m_ev2 + m_ev)]
                    if press:
                        tokp = f"<+{press}>" if press_in_stream else None
                        turns.append((tokp, "human",
                                      [("button", {"v": press,
                                                   "attr": False,
                                                   "stage":
                                                   stage.name})]))
                    emit_turns(turns)
                    stats["kept"] += 1
                    audit.maybe(stage.name, life_i, len(stream), q,
                                press, h_text, m_text, tail=tail)
            # --- close ritual + day event (flat: bare day event)
            if flat_life and len(stream) == day_t0:
                # 49g fix: a day that emitted NOTHING = the source is
                # dry. Without rituals nothing grows the stream, so the
                # loop would spin forever appending day events (exit
                # 137, the OOM killer — caught by the fixture dry-run).
                break
            if not flat_life:
                emit_turns([(t, w, []) for t, w in RITUAL_CLOSE])
            events.append({"pos": len(stream) - 1, "kind": "day",
                           "stage": stage.name, "life": life_i,
                           "day": day_i})
            day_i += 1
        # --- pad to EXACTLY life_budget with filler days, then eot
        while life_budget - (len(stream) - life_start) > 96:
            if flat_life:
                # v16: NO pad tokens (<pad> IS the silence token — a
                # pad run would plant the pause attractor) and NO
                # English filler. Fill the tail with real pairs that
                # FIT (pre-measured with the builder's own encoder —
                # stream is a TokenSink, it cannot be truncated); the
                # exact-fill eot tail below closes the last gap.
                src = _iter_stage_source(src_iters, stage.name)
                try:
                    convo = next(src)
                except StopIteration:
                    break
                if isinstance(convo, tuple):
                    convo = convo[0]
                cut = life_start + life_budget
                emitted = 0
                for i2 in range(0, len(convo) - 1, 2):
                    need = (len(enc(convo[i2])) +
                            len(enc(convo[i2 + 1])) + 8)
                    if len(stream) + need > cut:
                        break
                    emit_turns([(convo[i2], "human", []),
                                (convo[i2 + 1], "model", [])])
                    emitted += 1
                if emitted == 0:
                    break               # nothing fits: the eot tail
                continue
            emit_turns([(t, w, []) for t, w in
                        (RITUAL_OPEN[:1] + FILLER + RITUAL_CLOSE)])
            stats["filler_days"] += 1
        while len(stream) - life_start < life_budget:
            stream.append(eot_h)
        life_row = {
            "life": life_i, "start": life_start, "days": day_i,
            "stage_marks": stage_marks,
            "cast": [{k: f[k] for k in
                      ("name", "obj", "col", "cls", "era", "asks")}
                     for f in cast.roster]}
        if cast.epi is not None:
            # the bare-ask probe reads life["cast"]: a retired-fact
            # sample rides along as class "epi:<cls>" — weights that
            # hoard novel facts would show there as p_true > chance
            life_row["cast"] += [
                {"name": f["name"], "obj": f["obj"], "col": f["col"],
                 "cls": "epi:" + f["cls"], "era": False,
                 "asks": f["asks"]} for f in cast.epi_sample]
            life_row["episodic"] = {
                "n_facts": cast.epi_n, "n_asks": cast.epi_asks,
                "retired": cast.epi_retired, "live_at_end": cast.epi_live,
                "n_asks_range": list(cast.epi["n_asks"]),
                "plant_tokens": cast.epi["plant_tokens"],
                "roster_gap_mult": cast.epi["roster_gap_mult"]}
            stats["epi_facts"] = stats.get("epi_facts", 0) + cast.epi_n
            stats["epi_asks"] = stats.get("epi_asks", 0) + cast.epi_asks
        manifest["lives"].append(life_row)

    total = stream.close()
    audit.close()
    with open(os.path.join(out_dir, "events.jsonl"), "w") as f:
        for e in sorted(events, key=lambda e: e["pos"]):
            f.write(json.dumps(e) + "\n")
    manifest["total_tokens"] = total
    manifest["stats"] = stats
    manifest["stages"] = [
        {"name": s.name, "frac": s.frac, "day_units": s.day_units,
         "corr_rate": s.corr_rate, "gap_menu": s.gap_menu,
         "plant_every": s.plant_every} for s in stages]
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {total:,} tokens, {n_lives} lives x "
          f"{life_budget:,}, {len(events)} events, kept/dropped "
          f"{stats['kept']}/{stats['dropped']}, presses "
          f"{stats['presses']}, corrections {stats['corrections']} "
          f"-> {out_dir}")
    return out_dir


# SmolTalk2 subset ORDER (2026-08-21): v10 consumed the no_think
# parquets in sorted-filename order, which put LongAlign_64k (64k-
# context document QA: 2-turn "exchanges" of ~15k tokens) FIRST in
# adolescence and table_gpt / OpenThoughts3 wherever the alphabet fell.
# The ratified curriculum is flat floor / rising ceiling, so the order
# is now explicit: everyday conversation -> rewriting / summarizing ->
# personas -> Hermes -> science -> reasoning -> long-context LAST.
ST2_ORDER = ("everyday", "explore_instruct", "smol_rewrite",
             "smol_summarize", "tulu_3", "OpenHermes", "table_gpt",
             "Mixture_of_Thoughts", "OpenThoughts3", "LongAlign")


def st2_ordered(paths):
    """Deterministic curriculum order for SmolTalk2 parquet paths:
    ST2_ORDER rank, then filename (unknown subsets sort mid-list)."""
    import os as _os

    def key(pth):
        name = _os.path.basename(pth)
        rank = next((i for i, pat in enumerate(ST2_ORDER) if pat in name),
                    len(ST2_ORDER) // 2)
        return (rank, name)
    return sorted(paths, key=key)


QUALITY_COLS = ("quality", "instruct_reward", "reward", "score",
                "conversation_quality", "overall_score")


def smoltalk2_source(paths, en_only=True, quality=False):
    """Conversations from SmolTalk2-format parquet (messages =
    [{content, role}]). Serves both the SmolTalk2 no_think subsets
    and Smol-Magpie-Ultra (smoltalk2's newest curation of it).
    Convs containing system/tool roles are dropped WHOLE (a being
    with no tools and no system channel); EN gate = deterministic
    heuristic (ascii ratio + judge stopword rate) on the first
    exchange. Yields alternating [u, a, u, a, ...]."""
    import pyarrow.parquet as pq

    def _en(text):
        if not text:
            return False
        ascii_r = sum(1 for c in text if ord(c) < 128) / len(text)
        w = J._words(text)
        stop = (sum(1 for x in w if x in J.STOPWORDS) / len(w)) \
            if w else 0.0
        return ascii_r >= 0.9 and stop >= 0.12

    def gen():
        for path in paths:
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            qcol = next((c for c in QUALITY_COLS if c in names), None) \
                if quality else None
            cols = ["messages"] + ([qcol] if qcol else [])
            for batch in pf.iter_batches(batch_size=256, columns=cols):
                d = batch.to_pydict()
                qs = d.get(qcol) if qcol else None
                for bi, msgs in enumerate(d["messages"]):
                    roles = [m["role"] for m in msgs]
                    if any(r not in ("user", "assistant")
                           for r in roles):
                        continue
                    if not msgs or roles[0] != "user":
                        continue
                    texts = [m["content"] or "" for m in msgs]
                    if en_only and not _en(" ".join(texts[:2])):
                        continue
                    n = len(texts) - len(texts) % 2
                    if n >= 2:
                        if quality:
                            q = qs[bi] if qs is not None else None
                            try:
                                q = None if q is None else float(q)
                            except (TypeError, ValueError):
                                q = None
                            yield texts[:n], q
                        else:
                            yield texts[:n]
    return gen()


def press_from_quality(score, thresholds):
    """49d: dataset-grounded press level from a quality score.
    thresholds = (t_neg, t_pos, t_top) — score < t_neg -> -1,
    < t_pos -> 0 (no press), < t_top -> +1, else +2. None -> 0."""
    if score is None:
        return 0
    t_neg, t_pos, t_top = thresholds
    if score < t_neg:
        return -1
    if score < t_pos:
        return 0
    if score < t_top:
        return 1
    return 2


def simple_only(src, max_words=80):
    """Infancy filter: only the spine's shortest, simplest
    exchanges (every turn under max_words) — 'least to best, like
    learning' starts with the simplest conversations."""
    def gen():
        for turns in src:
            if all(len(t.split()) <= max_words for t in turns):
                yield turns
    return gen()


_FACE_RE = re.compile(r"\s*<([+-])([12])>\s*")


def _faces_out(text, press_in_stream):
    """strip inline caretaker faces from authored text; when the grade
    is a sense (press_in_stream False) return them as button events
    positioned by the text prefix they were felt on. When faces stay in
    the stream, the text is returned untouched (they are tokens)."""
    if press_in_stream or not text or "<" not in text:
        return text, []
    out, evs, pos = [], [], 0
    for m in _FACE_RE.finditer(text):
        out.append(text[pos:m.start()])
        prefix = "".join(out).rstrip()
        v = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
        evs.append(("button", {"v": v, "attr": False, "prefix": prefix}))
        out.append(" ")
        pos = m.end()
    out.append(text[pos:])
    clean = re.sub(r"\s+", " ", "".join(out)).strip()
    return clean, evs


def lives_source(path, skip=0):
    """Authored childhoods (GESTATION.md): lists of alternating turn
    strings, caretaker first, from scripts/author_lives.py JSONL."""
    def gen():
        with open(path) as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                turns = r.get("data") or []
                if len(turns) >= 2:
                    yield turns[:len(turns) - len(turns) % 2]
    return gen()

def _measure(src, stages, tok=None, sample_convs=400):
    """One-epoch yield per stage source (the A12 law's input): full
    word count per stage + a tokens/word ratio measured on a sample
    through the real BPE family -> estimated token yield. The
    builder's ritual/cast/press overhead ADDS stream tokens, so
    budgets sized from this are conservative."""
    out = {}
    seen = {}                    # stages sharing one iterator share
    for s in stages:             # its (single-pass) measurement
        it = src.get(s.name) or src.get("default")
        if it is None:
            continue
        if id(it) in seen:
            out[s.name] = dict(seen[id(it)])
            continue
        convs = words = 0
        ratio_w = ratio_t = 0
        for turns in it:
            w = sum(len(t.split()) for t in turns)
            convs += 1
            words += w
            if tok is not None and convs % 97 == 1 \
                    and ratio_w < sample_convs * 200:
                ratio_t += sum(len(tok.encode(t).ids) for t in turns)
                ratio_w += w
        r = (ratio_t / ratio_w) if ratio_w else 1.35
        out[s.name] = {"convs": convs, "words": words,
                       "tok_per_word": round(r, 4),
                       "est_tokens": int(words * r)}
        seen[id(it)] = out[s.name]
    # the binding constraint: budget * frac_s <= yield_s for every
    # stage; 0.85 margin covers estimator error (overhead helps).
    budgets = [v["est_tokens"] / s.frac for s in stages
               for n, v in out.items() if n == s.name]
    out["_max_budget"] = int(min(budgets) * 0.85) if budgets else 0
    return out


def _freeze_judge(src, stages, per_stage=4000, per_conv=3):
    """Grade real per-stage exchanges with the frozen scorer and
    quantile the pre-registered density targets into stage
    thresholds (lm_judge.stage_thresholds). Returns the freeze dict
    to be written to json, committed, and passed to prepare via
    --judge-thresholds."""
    qs = {}
    seen = {}                    # shared iterators grade one sample
    for s in stages:
        it = src.get(s.name) or src.get("default")
        if it is None:
            continue
        if id(it) in seen:
            qs[s.name] = list(seen[id(it)])
            continue
        got = []
        for turns in it:
            for i in range(0, min(len(turns) - 1, 2 * per_conv), 2):
                got.append(J.grade_dialogue(turns[i], turns[i + 1]))
            if len(got) >= per_stage:
                break
        qs[s.name] = got[:per_stage]
        seen[id(it)] = qs[s.name]
    t = J.stage_thresholds(qs)
    t["judge_version"] = J.JUDGE_VERSION
    t["density"] = {k: list(v) for k, v in J.DENSITY.items()}
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "measure",
                                     "freeze-judge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--stages", default="gate",
                    choices=["gate", "flash"],
                    help="gate=STAGES_V10 (parity), flash=the "
                         "supply-fitted 5.2B table")
    ap.add_argument("--st2-epochs", type=int, default=1)
    ap.add_argument("--magpie-epochs", type=int, default=1)
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world-seed", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--lives-file", default="data/lives.jsonl",
                    help="authored childhoods (scripts/author_lives.py)")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--shuffle-sessions", action="store_true")
    ap.add_argument("--episodic", action="store_true",
                    help="v10.1 episodic cast (novel facts, quotas, "
                         "retirement; roster demoted to long gaps)")
    ap.add_argument("--hot-frac", type=float, default=0.0,
                    help="fraction of corrections pressing <-2> (A72 hot)")
    ap.add_argument("--no-press-tokens", action="store_true",
                    help="grades as events only: no approval token in the stream")
    ap.add_argument("--silence-mean", type=float, default=0.0,
                    help="v14: mean silence ticks before a human turn (2x "
                         "before a model turn — thinking time); geometric "
                         "hazard, <pad> as the tick; 0 = none")
    ap.add_argument("--silence-side", default="both",
                    choices=["both", "model"],
                    help="model: pauses only before MODEL turns — "
                         "silence is the organism's own thinking act")
    ap.add_argument("--childhood-source", default="uc",
                    choices=["uc", "magpie"],
                    help="v15: magpie = childhood is Smol-Magpie-Ultra")
    ap.add_argument("--cast-off", action="store_true",
                    help="v15: no plants/asks/corrections in TRAINING "
                         "lives (the eval lives keep them as the ruler)")
    ap.add_argument("--flat-mix-everyday", type=float, default=0.0,
                    help="49e: fraction of flat-world conversations "
                         "drawn from the SmolTalk2 everyday/casual "
                         "splits (st2-dir, non-magpie) — the normal-"
                         "person register; 0 = pure Magpie")
    ap.add_argument("--press-quality", action="store_true",
                    help="49d: map the dataset's quality annotation to "
                         "a press event on each conversation's last "
                         "model turn (quantiles 10/50/90 from a 2000-"
                         "conv pre-scan -> -1/0/+1/+2); absent columns "
                         "-> loud manifest note, zero presses")
    ap.add_argument("--flat-life", action="store_true",
                    help="v16: no stages, no rituals, no filler — one "
                         "conversation per day (store wipes per "
                         "example), lives packed edge-to-edge")
    ap.add_argument("--no-grades", action="store_true",
                    help="v15: no press events at all — pretraining is "
                         "prediction; the felt sense trains live")
    ap.add_argument("--epi-asks", default="2,6",
                    help="episodic ask quota range lo,hi")
    ap.add_argument("--st2-dir", default=None,
                    help="dir of SmolTalk2 no_think parquet files "
                         "(adolescence); magpie files excluded")
    ap.add_argument("--magpie-dir", default=None,
                    help="dir holding smol_magpie_ultra parquet "
                         "(the tail)")
    ap.add_argument("--judge-thresholds", default=None,
                    help="stage-threshold freeze json (freeze-judge "
                         "output) loaded before grading")
    ap.add_argument("--tok-sample", type=int, default=1200)
    ap.add_argument("--per-stage", type=int, default=4000,
                    help="freeze-judge: graded exchanges per stage")
    a = ap.parse_args()
    import glob as _glob
    pq_ = None            # quality-press thresholds (set inside _src)

    def _src(ep=True):
        global pq_
        pq_ = None
        # v17: THE DIET IS LIVES (GESTATION.md). One authored source
        # feeds every stage; staging is authored into the lives
        # themselves. The document-diet branches are retired.
        lv = epochs(lambda: lives_source(a.lives_file, a.skip),
                    a.st2_epochs if ep else 1)
        return {k: lv for k in ("default", "tokenizer", "infancy",
                                "childhood", "adolescence", "tail")}

    stages_sel = STAGES_V10_FLASH if a.stages == "flash" \
        else STAGES_V10

    if a.mode == "measure":
        if a.tokenizer:
            tok = load_tokenizer(a.tokenizer)
        else:
            # throwaway BPE from the same sample pipeline prepare
            # uses — the tokens/word ratio must come from the real
            # tokenizer family, not a guess
            sample = []
            for turns in _src()["tokenizer"]:
                sample.extend(turns)
                if len(sample) >= a.tok_sample:
                    break
            tok = train_tokenizer(iter(sample), a.out + ".tok.json",
                                  a.vocab, specials=SPECIALS_LIFE)
        rep = _measure(_src(ep=False), stages_sel, tok=tok)
        # the honest ceiling: UltraChat is one shared pool for
        # infancy+childhood; late stages scale by their epochs
        rep["_max_budget"] = feasible_budget(
            rep, stages_sel, a.st2_epochs, a.magpie_epochs)
        rep["_epochs"] = {"st2": a.st2_epochs,
                          "magpie": a.magpie_epochs}
        rep["_stages"] = a.stages
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
    elif a.mode == "freeze-judge":
        t = _freeze_judge(_src(ep=False), stages_sel,
                          per_stage=a.per_stage)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(t, f, indent=1)
        print(json.dumps(t, indent=1))
    else:
        assert a.budget > 0, "prepare needs --budget"
        if a.judge_thresholds:
            J.freeze_stage_thresholds(a.judge_thresholds)
        prepare_life(a.out, a.budget, a.lives, seed=a.seed,
                     world_seed=a.world_seed, vocab=a.vocab,
                     tokenizer_path=a.tokenizer, sources=_src(),
                     stages=stages_sel,
                     tok_sample=a.tok_sample,
                     shuffle_sessions=a.shuffle_sessions,
                     hot_frac=a.hot_frac,
                     press_in_stream=not a.no_press_tokens,
                     silence_mean=a.silence_mean,
                     silence_side=a.silence_side,
                     cast_off=a.cast_off, grades_off=a.no_grades,
                     flat_life=a.flat_life,
                     press_quality=(pq_ if a.press_quality else None),
                     episodic=({"n_asks": tuple(int(x) for x in
                                          a.epi_asks.split(","))}
                               if a.episodic else None))
