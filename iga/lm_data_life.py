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
import os
import random
from dataclasses import dataclass

from .lm_data_ultrachat import (COLORS, NAMES, OBJECTS, ROOMS,
                                TokenSink, iter_convos,
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


def split_ultrachat(path, out_simple, out_rest, infancy_words,
                    skip=0):
    """ONE pass over the shared UltraChat file -> two disjoint
    files: the SIMPLEST conversations (smallest per-conv max turn
    length, by an adaptive threshold that just covers the infancy
    word budget) and everything else. Fixes the flash-scale A12
    violation where infancy and childhood re-read the same rows,
    and replaces the fixed 80-word filter that passed only ~1% of
    UltraChat (the 44M-budget collapse, 2026-08-20).

    Two passes over the jsonl (cheap line streams): pass 1 builds a
    histogram of per-conv max-turn-words vs words; pass 2 routes.
    Returns {"threshold": T, "simple_words": w, "rest_words": w2,
    "simple_convs": n, "rest_convs": n2}."""
    hist = {}
    for turns in ultrachat_source(path, skip):
        mx = max(len(t.split()) for t in turns)
        w = sum(len(t.split()) for t in turns)
        b = min(mx // 20, 400)          # 20-word bins, cap 8k words
        c, ww = hist.get(b, (0, 0))
        hist[b] = (c + 1, ww + w)
    acc = 0
    thr_bin = max(hist)
    for b in sorted(hist):
        acc += hist[b][1]
        if acc >= infancy_words:
            thr_bin = b
            break
    T = (thr_bin + 1) * 20
    res = {"threshold": T, "simple_words": 0, "rest_words": 0,
           "simple_convs": 0, "rest_convs": 0}
    with open(out_simple, "w") as fs, open(out_rest, "w") as fr:
        for turns in ultrachat_source(path, skip):
            mx = max(len(t.split()) for t in turns)
            w = sum(len(t.split()) for t in turns)
            simple = mx <= T and \
                res["simple_words"] + w <= infancy_words * 1.05
            f = fs if simple else fr
            k = "simple" if simple else "rest"
            f.write(json.dumps({"data": turns}) + "\n")
            res[k + "_words"] += w
            res[k + "_convs"] += 1
    return res


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

    def __init__(self, rng, world_seed, n_roster=24, n_era=4):
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
        if fact["era"]:
            menu = [g for g in menu if g[0] >= menu[-1][0] // 8] \
                or menu
        gaps, ws = zip(*menu)
        gap = self.rng.choices(gaps, weights=ws)[0]
        self.pending.append((pos + gap, fact))

    def plant_unit(self, pos, stage):
        """One plant: 'by the way NAME kept a COL OBJ in the ROOM .'
        + 'noted .' — returns (turns, meta) or None."""
        if not self.unplanted:
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
            turns.append(self._press(-1))
            turns.append((f"not right . the {fact['obj']} was "
                          f"{fact['col']} .", "human", []))
            turns.append(self._press(2))
        fact["asks"] += 1
        return turns, {"fact": fact, "kind": "ask",
                       "correct": correct}

    @staticmethod
    def _press(v):
        tok = f"<{'+' if v > 0 else '-'}{abs(v)}>"
        return (tok, "human", [("button", {"v": int(v)})])


def _iter_stage_source(sources, stage_name):
    """sources: {stage_name: iterator of conversations (list of turn
    strings, human first)}. Falls back to 'default'."""
    it = sources.get(stage_name) or sources["default"]
    return it


def prepare_life(out_dir, budget_tokens, n_lives, seed=0,
                 world_seed=None, vocab=16384, tokenizer_path=None,
                 stages=STAGES_V10, sources=None, tok_sample=1200,
                 spill=4_000_000, shuffle_sessions=False):
    """Writes tokenizer.json, tokens.bin (n_lives equal lives,
    back-to-back), events.jsonl (sorted, absolute pos), manifest.json,
    judge_audit.jsonl.

    shuffle_sessions: the G2 ordering CONTROL — the same world
    (world_seed) and the same day/exchange machinery, but each cast
    ask's recurrence gap is drawn from a SHUFFLED (uniform) menu and
    era facts lose their long-gap privilege, destroying the ordered
    recurrence structure while keeping the world stable (A69-R4:
    shuffle sessions, keep the world)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    world_seed = seed if world_seed is None else world_seed
    sources = sources or {"default": iter_convos(10 ** 9)}
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
                "stages": [], "lives": []}

    def emit_turns(turns):
        """Encode a unit's turns, place events at absolute
        positions. turns: [(text, who, [(kind, meta)])].
        NOTE for the 10B build (item 8): switch the hot path to
        tok.encode_batch over whole days — per-turn encode is fine
        at gate scale only."""
        enc_all = [tok.encode(t).ids for t, _, _ in turns]
        for (text, who, tevs), tids in zip(turns, enc_all):
            t0 = len(stream)
            stream.extend(tids)
            stream.append(eot_m if who == "model" else eot_h)
            for kind, meta in tevs:
                if kind == "button":
                    events.append({"pos": t0, "kind": "button",
                                   **meta})
                    k = str(meta["v"])
                    stats["presses"][k] = \
                        stats["presses"].get(k, 0) + 1
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
                        world_seed * 1000 + life_i + 1)
        if shuffle_sessions:
            for f in cast.roster:
                f["era"] = False
        stage_marks = {}
        day_i = 0
        while len(stream) - life_start < life_budget - 512:
            life_pos = len(stream) - life_start
            stage = sched.stage_at(life_pos)
            stage_marks.setdefault(stage.name, life_pos)
            n_units = rng.randint(*stage.day_units)
            # --- open ritual
            emit_turns([(t, w, []) for t, w in RITUAL_OPEN])
            unit_i = 0
            src = _iter_stage_source(src_iters, stage.name)
            while unit_i < n_units:
                if len(stream) - life_start >= life_budget - 512:
                    break
                unit_i += 1
                # cast plant cadence
                if unit_i % stage.plant_every == 0:
                    u = cast.plant_unit(len(stream), stage)
                    if u:
                        emit_plant(u, life_i)
                # due asks fire first (spacing is the curriculum) —
                # DRAINED in a bounded loop so a 96-token gap drawn
                # off a just-fired ask realizes IN-CONTEXT (an ask is
                # ~30-60 tokens; single-pass draining made sub-256
                # gaps unrealizable and starved G1's in-ctx bin)
                drained = 0
                while drained < 64:
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
                pairs = [(convo[i], convo[i + 1])
                         for i in range(0, len(convo) - 1, 2)]
                for h_text, m_text in pairs:
                    if (life_budget - (len(stream) - life_start)
                            < 4096):
                        break     # life boundary: no pair may cross
                    q = J.grade_dialogue(h_text, m_text)
                    tail = stage.name == "tail"
                    if not J.passes_floor(q):
                        stats["dropped"] += 1
                        audit.maybe(stage.name, life_i, len(stream),
                                    q, "DROP+TRUNC", h_text, m_text,
                                    tail=tail)
                        break        # truncate: coherence preserved
                    press = J.press_for(q, stage.name)
                    turns = [(h_text, "human", []),
                             (m_text, "model", [])]
                    if press:
                        tokp = f"<+{press}>"
                        turns.append((tokp, "human",
                                      [("button", {"v": press,
                                                   "attr": False,
                                                   "stage":
                                                   stage.name})]))
                    emit_turns(turns)
                    stats["kept"] += 1
                    audit.maybe(stage.name, life_i, len(stream), q,
                                press, h_text, m_text, tail=tail)
            # --- close ritual + day event
            emit_turns([(t, w, []) for t, w in RITUAL_CLOSE])
            events.append({"pos": len(stream) - 1, "kind": "day",
                           "stage": stage.name, "life": life_i,
                           "day": day_i})
            day_i += 1
        # --- pad to EXACTLY life_budget with filler days, then eot
        while life_budget - (len(stream) - life_start) > 96:
            emit_turns([(t, w, []) for t, w in
                        (RITUAL_OPEN[:1] + FILLER + RITUAL_CLOSE)])
            stats["filler_days"] += 1
        while len(stream) - life_start < life_budget:
            stream.append(eot_h)
        manifest["lives"].append({
            "life": life_i, "start": life_start, "days": day_i,
            "stage_marks": stage_marks,
            "cast": [{k: f[k] for k in
                      ("name", "obj", "col", "cls", "era", "asks")}
                     for f in cast.roster]})

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


def smoltalk2_source(paths, en_only=True):
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
            for batch in pf.iter_batches(batch_size=256,
                                         columns=["messages"]):
                for msgs in batch.to_pydict()["messages"]:
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
                        yield texts[:n]
    return gen()


def simple_only(src, max_words=80):
    """Infancy filter: only the spine's shortest, simplest
    exchanges (every turn under max_words) — 'least to best, like
    learning' starts with the simplest conversations."""
    def gen():
        for turns in src:
            if all(len(t.split()) <= max_words for t in turns):
                yield turns
    return gen()


def ultrachat_source(path, skip=0):
    """Conversations (lists of alternating turn strings, human
    first) from a local UltraChat jsonl."""
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
                                     "freeze-judge", "split-uc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--stages", default="gate",
                    choices=["gate", "flash"],
                    help="gate=STAGES_V10 (parity), flash=the "
                         "supply-fitted 5.2B table")
    ap.add_argument("--st2-epochs", type=int, default=1)
    ap.add_argument("--magpie-epochs", type=int, default=1)
    ap.add_argument("--ultrachat-simple", default=None,
                    help="split-uc output: pre-filtered infancy file")
    ap.add_argument("--ultrachat-rest", default=None,
                    help="split-uc output: childhood file")
    ap.add_argument("--infancy-tokens", type=int, default=0,
                    help="split-uc: infancy token budget to cover")
    ap.add_argument("--lives", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world-seed", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--ultrachat", default="data/ultrachat_raw.jsonl")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--shuffle-sessions", action="store_true")
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

    def _src(ep=True):
        if a.ultrachat_simple and a.ultrachat_rest:
            # split-uc products: disjoint files, one shared pass
            # already done — no filter, no double consumption
            s = {"default": ultrachat_source(a.ultrachat_rest),
                 "tokenizer": ultrachat_source(a.ultrachat_rest),
                 "infancy": ultrachat_source(a.ultrachat_simple),
                 "childhood": ultrachat_source(a.ultrachat_rest)}
        else:
            s = {"default": ultrachat_source(a.ultrachat, a.skip),
                 "tokenizer": ultrachat_source(a.ultrachat, a.skip),
                 "infancy": simple_only(
                     ultrachat_source(a.ultrachat, a.skip)),
                 "childhood": ultrachat_source(a.ultrachat,
                                               a.skip + 40)}
        if a.st2_dir:
            st2 = sorted(f for f in
                         _glob.glob(f"{a.st2_dir}/*.parquet")
                         if "magpie" not in f)
            s["adolescence"] = epochs(
                lambda: smoltalk2_source(st2),
                a.st2_epochs if ep else 1)
        if a.magpie_dir:
            mag = sorted(
                _glob.glob(f"{a.magpie_dir}/*magpie*.parquet"))
            s["tail"] = epochs(
                lambda: smoltalk2_source(mag),
                a.magpie_epochs if ep else 1)
        return s

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
    elif a.mode == "split-uc":
        assert a.infancy_tokens > 0, "split-uc needs --infancy-tokens"
        assert a.ultrachat_simple and a.ultrachat_rest, \
            "split-uc needs --ultrachat-simple/--ultrachat-rest"
        # words needed ~ tokens / (tok/word of simple text, ~1.6)
        res = split_ultrachat(a.ultrachat, a.ultrachat_simple,
                              a.ultrachat_rest,
                              int(a.infancy_tokens / 1.55),
                              skip=a.skip)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
    else:
        assert a.budget > 0, "prepare needs --budget"
        if a.judge_thresholds:
            J.freeze_stage_thresholds(a.judge_thresholds)
        prepare_life(a.out, a.budget, a.lives, seed=a.seed,
                     world_seed=a.world_seed, vocab=a.vocab,
                     tokenizer_path=a.tokenizer, sources=_src(),
                     stages=stages_sel,
                     tok_sample=a.tok_sample,
                     shuffle_sessions=a.shuffle_sessions)
