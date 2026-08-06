"""v5.0 procedural generator — the all-synthetic conveyor's precision core.

Two scene types, every cue and label known by construction (card A5):
  saga    — multi-chapter prose; facts planted early, re-stated late at
            controlled gaps. Every re-statement is an annotated probe:
            (position of the answer token, the answer id, the gap back
            to the plant). Channels grade the model's probability of
            the answer token at that position — ground truth by
            construction, ungameable (the model cannot edit the text).
  episode — agent task narrated as human/model turns (<eot_human> /
            <eot_model>); a tiny world sim verifies the outcome, and
            <ok> appears iff the goal was actually achieved. Failures
            occur by construction (~1/3), so earned/not-earned varies.

Markers (pipeline-written): <scene> </scene> <eot_human> <eot_model> <ok>.
Closed vocabulary: the generator's lexicon IS the token set (word-level;
exact positions, exact channels — the A5 simplification).
"""

import random

SPECIALS = ["<pad>", "<scene>", "</scene>", "<eot_human>", "<eot_model>", "<ok>"]

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

def _lexicon():
    words = set(SPECIALS)
    words.update(NAMES + OBJECTS + COLORS + ROOMS)
    for s in FILL_SUBJ:
        words.update(s.split())
    for v in FILL_VERB:
        words.update(v.split())
    words.update("""a the kept was in her his it and then that day chapter
    one morning later found took gave to went looked at remembered still
    had said asked where is what color of answered i you please find bring
    the- go picked up put down opened not right good job thanks done young
    old quiet .""".split())
    words.discard("the-")
    words.add("?")
    return sorted(words)

LEXICON = _lexicon()


def _fill(rng, n):
    out = []
    while len(out) < n:
        out += rng.choice(FILL_SUBJ).split() + rng.choice(FILL_VERB).split() + ["."]
    return out[:n]


def saga(rng, target_len=2048, n_facts=6):
    """One prose scene with planted facts re-stated at controlled gaps.

    Returns (tokens, meta). meta['probes'] = list of dicts with absolute
    in-scene positions: pos (the answer token), answer (word), gap.
    """
    toks = ["<scene>"]
    facts, probes = [], []
    used = set()
    while len(facts) < n_facts:
        name, obj, col = rng.choice(NAMES), rng.choice(OBJECTS), rng.choice(COLORS)
        if (name, obj) in used:
            continue
        used.add((name, obj))
        facts.append({"name": name, "obj": obj, "col": col, "plant": None, "done": False})
    chapter = 1
    while facts or len(toks) < target_len:
        toks += ["chapter", "one", "."] if chapter == 1 else ["chapter", "later", "."]
        chapter += 1
        toks += _fill(rng, rng.randint(20, 60))
        unplanted = [f for f in facts if f["plant"] is None]
        if unplanted and rng.random() < 0.9:
            f = rng.choice(unplanted)
            toks += [f["name"], "kept", "a", f["col"], f["obj"], "in", "the",
                     rng.choice(ROOMS), "."]
            f["plant"] = len(toks) - 5  # position of the color token
        ripe = [f for f in facts if f["plant"] is not None and not f["done"]
                and len(toks) - f["plant"] > 64]
        if ripe and rng.random() < 0.6:
            f = rng.choice(ripe)
            toks += ["the", f["obj"], f["name"], "kept", "was", f["col"], "."]
            pos = len(toks) - 2  # the color token just emitted
            probes.append({"pos": pos, "answer": f["col"], "gap": pos - f["plant"]})
            f["done"] = True
        facts = [f for f in facts if not f["done"]]
        if len(toks) >= target_len and not facts:
            break
        if len(toks) > target_len * 3:  # safety
            break
    toks += ["</scene>"]
    return toks, {"type": "saga", "probes": probes, "ok": True,
                  "turn_ends": [], "goal": None}


def episode(rng, distract=120):
    """One agent scene: instruction, narrated actions, verified outcome.

    A tiny world: objects are in rooms; the goal names one object. The
    'model' fetches either the right object (success) or a wrong one
    (~1/3, failure by construction). <ok> appears iff verified success.
    The human closes success with the reward words as ordinary text.
    """
    world = {obj: rng.choice(ROOMS) for obj in rng.sample(OBJECTS, 6)}
    goal_obj = rng.choice(list(world))
    goal_col = rng.choice(COLORS)
    toks = ["<scene>"]
    toks += ["please", "find", "the", goal_col, goal_obj, "and", "bring",
             "it", "to", "the", "hall", ".", "<eot_human>"]
    turn_ends = [len(toks) - 1]
    succeed = rng.random() > 0.33
    fetched = goal_obj if succeed else rng.choice([o for o in world if o != goal_obj])
    toks += ["i", "looked", "in", "the", world[fetched], "."]
    toks += _fill(rng, rng.randint(10, distract))
    toks += ["i", "picked", "up", "the", goal_col if succeed else rng.choice(COLORS),
             fetched, "and", "went", "to", "the", "hall", ".", "<eot_model>"]
    turn_ends.append(len(toks) - 1)
    # probe: the human asks back what was asked for (recall inside dialogue)
    toks += ["what", "color", "of", goal_obj, "was", "asked", "?", "<eot_human>"]
    turn_ends.append(len(toks) - 1)
    ans_pos = len(toks) + 3  # i answered <col>
    toks += ["i", "answered", goal_col, ".", "<eot_model>"]
    turn_ends.append(len(toks) - 1)
    probes = [{"pos": ans_pos - 1, "answer": goal_col, "gap": ans_pos - 5}]
    if succeed:
        toks += ["good", "job", ".", "thanks", ".", "<eot_human>"]
        turn_ends.append(len(toks) - 1)
        toks += ["<ok>"]
    else:
        toks += ["that", "is", "not", "right", ".", "<eot_human>"]
        turn_ends.append(len(toks) - 1)
    toks += ["</scene>"]
    return toks, {"type": "episode", "probes": probes, "ok": succeed,
                  "turn_ends": turn_ends, "goal": (goal_obj, goal_col)}


SCENE_TYPES = {"saga": saga, "episode": episode}


def make_scene(rng, kind, **kw):
    return SCENE_TYPES[kind](rng, **kw)
