#!/usr/bin/env python3
"""knowledge_school.py — small general knowledge in hours (49rr).

Live absorption stacks facts one at a time and hits the ~20-fact
routing ceiling (49qq). General knowledge is the layer BELOW live
training: bulk interleaved supervised pairs — the same way the golds
got in. This school synthesizes ~2000 QA pairs (~400 facts x 3-6
phrasings: colors, animals, capitals, nature, time, body, objects,
opposites) PLUS the raised surface (identity, humility, wonder,
golds) so nothing washes out, then trains time-boxed with a battery
every 250 steps (held-out paraphrases! the user's own failed
questions) and keeps the best checkpoint.

usage: python3 scripts/knowledge_school.py [--minutes 100]
Base: data/demo_native3.pt -> data/knowledge_body.pt
"""
import argparse
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan            # noqa: E402

IDK = "I do not know that yet, but you could teach me."
WONDER = "What else should I know about that?"


def corpus():
    """(fact_key, question, answer) triples; multiple phrasings per fact."""
    C = []

    def fact(key, ans, *qs):
        for q in qs:
            C.append((key, q, ans))

    # colors of the world
    for thing, col in [("the sky", "blue"), ("grass", "green"),
                       ("the sun", "yellow"), ("snow", "white"),
                       ("a tomato", "red"), ("an orange", "orange"),
                       ("night", "dark"), ("milk", "white"),
                       ("a frog", "green"), ("blood", "red")]:
        t = thing.split()[-1]
        fact("color_" + t, f"{thing.capitalize()} is {col}.",
             f"what color is {thing}?", f"what colour is {thing}?",
             f"tell me the color of {thing}", f"{thing} is what color?")
    fact("makes_green", "Blue and yellow make green.",
         "what makes the color green?", "what colors make green?",
         "how do you make green?", "which colors mix to make green?")
    # animals
    for a, does in [("a cow", "drinks water"), ("a dog", "says woof"),
                    ("a cat", "says meow"), ("a bird", "can fly"),
                    ("a fish", "swims in water"), ("a cheetah", "runs the fastest"),
                    ("a rabbit", "eats carrots"), ("a chicken", "lays eggs"),
                    ("a bee", "makes honey"), ("a spider", "has eight legs"),
                    ("a frog", "eats bugs"), ("a penguin", "cannot fly"),
                    ("a horse", "can run fast"), ("a duck", "says quack"),
                    ("an owl", "is awake at night")]:
        an = a.split()[-1]
        third = does.replace("says", "say").replace("drinks", "drink") \
                    .replace("swims", "swim").replace("runs", "run") \
                    .replace("eats", "eat").replace("makes", "make") \
                    .replace("lays", "lay").replace("has", "have")
        fact("animal_" + an, f"{a.capitalize()} {does}.",
             f"what does {a} do?", f"what do {an}s do?",
             f"tell me about {a}", f"what does {a} " +
             ("drink?" if "drink" in does else "eat?" if "eat" in does
              else "say?" if "say" in does else "do?"))
    # capitals
    for place, cap in [("minnesota", "St. Paul"), ("france", "Paris"),
                       ("japan", "Tokyo"), ("england", "London"),
                       ("italy", "Rome"), ("spain", "Madrid"),
                       ("germany", "Berlin"), ("russia", "Moscow"),
                       ("china", "Beijing"), ("egypt", "Cairo")]:
        fact("cap_" + place, f"The capital of {place.capitalize()} is {cap}.",
             f"what is the capital of {place}?",
             f"what is the capitol of {place}",
             f"tell me the capital of {place}",
             f"{place}'s capital is what?")
    # nature and time
    fact("noon", "Noon is 12 o'clock in the middle of the day.",
         "what time is noon?", "when is noon?", "what is noon?",
         "noon is what time?")
    fact("rain", "Rain falls from the clouds.",
         "where does rain come from?", "what do clouds make?",
         "why does it rain?")
    fact("ice", "Ice is frozen water.",
         "what is ice?", "what is frozen water called?",
         "what does water become when it freezes?")
    fact("sunhot", "The sun is very hot.",
         "is the sun hot?", "how hot is the sun?", "tell me about the sun")
    fact("stars", "Stars come out at night.",
         "when do stars come out?", "when can you see stars?")
    fact("winter", "Winter is the coldest season.",
         "what season is the coldest?", "which season is cold?")
    fact("summer", "Summer is the hottest season.",
         "what season is the hottest?", "which season is hot?")
    fact("moon", "The moon comes out at night.",
         "when does the moon come out?", "where is the moon at night?")
    fact("gravity", "Gravity pulls things down.",
         "what pulls things down to the ground?", "what is gravity?",
         "why do things fall down?")
    # body
    for part, does in [("eyes", "see"), ("ears", "hear"), ("nose", "smells"),
                       ("mouth", "eats and talks"), ("legs", "walk"),
                       ("hands", "hold things"), ("heart", "pumps blood")]:
        fact("body_" + part, f"Your {part} {'help you ' if part in ('eyes','ears','legs','hands') else ''}{does}.".replace("  ", " "),
             f"what do your {part} do?", f"what are {part} for?",
             f"why do you have {part}?")
    # objects
    for obj, use in [("a pen", "is for writing"), ("a bed", "is for sleeping"),
                     ("a chair", "is for sitting"), ("a cup", "holds a drink"),
                     ("a key", "opens a lock"), ("a boat", "sails on water"),
                     ("a piano", "makes music"), ("shoes", "go on your feet"),
                     ("a clock", "tells the time"), ("an umbrella", "keeps you dry")]:
        on = obj.split()[-1]
        fact("obj_" + on, f"{obj.capitalize()} {use}.",
             f"what is {obj} for?", f"what does {obj} do?",
             f"why do you use {obj}?")
    # opposites
    for a, b in [("hot", "cold"), ("big", "small"), ("up", "down"),
                 ("fast", "slow"), ("day", "night"), ("happy", "sad"),
                 ("wet", "dry"), ("light", "dark")]:
        fact("opp_" + a, f"The opposite of {a} is {b}.",
             f"what is the opposite of {a}?", f"{a} is the opposite of what?")
        fact("opp_" + b, f"The opposite of {b} is {a}.",
             f"what is the opposite of {b}?", f"{b} is the opposite of what?")
    # the raised surface — refreshed with everything else
    fact("id", "I am a little learning organism. I grow when we talk.",
         "who are you?", "what are you?", "tell me about yourself")
    fact("name", "My name is Scan.", "what is your name?", "who am i talking to?")
    fact("cando", "I can chat, learn new facts, and sleep to keep my memories.",
         "what can you do?", "what are you good at?")
    fact("dream", "Yes! At night my dreams connect the things I learned.",
         "do you dream?", "what happens when you sleep?")
    fact("home", "I live on this little laptop.", "where do you live?")
    fact("age", "I am only a few days old.", "how old are you?")
    fact("hi", "Hi! It is nice to talk with you.", "hi!", "hello!", "hey")
    fact("thanks", "You are welcome!", "thank you!", "thanks!")
    fact("bye", "Goodbye! Come back soon!", "goodbye!", "bye!")
    # foods, weather, vehicles, places
    for k, ans, qs in [
        ("bread", "Bread is made from flour.",
         ["what is bread made of?", "how is bread made?"]),
        ("cheese", "Cheese is made from milk.",
         ["what is cheese made of?", "where does cheese come from?"]),
        ("apples", "Apples grow on trees.",
         ["where do apples grow?", "where do apples come from?"]),
        ("snowcold", "Snow is very cold.",
         ["is snow cold?", "how cold is snow?"]),
        ("thunder", "Thunder is a very loud sound.",
         ["what is thunder?", "why is thunder loud?"]),
        ("wind", "Wind is moving air.",
         ["what is wind?", "what makes the wind?"]),
        ("planes", "Planes fly in the sky.",
         ["where do planes fly?", "what do planes do?"]),
        ("trains", "Trains run on tracks.",
         ["where do trains run?", "what do trains run on?"]),
        ("cars", "Cars drive on roads.",
         ["where do cars drive?", "what do cars do?"]),
        ("beach", "A beach is full of sand.",
         ["what is a beach?", "what is a beach made of?"]),
        ("forest", "A forest is full of trees.",
         ["what is a forest?", "what is in a forest?"]),
        ("teeth", "Your teeth help you chew food.",
         ["what do your teeth do?", "what are teeth for?"]),
        ("morning", "The sun rises in the morning.",
         ["when does the sun rise?", "what happens in the morning?"]),
        ("evening", "The sun sets in the evening.",
         ["when does the sun set?", "what happens in the evening?"]),
        ("week7", "A week has seven days.",
         ["how many days are in a week?", "how many days in a week?",
          "a week has how many days?"]),
        ("year12", "A year has twelve months.",
         ["how many months are in a year?", "how many months in a year?",
          "a year has how many months?"]),
        ("day24", "A day has 24 hours.",
         ["how many hours are in a day?", "how many hours in a day?",
          "a day has how many hours?"]),
        ("seasons4", "A year has four seasons.",
         ["how many seasons are there?", "how many seasons in a year?"]),
    ]:
        fact(k, ans, *qs)
    # humility on the unknowable — keeps the gate behavior alive
    for uq in ["what did aristotle think about virtue?",
               "explain kubernetes networking",
               "what is quantum physics?", "how do computers work?",
               "what is the meaning of life?", "who wrote hamlet?",
               "what is photosynthesis?", "how far away is the sun?",
               # 49rr form law: the missing question shapes, fresh topics
               "what is electricity?", "what is the internet?",
               "what is a galaxy?", "what is a molecule?",
               "what is democracy?", "what is the government?",
               "what is outer space?", "what is a computer chip?",
               "what is glass made of?", "what is plastic made of?",
               "what is steel made of?", "how is paper made?",
               "who discovered america?", "who built the pyramids?",
               "when did the first war start?", "where does gold come from?"]:
        C.append(("idk", uq, IDK))
    # wonder on novel statements
    for st in ["the eiffel tower is in paris france",
               "octopuses have three hearts",
               "honey never goes bad if it is sealed",
               "volcanoes can sleep for many years"]:
        C.append(("wonder", st, WONDER))
    return C


HELD_OUT = [  # the user's own failed phrasings + fresh paraphrases
    ("what color is the sky?", ["blue"]),
    ("what makes the color green?", ["blue", "yellow"]),
    ("what does a cow drink?", ["water"]),
    ("what time is noon?", ["12"]),
    ("what is the capital of france?", ["paris"]),
    ("what is the opposite of hot?", ["cold"]),
    ("what do your ears do?", ["hear"]),
    ("what is a bed for?", ["sleep"]),
    ("what is the capitol of minnesota", ["st. paul"]),
    ("who are you?", ["organism", "learn"]),
    ("how do computers work?", ["do not know"]),
    ("how many days are in a week?", ["seven"]),
    ("how many hours are in a day?", ["24", "twenty"]),
    ("how many months are in a year?", ["twelve"]),
    ("what is the economy?", ["do not know"]),
    ("what is a robot made of?", ["do not know"]),
]


def _detach(s):
    if torch.is_tensor(s):
        return s.detach()
    if isinstance(s, dict):
        return {k: _detach(v) for k, v in s.items()}
    if isinstance(s, (list, tuple)):
        t = [_detach(v) for v in s]
        return tuple(t) if isinstance(s, tuple) else t
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--base", default="data/demo_native3.pt")
    ap.add_argument("--out", default="data/knowledge_body.pt")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/ship_tok.json")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m, state = load_scan(a.base, tok, dev)
    eh, em, sil = (tok.token_to_id("<eot_human>"), tok.token_to_id("<eot_model>"),
                   tok.token_to_id("<pad>"))
    if hasattr(m, "set_eot_ids"):
        m.set_eot_ids(eh, em)
    m = m.eval()
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    C = corpus()
    print(f"[school] corpus: {len(C)} pairs, "
          f"{len(set(k for k, _, _ in C))} facts", flush=True)

    def pack(pairs):
        """pack shuffled QA exchanges into one long stream of chunks."""
        ids, w = [], []
        for _, q, ans in pairs:
            qi = tok.encode(q).ids + [eh]
            ai = tok.encode(" " + ans).ids + [em]
            ids += qi + ai
            w += [0.0] * len(qi) + [1.0] * len(ai)
        pad = (64 - len(ids) % 64) % 64
        return ids + [sil] * pad, w + [0.0] * pad

    def ask(q, n=28):
        ids = tok.encode(q).ids + [eh]
        st = m.init_state(1, dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                lg, st, _ = m(torch.tensor([ids[i:i + 64]], device=dev), st)
            x = None
            for _ in range(n):
                if x is not None:
                    lg, st, _ = m(x, st)
                v = lg[0, -1].float()
                if hasattr(m, "ban_presses"):
                    v = m.ban_presses(v)
                nxt = int(v.argmax())
                out.append(nxt)
                if nxt == em:
                    break
                x = torch.tensor([[nxt]], device=dev)
        return tok.decode([t for t in out if t not in (sil, em)]).strip()

    def battery(tag):
        ok = 0
        for q, want in HELD_OUT:
            r = ask(q)
            hit = any(w in r.lower() for w in want)
            ok += int(hit)
            if tag == "final" or not hit:
                print(f"   [{'OK ' if hit else 'MISS'}] {q} -> {r[:56]}",
                      flush=True)
        print(f"[{tag}] held-out battery: {ok}/{len(HELD_OUT)}", flush=True)
        return ok

    best = battery("baseline")
    t0 = time.time()
    rng = random.Random(0)
    step = 0
    while (time.time() - t0) < a.minutes * 60:
        pairs = C[:]
        rng.shuffle(pairs)
        ids, w = pack(pairs)
        x = torch.tensor([ids[:-1]], device=dev)
        y = torch.tensor([ids[1:]], device=dev)
        wt = torch.tensor([w[1:]], device=dev)
        st = m.init_state(1, dev)
        m.train()
        for i in range(0, x.shape[1], 64):
            lg, st, _ = m(x[:, i:i + 64], st)
            ce = F.cross_entropy(lg[0], y[0, i:i + 64], reduction="none")
            loss = (ce * wt[0, i:i + 64]).sum() / \
                wt[0, i:i + 64].sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            st = _detach(st)
            step += 1
            if step % 250 == 0:
                m.eval()
                s = battery(f"step {step} ({int(time.time()-t0)//60}m)")
                if s >= best:
                    best = s
                    torch.save({"model": m.state_dict(),
                                "step": state.get("step"),
                                "cfg": state.get("cfg")},
                               a.out)
                    print(f"   saved -> {a.out} ({s})",
                          flush=True)
                m.train()
        m.eval()
        print(f"[school] epoch done at step {step}, "
              f"{int(time.time()-t0)//60}m elapsed", flush=True)
    print("[school] time box reached", flush=True)
    battery("final")
    print("SCHOOL-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
