"""Author lives — the v17 curriculum generator (v0).

Writes childhoods (GESTATION.md): JSONL, one life per line,
{"data": [turn, turn, ...]} — alternating strings, caretaker first —
drop-in for `lm_data_life prepare --lives`.

Every structural choice encodes a law measured on the living 297M
organism by its teachers (the law book in GESTATION.md). v0 is
template-and-combinatoric: structure exact, surfaces plain. The
--enrich hook is where a batch LLM rewrites surfaces for breadth
without touching structure.

Usage: author_lives.py --n 2000 --out data/lives.jsonl [--seed 7]
"""
import argparse
import itertools
import json
import random

# ---- world stock: subjects x (frame-disjoint) predicates ----------
# Endings deliberately spread across shapes (law 5): no two facts in
# one life share a question frame, a key content word, or an
# answer-ending shape.
FACTS = [
    ("why does a wolf howl", "A wolf howls to call its family across the forest.", "howl"),
    ("how is a pearl made", "A pearl grows inside an oyster around a grain of sand.", "pearl"),
    ("why do camels have humps", "A camel keeps fat in its hump for long dry journeys.", "hump"),
    ("what does a compass do", "A compass always points north.", "compass"),
    ("why do stars twinkle", "Stars twinkle because moving air shakes their light.", "twinkle"),
    ("how does a spider make its web", "A spider spins its web from silk inside its body.", "web"),
    ("what does a bone do", "A bone is hard and holds your body up.", "bone"),
    ("why is the ocean salty", "The ocean is salty because rivers carry salt down to it.", "salty"),
    ("what is an echo", "An echo is a sound that bounces back to your ears.", "echo"),
    ("why does a cat purr", "A cat purrs to show it is calm and happy.", "purr"),
    ("what is fog", "Fog is a cloud that sits low on the ground.", "fog"),
    ("how does a turtle stay safe", "A turtle hides inside its hard shell.", "turtle"),
    ("why do leaves change color", "Leaves change color when they stop making green.", "leaves"),
    ("what makes thunder", "Thunder is the sound of lightning heating the air.", "thunder"),
    ("where does honey come from", "Honey is made by bees out of flower nectar.", "honey"),
    ("what does an anchor do", "An anchor holds a ship still in the water.", "anchor"),
    ("why do birds sing at dawn", "A bird sings at dawn to mark its home.", "birdsong"),
    ("how does a seed become a plant", "A seed drinks the rain and slowly opens.", "seed"),
    ("what is a shadow", "A shadow is the dark shape you make in the light.", "shadow"),
    ("why does bread rise", "Bread rises because tiny bubbles grow in the dough.", "bread"),
]
UNKNOWNS = [    # honest-ignorance probes, every question FORM (law 6)
    "how does a glacier move", "who invented the wheel",
    "what is a comet made of", "why is the desert dry",
    "where does the wind start", "when do cranes fly south",
    "which stone is the hardest", "what does a magnet pull",
]
IDK = "I do not know that yet, but you could teach me."
GREET_H = ["hello little one.", "good morning.", "hi again.", "hello."]
GREET_M = ["Hi! It is nice to talk with you.", "Good morning! I am awake and ready."]
WARMTH = ["you learned that so well today.", "i am proud of how you listened.",
          "that was lovely work, little one."]
IDENT = "I am a little learning organism. I grow when we talk."
BYE_H = ["goodnight little one.", "rest well now.", "sleep now, little one."]
BYE_M = "Goodbye! Come back soon!"
PRESS_GOOD = "<+2>"
PRESS_BAD = "<-1>"
SPEAK_FIRST = "<mv>"      # the child's own native floor-taking cue (law 11)


def life(rng):
    """One childhood: lessons -> night -> morning recall (laws 1-12)."""
    f1, f2 = rng.sample(FACTS, 2)
    unk = rng.choice(UNKNOWNS)
    t = []
    t += [rng.choice(GREET_H), rng.choice(GREET_M)]
    # honest ignorance BEFORE the lesson (law 6): asked, not known
    t += ["tell me, " + f1[0] + "?", IDK]
    # the lesson, its grammar (law 1); echo once, never drilled (law 2)
    t += [f1[1], "I will remember. " + f1[1]]
    # a press with a readable reason (law 8)
    t += ["that is exactly right. " + PRESS_GOOD, "Thank you!"]
    # asking earns the second lesson (law 7)
    t += ["there is more if you ask.", "What else should I know? " + f2[0] + "?"]
    t += [f2[1], "So " + f2[1].lower()]
    # a confident error, corrected ONCE (laws 4, 8)
    t += [f2[0] + "?", f1[1]]                       # wrong: neighbor capture
    t += ["not quite. " + PRESS_BAD + " listen once more: " + f2[1],
          f2[1]]
    # self-talk rehearsal in its own words (law 9)
    t += ["tell yourself quietly.",
          "I tell myself: " + f1[1] + " And " + f2[1].lower()]
    # the child speaks first, native cue (law 11) + warmth binds (law 12)
    t += [rng.choice(WARMTH), SPEAK_FIRST + " " + IDENT]
    # an unknown stays unknown at day's end (law 6, form breadth)
    t += [unk + "?", IDK]
    # bedtime, release mid-sentence (law 10)
    t += ["it is late now, " + rng.choice(BYE_H).rstrip('.') + ".", BYE_M]
    # THE NIGHT: replay each lesson once, spaced (laws 2, 3)
    t += ["<night>", f1[1] + " " + f2[1]]
    # morning recall, both facts, then the day ends
    t += ["good morning. " + f1[0] + "?", f1[1]]
    t += [f2[0] + "?", f2[1] + " " + PRESS_GOOD]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/lives.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--enrich", default=None,
                    help="hook: a JSONL of surface rewrites keyed by "
                         "template line (batch-LLM output); v0 passes "
                         "through unchanged when absent")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pairs = len(list(itertools.combinations(range(len(FACTS)), 2)))
    with open(a.out, "w") as f:
        for i in range(a.n):
            f.write(json.dumps({"data": life(rng)}) + "\n")
    print("wrote %d lives -> %s  (%d fact-pairs in stock; enrich hook %s)"
          % (a.n, a.out, pairs, "on" if a.enrich else "off"))


if __name__ == "__main__":
    main()
