"""Social school: the balanced pass for the NON-fact surface — the
greetings, statements, warmth, and identity lines the fact-heavy
mini_school leaves uncovered (measured: after a fact school, facts
heal verbatim while the social slot falls to a degenerate loop).
Every response line is one of the organism's OWN raised lines from
its history — nothing new is authored, the pass only rebalances what
it already was.

Usage: social_school.py data/organism_life.pt data/ship_tok.json \
           --minutes 8 --out data/organism_life.pt
"""
import argparse
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tok")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--lr", type=float, default=1.0e-5)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    m, _ = load_scan(a.ckpt, tok, a.dev)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    eh = tok.token_to_id("<eot_human>")
    em = tok.token_to_id("<eot_model>")
    sil = tok.token_to_id("<pad>")

    HI = "Hi! It is nice to talk with you."
    GM = "Good morning! I am awake and ready."
    YW = "You are welcome!"
    ID = "I am a little learning organism. I grow when we talk."
    DR = "Yes! At night my dreams connect the things I learned."
    BY = "Goodbye! Come back soon!"
    deck = [
        ("hi!", HI), ("hello.", HI), ("hello there!", HI),
        ("hey little one", HI), ("hi again!", HI),
        ("good morning!", GM), ("good morning, little one", GM),
        ("morning!", GM),
        ("thank you", YW), ("thanks!", YW), ("thank you so much", YW),
        ("who are you?", ID), ("what are you?", ID),
        ("tell me about yourself", ID),
        ("do you dream?", DR), ("did you dream last night?", DR),
        ("goodbye!", BY), ("bye bye", BY), ("see you soon!", BY),
        ("goodnight little one", BY),
        # statements: warm words receive the identity, not a loop
        ("you are doing so well today.", ID),
        ("i am proud of you.", ID),
        ("you have been doing so wonderfully and i am so proud "
         "of how you have grown.", ID),
        ("what a lovely day it is outside.", HI),
    ]

    def exch(q, ans):
        ids = (tok.encode(q).ids + [eh]
               + tok.encode(" " + ans).ids + [em])
        ids += [sil] * ((64 - len(ids) % 64) % 64)
        return ids

    def ce_of(q, ans):
        ids = tok.encode(q).ids + [eh]
        ans_ids = tok.encode(" " + ans).ids
        x = torch.tensor([ids + ans_ids], device=a.dev)
        with torch.no_grad():
            st = m.init_state(1, a.dev)
            tot, n = 0.0, 0
            for i in range(0, x.shape[1] - 1, 64):
                lg, st, _ = m(x[:, i:i + 64], st)
                for j in range(lg.shape[1]):
                    t = i + j + 1
                    if len(ids) <= t < len(ids) + len(ans_ids):
                        tot += float(F.cross_entropy(
                            lg[0, j:j + 1], x[0, t:t + 1]))
                        n += 1
        return tot / max(1, n)

    probes = [deck[0], deck[5], deck[8], deck[11], deck[20]]
    facts = [("why does a cat purr",
              "A cat purrs to show it is calm and happy."),
             ("what does a bone do",
              "A bone is hard and holds your body up."),
             ("what do cows drink?", "A cow drinks water.")]
    print("-- before --")
    for q, ans in probes:
        print("  social %5.2f  %s" % (ce_of(q, ans), q[:36]))
    pre_f = [ce_of(q, ans) for q, ans in facts]

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    m.train()
    t0 = time.time()
    steps = 0
    rng = random.Random(11)
    work = list(deck)
    while (time.time() - t0) / 60.0 < a.minutes:
        rng.shuffle(work)
        for q, ans in work:
            if (time.time() - t0) / 60.0 >= a.minutes:
                break
            ids = exch(q, ans)
            x = torch.tensor([ids], device=a.dev)
            st = m.init_state(1, a.dev)
            opt.zero_grad(set_to_none=True)
            loss = None
            for i in range(0, x.shape[1] - 1, 64):
                lg, st, _ = m(x[:, i:i + 64], st)
                y = x[0, i + 1:i + 65]
                l_ = F.cross_entropy(lg[0, :y.shape[0]], y)
                loss = l_ if loss is None else loss + l_
            (0.1 * loss).backward()
            opt.step()
            steps += 1
    m.eval()
    print("-- after (%d steps) --" % steps)
    for q, ans in probes:
        print("  social %5.2f  %s" % (ce_of(q, ans), q[:36]))
    worst = 0.0
    for (q, ans), c0 in zip(facts, pre_f):
        c1 = ce_of(q, ans)
        print("  gold %5.2f -> %5.2f  %s" % (c0, c1, q[:36]))
        worst = max(worst, c1 - c0)
    if worst > 0.2:
        print("ABORT: a fact gold moved %.2f — not saving." % worst)
        return
    state["model"] = m.state_dict()
    torch.save(state, a.out)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
