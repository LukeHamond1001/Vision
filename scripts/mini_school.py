"""Mini-school (49yy): balanced interleaved re-grounding of the LIVE
body's router. The week's serial serve plasticity (concentrated
pursuit replay + stacked corrective unlikelihood) collapsed the
question->answer router onto single attractors while storage stayed
intact (the 49ss dissociation). The 49rr remedy applied to a living
body: every fact it knows -- session facts, old facts, its own noticed
statements, the innate golds, idk forms -- shuffled flat, small even
doses, no fact favored. School for balance; live for top-up.

Usage: mini_school.py data/organism_life.pt data/ship_tok.json \
           --minutes 14 --out data/organism_life.pt
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
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--lr", type=float, default=1.2e-5)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    m, _ = load_scan(a.ckpt, tok, a.dev)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    life = state.get("life") or {}
    facts = [tuple(x) for x in life.get("facts", [])]
    study = list(life.get("study", []))
    eh = tok.token_to_id("<eot_human>")
    em = tok.token_to_id("<eot_model>")
    sil = tok.token_to_id("<pad>")

    # ---- the curriculum: everything it knows, one flat deck ----
    innate = [
        ("what do cows drink?", "A cow drinks water."),
        ("what color is grass?", "Grass is green."),
        ("what do dogs say?", "A dog says woof."),
        ("what color is the sky?", "The sky is blue."),
        ("is the sun hot?", "The sun is very hot."),
        ("what is ice made of", "Ice is frozen water."),
        ("what color is snow?", "Snow is white."),
        ("what does a fish do?", "A fish swims in the water."),
        ("what color is milk?", "Milk is white."),
        ("what does a chicken lay?", "A chicken lays eggs."),
    ]
    social = [
        ("hi!", "Hi! It is nice to talk with you."),
        ("good morning!", "Good morning! I am awake and ready."),
        ("thank you", "You are welcome!"),
    ]
    idk = [
        ("how does a jet engine work?",
         "I do not know that yet, but you could teach me."),
        ("who wrote the first book?",
         "I do not know that yet, but you could teach me."),
        ("what is a glacier made of?",
         "I do not know that yet, but you could teach me."),
        ("why is the desert dry?",
         "I do not know that yet, but you could teach me."),
    ]
    deck = []
    for q, ans in facts:
        deck.append((q, ans))
    for s in study:
        deck.append((None, s))
    deck += innate + social + idk
    print("deck: %d taught + %d studied + %d innate + %d social + %d idk"
          % (len(facts), len(study), len(innate), len(social), len(idk)))

    def exch(q, ans):
        if q is None:
            ids = tok.encode(ans).ids + [eh]
        else:
            ids = tok.encode(q).ids + [eh] + tok.encode(ans).ids + [em]
        ids += [sil] * ((64 - len(ids) % 64) % 64)
        return ids

    def ce_of(q, ans):
        ids = tok.encode(q).ids + [eh]
        ans_ids = tok.encode(ans).ids
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
                            lg[0, j:j + 1],
                            x[0, t:t + 1]))
                        n += 1
        return tot / max(1, n)

    probes = facts[-7:] + innate[:4]
    print("-- before --")
    for q, ans in probes:
        print("  %5.2f  %s" % (ce_of(q, ans), q[:48]))

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    m.train()
    t0 = time.time()
    steps = 0
    rng = random.Random(7)
    while (time.time() - t0) / 60.0 < a.minutes:
        rng.shuffle(deck)
        for q, ans in deck:
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
            st = None
            steps += 1
        print("  epoch done at step %d (%.1f min)"
              % (steps, (time.time() - t0) / 60.0))
    m.eval()
    print("-- after (%d steps) --" % steps)
    for q, ans in probes:
        print("  %5.2f  %s" % (ce_of(q, ans), q[:48]))

    state["model"] = m.state_dict()
    torch.save(state, a.out)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
