#!/usr/bin/env python3
"""hpc_gym.py — live retrieval training for the hippocampus read path.

The 49bb question: can the store carry same-day recall natively? The
gestation diet never forced it (flat world, locally-predictable chat),
so the read learned to whisper (+1.7 logits, below argmax). Here we
apply the missing pressure post-hoc, WITHOUT touching the carpet:

  freeze all 297M weights EXCEPT the read pathway —
    query_proj   (what the council asks memory)         d x d
    alpha[band]  (how loudly each band's store answers)  scalars
    store_in     (read -> slot feedback)                 d x d
    stores.beta  (write strengths)                       scalars
  train on episodes where a fact is stated once, buried under filler
  chat, then queried — CE weight 1.0 on the answer span, 0.1 elsewhere
  (teaches shout-when-needed, whisper-when-not). Every point of CE the
  gym wins must flow through the store: nothing else can move.

Instruments (held-out facts never trained):
  - answer CE, store on vs off  (the lesion gap = memory's real work)
  - free recall (greedy) on held-outs
  - neutral CE drift on the morning stream + minnesota gold CE
    (the carpet check)

usage: python3 scripts/hpc_gym.py data/demo_body.pt data/ship_tok.json \
          --dev mps --steps 200 --save data/hpc_tuned.pt
"""
import argparse
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan            # noqa: E402

# (question, answer) — or (statement, question, answer) when the fact
# is stated differently than it is answered (names, possession)
TRAIN_FACTS = [
    ("My name is Karim.", "what is my name?", "Your name is Karim."),
    ("My name is Sana.", "what is my name?", "Your name is Sana."),
    ("how many legs does a spider have?", "A spider has eight legs."),
    ("what is the price of the book?", "The book costs five dollars."),
    ("how old is Ana?", "Ana is nine years old."),
    ("what time is lunch?", "Lunch is at one o'clock."),
    ("what is the gdp of france?", "France's GDP is about three trillion dollars."),
    ("what is the cat's name?", "The cat is called Momo."),
    ("where is the spoon?", "The spoon is in the blue cup."),
    ("what is the dog's name?", "The dog is called Pepper."),
    ("what did Mara buy?", "Mara bought three red apples."),
    ("who has the key?", "Sam has the key."),
    ("what is in the basket?", "The basket has two eggs."),
    ("where did Ben go?", "Ben went to the lake."),
    ("what color is the door?", "The door is yellow."),
    ("who made the soup?", "Nina made the soup."),
    ("what is on the table?", "A green book is on the table."),
    ("where is the cat?", "The cat is under the bed."),
    ("what did Tom eat?", "Tom ate warm bread."),
    ("who sings at night?", "The little bird sings at night."),
    ("what is in the box?", "The box holds a small bell."),
    ("where are the shoes?", "The shoes are by the door."),
    ("what did Lily draw?", "Lily drew a big boat."),
    ("who waters the plants?", "Rosa waters the plants."),
]
HELD_FACTS = [
    ("where is the hat?", "The hat is on the chair."),
    ("what is the fish's name?", "The fish is called Blue."),
    ("My name is Luke.", "what is my name?", "Your name is Luke."),
    ("what is the gdp of japan?", "Japan's GDP is about four trillion dollars."),
    ("what time is noon?", "Noon is 12 o'clock in the daytime."),
    ("who opened the window?", "Mia opened the window."),
]
FILLERS = [
    ("how are you today?", "I am happy today. Thank you for asking!"),
    ("do you like the sun?", "Yes, the sun is warm and bright."),
    ("what do you like to do?", "I like to chat and learn new things."),
    ("is it a nice day?", "It is a very nice day."),
    ("do you like music?", "Yes, music is fun to hear."),
    ("are you sleepy?", "Not yet. I sleep at night to keep my memories."),
]


class Gym:
    def __init__(self, a):
        from tokenizers import Tokenizer
        self.a = a
        self.tok = Tokenizer.from_file(a.tok)
        self.dev = a.dev
        self.m, self.state = load_scan(a.ckpt, self.tok, self.dev)
        t = self.tok.token_to_id
        self.eh, self.em, self.sil = (t("<eot_human>"), t("<eot_model>"),
                                      t("<pad>"))
        self.m = self.m.eval()          # reads always on; writes still live
        pat = re.compile(r"query_proj|^alpha\.|store_in|stores\..*\.beta")
        self.sel = []
        for n, p in self.m.named_parameters():
            if pat.search(n):
                p.requires_grad_(True)
                self.sel.append((n, p))
            else:
                p.requires_grad_(False)
        n_par = sum(p.numel() for _, p in self.sel)
        print(f"[gym] trainable read path: {len(self.sel)} tensors, "
              f"{n_par/1e6:.2f}M params "
              f"({[n for n, _ in self.sel if '.' not in n[:6]][:2]}...)",
              flush=True)
        self.opt = torch.optim.Adam([p for _, p in self.sel], lr=a.lr)
        self.g = torch.Generator().manual_seed(a.seed)
        self.morning = [l.strip() for l in open("data/qa_morning.txt")
                        if l.strip()][:12]

    def exch(self, q, ans):
        return (self.tok.encode(q).ids + [self.eh]
                + self.tok.encode(" " + ans).ids + [self.em])

    def episode(self, fact, fillers):
        """statement -> ack -> filler chat -> question -> answer.
        Returns ids, weights (1.0 on answer span + its eot, 0.1 else)."""
        if len(fact) == 3:
            stmt, q, ans = fact
        else:
            q, ans = fact
            stmt = ans
        ids = (self.tok.encode(stmt).ids + [self.eh]
               + self.tok.encode(" Okay!").ids + [self.em])
        for fq, fa in fillers:
            ids += self.exch(fq, fa)
        a_ids = self.tok.encode(" " + ans).ids
        ids += self.tok.encode(q).ids + [self.eh]
        a0 = len(ids)
        ids += a_ids + [self.em]
        w = [0.1] * len(ids)
        for i in range(a0, len(ids)):
            w[i] = 1.0
        ids += [self.sil] * ((64 - len(ids) % 64) % 64)
        w += [0.0] * (len(ids) - len(w))
        return ids, w

    def run_ce(self, ids, w, grad=False, store_off=False):
        """teacher-forced CE over the episode, weighted; fresh state."""
        x = torch.tensor([ids[:-1]], device=self.dev)
        y = torch.tensor([ids[1:]], device=self.dev)
        wt = torch.tensor([w[1:]], device=self.dev)
        self.m.store_read_off = store_off
        st = self.m.init_state(1, self.dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        tot, den = None, wt.sum().clamp_min(1.0)
        with ctx:
            for i in range(0, x.shape[1], 64):
                lg, st, _ = self.m(x[:, i:i + 64], st)
                ce = F.cross_entropy(lg[0], y[0, i:i + 64],
                                     reduction="none")
                pc = (ce * wt[0, i:i + 64]).sum()
                tot = pc if tot is None else tot + pc
        self.m.store_read_off = False
        return tot / den

    def answer_ce(self, fact, fillers, store_off=False):
        """CE on the answer span only (weight-1 tokens)."""
        ids, w = self.episode(fact, fillers)
        w = [1.0 if x == 1.0 else 0.0 for x in w]
        return float(self.run_ce(ids, w, grad=False,
                                 store_off=store_off).detach())

    def recall(self, fact, fillers, n=14):
        """greedy free recall after the question."""
        if len(fact) == 3:
            stmt, q, ans = fact
        else:
            q, ans = fact
            stmt = ans
        ids = (self.tok.encode(stmt).ids + [self.eh]
               + self.tok.encode(" Okay!").ids + [self.em])
        for fq, fa in fillers:
            ids += self.exch(fq, fa)
        ids += self.tok.encode(q).ids + [self.eh]
        st = self.m.init_state(1, self.dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                lg, st, _ = self.m(
                    torch.tensor([ids[i:i + 64]], device=self.dev), st)
            x = None
            for _ in range(n):
                if x is not None:
                    lg, st, _ = self.m(x, st)
                v = lg[0, -1].float()
                if hasattr(self.m, "ban_presses"):
                    v = self.m.ban_presses(v)
                nxt = int(v.argmax())
                if nxt == self.em:
                    break
                out.append(nxt)
                x = torch.tensor([[nxt]], device=self.dev)
        return self.tok.decode(
            [t for t in out if t != self.sil]).strip()

    def neutral_ce(self):
        ids = []
        for i, turn in enumerate(self.morning):
            ids += self.tok.encode(turn).ids + [self.eh if i % 2 == 0
                                                else self.em]
        ids += [self.sil] * ((64 - len(ids) % 64) % 64)
        w = [1.0] * len(ids)
        return float(self.run_ce(ids, w).detach())

    def gold_ce(self):
        ids, w = self.episode(
            ("what is the capitol of minnesota",
             "The capital of Minnesota is St. Paul."), FILLERS[:2])
        w = [1.0 if x == 1.0 else 0.0 for x in w]
        return float(self.run_ce(ids, w).detach())

    def eval_round(self, tag):
        fills = FILLERS[:3]
        on = sum(self.answer_ce(f, fills) for f in HELD_FACTS) / len(HELD_FACTS)
        off = sum(self.answer_ce(f, fills, store_off=True)
                  for f in HELD_FACTS) / len(HELD_FACTS)
        print(f"[{tag}] held-out answer CE: on {on:.3f} / off {off:.3f} "
              f"(lesion gap {off-on:+.3f}) | neutral {self.neutral_ce():.3f}"
              f" | gold {self.gold_ce():.3f}", flush=True)
        for f in HELD_FACTS[:4]:
            print(f"    recall '{f[0]}' -> {self.recall(f, fills)!r}",
                  flush=True)
        return on, off

    def train(self):
        self.eval_round("baseline")
        for step in range(1, self.a.steps + 1):
            fi = int(torch.randint(len(TRAIN_FACTS), (1,),
                                   generator=self.g))
            nf = 2 + int(torch.randint(2, (1,), generator=self.g))
            fis = torch.randperm(len(FILLERS), generator=self.g)[:nf]
            fills = [FILLERS[int(i)] for i in fis]
            ids, w = self.episode(TRAIN_FACTS[fi], fills)
            self.opt.zero_grad(set_to_none=True)
            loss = self.run_ce(ids, w, grad=True)
            loss.backward()
            self.opt.step()
            if step % 40 == 0 or step == self.a.steps:
                print(f"[step {step}] train loss {float(loss):.3f}",
                      flush=True)
                self.eval_round(f"step {step}")
        if self.a.save:
            sd = self.state
            sd["model"] = self.m.state_dict()
            torch.save(sd, self.a.save)
            print(f"[gym] saved -> {self.a.save}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tok")
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--new-only", action="store_true",
                    help="train only the newer families (names, numbers, "
                         "values) — curriculum expansion from a tuned body")
    ap.add_argument("--episodic-only", action="store_true",
                    help="train only the original episodic families — "
                         "the measured round-1 sweet-spot recipe")
    a = ap.parse_args()
    if a.new_only:
        del TRAIN_FACTS[8:]
    if a.episodic_only:
        del TRAIN_FACTS[:8]
    Gym(a).train()


if __name__ == "__main__":
    main()
