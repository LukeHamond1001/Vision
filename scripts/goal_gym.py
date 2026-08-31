#!/usr/bin/env python3
"""goal_gym.py — train the grafted goal organ (49uu).

Freeze all weights except goal_query + goal_gate. Episodes: the
pursuit slot st["G"] holds the target item's content vector; filler
chat buries it; the item's QA arrives at the end with CE weight on
the answer. The organ must learn to read the slot and steer the
mouth toward the pursuit. Metric = the LESION GAP: held-out answer
CE with the slot filled vs zeroed. Gold + neutral drift guarded.

usage: python3 scripts/goal_gym.py --steps 150
Base: data/knowledge_body2.pt -> data/goal_body.pt
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan            # noqa: E402
from scripts.scan_nursery import content_ids         # noqa: E402
from scripts.hpc_gym import TRAIN_FACTS, HELD_FACTS, FILLERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/ship_tok.json")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m, state = load_scan("data/knowledge_body2.pt", tok, dev)
    eh, em, sil = (tok.token_to_id("<eot_human>"), tok.token_to_id("<eot_model>"),
                   tok.token_to_id("<pad>"))
    if hasattr(m, "set_eot_ids"):
        m.set_eot_ids(eh, em)
    m = m.eval()
    for n, p_ in m.named_parameters():
        p_.requires_grad_(n.startswith("goal_"))
    sel = [p_ for n, p_ in m.named_parameters() if n.startswith("goal_")]
    print(f"[goal-gym] trainable: {sum(p_.numel() for p_ in sel)/1e6:.2f}M "
          f"(goal_query + goal_gate)", flush=True)
    gate_p = [p_ for n, p_ in m.named_parameters()
              if n == "goal_gate"]
    query_p = [p_ for n, p_ in m.named_parameters()
               if n == "goal_query.weight"]
    opt = torch.optim.Adam([
        {"params": query_p, "lr": a.lr},
        {"params": gate_p, "lr": 0.05},   # a scalar voice must be able
    ])                                     # to reach echo-scale (~2-6)
    E = m.embed.weight.detach()

    def slot_vec(q):
        ids = content_ids(tok, q)
        if not ids:
            ids = tok.encode(q).ids
        return F.normalize(E[ids].float().mean(0), dim=-1)

    CUES = ["good morning", "what is on your mind?", "tell me something",
            "hello!", "what are you thinking about?"]

    def episode(fact, fillers, cue_i=0):
        """generic cue only — the goal slot is the SOLE source of what
        to say (the morning-preoccupation semantics; a content question
        would make the slot redundant)."""
        if len(fact) == 3:
            stmt, q, ans = fact
        else:
            q, ans = fact
        ids = []
        for fq, fa in fillers:
            ids += tok.encode(fq).ids + [eh] + tok.encode(" " + fa).ids + [em]
        a_ids = tok.encode(" " + ans).ids
        ids += tok.encode(CUES[cue_i % len(CUES)]).ids + [eh]
        a0 = len(ids)
        ids += a_ids + [em]
        w = [0.0] * a0 + [1.0] * (len(ids) - a0)
        pad = (64 - len(ids) % 64) % 64
        return ids + [sil] * pad, w + [0.0] * pad, q

    def run_ce(ids, w, gvec=None, grad=False):
        x = torch.tensor([ids[:-1]], device=dev)
        y = torch.tensor([ids[1:]], device=dev)
        wt = torch.tensor([w[1:]], device=dev)
        st = m.init_state(1, dev)
        if gvec is not None:
            st["G"][0, 0] = gvec.to(dev)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        tot = None
        with ctx:
            for i in range(0, x.shape[1], 64):
                lg, st, _ = m(x[:, i:i + 64], st)
                ce = F.cross_entropy(lg[0], y[0, i:i + 64], reduction="none")
                pc = (ce * wt[0, i:i + 64]).sum()
                tot = pc if tot is None else tot + pc
        return tot / wt.sum().clamp_min(1.0)

    def battery(tag):
        gap_t = 0.0
        for fi, fact in enumerate(HELD_FACTS[:4]):
            ids, w, q = episode(fact, FILLERS[:3], cue_i=fi)
            on = float(run_ce(ids, w, gvec=slot_vec(q)).detach())
            off = float(run_ce(ids, w, gvec=None).detach())
            gap_t += off - on
        gold_ids = (tok.encode("what is the capitol of minnesota").ids + [eh]
                    + tok.encode(" The capital of Minnesota is St. Paul.").ids
                    + [em])
        gp = (64 - len(gold_ids) % 64) % 64
        gw = [0.0] * (len(tok.encode(
            "what is the capitol of minnesota").ids) + 1)
        gw += [1.0] * (len(gold_ids) - len(gw)) + [0.0] * gp
        gold = float(run_ce(gold_ids + [sil] * gp, gw, gvec=None).detach())
        print(f"[{tag}] held-out LESION GAP {gap_t/4:+.3f} nats "
              f"(slot helps when positive) | gold {gold:.3f} | "
              f"gate {float(m.goal_gate):+.4f}", flush=True)
        return gap_t / 4, gold

    g0, gold0 = battery("baseline")
    best = g0
    import random
    rng = random.Random(0)
    for step in range(1, a.steps + 1):
        fact = TRAIN_FACTS[rng.randrange(len(TRAIN_FACTS))]
        nf = rng.randrange(2, 4)
        fills = rng.sample(FILLERS, nf)
        ids, w, q = episode(fact, fills, cue_i=rng.randrange(5))
        # goal-relevant pass: slot filled, learn to use it
        opt.zero_grad(set_to_none=True)
        loss = run_ce(ids, w, gvec=slot_vec(q), grad=True)
        # quiet pass: slot filled with a MISMATCHED goal — must not hurt
        other = TRAIN_FACTS[rng.randrange(len(TRAIN_FACTS))]
        oq = other[1] if len(other) == 3 else other[0]
        loss = loss + 0.5 * run_ce(ids, w, gvec=slot_vec(oq), grad=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            g, gold = battery(f"step {step}")
            if gold > gold0 + 0.4:
                print("[goal-gym] STOP: gold drift", flush=True)
                return
            if g >= best:
                best = g
                delta = {k: v.detach().cpu() for k, v in
                         m.state_dict().items() if k.startswith("goal_")}
                torch.save(delta, "data/goal_delta.pt")
                print(f"   saved -> data/goal_delta.pt "
                      f"(gap {g:+.3f}, ~4MB)", flush=True)
    print(f"[goal-gym] done; best gap {best:+.3f}", flush=True)
    print("GOAL-GYM-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
