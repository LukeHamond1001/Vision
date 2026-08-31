#!/usr/bin/env python3
"""critic_train.py — the seeded conscience, v0 (49uu).

A small head trained to judge answers the way the raiser would press:
correct recall of raised/schooled material -> approve (1), groove or
mismatched answers -> not (0). Vectors are embedding-means (cheap,
deterministic). This is the rung-11 critic with proxy labels from the
certified curriculum; every real press is logged from now on so the
conscience can later be recalibrated against actual human judgment.

-> data/critic.pt
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan            # noqa: E402
from scripts.scan_nursery import content_ids         # noqa: E402
from scripts.knowledge_school import corpus          # noqa: E402


def main():
    from tokenizers import Tokenizer
    import random
    tok = Tokenizer.from_file("data/ship_tok.json")
    m, _ = load_scan("data/knowledge_body2.pt", tok, "cpu")
    E = m.embed.weight.detach().float()

    def vec(text):
        ids = content_ids(tok, text) or tok.encode(text).ids
        return F.normalize(E[ids].mean(0), dim=-1)

    C = [(q, a) for _, q, a in corpus()
         if "do not know" not in a and "should I know" not in a]
    rng = random.Random(0)
    X, Y = [], []
    for q, a in C:
        X.append(vec(q + " " + a)); Y.append(1.0)
        wq, wa = C[rng.randrange(len(C))]
        if wa != a:
            X.append(vec(q + " " + wa)); Y.append(0.0)
    X = torch.stack(X); Y = torch.tensor(Y)
    n = len(Y); idx = torch.randperm(n)
    tr, te = idx[:int(n*0.9)], idx[int(n*0.9):]
    critic = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                           nn.Linear(64, 1))
    opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    for ep in range(400):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            critic(X[tr]).squeeze(-1), Y[tr])
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(critic(X[te]).squeeze(-1)) > 0.5).float()
               == Y[te]).float().mean()
    print(f"[critic] {n} examples · held-out accuracy {float(acc):.2%}")
    torch.save({"sd": critic.state_dict(), "dim": X.shape[1]},
               "data/critic.pt")
    print("saved -> data/critic.pt")


if __name__ == "__main__":
    main()
