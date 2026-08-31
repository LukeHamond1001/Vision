"""Stutter repair (49mm method, applied): targeted unlikelihood on a
degenerate token loop -- here the "smartph" attractor that captures
weakly-conditioned states (post-teach acks, free-run). Suppresses
P(tok | tok) along a repeat run and P(tok | <eot_human>) at reply
start; nothing else. Golds are measured before and after inside the
same run -- the pass aborts (no save) if any moves more than 0.15.

Usage: stutter_repair.py data/organism_life.pt data/ship_tok.json \
           --word smartph --steps 40 --out data/organism_life.pt
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tok")
    ap.add_argument("--word", default="smartph")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--lr", type=float, default=1.0e-5)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    m, _ = load_scan(a.ckpt, tok, a.dev)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    eh = tok.token_to_id("<eot_human>")
    wid = tok.encode(a.word).ids
    assert len(wid) >= 1, "word did not tokenize"
    print("target ids for %r: %s" % (a.word, wid))

    def loop_prob():
        """P(word-start | a run of the word) and P(word-start | <eh>)."""
        run = (wid * 24)[:48]
        x = torch.tensor([[eh] + run], device=a.dev)
        with torch.no_grad():
            st = m.init_state(1, a.dev)
            lg, st, _ = m(x, st)
            p_run = float(torch.softmax(lg[0, -1].float(), -1)[wid[0]])
            p_eh = float(torch.softmax(lg[0, 0].float(), -1)[wid[0]])
        return p_run, p_eh

    def gold_ce(q, ans):
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
                        tot += float(F.cross_entropy(lg[0, j:j + 1],
                                                     x[0, t:t + 1]))
                        n += 1
        return tot / max(1, n)

    GOLDS = [
        ("what do cows drink?", "A cow drinks water."),
        ("what makes thunder", "Thunder is the sound of lightning heating the air."),
        ("why does a cat purr", "A cat purrs to show it is calm and happy."),
        ("why do stars twinkle", "Stars twinkle because moving air shakes their light."),
        ("what is fog", "Fog is a cloud that sits on the ground."),
        ("how does a jet engine work?",
         "I do not know that yet, but you could teach me."),
    ]
    pre_g = [gold_ce(q, ans) for q, ans in GOLDS]
    pr, pe = loop_prob()
    print("before: P(loop) %.4f · P(reply-start) %.4f" % (pr, pe))
    for (q, _), c in zip(GOLDS, pre_g):
        print("   gold %5.2f  %s" % (c, q[:44]))

    # the unlikelihood pass: a run of the word after <eh>; suppress the
    # word at every continuation position (the loop) incl. position 0
    # (reply start). 0.3x like the serve's own corrective unlearning.
    run = (wid * 24)[:48]
    x = torch.tensor([[eh] + run], device=a.dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    m.train()
    for s in range(a.steps):
        st = m.init_state(1, a.dev)
        opt.zero_grad(set_to_none=True)
        lg, st, _ = m(x, st)
        logp = torch.log_softmax(lg[0].float(), -1)
        loss = None
        for j in range(lg.shape[1] - 1):
            tgt = x[0, j + 1]
            if int(tgt) != wid[0] and int(tgt) not in wid:
                continue
            p_ = logp[j, tgt].exp().clamp(max=0.999)
            ul = -torch.log1p(-p_)
            loss = ul if loss is None else loss + ul
        (0.3 * loss / max(1, lg.shape[1])).backward()
        opt.step()
        if (s + 1) % 10 == 0:
            m.eval()
            pr, pe = loop_prob()
            print("  step %d: P(loop) %.4f · P(start) %.4f" % (s + 1, pr, pe))
            m.train()
    m.eval()
    post_g = [gold_ce(q, ans) for q, ans in GOLDS]
    pr, pe = loop_prob()
    print("after : P(loop) %.4f · P(reply-start) %.4f" % (pr, pe))
    worst = 0.0
    for (q, _), c0, c1 in zip(GOLDS, pre_g, post_g):
        print("   gold %5.2f -> %5.2f  %s" % (c0, c1, q[:44]))
        worst = max(worst, c1 - c0)
    if worst > 0.15:
        print("ABORT: a gold moved %.2f — not saving." % worst)
        return
    state["model"] = m.state_dict()
    torch.save(state, a.out)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
