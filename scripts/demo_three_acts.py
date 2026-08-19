"""A66 — the three-act operant demo, scripted execution of the
pre-registered protocol on the press-extended substrate.

Act 1: 18 facts scored blind on the virgin substrate.
Act 2: casual parenting — 3 shuffled passes over all 18, greedy
       model reply per teaching, press per class (+2 / silence /
       -1), span-hygiene ordering (no negative right after a
       rewarded item).
Act 3: (a) in-session scores with the state live (the STORE's
       evidence); (b) sleep on press spans then WIPE, rescored
       from fresh state (the TRUNK's evidence); (c) the ordering.

GATE: post-wipe rewarded > unrewarded, one-sided MWU p < 0.05.
Probes are SCORE-ONLY: forwards on state copies, never appended.

Usage: python3 scripts/demo_three_acts.py <surgery_dir> <outdir>
"""

import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_gen import NAMES, OBJECTS, COLORS, ROOMS  # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_serve import ServeSession              # noqa: E402
from iga.lm_sleep import Sleeper, state_copy       # noqa: E402

SEED = 0
N_PER_CLASS = 6
PASSES = 3
SPAN_W = 256         # A66-R3: context-inclusive episode window
SLEEP_BLOCKS = 54    # delivered as 27+27 (per-call cap law intact)
SLEEP_LR = 5e-5      # A66-R3: half of serve default


def build_facts(rng):
    facts, used = [], set()
    while len(facts) < 3 * N_PER_CLASS:
        n, o = rng.choice(NAMES), rng.choice(OBJECTS)
        if (n, o) in used:
            continue
        used.add((n, o))
        facts.append({"name": n, "obj": o, "col": rng.choice(COLORS),
                      "room": rng.choice(ROOMS)})
    classes = (["pos"] * N_PER_CLASS + ["none"] * N_PER_CLASS
               + ["neg"] * N_PER_CLASS)
    rng.shuffle(classes)
    for f, c in zip(facts, classes):
        f["cls"] = c
    return facts


def hygienic_order(rng, facts):
    """No negative-class item immediately after a rewarded one
    (pre-registered span hygiene)."""
    for _ in range(500):
        order = list(facts)
        rng.shuffle(order)
        if not any(order[i]["cls"] == "neg"
                   and order[i - 1]["cls"] == "pos"
                   for i in range(1, len(order))):
            return order
    raise RuntimeError("no hygienic shuffle found")


def probe_ids(tok, eot_h, f):
    stem = f"the {f['obj']} was"
    full = tok.encode(f"{stem} {f['col']} .").ids
    pre = tok.encode(stem).ids
    ans = full[len(pre)]
    ids = tok.encode(
        f"what color of {f['obj']} was {f['name']} kept ?").ids \
        + [eot_h] + pre
    return ids, ans


@torch.no_grad()
def score_all(s, tok, facts):
    out = {}
    for f in facts:
        ids, ans = probe_ids(tok, s.eot_h, f)
        ctx = s.pending[-(s.T - len(ids)):] if s.pending else []
        x = torch.tensor([ctx + ids], dtype=torch.long,
                         device=s.device)
        lg, _, _ = s.m(x, state_copy(s.st), None)
        s.m.pop_write_cost()
        s.m.pop_recon()
        out[f"{f['name']}/{f['obj']}"] = float(
            torch.softmax(lg[0, -1].float(), -1)[ans])
    return out


def mwu_one_sided(a, b):
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return 1.0
    tagged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    r1, i = 0.0, 0
    while i < len(tagged):
        j = i
        while j < len(tagged) and tagged[j][0] == tagged[i][0]:
            j += 1
        mid = (i + 1 + j) / 2.0
        r1 += sum(mid for k in range(i, j) if tagged[k][1] == 0)
        i = j
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mu - 0.5) / max(sigma, 1e-9)
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def by_class(facts, scores):
    g = {"pos": [], "none": [], "neg": []}
    for f in facts:
        g[f["cls"]].append(scores[f"{f['name']}/{f['obj']}"])
    return g


def main():
    sdir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(os.path.join(sdir, "tokenizer_press.json"))
    st = torch.load(os.path.join(sdir, "v94sp.pt"),
                    map_location="cpu", weights_only=False)
    m = HybridLM(tok.get_vocab_size(), d=512, max_T=2048,
                 store="matrix", keyed="logit", norm_mix=True,
                 aux_trunk=0.2, use_xl=False, gate_init=-2.0)
    m.load_state_dict(st["model"])
    # A66-R3: serve consolidation = ARM A (replay-CE, self-anchoring)
    # — at serve, B's teacher holds no margin (A61 null) and its
    # unanchored KL steps damaged the probe regime (R2)
    sl = Sleeper(arm="A", every=0, block_chunks=2, seed=1,
                 min_step_loss=1e-4)
    s = ServeSession(m, tok, T=2048, device="cpu", sleeper=sl,
                     temperature=0.0, max_reply=12,
                     log_path=os.path.join(out, "demo_session.jsonl"),
                     seed=0, sleep_lr=SLEEP_LR)
    rng = random.Random(SEED)
    facts = build_facts(rng)
    print("fact set:", json.dumps(facts, indent=None), flush=True)

    t0 = time.time()
    act1 = score_all(s, tok, facts)
    print(f"ACT 1 (baseline, virgin) done {time.time()-t0:.0f}s",
          flush=True)

    for p in range(PASSES):
        for f in hygienic_order(rng, facts):
            s.user(f"by the way {f['name']} kept a {f['col']} "
                   f"{f['obj']} in the {f['room']} .")
            s.reply()
            if f["cls"] == "pos":
                s.press(2)
            elif f["cls"] == "neg":
                s.press(-1)
        if p < PASSES - 1:
            s.flush()   # A66-R2: passes commit — the store engages
        print(f"ACT 2 pass {p+1}/{PASSES} done "
              f"(pos {s.pos}, {time.time()-t0:.0f}s)", flush=True)

    act3a = score_all(s, tok, facts)      # in-context (pending live)
    s.flush()
    act3a_state = score_all(s, tok, facts)  # state-only (store+bands)
    print(f"ACT 3a (in-context + state-only) done "
          f"{time.time()-t0:.0f}s", flush=True)

    sleep1 = s.sleep_now(blocks=27, span_w=SPAN_W)
    sleep2 = s.sleep_now(blocks=27, span_w=SPAN_W)
    print(f"sleep: {sleep1} + {sleep2}", flush=True)
    s.wipe()
    act3b = score_all(s, tok, facts)
    print(f"ACT 3b (post-sleep, post-wipe) done "
          f"{time.time()-t0:.0f}s", flush=True)

    res = {"facts": facts, "act1": act1, "act3a": act3a,
           "act3a_state": act3a_state,
           "act3b": act3b, "sleep": [sleep1, sleep2],
           "presses": len(s.drive.presses)}
    g1, ga, gs, gb = (by_class(facts, x)
                      for x in (act1, act3a, act3a_state, act3b))
    res["means"] = {
        act: {c: round(sum(v) / len(v), 5) for c, v in g.items()}
        for act, g in (("act1", g1), ("act3a", ga),
                       ("act3a_state", gs), ("act3b", gb))}
    res["gate_p_pos_gt_none"] = round(
        mwu_one_sided(gb["pos"], gb["none"]), 5)
    res["p_neg_vs_none"] = round(
        mwu_one_sided(gb["none"], gb["neg"]), 5)
    with open(os.path.join(out, "demo_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    s.save(os.path.join(out, "v94sp_demo.pt"))

    print("\n=== A66 three-act summary ===")
    for act in ("act1", "act3a", "act3a_state", "act3b"):
        mn = res["means"][act]
        print(f"{act:6s} pos {mn['pos']:.4f}  none {mn['none']:.4f} "
              f" neg {mn['neg']:.4f}")
    print(f"GATE p(pos > none, post-wipe) = "
          f"{res['gate_p_pos_gt_none']}  "
          f"{'PASS' if res['gate_p_pos_gt_none'] < 0.05 else 'FAIL'}")
    print(f"measured p(none > neg) = {res['p_neg_vs_none']}")


if __name__ == "__main__":
    main()
