"""A64-R debug gate — scripted parenting, LOCAL, $0.

Two arms, same seed: sleep-ON (ARM B at the frozen dose) and
sleep-OFF, both in button mode (~30/50/20 rewarded/unrewarded/
negative planted items, all-good answers, alphas seeded live per
the A62 finding), both with a spectator PressProphet.

Readout: STORE-WIPED re-ask of every logged item from fresh state
(the sharp teach-probe carried from A63): the ask turn is replayed
verbatim and p(color) read at the answer slot. Pre-registered:
rewarded > unrewarded (one-sided Mann-Whitney p < 0.05, sleep
arm); negative <= unrewarded (measured); ordering stronger with
sleep than without; replay mass concentrated on rewarded spans.

Usage: python3 scripts/parenting_debug.py <outdir> [steps]
"""

import json
import math
import os
import sys

import torch

from iga.lm_hybrid import HybridLM
from iga.lm_press import PressProphet
from iga.lm_sleep import Sleeper
from iga.lm_train import train, atomic_save
from iga.lm_conveyor import Vocab

D, LANES, T = 64, 4, 128
SEED = 0
ALPHA_SEED = 2.0
# A64-R2: sparse presses (ratified) + multi-ask curriculum — the
# round-1 chance floor convicted single-exposure teaching
BUTTON_CFG = {"pos": 0.15, "neg": 0.10, "pos_v": 2, "neg_v": 1,
              "asks": 3, "press_p": 0.25}


def seeded_init(path):
    torch.manual_seed(SEED)
    vocab = Vocab()
    m = HybridLM(len(vocab), d=D, max_T=T, store="matrix",
                 keyed="logit", norm_mix=True, aux_trunk=0.2)
    with torch.no_grad():
        for a in m.alpha.values():
            a.fill_(ALPHA_SEED)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    atomic_save({"model": m.state_dict(),
                 "opt": opt.state_dict(), "step": 0}, path)


@torch.no_grad()
def item_recall(m, items, vocab):
    """Store-wiped, fresh-state re-ask of each item — the ask turn
    verbatim, p(color) at the answer slot."""
    m.eval()
    m.store_read_off = True
    rows = []
    for it in items:
        words = ["what", "color", "of", it["obj"], "was", it["name"],
                 "kept", "?", "<eot_human>", "the", it["obj"], "was"]
        x = torch.tensor([vocab.encode(words)])
        logits, _, _ = m(x, m.init_state(1, "cpu"), None)
        m.pop_write_cost()
        m.pop_recon()
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        rows.append({"cls": it["cls"],
                     "p": float(lp[vocab.idx[it["col"]]].exp())})
    m.store_read_off = False
    return rows


def mwu_one_sided(a, b):
    """P(one-sided) for H1: a stochastically > b. Normal approx
    with midranks + continuity correction."""
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return 1.0
    tagged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    r1 = 0.0
    while i < len(tagged):
        j = i
        while j < len(tagged) and tagged[j][0] == tagged[i][0]:
            j += 1
        mid = (i + 1 + j) / 2.0
        for k in range(i, j):
            if tagged[k][1] == 0:
                r1 += mid
        i = j
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mu - 0.5) / max(sigma, 1e-9)
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def replay_coverage(sleeper, items):
    """Fraction of items (per class) whose [plant, ask] intersects a
    replayed window on their lane."""
    cov = {"pos": [0, 0], "none": [0, 0], "neg": [0, 0]}
    for it in items:
        hit = any(r["lane"] == it["lane"] and r["lo"] < it["ask"] + T
                  and it["plant"] < r["hi"]
                  for r in sleeper.replayed)
        cov[it["cls"]][0] += int(hit)
        cov[it["cls"]][1] += 1
    return {c: (round(h / max(n, 1), 4), n) for c, (h, n) in cov.items()}


def main():
    global D
    out = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    if len(sys.argv) > 3:
        D = int(sys.argv[3])   # A64-R3: substrate maturity knob
    os.makedirs(out, exist_ok=True)
    init = os.path.join(out, "par_init.pt")
    seeded_init(init)
    results = {}
    for name, sleep_on in (("sleep", True), ("nosleep", False)):
        print(f"=== {name} ===", flush=True)
        buttons = dict(BUTTON_CFG, log=[])
        prophet = PressProphet(d=D)
        sl = Sleeper(arm="B", every=8, block_chunks=2, seed=1) \
            if sleep_on else None
        m, drive, vocab, *_ = train(
            d=D, lanes=LANES, T=T, steps=steps, seed=SEED,
            device="cpu", arch="hybrid", store="matrix",
            keyed="logit", norm_mix=True, aux_trunk=0.2,
            resume=init, log_every=400, buttons=buttons,
            prophet=prophet, sleep=sl,
            ckpt=os.path.join(out, f"par_{name}.pt"))
        # R2: multi-ask logs one row per ask; roster reuse can rebind
        # a freed (name, obj) to a new color — keep the LAST entry
        seen = {}
        for i in buttons["log"]:
            if i["correct"]:
                seen[(i["name"], i["obj"], i["lane"])] = i
        items = list(seen.values())
        rows = item_recall(m, items, vocab)
        grp = {c: [r["p"] for r in rows if r["cls"] == c]
               for c in ("pos", "none", "neg")}
        mean = {c: (sum(v) / len(v) if v else 0.0)
                for c, v in grp.items()}
        r = {"items": {c: len(v) for c, v in grp.items()},
             # A64-R3 base-faculty gauge: the contrast can only exist
             # on a formed recall organ (R1/R2 floor conviction)
             "recall_ema": {k: round(float(v), 4)
                            for k, v in drive.ema.items()
                            if k.startswith("recall:")},
             "mean_p": {c: round(m_, 5) for c, m_ in mean.items()},
             "p_pos_gt_none": round(mwu_one_sided(grp["pos"],
                                                  grp["none"]), 5),
             "p_none_gt_neg": round(mwu_one_sided(grp["none"],
                                                  grp["neg"]), 5),
             "effect_pos_none": round(mean["pos"] - mean["none"], 5),
             "presses": len(drive.presses),
             "audit": drive.audit(),
             "prophet": prophet.report()}
        if sl is not None:
            r["sleep_audit"] = sl.audit()
            r["replay_coverage"] = replay_coverage(sl, items)
        results[name] = r
        print(json.dumps({name: r}, indent=1, default=str), flush=True)
    with open(os.path.join(out, "parenting_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("\n=== A64-R summary ===")
    for name, r in results.items():
        print(f"{name:8s} mean_p pos {r['mean_p']['pos']:.4f} "
              f"none {r['mean_p']['none']:.4f} "
              f"neg {r['mean_p']['neg']:.4f} | "
              f"p(pos>none) {r['p_pos_gt_none']} | "
              f"effect {r['effect_pos_none']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
