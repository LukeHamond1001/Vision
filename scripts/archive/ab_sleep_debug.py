"""A62 debug A/B — LOCAL (user directive: next step local; CPU
training at debug tier is minutes on the Mac, no pod needed).

Three arms, same seed, same wake stream: control (no sleep),
ARM A (paid-span replay), ARM B (store->trunk distillation), at
the top of the pre-registered dose ladder (1:4 — strongest
mechanism signal; the ladder proper runs at v-scale).

Deviation, documented: store alphas are seeded to 2.0 in ALL arms
via an init checkpoint. A fresh init's alpha=0 makes ARM B a
structural no-op (KL==0 — law-test finding, 2026-08-19); at
v-scale the arms resume from v94-best whose alphas are live
(5.61/3.89/3.00), so a live debug store is the faithful
miniature. Alphas stay TRAINABLE in wake (the certified economy
prices them); sleep freezes them (L3).

Readout (held-out weaver stream, eval seed, fresh state): CE and
probe recall, full vs store-wiped (store_read_off). The
consolidation claim is exactly that the WIPED rows move toward
the full ones: prediction B > A > control on wiped-row recovery.

Usage: python3 scripts/ab_sleep_debug.py <outdir> [steps]
"""

import json
import os
import sys

import torch

from iga.lm_conveyor import Vocab, Conveyor, splits
from iga.lm_drive import Drive          # noqa: F401 (readability)
from iga.lm_hybrid import HybridLM
from iga.lm_sleep import Sleeper
from iga.lm_train import train, atomic_save

D, LANES, T = 64, 4, 128
SEED = 0
ALPHA_SEED = 2.0
EVAL_CHUNKS = 40


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
def readout(m):
    m.eval()
    vocab = Vocab()
    rows = {}
    for wiped in (False, True):
        m.store_read_off = wiped
        conv = Conveyor(vocab, n_lanes=2,
                        seed=splits(SEED)["eval"])  # same text both
        st = m.init_state(2, "cpu")
        ces, prec, n = [], 0.0, 0
        for _ in range(EVAL_CHUNKS):
            x, y, events = conv.chunk(T)
            logits, st, _ = m(x, st, None)
            m.pop_write_cost()
            m.pop_recon()
            st = m.detach_state(st)
            ces.append(float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, m.vocab_size), y.reshape(-1))))
            lp = torch.log_softmax(logits, -1)
            for lane, evs in enumerate(events):
                for p, kind, dd in evs:
                    if kind == "probe" and p > 0:
                        prec += float(
                            lp[lane, p - 1, dd["answer"]].exp())
                        n += 1
        rows["wiped" if wiped else "full"] = {
            "ce": round(sum(ces) / len(ces), 4),
            "probe": round(prec / max(n, 1), 4), "n_probes": n}
    m.store_read_off = False
    return rows


def main():
    out = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    os.makedirs(out, exist_ok=True)
    init = os.path.join(out, "ab_init.pt")
    seeded_init(init)
    arms = [
        ("control", None),
        ("armA", Sleeper(arm="A", every=8, block_chunks=2, seed=1)),
        ("armB", Sleeper(arm="B", every=8, block_chunks=2, seed=1)),
    ]
    results = {}
    for name, sl in arms:
        print(f"=== {name} ===", flush=True)
        m, drive, *_ = train(
            d=D, lanes=LANES, T=T, steps=steps, seed=SEED,
            device="cpu", arch="hybrid", store="matrix",
            keyed="logit", norm_mix=True, aux_trunk=0.2,
            resume=init, log_every=200, sleep=sl,
            ckpt=os.path.join(out, f"ab_{name}.pt"))
        r = {"readout": readout(m),
             "alphas": {k: round(float(a), 3)
                        for k, a in m.alpha.items()},
             "ledger": len(drive.ledger)}
        if sl is not None:
            r["sleep"] = sl.audit()
            r["sleep_blocks"] = sl.stats[-3:]
            assert sl.audit()["only_paid"], f"{name}: L1 violated"
        results[name] = r
        print(json.dumps({name: r}, indent=1), flush=True)
    with open(os.path.join(out, "ab_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\n=== A62 debug A/B summary ===")
    for name, r in results.items():
        ro = r["readout"]
        print(f"{name:8s} full ce {ro['full']['ce']:.4f} "
              f"probe {ro['full']['probe']:.4f} | wiped ce "
              f"{ro['wiped']['ce']:.4f} probe "
              f"{ro['wiped']['probe']:.4f} | alphas "
              f"{r['alphas']}", flush=True)


if __name__ == "__main__":
    main()
