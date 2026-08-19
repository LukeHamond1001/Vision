"""A69-R1 — the biography ablation, the gate everything downstream
rides on: does training on ONE LIFE (facts recurring across days at
band-timescale gaps) teach cross-session recall that day-locked
training cannot?

Two debug models, identical but for life["cross"]:
  BIO   cross=True  — recurrence across days, long bin 16k
  CTRL  cross=False — same day structure, same exchange shapes,
        pending flushed at day close (nothing crosses a boundary)

Both evaluate on the SAME held-out biography stream (eval seed),
probe mass and color-argmax accuracy binned by gap, under lesions:
  full   — the whole machine
  nostore— store reads off (recall carried by bands/trunk only)
  fresh  — state reset every chunk (the no-carry floor)

Sleep and corrections are OFF in both arms: the data regime is the
only variable. Usage: python3 scripts/bio_ablation.py [steps]
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_conveyor import Conveyor, Vocab, splits   # noqa: E402
from iga.lm_gen import COLORS                          # noqa: E402
from iga.lm_train import train                         # noqa: E402

D, LANES, T = 64, 4, 256
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
if len(sys.argv) > 2:
    D = int(sys.argv[2])
ARMS = (sys.argv[3].split(",") if len(sys.argv) > 3
        else ["bio", "ctrl"])
# A69-R2: the R1 null — in-ctx recall AT CHANCE at d=64/3k steps
# (the binder never formed; every lesion identical) — makes the
# precondition explicit: the ablation only means something once
# in-ctx accuracy clears ~2x chance (>=0.17). Sweep d/steps first.
SEED = 7
SESS = (20, 40)
BUTTONS = {"pos": .3, "neg": .1, "pos_v": 2, "neg_v": 1}
LIFE_BIO = {"sess": SESS, "cross": True, "long_gap": 16000,
            "long_w": 3}
LIFE_CTRL = {"sess": SESS, "cross": False}
EVAL_CHUNKS = 500
BINS = [(1, 256, "in-ctx"), (257, 2048, "short"),
        (2049, 8192, "band3"), (8193, 40000, "band4+")]


def train_arm(life, tag):
    t0 = time.time()
    model, drive, vocab, ce0, ce1 = train(
        d=D, lanes=LANES, T=T, steps=STEPS, seed=SEED, device="cpu",
        arch="hybrid", store="matrix", keyed="logit", norm_mix=True,
        aux_trunk=0.2, use_xl=False, gate_init=-2.0,
        log_every=max(STEPS // 4, 1), buttons=dict(BUTTONS),
        life=dict(life))
    print(f"[{tag}] trained {STEPS} steps  ce {ce0:.3f}->{ce1:.3f} "
          f" {time.time() - t0:.0f}s", flush=True)
    return model, vocab, {"ce_first": ce0, "ce_last": ce1}


@torch.no_grad()
def score(model, vocab, lesion):
    conv = Conveyor(Vocab(), n_lanes=LANES,
                    seed=splits(SEED)["eval"],
                    buttons=dict(BUTTONS), life=dict(LIFE_BIO))
    color_ids = [conv.vocab.idx[c] for c in COLORS]
    model.eval()
    model.store_read_off = (lesion == "nostore")
    st = model.init_state(LANES, "cpu")
    acc = {name: [0.0, 0, 0] for _, _, name in BINS}  # mass, hits, n
    for _ in range(EVAL_CHUNKS):
        x, _, events = conv.chunk(T)
        if lesion == "fresh":
            st = model.init_state(LANES, "cpu")
        logits, st, _ = model(x, st, None)
        model.pop_write_cost()
        model.pop_recon()
        st = model.detach_state(st)
        lp = torch.log_softmax(logits.float(), -1)
        for lane, evs in enumerate(events):
            for p, kind, d in evs:
                if kind != "probe" or p <= 0:
                    continue
                for lo, hi, name in BINS:
                    if lo <= d["gap"] <= hi:
                        row = lp[lane, p - 1]
                        acc[name][0] += float(row[d["answer"]].exp())
                        best = max(color_ids, key=lambda i:
                                   float(row[i]))
                        acc[name][1] += int(best == d["answer"])
                        acc[name][2] += 1
                        break
    model.store_read_off = False
    return {name: {"mean_p": round(m / n, 4) if n else None,
                   "acc": round(h / n, 3) if n else None, "n": n}
            for name, (m, h, n) in acc.items()}


def main():
    out = {"steps": STEPS, "d": D, "T": T, "lanes": LANES,
           "arms": {}}
    for tag, life in (("bio", LIFE_BIO), ("ctrl", LIFE_CTRL)):
        if tag not in ARMS:
            continue
        model, vocab, meta = train_arm(life, tag)
        arm = {"train": meta, "eval": {}}
        for lesion in ("full", "nostore", "fresh"):
            t0 = time.time()
            arm["eval"][lesion] = score(model, vocab, lesion)
            print(f"[{tag}/{lesion}] "
                  f"{json.dumps(arm['eval'][lesion])} "
                  f" {time.time() - t0:.0f}s", flush=True)
        out["arms"][tag] = arm
        torch.save(model.state_dict(),
                   f"/Users/lukehamond/Projects/project/data/"
                   f"a69_{tag}.pt")
    path = ("/Users/lukehamond/Projects/project/results/evidence/"
            "a69_bio_ablation.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
