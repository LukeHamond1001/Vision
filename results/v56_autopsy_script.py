"""v5.6 autopsy: why did training binding (b0 0.72) not generalize?

Discriminators:
  1. straddle split — in-window probes whose plant sits in the SAME
     chunk (pure attention's job) vs the PREVIOUS chunk (matrix-
     reachable). If same-chunk ~ floor: induction never formed and
     training binding was matrix-borne (crowding-out story).
  2. pathway lesion — matrix content zeroed vs everything zeroed.
  3. warmup sweep — score after 0 vs 100 unscored warm chunks.
  4. v5.5 (vector) contrast on the same split.
"""
import sys, os
sys.path.insert(0, "/Users/lukehamond/Projects/project")
os.chdir("/Users/lukehamond/Projects/project")

import torch
from iga.lm_hybrid import HybridLM
from iga.lm_data_ultrachat import load_tokenizer, UltraConveyor

S = ("/private/tmp/claude-501/-Users-lukehamond-Projects-project/"
     "f1855516-28a1-49ca-8616-f110bf224fbc/scratchpad")
SHARD = os.path.join(S, "uc_v56_eval")
tok = load_tokenizer(os.path.join(SHARD, "tokenizer.json"))


@torch.no_grad()
def table(model, warmup=0, zero_M=False, n_chunks=200):
    conveyor = UltraConveyor(SHARD, n_lanes=4)
    st = model.init_state(4, "cpu")
    seg = conveyor.seg
    stats = {}
    for ci in range(warmup + n_chunks):
        x, y, events = conveyor.chunk(512)
        if zero_M and "M" in st:
            st["M"] = {k: torch.zeros_like(v)
                       for k, v in st["M"].items()}
        logits, st, _ = model(x, st, None)
        if ci < warmup:
            continue
        logp = torch.log_softmax(logits, dim=-1)
        for lane, evs in enumerate(events):
            lo = lane * seg
            for p, kind, d in evs:
                if kind != "probe" or p <= 0 or \
                        not d.get("answerable", True):
                    continue
                pos = d["pos"]
                plant = pos - d["gap"]
                if d["gap"] < 256:
                    same = (pos - lo) // 512 == (plant - lo) // 512
                    key = "inwin-same" if same else "inwin-straddle"
                elif d["gap"] < 2048:
                    key = "b1"
                else:
                    key = "b2+"
                prob = float(logp[lane, p - 1, d["answer"]].exp())
                top1 = int(logits[lane, p - 1].argmax()) == d["answer"]
                s = stats.setdefault(key, [0.0, 0, 0])
                s[0] += prob
                s[1] += int(top1)
                s[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2], v[2])
            for k, v in sorted(stats.items())}


def show(name, t):
    print(f"-- {name}")
    for k, (p, t1, n) in t.items():
        print(f"   {k:15s} p {p:.3f} top1 {t1:.2f} n={n}")


m56 = HybridLM(tok.get_vocab_size(), d=128, max_T=512, store="matrix")
m56.load_state_dict(torch.load(os.path.join(S, "v56_final.pt"),
                               map_location="cpu",
                               weights_only=False)["model"])
m56.eval()
show("v5.6 matrix FULL", table(m56))
show("v5.6 matrix M-ZEROED (mem tokens alive)", table(m56, zero_M=True))
m56.lesioned = {3, 4, 5}
show("v5.6 matrix ALL LESIONED", table(m56))
m56.lesioned = set()
show("v5.6 matrix FULL warmup=100", table(m56, warmup=100))

m55 = HybridLM(tok.get_vocab_size(), d=128, max_T=512, store="vector")
m55.load_state_dict(torch.load(os.path.join(S, "v55_final.pt"),
                               map_location="cpu",
                               weights_only=False)["model"])
m55.eval()
show("v5.5 vector FULL (contrast)", table(m55))
