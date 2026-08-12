"""v8.0 autopsy: NAT/PLANTED split, T=2048 bins, lesions, A50 scoring."""
import sys, os
sys.path.insert(0, "/Users/lukehamond/Projects/project")
os.chdir("/Users/lukehamond/Projects/project")

import torch
from iga.lm_hybrid import HybridLM
from iga.lm_data_ultrachat import load_tokenizer, UltraConveyor

S = ("/private/tmp/claude-501/-Users-lukehamond-Projects-project/"
     "6a660d03-4ba8-4edb-8b91-6c006b236602/scratchpad")
SHARD = os.path.join(S, "mix_v80_eval")
tok = load_tokenizer(os.path.join(SHARD, "tokenizer.json"))
T = 2048


@torch.no_grad()
def table(model, warmup=4, n_chunks=140):
    conv = UltraConveyor(SHARD, n_lanes=2)
    st = model.init_state(2, "cpu")
    seg = conv.seg
    stats = {}
    for ci in range(warmup + n_chunks):
        x, y, events = conv.chunk(T)
        logits, st, _ = model(x, st, None)
        if ci < warmup:
            continue
        logp = torch.log_softmax(logits.float(), dim=-1)
        for lane, evs in enumerate(events):
            lo = lane * seg
            for p, kind, d in evs:
                if kind != "probe" or p <= 0 or \
                        not d.get("answerable", True):
                    continue
                pos, plant = d["pos"], d["pos"] - d["gap"]
                if d.get("nat"):
                    g = d["gap"]
                    key = ("nat<2k" if g < 2048 else
                           "nat2-8k" if g < 8192 else "nat>8k")
                else:
                    if (pos - lo) // T == (plant - lo) // T:
                        key = "pl-same"
                    elif d["gap"] <= T:
                        key = "pl-straddle"
                    else:
                        key = "pl-cross"
                s = stats.setdefault(key, [0.0, 0, 0])
                s[0] += float(logp[lane, p - 1, d["answer"]].exp())
                s[1] += int(int(logits[lane, p - 1].argmax())
                            == d["answer"])
                s[2] += 1
    return {k: (round(v[0] / v[2], 3), round(v[1] / v[2], 2), v[2])
            for k, v in sorted(stats.items())}


for name in ("v80_best", "v80_final"):
    ck = torch.load(os.path.join(S, name + ".pt"), map_location="cpu",
                    weights_only=False)
    m = HybridLM(tok.get_vocab_size(), d=512, max_T=T, store="matrix",
                 use_xl=False)
    m.load_state_dict(ck["model"])
    m.eval()
    print(f"== {name} step {ck.get('step')}")
    print("gates:", {k: round(float(torch.sigmoid(v)), 4)
                     for k, v in m.read_gate.items()})
    print("full        :", table(m))
    for band in (3, 4, 5):
        m.lesioned = {band}
        print(f"lesion b{band}   :", table(m))
    m.lesioned = {3, 4, 5}
    print("lesion ALL  :", table(m))
    m.lesioned = set()
