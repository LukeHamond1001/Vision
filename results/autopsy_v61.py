"""v6.0 autopsy: gates + fine splits (same/straddle/3.2k/12.8k) + lesions."""
import sys, os
sys.path.insert(0, "/Users/lukehamond/Projects/project")
os.chdir("/Users/lukehamond/Projects/project")

import torch
from iga.lm_hybrid import HybridLM
from iga.lm_data_ultrachat import load_tokenizer, UltraConveyor

S = ("/private/tmp/claude-501/-Users-lukehamond-Projects-project/"
     "6a660d03-4ba8-4edb-8b91-6c006b236602/scratchpad")
SHARD = os.path.join(S, "uc_v61_eval")
tok = load_tokenizer(os.path.join(SHARD, "tokenizer.json"))


@torch.no_grad()
def table(model, warmup=12, n_chunks=200):
    conv = UltraConveyor(SHARD, n_lanes=4)
    st = model.init_state(4, "cpu")
    seg = conv.seg
    stats = {}
    for ci in range(warmup + n_chunks):
        x, y, events = conv.chunk(512)
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
                pos, plant = d["pos"], d["pos"] - d["gap"]
                if (pos - lo) // 512 == (plant - lo) // 512:
                    key = "same"
                elif d["gap"] <= 512:
                    key = "straddle"
                elif d["gap"] <= 6400:
                    key = "cross3k"
                else:
                    key = "cross13k"
                s = stats.setdefault(key, [0.0, 0, 0])
                s[0] += float(logp[lane, p - 1, d["answer"]].exp())
                s[1] += int(int(logits[lane, p - 1].argmax())
                            == d["answer"])
                s[2] += 1
    return {k: (round(v[0] / v[2], 3), round(v[1] / v[2], 2), v[2])
            for k, v in sorted(stats.items())}


ck = torch.load(os.path.join(S, "v61_final.pt"), map_location="cpu",
                weights_only=False)
m = HybridLM(tok.get_vocab_size(), d=128, max_T=512, store="matrix",
             use_xl=False)
m.load_state_dict(ck["model"])
m.eval()
print("step:", ck.get("step"))
print("read gates:", {k: round(float(torch.sigmoid(v)), 4)
                      for k, v in m.read_gate.items()})
print("beta:", {k: round(float(torch.sigmoid(b.beta)), 3)
                for k, b in m.mats.items()})
print("full           :", table(m))
for band in (3, 4, 5):
    m.lesioned = {band}
    print(f"lesion band {band} :", table(m))
m.lesioned = {3, 4, 5}
print("lesion ALL     :", table(m))
m.lesioned = set()
