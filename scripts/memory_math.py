"""Memory math, measured: the 78M raised life's weights (v94sp, step 296k)
on UltraChat held-out text in its own tokenizer.

(1) the store's learned key mix (qmix softmax, entropy), tok_u by word
    class, alpha, read gates — the "able" side;
(2) CE by position-in-chunk under base / thread off (mem_off) / stores
    off / both off, state carried across chunks — the "necessary" side:
    how much does the cortex lose at a chunk boundary, and which organ
    covers it.
"""
import json, sys, time, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer
from iga.lm_hybrid import HybridLM
from iga.lm_gen import NAMES, OBJECTS, COLORS

torch.set_num_threads(8)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 2048
N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 24

tok = Tokenizer.from_file(f"{ROOT}/data/life/tokenizer_press.json")
blob = torch.load(f"{ROOT}/data/life/v94sp.pt", map_location="cpu", weights_only=False)
m = HybridLM(tok.get_vocab_size(), d=512, max_T=T, store="matrix", keyed="logit",
             norm_mix=True, aux_trunk=0.2, use_xl=False, gate_init=-2.0)
m.load_state_dict(blob["model"]); m.eval()

# ---- (1) parameters of the store's key and read path
out = {"step": blob.get("step")}
w = torch.softmax(m.qmix.detach().float(), 0)
top = torch.topk(w, 5)
out["qmix_top"] = [[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)]
out["qmix_entropy"] = round(float(-(w * (w + 1e-12).log()).sum()), 4)
tu = m.tok_u.detach().float()
def ids_of(words):
    r = []
    for wd in words:
        for form in (" " + wd, wd):
            i = tok.token_to_id(form)
            if i is not None: r.append(i); break
    return r
out["tok_u"] = {"mean": round(float(tu.mean()), 3), "std": round(float(tu.std()), 3),
                "names": round(float(tu[ids_of(NAMES)].mean()), 3),
                "objects": round(float(tu[ids_of(OBJECTS)].mean()), 3),
                "colors": round(float(tu[ids_of(COLORS)].mean()), 3),
                "n_found": [len(ids_of(NAMES)), len(ids_of(OBJECTS)), len(ids_of(COLORS))]}
out["alpha"] = {k: round(float(v), 3) for k, v in m.alpha.items()}
out["read_gate"] = {k: round(float(torch.sigmoid(v)), 3) for k, v in m.read_gate.items()}
out["store_beta"] = {k: round(float(torch.sigmoid(s.beta)), 3) for k, s in m.stores.items()}
out["store_decay"] = {k: s.decay for k, s in m.stores.items()}
out["cell_gate_bias"] = {k: round(float(c.z.bias.mean()), 3) for k, c in m.cells.items()}
print("PARAMS", json.dumps(out), flush=True)

# ---- (2) a stream of held-out dialogue in the training grammar
eot_h, eot_m = tok.token_to_id("<eot_human>"), tok.token_to_id("<eot_model>")
ids = []
with open(f"{ROOT}/data/ultrachat_heldout.jsonl") as f:
    for line in f:
        turns = json.loads(line)["data"]
        for i, t in enumerate(turns):
            ids += tok.encode(t).ids + [eot_h if i % 2 == 0 else eot_m]
        if len(ids) >= (N_CHUNKS + 1) * T + 1:
            break
ids = ids[: (N_CHUNKS + 1) * T + 1]
x_all = torch.tensor(ids, dtype=torch.long)
print(f"stream {len(ids)} tokens, {N_CHUNKS} chunks", flush=True)

BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 2048)]
CONDS = ["base", "thread_off", "store_off", "both_off"]

def run(cond):
    st = m.init_state(1, "cpu")
    m.lesioned, m.store_read_off, m.mem_off = set(), False, False
    if cond == "thread_off": m.mem_off = True
    elif cond == "store_off": m.store_read_off = True
    elif cond == "both_off": m.lesioned = set(m.bands)
    sums = torch.zeros(T); cnt = 0
    with torch.no_grad():
        for c in range(N_CHUNKS):
            x = x_all[c * T:(c + 1) * T].unsqueeze(0)
            y = x_all[c * T + 1:(c + 1) * T + 1]
            lg, st, _ = m(x, st, None)
            m.pop_write_cost(); m.pop_recon()
            st = m.detach_state(st)
            ce = torch.nn.functional.cross_entropy(lg[0].float(), y, reduction="none")
            sums += ce; cnt += 1
    m.lesioned, m.store_read_off, m.mem_off = set(), False, False
    per = sums / cnt
    return {f"{a}-{b}": round(float(per[a:b].mean()), 4) for a, b in BUCKETS} | {"all": round(float(per.mean()), 4)}

t0 = time.time()
res = {}
for cond in CONDS:
    res[cond] = run(cond)
    print(cond, json.dumps(res[cond]), f"{time.time()-t0:.0f}s", flush=True)
print("DELTAS (cond - base), nats:")
for cond in CONDS[1:]:
    print(" ", cond, {k: round(res[cond][k] - res["base"][k], 4) for k in res["base"]})
json.dump({"params": out, "ce_by_position": res, "n_chunks": N_CHUNKS},
          open(f"{ROOT}/results/evidence/memory_math_v94sp.json", "w"), indent=1)
