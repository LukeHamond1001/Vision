"""v5.0 eval — recall-at-distance, the talking eval, the lesion, the panel.

Held-out conveyor (eval seed; the instrument discipline). Reports:
  recall table   — per scene type x gap bin: mean p(answer) and top-1
                   accuracy at annotated probes (generator ground truth).
  lesion         — same table with slow bands zeroed: what dies was
                   band-carried (the control, card A2).
  talk           — greedy continuation from a held-out prompt, panel
                   printed beside it (persistence, not eloquence).
"""

import torch

from .lm_conveyor import Vocab, Conveyor, splits
from .lm_drive import gap_bin, GAP_BINS


@torch.no_grad()
def recall_table(model, vocab, n_chunks=40, T=512, lanes=4, seed=0,
                 lesion=()):
    model.lesioned = set(lesion)
    conveyor = Conveyor(vocab, n_lanes=lanes, seed=splits(seed)["eval"])
    st = model.init_state(lanes, next(model.parameters()).device)
    stats = {}
    for _ in range(n_chunks):
        x, y, events = conveyor.chunk(T)
        x = x.to(next(model.parameters()).device)
        scene_starts = {}
        for lane, evs in enumerate(events):
            for p, k, d in evs:
                if k == "scene_start":
                    scene_starts.setdefault(p, []).append(lane)
        logits, st, _ = model(x, st, scene_starts)
        logp = torch.log_softmax(logits, dim=-1)
        for lane, evs in enumerate(events):
            for p, kind, d in sorted(evs):
                if kind == "probe" and p > 0:
                    key = (d["type"], gap_bin(d["gap"]))
                    prob = float(logp[lane, p - 1, d["answer"]].exp())
                    top1 = int(logits[lane, p - 1].argmax()) == d["answer"]
                    s = stats.setdefault(key, [0.0, 0, 0])
                    s[0] += prob
                    s[1] += int(top1)
                    s[2] += 1
    model.lesioned = set()
    rows = []
    for (stype, b), (psum, hits, n) in sorted(stats.items()):
        lo, hi = GAP_BINS[b]
        rows.append({"type": stype, "bin": b, "gap": f"{lo}-{hi}",
                     "p": psum / n, "top1": hits / n, "n": n})
    return rows


def print_table(rows, title):
    print(f"\n== {title} ==")
    for r in rows:
        print(f"  {r['type']:8s} gap {r['gap']:>14s}  p(ans) {r['p']:.3f}  "
              f"top1 {r['top1']:.2f}  n={r['n']}")


@torch.no_grad()
def talk(model, vocab, prompt_words, n_new=40, temperature=0.0):
    device = next(model.parameters()).device
    ids = vocab.encode(["<scene>"] + prompt_words)
    st = model.init_state(1, device)
    x = torch.tensor([ids], device=device)
    logits, st, _ = model(x, st, {0: [0]})
    out = []
    tok = logits[0, -1].argmax() if temperature == 0 else \
        torch.multinomial(torch.softmax(logits[0, -1] / temperature, -1), 1)[0]
    for _ in range(n_new):
        out.append(int(tok))
        logits, st, _ = model(tok.view(1, 1), st, None)
        tok = logits[0, -1].argmax() if temperature == 0 else \
            torch.multinomial(torch.softmax(logits[0, -1] / temperature, -1), 1)[0]
    return " ".join(vocab.decode(out))


def full_eval(model, vocab, seed=0, lanes=4, n_chunks=40, T=512):
    base = recall_table(model, vocab, n_chunks, T, lanes, seed)
    print_table(base, "recall at distance (full model)")
    les = recall_table(model, vocab, n_chunks, T, lanes, seed, lesion=(3, 4, 5))
    print_table(les, "recall at distance (bands 4-6 lesioned)")
    print("\n== talk (greedy continuation) ==")
    print(talk(model, vocab, "mira kept a silver key in the cellar .".split()))
    return base, les
