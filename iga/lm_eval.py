"""v5.0 eval — recall-at-distance, the talking eval, the lesion, the panel.

Held-out conveyor (eval seed; the instrument discipline). The stream
is one endless dialogue, so evaluation is just more conversation:
  recall table — per gap bin: mean p(answer) and top-1 accuracy at
                 annotated probes (generator ground truth).
  lesion       — same table with slow bands zeroed: what dies was
                 band-carried (the control, card A2).
  talk         — the agent's turn generated after a human turn ends;
                 persistence, not eloquence, is what is being read.
"""

import torch

from .lm_conveyor import Vocab, Conveyor, splits
from .lm_drive import gap_bin, GAP_BINS


@torch.no_grad()
def recall_table(model, vocab, n_chunks=40, T=512, lanes=4, seed=0,
                 lesion=(), data=None):
    model.lesioned = set(lesion)
    if data:  # held-out real shard (A12)
        from .lm_data_ultrachat import UltraConveyor
        conveyor = UltraConveyor(data, n_lanes=lanes)
    else:
        conveyor = Conveyor(vocab, n_lanes=lanes, seed=splits(seed)["eval"])
    device = next(model.parameters()).device
    st = model.init_state(lanes, device)
    stats = {}
    for _ in range(n_chunks):
        x, y, events = conveyor.chunk(T)
        x = x.to(device)
        logits, st, _ = model(x, st, None)
        logp = torch.log_softmax(logits, dim=-1)
        for lane, evs in enumerate(events):
            for p, kind, d in evs:
                if kind == "probe" and p > 0:
                    b = gap_bin(d["gap"])
                    prob = float(logp[lane, p - 1, d["answer"]].exp())
                    top1 = int(logits[lane, p - 1].argmax()) == d["answer"]
                    s = stats.setdefault(b, [0.0, 0, 0])
                    s[0] += prob
                    s[1] += int(top1)
                    s[2] += 1
    model.lesioned = set()
    rows = []
    for b, (psum, hits, n) in sorted(stats.items()):
        lo, hi = GAP_BINS[b]
        rows.append({"bin": b, "gap": f"{lo}-{hi}",
                     "p": psum / n, "top1": hits / n, "n": n})
    return rows


def print_table(rows, title):
    print(f"\n== {title} ==")
    for r in rows:
        print(f"  gap {r['gap']:>14s}  p(ans) {r['p']:.3f}  "
              f"top1 {r['top1']:.2f}  n={r['n']}")


@torch.no_grad()
def talk(model, vocab, human_words, n_new=30, temperature=0.0):
    """One human turn in, the agent's turn out. vocab may be the
    weaver Vocab or an HF tokenizer (real shards)."""
    device = next(model.parameters()).device
    if hasattr(vocab, "token_to_id"):  # HF tokenizer
        ids = vocab.encode(" ".join(human_words)).ids \
            + [vocab.token_to_id("<eot_human>")]
    else:
        ids = vocab.encode(human_words + ["<eot_human>"])
    st = model.init_state(1, device)
    logits, st, _ = model(torch.tensor([ids], device=device), st, None)
    out = []
    tok = logits[0, -1].argmax() if temperature == 0 else \
        torch.multinomial(torch.softmax(logits[0, -1] / temperature, -1), 1)[0]
    if hasattr(vocab, "token_to_id"):
        stop = vocab.token_to_id("<eot_model>")
    else:
        stop = vocab.idx["<eot_model>"]
    for _ in range(n_new):
        out.append(int(tok))
        if int(tok) == stop:
            break
        logits, st, _ = model(tok.view(1, 1), st, None)
        tok = logits[0, -1].argmax() if temperature == 0 else \
            torch.multinomial(torch.softmax(logits[0, -1] / temperature, -1), 1)[0]
    if hasattr(vocab, "token_to_id"):
        return vocab.decode(out, skip_special_tokens=False)
    return " ".join(vocab.decode(out))


def full_eval(model, vocab, seed=0, lanes=4, n_chunks=40, T=512, data=None):
    base = recall_table(model, vocab, n_chunks, T, lanes, seed, data=data)
    print_table(base, "recall at distance (full model)")
    les = recall_table(model, vocab, n_chunks, T, lanes, seed,
                       lesion=(3, 4, 5), data=data)
    print_table(les, "recall at distance (bands 4-6 lesioned)")
    print("\n== talk (one agent turn, greedy) ==")
    print(talk(model, vocab,
               "by the way mira kept a silver key in the cellar .".split()))
    return base, les


def main():
    """Registered-run evaluation: load a checkpoint, run the tables,
    the lesion, and the talk on the held-out real shard."""
    import argparse
    import os
    from .lm_bands import BandLM
    from .lm_data_ultrachat import load_tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--talk", default="tick")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=60)
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tok = load_tokenizer(os.path.join(a.data, "tokenizer.json"))
    model = BandLM(tok.get_vocab_size(), d=a.d, talk=a.talk).to(a.device)
    state = torch.load(a.ckpt, map_location=a.device)
    model.load_state_dict(state["model"])
    print(f"loaded {a.ckpt} (step {state.get('step','?')}), "
          f"{model.n_params():,} params")
    full_eval(model, tok, lanes=a.lanes, n_chunks=a.chunks, data=a.data)


if __name__ == "__main__":
    main()
