"""trunk_alone.py — the trunk-alone gauge on a saved diary body.

Each cue is typed as the ear on a new line, one symbol per tick, and the
mouth then continues on its own ticks: once with memory set aside (the
trunk alone: what the weights know) and once with memory on (the day's
store, as it lives in the saved state). Greedy, no stamina, no faces.
Read-only: the body file is never written. Run it on a scratch copy.

  python3 scripts/trunk_alone.py BODY.pt data/tok_char.json --dev cpu \
      --cues "dog will |sad |why dog up? |what? " --n 14
"""
import argparse
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.scan_infer import load_scan, _to_dev, _lane0  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tok")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--cues", default="dog will |sad |why dog up? |dog go in then |happy |I can |what? |where ball")
    ap.add_argument("--store-boost", type=float, default=4.0)
    ap.add_argument("--store-read-beta", type=float, default=1.0)
    ap.add_argument("--sil-decay", type=float, default=None)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    m, state = load_scan(a.ckpt, tok, a.dev)
    m.eval()
    sil = tok.token_to_id("<pad>")
    nl = tok.token_to_id("\n")
    m.kc_break_ids = {int(nl)} if nl is not None else set()
    m.sil_id = int(sil)
    if a.sil_decay:
        m.kc_sil_decay = float(a.sil_decay)
    m.store_boost = float(a.store_boost)
    m.read_beta = float(a.store_read_beta)
    src = state.get("st_live") or state.get("st")
    base = _to_dev(src if state.get("st_live") else _lane0(src), a.dev) if src is not None \
        else m.init_state(1, a.dev)
    who0 = torch.tensor([[0]], device=a.dev)
    who2 = torch.tensor([[2]], device=a.dev)
    print(f"[body] {sum(p.numel() for p in m.parameters())/1e6:.0f}M on {a.dev}; store {'saved' if src is not None else 'empty'}",
          file=sys.stderr)

    def run(cue, mem_on):
        st = m.lane_state(base, 0) if isinstance(base, dict) else m.init_state(1, a.dev)
        m.reset_bag(st)
        st["mouth_floor"] = False
        m.store_read_off = not mem_on
        m.store_write_off = True
        out = []
        with torch.no_grad():
            for i in tok.encode("\n" + cue).ids:          # the ear, one symbol per tick, the mouth silent under it
                _, st, _ = m(torch.tensor([[i]], device=a.dev), st, who=who0)
                _, st, _ = m(torch.tensor([[sil]], device=a.dev), st, who=who0)
            for _ in range(a.n):
                lg, st, _ = m(torch.tensor([[sil]], device=a.dev), st, who=who0)   # the ear is quiet
                v = lg[0, -1].float().clone()
                v[[i for i in range(11) if i != sil]] = float("-inf")
                nxt = int(v.argmax())
                out.append(nxt)
                _, st, _ = m(torch.tensor([[nxt]], device=a.dev), st, who=(who2 if nxt != sil else who0))
        return "".join("·" if i == sil else tok.decode([i]) for i in out)

    for cue in a.cues.split("|"):
        alone = run(cue, mem_on=False)
        withm = run(cue, mem_on=True)
        print(f"{cue!r:>20}  trunk alone: {alone!r:>18}   with memory: {withm!r}")


if __name__ == "__main__":
    main()
