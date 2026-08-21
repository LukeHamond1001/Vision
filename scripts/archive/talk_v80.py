"""Talk to v8.0 (banked best) in your terminal — architecture-native.

The session lives in the BAND STATE: each completed turn is committed
through the machine once (bands tick, matrices write), and generation
attends only the recent window — older context is carried by the
organs, exactly as designed. Generation itself uses a throwaway copy
of the state so the clocks don't fast-forward while sampling.

Usage:
  python3 scripts/talk_v80.py [--ckpt PATH] [--temp 0.8]
Commands inside: /reset  /temp X  /state  /quit
"""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # scripts/archive/ -> repo root

import torch

from iga.lm_hybrid import HybridLM
from iga.lm_data_ultrachat import load_tokenizer

S = ("/private/tmp/claude-501/-Users-lukehamond-Projects-project/"
     "6a660d03-4ba8-4edb-8b91-6c006b236602/scratchpad")
DEF_CKPT = os.path.join(S, "v80_best.pt")
DEF_TOK = os.path.join(S, "mix_v80_eval", "tokenizer.json")
WINDOW = 512          # attention's live window; bands carry the rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--tok", default=DEF_TOK)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=60)
    a = ap.parse_args()

    tok = load_tokenizer(a.tok)
    eot_h = tok.token_to_id("<eot_human>")
    eot_m = tok.token_to_id("<eot_model>")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = HybridLM(tok.get_vocab_size(), d=512, max_T=2048,
                 store="matrix", use_xl=False)
    m.load_state_dict(ck["model"])
    m.eval()
    torch.set_grad_enabled(False)
    print(f"v8.0 banked best (step {ck.get('step')}) loaded. "
          f"temp={a.temp}. /reset /temp /state /quit")

    st = m.init_state(1, "cpu")
    buf = []              # committed token history (for the window)
    temp = a.temp

    def commit(ids):
        nonlocal st
        for i in range(0, len(ids), WINDOW):
            piece = ids[i:i + WINDOW]
            x = torch.tensor([piece])
            _, st, _ = m(x, st, None)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            st = m.init_state(1, "cpu")
            buf = []
            print("(session state cleared)")
            continue
        if user.startswith("/temp"):
            try:
                temp = float(user.split()[1])
                print(f"(temp={temp})")
            except (IndexError, ValueError):
                print("(usage: /temp 0.8)")
            continue
        if user == "/state":
            for k in m.bands:
                hn = float(st["h"][k].norm())
                mn = float(st["M"][k].norm())
                print(f"  band {k}: vector |h|={hn:.2f}  "
                      f"matrix |M|={mn:.2f}")
            print(f"  committed tokens: {len(buf)}")
            continue

        turn = tok.encode(user).ids + [eot_h]
        commit(turn)
        buf.extend(turn)

        out = []
        gen_st = copy.deepcopy(st)
        ctx = list(buf[-WINDOW:])
        print("v80> ", end="", flush=True)
        for _ in range(a.max_new):
            x = torch.tensor([ctx[-WINDOW:]])
            logits, _, _ = m(x, copy.deepcopy(gen_st), None)
            logit = logits[0, -1]
            if temp <= 0:
                nxt = int(logit.argmax())
            else:
                nxt = int(torch.multinomial(
                    torch.softmax(logit / temp, -1), 1))
            if nxt == eot_m:
                break
            out.append(nxt)
            ctx.append(nxt)
            piece = tok.decode(out)
            sys.stdout.write("\r" + "v80> " + piece)
            sys.stdout.flush()
        print()
        reply = out + [eot_m]
        commit(reply)
        buf.extend(reply)


if __name__ == "__main__":
    main()
