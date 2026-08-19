"""A65 — the parenting room. Live serving REPL on the
press-extended substrate.

  python3 scripts/serve_v94s.py --dir <surgery outdir>

  you> hello                 your turn; the model replies
  you> /+2  /+1  /-1  /-2    press the graded button
  you> /sleep [blocks]       consolidate pressed episodes (ARM B)
  you> /wipe                 context wipe (act-3 readout state)
  you> /state                economy panel
  you> /save [path]          save weights + session record
  you> /quit                 save and exit

Generation runs at the faithful regime (state advances only on
exact-T commits); ~0.2-0.5s per token on this Mac's CPU — short
replies land in a few seconds.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_serve import ServeSession              # noqa: E402
from iga.lm_sleep import Sleeper                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="surgery outdir (v94sp.pt + tokenizer)")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40, dest="top_k")
    ap.add_argument("--max-reply", type=int, default=48,
                    dest="max_reply")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    tok = load_tokenizer(os.path.join(a.dir, "tokenizer_press.json"))
    V = tok.get_vocab_size()
    st = torch.load(os.path.join(a.dir, "v94sp.pt"),
                    map_location="cpu", weights_only=False)
    m = HybridLM(V, d=a.d, max_T=a.T, store="matrix", keyed="logit",
                 norm_mix=True, aux_trunk=0.2, use_xl=False,
                 gate_init=-2.0)
    m.load_state_dict(st["model"])
    print(f"substrate loaded: step {st.get('step')}, vocab {V}, "
          f"{m.n_params():,} params", flush=True)
    log = a.log or os.path.join(a.dir, "session.jsonl")
    s = ServeSession(
        m, tok, T=a.T, device="cpu",
        sleeper=Sleeper(arm="B", every=0, block_chunks=2, seed=1,
                        min_step_loss=1e-4),
        temperature=0.0 if a.greedy else a.temp, top_k=a.top_k,
        max_reply=a.max_reply, log_path=log, seed=a.seed)
    print(f"session log -> {log}\nready. /quit to save+exit.",
          flush=True)
    presses = {"/+1": 1, "/+2": 2, "/-1": -1, "/-2": -2}
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/quit"
        if not line:
            continue
        if line in presses:
            s.press(presses[line])
            print(f"  [press {line[1:]} at {s.pos}]", flush=True)
        elif line.startswith("/sleep"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 else 8
            t0 = time.time()
            out = s.sleep_now(blocks=n)
            print(f"  [sleep: {out.get('blocks', 0)} blocks over "
                  f"{out.get('spans', 0)} spans in "
                  f"{time.time() - t0:.1f}s]", flush=True)
        elif line == "/wipe":
            s.wipe()
            print("  [context wiped: fresh state]", flush=True)
        elif line.startswith("/save"):
            parts = line.split()
            path = parts[1] if len(parts) > 1 else \
                os.path.join(a.dir, "v94sp_parented.pt")
            s.save(path)
            print(f"  [saved {path}]", flush=True)
        elif line == "/state":
            print("  " + json.dumps(s.panel()), flush=True)
        elif line == "/quit":
            path = os.path.join(a.dir, "v94sp_parented.pt")
            s.save(path)
            print(f"  [saved {path}; bye]", flush=True)
            break
        else:
            s.user(line)
            t0 = time.time()
            text = s.reply()
            dt = time.time() - t0
            print(f"model> {text}", flush=True)
            print(f"  [{dt:.1f}s, pos {s.pos}]", flush=True)


if __name__ == "__main__":
    main()
