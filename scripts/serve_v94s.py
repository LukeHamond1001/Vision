"""A65/A67 — the parenting room. One continuous life on the
press-extended substrate.

  python3 scripts/serve_v94s.py --dir <surgery outdir> \
      [--resume <v94sp_life.pt>]

  you> hello                 your turn; the model replies
  you> /+2  /+1  /-1  /-2    press the graded button
  you> /sleep [blocks]       consolidate pressed episodes (ARM A)
  you> /wipe                 context wipe (retention testing)
  you> /state                economy panel
  you> /prophet              band press-predictor fidelity
  you> /save [path]          save the whole life
  you> /quit                 save the life and exit

/quit writes v94sp_life.pt — band/store state, press ledger,
sleeper memory, prophet heads, optimizer moments, RNG streams.
--resume continues that exact life (A67 law: bit-equal to never
having stopped). The pristine substrate v94sp.pt is never
overwritten. Replies ~0.1-0.5s on the Mac.
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
    ap.add_argument("--resume", default=None,
                    help="A67: a saved life (v94sp_life.pt) — the "
                         "same existence continues")
    a = ap.parse_args()
    tok = load_tokenizer(os.path.join(a.dir, "tokenizer_press.json"))
    V = tok.get_vocab_size()
    m = HybridLM(V, d=a.d, max_T=a.T, store="matrix", keyed="logit",
                 norm_mix=True, aux_trunk=0.2, use_xl=False,
                 gate_init=-2.0)
    resume_state = None
    if a.resume:
        resume_state = torch.load(a.resume, map_location="cpu",
                                  weights_only=False)
        print(f"resuming life from {a.resume}", flush=True)
    else:
        st = torch.load(os.path.join(a.dir, "v94sp.pt"),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(st["model"])
        print(f"fresh substrate: step {st.get('step')}", flush=True)
    from iga.lm_press import PressProphet
    log = a.log or os.path.join(a.dir, "session.jsonl")
    # A66-R3: serve consolidation is ARM A (replay-CE, self-anchoring)
    s = ServeSession(
        m, tok, T=a.T, device="cpu",
        sleeper=Sleeper(arm="A", every=0, block_chunks=2, seed=1,
                        min_step_loss=1e-4),
        temperature=0.0 if a.greedy else a.temp, top_k=a.top_k,
        max_reply=a.max_reply, log_path=log, seed=a.seed,
        sleep_lr=5e-5, prophet=PressProphet(d=a.d),
        resume_state=resume_state)
    print(f"vocab {V}, {m.n_params():,} params | life: {s.pos} "
          f"tokens, {len(s.drive.presses)} presses", flush=True)
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
                os.path.join(a.dir, "v94sp_life.pt")
            s.save(path)
            print(f"  [life saved {path}]", flush=True)
        elif line == "/state":
            print("  " + json.dumps(s.panel()), flush=True)
        elif line == "/prophet":
            print("  " + json.dumps(s.prophet.report()), flush=True)
        elif line == "/quit":
            path = os.path.join(a.dir, "v94sp_life.pt")
            s.save(path)
            print(f"  [life saved -> {path}; resume with "
                  f"--resume {path}]", flush=True)
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
