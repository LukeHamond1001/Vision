#!/usr/bin/env python3
"""scan_chat — talk to a scan organism, turn by turn.

The conversation harness scan_infer cannot be: state persists across
the whole session (one continuous life), generation stops at
<eot_model> (no fixed-N self-talk), silence is allowed but budgeted
(the pause attractor cannot stall a turn), and the brain wakes from
the checkpoint's lived lane-0 state by default (a model turn never
trains on a cold brain, so we never serve one).

  python scripts/scan_chat.py CKPT TOK [--temp .9] [--pause-budget 8]
         [--max-new 300] [--dev cpu|mps] [--lesion none|bands|store|both]
         [--cold] [--seed 0] [--transcript out.txt]

Type; it answers. /quit ends. [pause]s are shown dim as ellipses.
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from scripts.scan_infer import load_scan, _lane0, _to_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tok")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--pause-budget", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--dev", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--lesion", default="none",
                    choices=["none", "bands", "store", "both"])
    ap.add_argument("--cold", action="store_true",
                    help="init state instead of the lived wake state")
    ap.add_argument("--fresh", action="store_true",
                    help="fresh morning: wake the lived state, then "
                         "wipe the store (episodic slate clean, bands "
                         "keep their warmth) — the day-boundary op")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--store-boost", type=float, default=1.0,
                    help="49q: amplify the hippocampus's logit vote at "
                         "serve (sparse top-k form; 1.0 = natural gain)")
    ap.add_argument("--eot-boost", type=float, default=0.0,
                    help="soft wrap-up pressure: adds boost x "
                         "(content_tokens/100) to the <eot_model> "
                         "logit as the reply grows (0 = off)")
    ap.add_argument("--preamble", default=None,
                    help="text file of alternating human/model lines "
                         "fed into the state before the REPL — a "
                         "lived morning, the only system prompt this "
                         "architecture understands")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    dev = a.dev
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    m, state = load_scan(a.ckpt, tok, dev)
    m = m.eval()
    m.lesioned = {3, 4, 5, 6, 7, 8} if a.lesion in ("bands", "both") else set()
    m.store_boost = float(a.store_boost)
    if a.lesion in ("store", "both"):
        m.store_read_off = True
    eh, em = tok.token_to_id("<eot_human>"), tok.token_to_id("<eot_model>")
    sil = tok.token_to_id("<pad>")

    if not a.cold and state.get("st") is not None:
        st = _lane0(state["st"])
        mode = "woken from the lived lane-0 state"
        if a.fresh:
            m.wipe_stores(st, [0])      # wipe on CPU first: _to_dev
            mode += " + store wiped (fresh morning)"   # skips tuples
        st = _to_dev(st, dev)
    else:
        st = m.init_state(1, dev)
        mode = "COLD state (newborn)"
    print(f"[scan_chat] step {state.get('step')} on {dev} — {mode}; "
          f"t={a.temp} pause_budget={a.pause_budget} lesion={a.lesion}",
          file=sys.stderr)

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    lines = []

    def feed(ids):
        nonlocal st
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                x = torch.tensor([ids[i:i + 64]], device=dev)
                logits, st, _ = m(x, st)
        return logits

    def turn():
        nonlocal st
        out, pauses, shown = [], 0, ""
        x = None
        print("scan15: ", end="", flush=True)
        with torch.no_grad():
            for _ in range(a.max_new + a.pause_budget + 4):
                if x is not None:
                    logits, st, _ = m(x, st)
                else:
                    logits = last[0]
                lg = logits[0, -1].float()
                if hasattr(m, "ban_presses"):
                    lg = m.ban_presses(lg)
                if pauses >= a.pause_budget and sil is not None:
                    lg[sil] = float("-inf")
                if a.eot_boost > 0 and em is not None:
                    n_c = len([t for t in out if t != sil])
                    lg[em] = lg[em] + a.eot_boost * (n_c / 100.0)
                pr = torch.softmax(lg / max(a.temp, 1e-4), -1).cpu()
                nxt = int(torch.multinomial(pr, 1, generator=gen))
                out.append(nxt)
                if nxt == sil:
                    pauses += 1
                    print("\u2026", end="", flush=True)   # it is thinking
                elif nxt != em:
                    text = tok.decode(
                        [t for t in out if t not in (sil, em)])
                    print(text[len(shown):], end="", flush=True)
                    shown = text
                if nxt == em or \
                        len([t for t in out if t != sil]) >= a.max_new:
                    break
                x = torch.tensor([[nxt]], device=dev)
        print(flush=True)
        if not out or out[-1] != em:
            feed([em])                      # the turn always closes
        return shown, pauses

    if a.preamble:
        pre = [l.strip() for l in open(a.preamble) if l.strip()]
        ids = []
        for i, t in enumerate(pre):
            ids += tok.encode(t).ids + [eh if i % 2 == 0 else em]
        feed(ids)
        print(f"[preamble: {len(pre)} turns, {len(ids)} tokens lived]",
              file=sys.stderr)

    print("you: ", end="", flush=True)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            print("you: ", end="", flush=True)
            continue
        if text in ("/quit", "/q"):
            break
        last = [feed(tok.encode(text).ids + [eh])]
        words, pauses = turn()
        lines.append(f"you: {text}")
        lines.append(f"scan15: {'[pause]' * pauses}{words}")
        print("you: ", end="", flush=True)
    if a.transcript:
        open(a.transcript, "w").write("\n".join(lines) + "\n")
        print(f"\n[transcript -> {a.transcript}]", file=sys.stderr)


if __name__ == "__main__":
    main()
