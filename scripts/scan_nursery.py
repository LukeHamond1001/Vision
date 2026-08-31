#!/usr/bin/env python3
"""scan_nursery.py — the two-button nursery (Phase 2, first light).

The organism converses; you press. A press is BOTH experience and
lesson:
  * it enters the live stream as a press token, so the reward slot /
    dopamine machinery (trained-but-unclaimed since birth) finally
    receives a real external signal in-context;
  * a POSITIVE press triggers live gradient steps on the exchange as
    a standalone flat day (Q<eot_h>A<eot_m><press>) — the same shape
    as every day it ever lived. +1 = one step, +2 = two.
  * a NEGATIVE press is felt (token in stream) but never reinforced —
    no unlikelihood in v1; cortisol is experience, not anti-training.

Lever 1 (question-echo anchoring) is a serve knob: content tokens of
your last turn get a logit bonus early in the reply, so answers
anchor to what you actually asked. Weights untouched by the knob.

The input bodies are NEVER overwritten: /save writes to --save.

usage:
  python3 scripts/scan_nursery.py data/ship_scan16_schooled.pt \
      data/ship_tok.json --dev mps --save data/nursery_body.pt
commands at the you: prompt: /quit /save
press prompt after each reply: +1 +2 -1 -2 or enter to skip
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from scripts.scan_infer import load_scan          # noqa: E402
from scripts.scan_chat import _lane0, _to_dev      # noqa: E402

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "what", "who", "why", "how", "when", "where", "which", "i", "you",
    "my", "me", "we", "it", "to", "of", "in", "on", "for", "and", "or",
    "can", "could", "should", "would", "that", "this", "with", "at",
    "be", "have", "has", "your", "s", "t", "?", ".", ",", "!", "one",
    "fine", "please", "tell", "say", "said",
}


def content_ids(tok, text):
    ids = []
    for t in tok.encode(text).ids:
        w = tok.decode([t]).strip().lower()
        if len(w) >= 3 and w not in STOP:
            ids.append(t)
    return list(dict.fromkeys(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tok")
    ap.add_argument("--dev", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--temp", type=float, default=0.65)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--pause-budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--echo-boost", type=float, default=3.0,
                    help="lever 1: logit bonus on your last turn's "
                         "content tokens (0 = off)")
    ap.add_argument("--echo-n", type=int, default=24,
                    help="boosted content-token count at reply start")
    ap.add_argument("--live-lr", type=float, default=1e-5)
    ap.add_argument("--rem-w", type=float, default=0.1,
                    help="/sleep: REM loss weight (pretraining value)")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe the store at boot (default: KEEP the "
                         "hippocampus alive — the lived state carries)")
    ap.add_argument("--save", default="data/nursery_body.pt")
    ap.add_argument("--transcript", default=None)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    dev = a.dev
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    m, state = load_scan(a.ckpt, tok, dev)
    eh, em = tok.token_to_id("<eot_human>"), tok.token_to_id("<eot_model>")
    sil = tok.token_to_id("<pad>")
    press_ids = {s: tok.token_to_id(s) for s in ("<+1>", "<+2>", "<-1>", "<-2>")}
    press_lvl = {"<+1>": 1, "<+2>": 2, "<-1>": 3, "<-2>": 4}
    if hasattr(m, "set_reward_tokens"):
        m.set_reward_tokens({i: press_lvl[s] for s, i in press_ids.items()
                             if i is not None})
    if eh is not None and em is not None and hasattr(m, "set_eot_ids"):
        m.set_eot_ids(eh, em)
    m = m.eval()

    st = m.init_state(1, dev)
    src_st = state.get("st_live") or state.get("st")
    if src_st is not None:
        st0 = src_st if state.get("st_live") else _lane0(src_st)
        if a.fresh:
            m.wipe_stores(st0, [0])
        st = _to_dev(st0, dev)   # default: the hippocampus STAYS alive
    opt = torch.optim.Adam(m.parameters(), lr=a.live_lr)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    lines, n_press, n_steps = [], 0, 0
    print(f"[nursery] {a.ckpt} on {dev} — echo {a.echo_boost} "
          f"live-lr {a.live_lr}; presses train LIVE. /save -> {a.save}",
          file=sys.stderr)

    day_buf = []   # every lived token since the last /sleep — dream fuel

    def feed(ids):
        nonlocal st
        day_buf.extend(ids)
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                x = torch.tensor([ids[i:i + 64]], device=dev)
                logits, st, _ = m(x, st)
        return logits

    def _detach_state(s):
        if torch.is_tensor(s):
            return s.detach()
        if isinstance(s, dict):
            return {k: _detach_state(v) for k, v in s.items()}
        if isinstance(s, (list, tuple)):
            t = [_detach_state(v) for v in s]
            return type(s)(t) if isinstance(s, tuple) else t
        return s

    def rem_sleep():
        """the /sleep button: run the organism's own REM machinery over
        the lived day — training-mode forwards repopulate the capture
        record chunk by chunk, and each chunk's dreams (rem_loss: the
        planner rolling captured states closed-loop against the wake
        trajectory) backprop onto the cortex. Pretraining's night, live."""
        nonlocal n_steps
        if len(day_buf) < 65:
            print("  [not enough lived tokens to dream yet]", file=sys.stderr)
            return
        ids = day_buf[-4096:]
        ids = ids + [sil] * ((64 - len(ids) % 64) % 64)
        m.train()
        st_s = m.init_state(1, dev)
        cycles = 0
        for i in range(0, len(ids), 64):
            x = torch.tensor([ids[i:i + 64]], device=dev)
            _, st_s, _ = m(x, st_s)
            rl = m.rem_loss()
            if rl is not None:
                opt.zero_grad(set_to_none=True)
                (a.rem_w * rl).backward()
                opt.step()
                n_steps += 1
                cycles += 1
            st_s = _detach_state(st_s)
        m.eval()
        fid = {k: round(float(v), 3)
               for k, v in getattr(m, "rem_fid", {}).items()}
        print(f"  [REM: {cycles} dream cycles over {len(day_buf)} lived "
              f"tokens; dream fidelity {fid}]", file=sys.stderr)
        del day_buf[:]

    def reply(last_logits, boost_ids):
        nonlocal st
        out, pauses, shown = [], 0, ""
        used = set()          # use-once echo: a word's pull dies once spoken
        lg = last_logits
        x = None
        print("model: ", end="", flush=True)
        with torch.no_grad():
            for _ in range(a.max_new + a.pause_budget + 4):
                if x is not None:
                    lg, st, _ = m(x, st)
                v = lg[0, -1].float()
                if hasattr(m, "ban_presses"):
                    v = m.ban_presses(v)
                n_c = len([t for t in out if t != sil])
                live_boost = [i for i in boost_ids if i not in used]
                if a.echo_boost > 0 and n_c < a.echo_n and live_boost:
                    v[live_boost] = v[live_boost] + a.echo_boost
                if pauses >= a.pause_budget and sil is not None:
                    v[sil] = float("-inf")
                pr = torch.softmax(v / max(a.temp, 1e-4), -1).cpu()
                nxt = int(torch.multinomial(pr, 1, generator=gen))
                if nxt in boost_ids:
                    used.add(nxt)
                out.append(nxt)
                if nxt == sil:
                    pauses += 1
                elif nxt != em:
                    text = tok.decode([t for t in out if t not in (sil, em)])
                    print(text[len(shown):], end="", flush=True)
                    shown = text
                if nxt == em or n_c + 1 >= a.max_new:
                    break
                x = torch.tensor([[nxt]], device=dev)
        print(flush=True)
        day_buf.extend(out)      # its own words are lived experience too
        if not out or out[-1] != em:
            feed([em])
        return shown

    def live_step(q_text, ans_text, press_tok, k):
        """k gradient steps on the exchange as ONE standalone flat day."""
        nonlocal n_steps
        ids = (tok.encode(q_text).ids + [eh]
               + tok.encode(" " + ans_text.strip()).ids + [em]
               + [press_ids[press_tok]])
        pad = (64 - len(ids) % 64) % 64
        ids = ids + [sil] * pad
        x = torch.tensor([ids[:-1]], device=dev)
        y = torch.tensor([ids[1:]], device=dev)
        ans_from = len(tok.encode(q_text).ids) + 1
        w = torch.zeros_like(y, dtype=torch.float)
        w[0, ans_from - 1:ans_from - 1 + len(tok.encode(" " + ans_text.strip()).ids) + 1] = 1.0
        m.train()
        for _ in range(k):
            opt.zero_grad(set_to_none=True)
            st_d = m.init_state(1, dev)          # day law: fresh day
            loss_sum, n_tok = None, w.sum().clamp_min(1.0)
            for i in range(0, x.shape[1], 64):
                lg, st_d, _ = m(x[:, i:i + 64], st_d)
                ce = torch.nn.functional.cross_entropy(
                    lg[0], y[0, i:i + 64], reduction="none")
                piece = (ce * w[0, i:i + 64]).sum()
                loss_sum = piece if loss_sum is None else loss_sum + piece
            loss = loss_sum / n_tok
            loss.backward()
            opt.step()
            n_steps += 1
            print(f"  [absorbed: step {n_steps}, loss {float(loss):.3f}]",
                  file=sys.stderr)
        m.eval()

    print("you: ", end="", flush=True)
    q_prev = None
    for line in sys.stdin:
        text = line.strip()
        if not text:
            print("you: ", end="", flush=True)
            continue
        if text in ("/quit", "/q"):
            break
        if text == "/save":
            torch.save({"model": m.state_dict(), "step": state.get("step"),
                        "cfg": state.get("cfg"), "nursery_steps": n_steps,
                        "st_live": _detach_state(_to_dev(st, "cpu"))},
                       a.save)
            print(f"[saved body + lived state -> {a.save}]", file=sys.stderr)
            print("you: ", end="", flush=True)
            continue
        if text == "/sleep":
            rem_sleep()
            print("you: ", end="", flush=True)
            continue
        if text in ("+1", "+2", "-1", "-2") and q_prev is not None:
            ptok = f"<{text}>"
            n_press += 1
            feed([press_ids[ptok]])              # felt in the live stream
            if text in ("+1", "+2"):
                live_step(q_prev[0], q_prev[1], ptok, k=int(text[1]))
            else:
                print("  [felt. not reinforced.]", file=sys.stderr)
            print("you: ", end="", flush=True)
            continue
        boost = content_ids(tok, text)
        last = feed(tok.encode(text).ids + [eh])
        ans = reply(last, boost)
        lines.append(f"you: {text}")
        lines.append(f"model: {ans}")
        q_prev = (text, ans)
        print("press [+1/+2/-1/-2 or enter]> ", end="", flush=True)
    if a.transcript:
        open(a.transcript, "w").write("\n".join(lines) + "\n")
    print(f"\n[nursery closed: {n_press} presses, {n_steps} live steps. "
          f"/save target was {a.save}]", file=sys.stderr)


if __name__ == "__main__":
    main()
