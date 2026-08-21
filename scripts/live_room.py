"""A67 live room — a mailbox bridge so a caregiver process (me)
can sit with the life turn by turn: read what it says, think, and
answer sentence by sentence. Commands arrive on <life>.inbox as
"seq|cmd|payload" lines; results append to <life>.outbox as
"seq|kind|text". Commands: say, press, sleep, lesion (none|bands|
store|both — the centerpiece's removals, read path only), probe (a silent
belief read — the parent glancing at the child's face), probe0
(the weights-only read: fresh state, no pending — for day-to-day
comparisons), state, save, quit.

Usage: python3 scripts/live_room.py <surgery_dir> <life_path>
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_press import PressProphet              # noqa: E402
from iga.lm_serve import ServeSession              # noqa: E402
from iga.lm_sleep import Sleeper, state_copy       # noqa: E402


def main():
    sdir, life = sys.argv[1], sys.argv[2]
    inbox, outbox = life + ".inbox", life + ".outbox"
    open(inbox, "a").close()
    open(outbox, "w").close()
    # The inbox is an append-only history. Lines already on file are
    # PAST sessions' commands — including their final quit — and must
    # never be replayed into a resumed life (A67-P3 incident).
    with open(inbox) as f:
        stale = sum(1 for _ in f)
    tok = load_tokenizer(os.path.join(sdir, "tokenizer_press.json"))
    m = HybridLM(tok.get_vocab_size(), d=512, max_T=2048,
                 store="matrix", keyed="logit", norm_mix=True,
                 aux_trunk=0.2, use_xl=False, gate_init=-2.0)
    resume = None
    if os.path.exists(life):
        resume = torch.load(life, map_location="cpu",
                            weights_only=False)
    else:
        st = torch.load(os.path.join(sdir, "v94sp.pt"),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(st["model"])
    s = ServeSession(
        m, tok, T=2048, device="cpu",
        sleeper=Sleeper(arm="C", every=0, block_chunks=2, seed=1,
                        min_step_loss=1e-4),
        temperature=0.6, top_k=40, max_reply=24,
        log_path=life + ".sessions.jsonl", seed=7,
        sleep_lr=5e-5, prophet=PressProphet(d=512),
        resume_state=resume)

    def out(n, kind, text):
        with open(outbox, "a") as f:
            f.write(f"{n}|{kind}|{str(text).replace(chr(10), ' ')}\n")

    @torch.no_grad()
    def belief(name, obj, col, bare=False):
        # bare=True is the weights-only read: fresh band state, no
        # pending tail — immune to whatever today's conversation
        # left in context (the day-9 confound).
        stem = f"the {obj} was"
        full = tok.encode(f"{stem} {col} .").ids
        pre = tok.encode(stem).ids
        ans = full[len(pre)]
        ids = tok.encode(
            f"what color of {obj} was {name} kept ?").ids \
            + [s.eot_h] + pre
        if bare:
            ctx, st = [], s.m.init_state(1, "cpu")
        else:
            ctx = s.pending[-(s.T - len(ids)):] if s.pending else []
            st = state_copy(s.st)
        x = torch.tensor([ctx + ids], dtype=torch.long)
        with s.lesion_scope():          # the removal, if one is set
            lg, _, _ = s.m(x, st, None)
        s.m.pop_write_cost()
        s.m.pop_recon()
        return float(torch.softmax(lg[0, -1].float(), -1)[ans])

    out(0, "ready", f"life {s.pos} tokens, "
        f"{len(s.drive.presses)} presses, "
        f"{stale} stale inbox lines skipped")
    seen = stale
    while True:
        with open(inbox) as f:
            lines = f.read().splitlines()
        for line in lines[seen:]:
            seen += 1
            try:
                n, cmd, payload = (line.split("|", 2) + [""])[:3]
                if cmd == "say":
                    s.user(payload)
                    out(n, "reply", s.reply())
                elif cmd == "press":
                    s.press(int(payload))
                    out(n, "pressed", payload)
                elif cmd == "sleep":
                    o = s.sleep_now(blocks=int(payload or "12"),
                                    span_w=256)
                    out(n, "slept", json.dumps(
                        {k: o.get(k)
                         for k in ("spans", "pairs", "blocks")}))
                elif cmd == "probe":
                    nm, ob, co = payload.split()
                    out(n, "belief", f"{belief(nm, ob, co):.4f}")
                elif cmd == "probe0":
                    nm, ob, co = payload.split()
                    out(n, "belief0",
                        f"{belief(nm, ob, co, bare=True):.4f}")
                elif cmd == "lesion":
                    # CENTERPIECE: none | bands | store | both — read path only
                    out(n, "lesion", s.lesion(payload))
                elif cmd == "state":
                    out(n, "state", json.dumps(s.panel()))
                elif cmd == "save":
                    s.save(life)
                    out(n, "saved", life)
                elif cmd == "quit":
                    s.save(life)
                    out(n, "bye", "life saved")
                    return
                else:
                    out(n, "err", f"unknown cmd {cmd}")
            except Exception as e:  # keep the room alive
                out(n if "n" in dir() else "?", "err", repr(e))
        time.sleep(0.2)


if __name__ == "__main__":
    main()
