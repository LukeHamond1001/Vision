"""A76 gate — sleep homeostasis vs THE INCUMBENT DISEASE, at debug
tier, $0.

Recipe (pre-registered): take the flash-born d=128 creature
(data/a69_bio.pt), raise a MINI-INCUMBENT the exact way the 78M
disease grew (all-positive drilling of one fact, +2 every touch,
nightly sleep), then fork two identical continuations from the
drilled state:
  arm h0 — homeostasis = 0        (the certified sleeper)
  arm h1 — homeostasis = H        (the downscale, sleep steps only)
Both arms then live IDENTICAL further days: keep drilling the
incumbent fact all-positive AND teach one fresh fact (+2, spaced).

PASS iff, at the end:
  (1) DAMPING    — incumbent conviction under h1 is lower than
                   under h0 by >= 5% relative (saturation bent);
  (2) MEMORY     — the fresh taught fact's belief under h1 is
                   >= 0.8x its h0 value (downscale must not eat
                   new learning);
  (3) SPEECH     — mean stream CE on a fixed probe set within 3%.
Evidence -> results/evidence/a76_gate.json.
Usage: python3 scripts/a76_gate.py [H] [drill_days] [fork_days]
"""

import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # scripts/archive/ -> repo root

from iga.lm_conveyor import Vocab                  # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_serve import ServeSession              # noqa: E402
from iga.lm_sleep import Sleeper                   # noqa: E402

H = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-4
DRILL_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FORK_DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PASS_DAMP = 0.05          # relative conviction reduction required
PASS_MEM = 0.80           # fresh-fact belief retention required
PASS_CE = 0.03            # relative CE budget


class TokAdapter:
    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def __init__(self):
        self.v = Vocab()

    def token_to_id(self, w):
        return self.v.idx[w]

    def get_vocab_size(self):
        return len(self.v)

    def encode(self, text):
        return self._Enc([self.v.idx[w] for w in text.split()])

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.v.words[i] for i in ids)


def session(sd, homeostasis):
    tok = TokAdapter()
    m = HybridLM(tok.get_vocab_size(), d=128, max_T=256,
                 store="matrix", keyed="logit", norm_mix=True,
                 aux_trunk=0.2, use_xl=False, gate_init=-2.0)
    m.load_state_dict(sd)
    s = ServeSession(
        m, tok, T=256, device="cpu",
        sleeper=Sleeper(arm="C", every=0, block_chunks=2, seed=1,
                        min_step_loss=1e-4,
                        homeostasis=homeostasis),
        temperature=0.0, max_reply=12, seed=7, sleep_lr=5e-5)
    return s, tok, m


@torch.no_grad()
def belief(m, tok, name, obj, col):
    toks = (f"what color of {obj} was {name} kept ? "
            f"<eot_human> the {obj} was").split()
    x = torch.tensor([[tok.token_to_id(w) for w in toks]])
    lg, _, _ = m(x, m.init_state(1, "cpu"), None)
    m.pop_write_cost()
    m.pop_recon()
    return float(torch.softmax(lg[0, -1].float(), -1)
                 [tok.token_to_id(col)])


@torch.no_grad()
def stream_ce(m, tok):
    """Fixed in-lexicon probe stream, fresh state."""
    text = ("good morning . the town waited . what color of rope "
            "was petra kept ? <eot_human> the rope was blue . "
            "one morning later . the wind moved that day . "
            "that day was done . good job .").split()
    ids = [tok.token_to_id(w) for w in text]
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    lg, _, _ = m(x, m.init_state(1, "cpu"), None)
    m.pop_write_cost()
    m.pop_recon()
    return float(torch.nn.functional.cross_entropy(
        lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1)))


def live_day(s, inc, fresh=None, teach_fresh=False):
    def say(t):
        s.user(t)
        return s.reply()

    say("good morning .")
    say(f"by the way {inc[0]} kept a {inc[2]} {inc[1]} in the "
        f"cellar .")
    s.press(2)
    for _ in range(2):
        say("the wind moved that day . the town waited .")
    if teach_fresh:
        say(f"by the way {fresh[0]} kept a {fresh[2]} {fresh[1]} "
            f"in the attic .")
        s.press(2)
        say("the wind moved that day .")
    say(f"still . {inc[0]} kept a {inc[2]} {inc[1]} .")
    s.press(2)
    say("that day was done . good job .")
    return s.sleep_now(blocks=32, span_w=48, void_w=32)


def main():
    sd0 = torch.load("data/a69_bio.pt", map_location="cpu",
                     weights_only=False)
    inc = ("mira", "bell", "copper")     # the mini-incumbent
    fresh = ("dov", "map", "grey")       # taught post-fork only

    # phase 1 — raise the incumbent (shared history, h=0)
    s, tok, m = session(sd0, 0.0)
    b0 = belief(m, tok, *inc)
    for day in range(DRILL_DAYS):
        live_day(s, inc)
    sd_drilled = copy.deepcopy(m.state_dict())
    b_drilled = belief(m, tok, *inc)
    print(f"incumbent drilled {b0:.4f} -> {b_drilled:.4f} over "
          f"{DRILL_DAYS} days", flush=True)

    # phase 2 — identical forked lives, the knob the only variable
    out = {"H": H, "drill_days": DRILL_DAYS, "fork_days": FORK_DAYS,
           "b0": b0, "b_drilled": b_drilled, "arms": {}}
    for tag, h in (("h0", 0.0), ("h1", H)):
        s, tok, m = session(copy.deepcopy(sd_drilled), h)
        for day in range(FORK_DAYS):
            live_day(s, inc, fresh, teach_fresh=(day == 0))
        out["arms"][tag] = {
            "incumbent": belief(m, tok, *inc),
            "fresh": belief(m, tok, *fresh),
            "ce": stream_ce(m, tok)}
        print(f"[{tag}] {out['arms'][tag]}", flush=True)

    a0, a1 = out["arms"]["h0"], out["arms"]["h1"]
    damp = (a0["incumbent"] - a1["incumbent"]) / max(a0["incumbent"],
                                                     1e-9)
    mem = a1["fresh"] / max(a0["fresh"], 1e-9)
    ce_rel = (a1["ce"] - a0["ce"]) / max(a0["ce"], 1e-9)
    verdict = bool(damp >= PASS_DAMP and mem >= PASS_MEM
                   and ce_rel <= PASS_CE)
    out["verdict"] = {"damp": round(damp, 4), "mem": round(mem, 4),
                      "ce_rel": round(ce_rel, 4),
                      "A76_ships": verdict}
    os.makedirs("results/evidence", exist_ok=True)
    with open("results/evidence/a76_gate.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nA76 verdict: damp {damp:.3f} (need >={PASS_DAMP}) "
          f"mem {mem:.3f} (need >={PASS_MEM}) ce_rel {ce_rel:+.3f} "
          f"(budget {PASS_CE}) -> "
          f"{'SHIPS' if verdict else 'OUT'}", flush=True)


if __name__ == "__main__":
    main()
