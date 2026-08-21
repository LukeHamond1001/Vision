"""A69-R6 — the end-to-end debug gate: FLASH -> LIFE -> FACT IN
WEIGHTS. R5 proved pretraining cannot concentrate replay on
individual facts (0.1 replays/fact); the serve life can (dozens).
Division-of-labor law: the flash builds faculties, the life writes
biography. This smoke closes the loop: the R5 flash-born d=128
creature enters a serve room, is taught one rewarded fact and one
unrewarded control, sleeps, and is probed WEIGHTS-ONLY (fresh
state, bare question) before and after.

Gate R6-G1: rewarded belief rises >=5x from its pre-teach baseline
AND the unrewarded control stays within 2x of its own baseline
(A66 selectivity, on a flash-born being).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))   # scripts/archive/ -> repo root

from iga.lm_conveyor import Vocab                  # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_serve import ServeSession              # noqa: E402
from iga.lm_sleep import Sleeper                   # noqa: E402


class TokAdapter:
    """ServeSession-shaped view of the synthetic word vocab."""

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


def main():
    tok = TokAdapter()
    m = HybridLM(tok.get_vocab_size(), d=128, max_T=256,
                 store="matrix", keyed="logit", norm_mix=True,
                 aux_trunk=0.2, use_xl=False, gate_init=-2.0)
    sd = torch.load(
        "/Users/lukehamond/Projects/project/data/a69_bio.pt",
        map_location="cpu", weights_only=False)
    m.load_state_dict(sd)
    s = ServeSession(
        m, tok, T=256, device="cpu",
        sleeper=Sleeper(arm="C", every=0, block_chunks=2, seed=1,
                        min_step_loss=1e-4),
        temperature=0.0, max_reply=12, seed=7, sleep_lr=5e-5)

    @torch.no_grad()
    def belief(name, obj, col):
        toks = (f"what color of {obj} was {name} kept ? "
                f"<eot_human> the {obj} was").split()
        x = torch.tensor([[tok.token_to_id(w) for w in toks]])
        lg, _, _ = m(x, m.init_state(1, "cpu"), None)
        m.pop_write_cost()
        m.pop_recon()
        return float(torch.softmax(lg[0, -1].float(), -1)
                     [tok.token_to_id(col)])

    rew = ("nedra", "coin", "golden")
    ctl = ("arlen", "candle", "blue")
    base_r, base_c = belief(*rew), belief(*ctl)
    print(f"baseline  rewarded {base_r:.4f}  control {base_c:.4f}",
          flush=True)

    def say(t):
        s.user(t)
        return s.reply()

    def spacer(n):
        for _ in range(n):
            say("the wind moved that day . the town waited .")

    say("good morning .")
    say(f"by the way {rew[0]} kept a {rew[2]} {rew[1]} in the "
        f"cellar .")
    s.press(2)
    spacer(3)                           # push control out of reach
    say(f"by the way {ctl[0]} kept a {ctl[2]} {ctl[1]} in the "
        f"attic .")                     # taught, NEVER pressed
    spacer(3)
    say(f"still . {rew[0]} kept a {rew[2]} {rew[1]} .")
    s.press(2)
    spacer(3)
    say(f"one morning later . {rew[0]} still kept a {rew[2]} "
        f"{rew[1]} .")
    s.press(2)
    say("that day was done . good job .")
    # tight spans: a press pays its own teach turn, never the
    # whole session (the first R6 shape voided selectivity by
    # engulfing the control in every paid window)
    out = s.sleep_now(blocks=32, span_w=48, void_w=32)
    print(f"slept {out}", flush=True)

    # days 2-4: the life's actual protocol — spaced touches, nights
    # compounding (the 78M curve: 8.7x, 17x on later touches). The
    # control is never touched again.
    for day in (2, 3, 4):
        say("one morning later . good morning .")
        say(f"still . {rew[0]} kept a {rew[2]} {rew[1]} .")
        s.press(2)
        spacer(2)
        say("that day was done . good job .")
        out = s.sleep_now(blocks=32, span_w=48, void_w=32)
        r, c = belief(*rew), belief(*ctl)
        print(f"day {day}  rewarded {r:.4f}  control {c:.4f}",
              flush=True)

    post_r, post_c = belief(*rew), belief(*ctl)
    print(f"post      rewarded {post_r:.4f}  control {post_c:.4f}",
          flush=True)
    lift_r = post_r / max(base_r, 1e-6)
    lift_c = post_c / max(base_c, 1e-6)
    g1 = lift_r >= 5.0 and lift_c <= 2.0
    print(f"lift      rewarded {lift_r:.1f}x  control {lift_c:.1f}x"
          f"  ->  R6-G1 {'PASS' if g1 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
