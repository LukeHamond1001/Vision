"""A63 v-scale Phase 1 application driver (pod-side).

Wake continues v94-best (step 266k) on the CERTIFIED v9.4 config —
lanes 6 x T=2048, lr 1e-4 on the 488k cosine, fp32, offset 0.545 so
the continuation rides the tail v94-best never saw — with ARM B
sleep interleaved at the debug-tested winning dose (block of 2
chunks every 8 wake steps = 1:4 sleep:wake in chunks... measured in
B=1 sleep chunks vs 6-lane wake chunks the compute overhead is ~4%).

Seed: v94.pt.best.pt holds model+step only (F4 banking format), so
this driver builds a resume bundle with a FRESH AdamW state — the
model is reconstructed at the certified shape first, so a state-dict
mismatch aborts before any GPU time is spent.

Gate instrument (banked live, scored LOCALLY later): every 4th
sleep block, the replayed window's tokens AND a matched control
window (same lane, 3 window-lengths earlier, verified to overlap no
replayed window) are appended to v94s_windows.jsonl. The A63 gate:
store-wiped CE improvement (v94-best -> final) on replayed windows
vs their matched controls, paired sign test p < 0.05; guard =
mix_r1_eval CE regression < 1%.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_sleep import Sleeper                   # noqa: E402
from iga.lm_train import train, atomic_save        # noqa: E402

BEST = "/workspace/w-v94/iga-scale/v94.pt.best.pt"
DATA = "/workspace/rmix/mix_v9"
EVAL = "/workspace/rmix/mix_r1_eval"
SEED_STEP_MIN = 200_000
TOTAL = 488_000
STEPS = 30_000


class BankingSleeper(Sleeper):
    """Sleeper + the gate's paired-window instrument."""

    def __init__(self, windows_path, **kw):
        super().__init__(**kw)
        self.windows_path = windows_path
        self.banked = 0
        self.bank_skipped = 0

    def _block(self, model, opt, step):
        row = super()._block(model, opt, step)
        if row is None or len(self.stats) % 4:
            return row
        lane = row["lane"]
        lo, hi = row["window"]
        W = hi - lo
        clo = lo - 3 * W
        overlaps = any(r["lane"] == lane and r["lo"] < clo + W
                       and clo < r["hi"] for r in self.replayed)
        if clo < self.start or overlaps:
            self.bank_skipped += 1
            return row
        buf = self.buffers[lane]
        rec = {"step": step, "lane": lane, "lo": lo, "hi": hi,
               "pay": row["pay"], "clo": clo, "chi": clo + W,
               "toks": buf[lo - self.start: hi - self.start],
               "ctoks": buf[clo - self.start: clo + W - self.start]}
        with open(self.windows_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.banked += 1
        return row


def build_seed(seed_path):
    st = torch.load(BEST, map_location="cpu", weights_only=False)
    step = st.get("step", 0)
    assert step >= SEED_STEP_MIN, f"seed step {step} < {SEED_STEP_MIN}"
    tok = load_tokenizer(os.path.join(DATA, "tokenizer.json"))
    m = HybridLM(tok.get_vocab_size(), d=512, max_T=2048,
                 store="matrix", keyed="logit", norm_mix=True,
                 aux_trunk=0.2, use_xl=False, gate_init=-2.0)
    m.load_state_dict(st["model"])   # certified-shape validation
    alphas = {k: round(float(a), 3) for k, a in m.alpha.items()}
    print(f"seed: step {step}, alphas {alphas}", flush=True)
    assert any(abs(v) > 0.5 for v in alphas.values()), \
        "store not live — ARM B would be a no-op"
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    atomic_save({"model": m.state_dict(), "opt": opt.state_dict(),
                 "step": step, "peval_best": None, "drive": {}},
                seed_path)
    return step


def main():
    work = os.getcwd()
    seed_path = os.path.join(work, "v94s_seed.pt")
    if not os.path.exists(seed_path):
        step0 = build_seed(seed_path)
    else:
        step0 = torch.load(seed_path, map_location="cpu",
                           weights_only=False).get("step", 0)
        print(f"seed exists (step {step0})", flush=True)
    resume = seed_path
    ckpt = os.path.join(work, "v94s.pt")
    if os.path.exists(ckpt):
        cur = torch.load(ckpt, map_location="cpu",
                         weights_only=False).get("step", 0)
        if cur > step0 + 100:      # crash resume: continue the run
            resume = ckpt
            print(f"resuming crashed run at {cur}", flush=True)
    done = torch.load(resume, map_location="cpu",
                      weights_only=False).get("step", 0) - step0
    sl = BankingSleeper(os.path.join(work, "v94s_windows.jsonl"),
                        arm="B", every=8, block_chunks=2, seed=1)
    train(d=512, lanes=6, T=2048, steps=STEPS - done, seed=0,
          device="cuda", ckpt=ckpt, log_every=50, data=DATA,
          talk="tick", arch="hybrid", resume=resume,
          offset_frac=266_000 / TOTAL, store="matrix",
          eval_data=EVAL, use_xl=False, gate_init=-2.0,
          lr=1e-4, lam=0.02, keyed="logit", lr_decay="cosine",
          lr_total_steps=TOTAL, norm_mix=True, aux_trunk=0.2,
          sleep=sl)
    with open(os.path.join(work, "v94s_sleep.json"), "w") as f:
        json.dump({"audit": sl.audit(), "banked": sl.banked,
                   "bank_skipped": sl.bank_skipped,
                   "stats": sl.stats, "replayed": sl.replayed}, f)
    print(f"SLEEP AUDIT {sl.audit()} banked={sl.banked} "
          f"skipped={sl.bank_skipped}", flush=True)


if __name__ == "__main__":
    main()
