"""From-random gestation of the live body (LIVE_BODY.md): the same
train() recipe the base was raised on (pod_scan.sh's smoke call), at the
chosen shape, on real-text shards (iga/lm_data_text.py) that carry the
lives diet and its face events. The new organs (your face as a sense,
its face as a forecast) train from token one via lm_train's affect path.

    python3 scripts/gestate_1b.py --data data/text_1b --d 1536 --n-layers 28 \
        --lanes 32 --steps 200000 --device cuda --precision bf16 \
        --ckpt /workspace/life_1b/scan.pt --resume
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-sleep", action="store_true", help="no nights (smokes)")
    a = ap.parse_args()
    import torch
    from iga.lm_train import train
    clocks = {3: 1, 4: 8, 5: 64, 6: 512, 7: 4096, 8: 32768}
    opts = {"order": "pfc_first", "n_council": 4, "slot_every": 1, "write_every": 1,
            "compile_council": a.device == "cuda", "compile_read": a.device == "cuda",
            "store_exact": True, "tie_embed": True, "z_w": 1e-4, "ponder": 3,
            "ponder_mode": "route", "ponder_reenter": "token", "ponder_aux": 0.5,
            "route_cap": 0.125, "store_wipe": "day", "write_surprise": 1.0,
            "press_unwrite": True, "plan_m": 4, "plan_cand": 4, "rem_k": 32,
            "intrinsic_w": 0.5, "dopamine": 1.0, "bg_w": 0.01, "imag_k": 4}
    sl = None
    if not a.no_sleep:
        from iga.lm_sleep import Sleeper
        sl = Sleeper(arm="C", every=8, block_chunks=2, seed=1, homeostasis=1e-3)
        sl.press_pay = (a.T, a.T // 8)
    os.makedirs(os.path.dirname(os.path.abspath(a.ckpt)), exist_ok=True)
    resume = a.ckpt if (a.resume and os.path.exists(a.ckpt)) else None
    t0 = time.time()
    out = train(d=a.d, n_layers=a.n_layers, lanes=a.lanes, T=a.T, steps=a.steps, seed=a.seed,
                device=a.device, arch="scan", store="matrix", keyed="hidden", scan=opts,
                norm_mix=True, aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
                clocks=clocks, precision=a.precision, data=a.data, sleep=sl,
                lr=a.lr, log_every=a.log_every, ckpt=a.ckpt, resume=resume)
    model = out[0]
    n = sum(p.numel() for p in model.parameters())
    print(json.dumps({"params_m": round(n / 1e6, 1), "steps": a.steps, "hours": round((time.time() - t0) / 3600, 3),
                      "tokens": a.steps * a.lanes * a.T, "ckpt": a.ckpt}))


if __name__ == "__main__":
    main()
