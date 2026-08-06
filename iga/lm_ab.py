"""v5.0 debug A/B — talk mode x width shape, matched params (A10).

Four cells: {dense, tick} x {uniform, slowheavy}. Each cell's base
width d is searched so total params match the reference cell within
~3% (fair fight). Reports final CE, short-gap recall EMA, and tok/s.
The winners get frozen into the card before any registered run.

Run (debug pod):
  python -m iga.lm_ab --steps 400 --d 128 [--data data/uc_shard]
"""

import argparse

import torch

from .lm_bands import BandLM, shape_widths
from .lm_train import train


def match_d(vocab_size, talk, shape, target, d0):
    d, best = d0, None
    for d in range(max(16, d0 // 2), d0 * 2, 8):
        n = BandLM(vocab_size, d=d, talk=talk,
                   widths=shape_widths(d, shape)).n_params()
        gap = abs(n - target) / target
        if best is None or gap < best[0]:
            best = (gap, d, n)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()
    if a.data:
        from .lm_data_ultrachat import load_tokenizer
        import os
        vocab_size = load_tokenizer(
            os.path.join(a.data, "tokenizer.json")).get_vocab_size()
    else:
        from .lm_conveyor import Vocab
        vocab_size = len(Vocab())
    target = BandLM(vocab_size, d=a.d, talk="dense",
                    widths=shape_widths(a.d, "uniform")).n_params()
    rows = []
    for talk in ("dense", "tick"):
        for shape in ("uniform", "slowheavy"):
            d, n = match_d(vocab_size, talk, shape, target, a.d)
            print(f"\n=== cell talk={talk} shape={shape} "
                  f"d={d} params={n:,} ===")
            model, drive, _, ce0, ce1 = train(
                d=d, lanes=a.lanes, T=a.chunk, steps=a.steps, seed=a.seed,
                device=a.device, log_every=max(a.steps // 4, 1),
                data=a.data, talk=talk, widths=shape_widths(d, shape))
            b0 = drive.ema.get("recall:b0", 0.0)
            b1 = drive.ema.get("recall:b1", 0.0)
            rows.append((talk, shape, d, n, ce1, b0, b1))
    print("\n== A/B results (matched params; freeze winners in the card) ==")
    print(f"{'talk':6s} {'shape':10s} {'d':>4s} {'params':>10s} "
          f"{'ce':>7s} {'recall b0':>10s} {'recall b1':>10s}")
    for talk, shape, d, n, ce, b0, b1 in rows:
        print(f"{talk:6s} {shape:10s} {d:4d} {n:10,d} {ce:7.3f} "
              f"{b0:10.4f} {b1:10.4f}")
    best = min(rows, key=lambda r: r[4] - 10 * r[5])
    print(f"\nsuggested winner (ce - 10*b0 recall): "
          f"talk={best[0]} shape={best[1]}")

    # throughput dialing on the winner: lanes sweep + compile probe.
    # tok/s here sizes the ONE registered run (A11).
    import time as _t
    from .lm_train import process_chunk
    print("\n== throughput sweep (winner cell) ==")
    for lanes, comp in ((8, False), (32, False), (128, False), (128, True)):
        try:
            t0 = _t.time()
            train(d=best[2], lanes=lanes, T=a.chunk, steps=12, seed=1,
                  device=a.device, log_every=999, data=a.data,
                  talk=best[0],
                  widths=shape_widths(best[2], best[1]),
                  compile_model=comp)
            dt = _t.time() - t0
            toks = lanes * a.chunk * 12
            print(f"  lanes={lanes:4d} compile={comp}  "
                  f"{toks/dt:,.0f} tok/s (incl. warmup)")
        except Exception as e:
            print(f"  lanes={lanes:4d} compile={comp}  FAILED: {e}")


if __name__ == "__main__":
    main()
