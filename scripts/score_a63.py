"""A63 gate scorer — LOCAL (ops law: measurement never rents).

Input: v94s_windows.jsonl (paired replayed/control token windows
banked live by the pod's BankingSleeper), v94-best and the sleep
run's final checkpoint. For each pair, store-wiped CE under both
models; the gate statistic is the paired comparison of improvement
(best -> final) on replayed vs control windows.

  PASS = wins(replayed improvement > control improvement) sign
  test p < 0.05 (one-sided)  [the mix_r1_eval CE guard is scored
  separately with autopsy_v9 against the banked 1.9242 row]

Cold-start note: windows are scored from fresh state (bands cold,
store wiped). The bias is identical across models and window
types; the paired design cancels it.

Usage: python3 scripts/score_a63.py --windows W.jsonl \
         --best v94_best.pt --final v94s_final.pt \
         --tokenizer tokenizer.json [--pairs 500] [--d 512]
"""

import argparse
import json
from math import comb

import torch


def build(ckpt_path, vocab, d, max_t, device):
    from iga.lm_hybrid import HybridLM
    m = HybridLM(vocab, d=d, max_T=max_t, store="matrix",
                 keyed="logit", norm_mix=True, aux_trunk=0.2,
                 use_xl=False, gate_init=-2.0)
    st = torch.load(ckpt_path, map_location="cpu",
                    weights_only=False)
    m.load_state_dict(st["model"])
    step = st.get("step", "?")
    m.to(device).eval()
    m.store_read_off = True          # the store-wipe, by the A62 switch
    return m, step


@torch.no_grad()
def window_ce(m, toks, T, device):
    st = m.init_state(1, device)
    tot, n = 0.0, 0
    for off in range(0, len(toks) - 1, T):
        xs = toks[off: off + T + 1]
        if len(xs) < 2:
            break
        x = torch.tensor([xs[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([xs[1:]], dtype=torch.long, device=device)
        logits, st, _ = m(x, st, None)
        m.pop_write_cost()
        m.pop_recon()
        st = m.detach_state(st)
        ce = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, m.vocab_size), y.reshape(-1),
            reduction="sum")
        tot += float(ce)
        n += y.numel()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--best", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from iga.lm_data_ultrachat import load_tokenizer
    vocab = load_tokenizer(a.tokenizer).get_vocab_size()
    mb, sb = build(a.best, vocab, a.d, a.chunk, a.device)
    mf, sf = build(a.final, vocab, a.d, a.chunk, a.device)
    print(f"best @ {sb}, final @ {sf}, vocab {vocab}", flush=True)
    rows = [json.loads(x) for x in open(a.windows)]
    if len(rows) > a.pairs:            # deterministic thinning
        stride = len(rows) / a.pairs
        rows = [rows[int(i * stride)] for i in range(a.pairs)]
    wins = ties = 0
    dr_sum = dc_sum = 0.0
    out = []
    for i, r in enumerate(rows):
        ce = {}
        for tag, m in (("b", mb), ("f", mf)):
            for wtag, tk in (("r", r["toks"]), ("c", r["ctoks"])):
                ce[tag + wtag] = window_ce(m, tk, a.chunk, a.device)
        dr = ce["br"] - ce["fr"]       # improvement on replayed
        dc = ce["bc"] - ce["fc"]       # improvement on control
        dr_sum += dr
        dc_sum += dc
        if abs(dr - dc) < 1e-9:
            ties += 1
        elif dr > dc:
            wins += 1
        out.append({"step": r["step"], "lane": r["lane"],
                    "dr": round(dr, 5), "dc": round(dc, 5),
                    **{k: round(v, 5) for k, v in ce.items()}})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} wins {wins}", flush=True)
    n = len(rows) - ties
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n \
        if n else 1.0
    print(f"\nA63 GATE: pairs {len(rows)} (ties {ties})  "
          f"wins {wins}/{n}  sign p = {p:.5f}  "
          f"mean d_replayed {dr_sum/len(rows):+.5f}  "
          f"mean d_control {dc_sum/len(rows):+.5f}")
    print("PASS" if (n and wins / n > 0.5 and p < 0.05) else "FAIL",
          "(gate half; CE guard scored separately)")
    with open(a.windows + ".scored.json", "w") as f:
        json.dump({"pairs": len(rows), "ties": ties, "wins": wins,
                   "p": p, "rows": out}, f)


if __name__ == "__main__":
    main()
