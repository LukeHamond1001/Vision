#!/usr/bin/env python3
"""conceive.py — a life from NOTHING: a random body at a chosen shape,
the base's genome (cfg) otherwise, the organism's own tokenizer, no
facts, no days. Everything it will ever know, it learns live.

  python3 scripts/conceive.py data/organism_life_blank_0p5b.pt data/tok_0p5b.json \
      --d 1024 --n-layers 29 --content-keys
"""
import argparse, json, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iga.lm_scan import ScanLM

BASE_SCAN = {"order": "pfc_first", "n_council": 4, "slot_every": 1, "write_every": 1,
             "compile_council": True, "compile_read": True, "store_exact": True,
             "tie_embed": True, "z_w": 0.0001, "ponder": 3, "ponder_mode": "route",
             "ponder_reenter": "token", "ponder_aux": 0.5, "route_cap": 0.125,
             "store_wipe": "day", "write_surprise": 1.0, "press_unwrite": True,
             "plan_m": 4, "plan_cand": 4, "rem_k": 32, "intrinsic_w": 0.5,
             "dopamine": 1.0, "bg_w": 0.01, "imag_k": 4}
CLOCKS = {"3": 1, "4": 8, "5": 64, "6": 512, "7": 4096, "8": 32768}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst"); ap.add_argument("tokenizer")
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=29)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--content-keys", action="store_true", help="hippocampus keyed by the words themselves")
    ap.add_argument("--kernel", type=float, default=3.0, help="memory kernel sharpness (random-feature scale; the base had 1.4)")
    ap.add_argument("--tied-head", action="store_true", help="tie the head to the embedding (a random tied head copies its input; default untied)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    V = tok.get_vocab_size()
    for s in ("<pad>", "<eot_human>", "<eot_model>", "<+1>", "<+2>", "<-1>", "<-2>"):
        assert tok.token_to_id(s) is not None, s
    scan = dict(BASE_SCAN)
    scan["tie_embed"] = bool(a.tied_head)          # the mouth is not the ear
    if a.content_keys:
        scan["keyed_content"] = True
    torch.manual_seed(a.seed)
    m = ScanLM(V, d=a.d, n_layers=a.n_layers, n_heads=a.n_heads, max_T=64,
               clocks={int(k): v for k, v in CLOCKS.items()}, mlp="gelu", aux_trunk=0.2,
               gate_init=-2.0, **scan)
    with torch.no_grad():
        # the genome: a store that writes and reads from day one (the base
        # LEARNED these: beta ~0.6-0.9 -> sigmoid ~0.7, alpha ~1.0-1.35)
        for k_, stn in m.stores.items():
            stn.beta.fill_(0.85)
        for k_, al in m.alpha.items():
            al.fill_(1.0)
        if a.kernel != 1.4:
            for k_, stn in m.stores.items():
                stn.proj.mul_(a.kernel / 1.4)          # sharper matching: near-exact recall
    cfg = {"arch": "scan", "d": a.d, "n_layers": a.n_layers, "n_heads": a.n_heads, "T": 64,
           "precision": "bf16", "clocks": CLOCKS, "scan": scan, "mlp": "gelu", "store": "matrix",
           "keyed": "hidden", "aux_trunk": 0.2, "gate_init": -2.0,
           "conceived": {"tokenizer": os.path.basename(a.tokenizer), "seed": a.seed, "from": "nothing",
                         "kernel": a.kernel, "genome": {"beta": 0.85, "alpha": 1.0}}}
    life = {"model": m.state_dict(), "cfg": cfg, "step": 0, "nursery_steps": 0, "st": None,
            "life": {"facts": [], "study": [], "progress": {}, "surp_mu": None,
                     "pursuit": None, "pursuit_installment": False, "press_log": [],
                     "notice_peak_dyn": None, "budget_history": [], "day_n": 0,
                     "saliences": {}, "n_human_presses": 0}}
    torch.save(life, a.dst)
    n = sum(p.numel() for p in m.parameters())
    print(f"conceived: {a.dst} | {n/1e6:.1f}M params, d={a.d} x {a.n_layers}, V={V}, content keys={a.content_keys}, "
          f"head {'tied' if a.tied_head else 'untied'}, kernel {a.kernel}, from nothing")


if __name__ == "__main__":
    main()
