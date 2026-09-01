#!/usr/bin/env python3
"""birth.py — a fresh life from the birth body. Copies the gestated
weights (never touching the source checkpoint) into a new life file
with no facts, no ledger, no days: a body that has been born and not
yet raised. Everything it will know, it learns live.

  python3 scripts/birth.py data/ship_scan16_final.pt data/organism_life_v2.pt
"""
import sys
import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    life = {"model": ck["model"], "cfg": ck["cfg"], "step": ck.get("step"),
            "nursery_steps": 0,
            "st": ck.get("st"),           # the birth state (lane 0 is taken at load)
            "life": {"facts": [], "study": [], "progress": {}, "surp_mu": None,
                     "pursuit": None, "pursuit_installment": False,
                     "press_log": [], "notice_peak_dyn": None,
                     "budget_history": [], "day_n": 0, "saliences": {},
                     "n_human_presses": 0}}
    torch.save(life, dst)
    V, d = ck["model"]["embed.weight"].shape
    print("born: %s -> %s | %dM params-ish body, V=%d, step %s, no facts, day 0"
          % (src, dst, sum(p.numel() for p in ck["model"].values()) // 1_000_000, V, ck.get("step")))


if __name__ == "__main__":
    main()
