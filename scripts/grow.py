#!/usr/bin/env python3
"""grow.py — scale a little, without a gestation, without changing what
it knows. Two organs can grow FUNCTION-PRESERVING:

  * the trunk: new ScanBlocks appended with their attention output and
    MLP output zeroed — the block adds exactly nothing at first, so the
    body answers as before; live doses and the nights train the new
    capacity from there (net2net-style growth);
  * the hippocampus: the store's addresses are FROZEN random Fourier
    features (nothing between store and prediction is learned but one
    scalar), and capacity is ~D pairs per band, so doubling D doubles
    one-shot capacity with no training at all. New addresses mean a
    new hippocampus: the episodic state M restarts empty (the night
    would have decayed it anyway); every fact in the weights stays.

  python3 scripts/grow.py data/organism_life.pt data/organism_life_grown.pt \
      --blocks 2 --slots 2
"""
import argparse
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from iga.lm_scan import ScanLM, ScanBlock                 # noqa: E402
from iga.lm_hybrid import LogitStore                      # noqa: E402


def build(cfg, V):
    kw = dict(d=cfg["d"], n_layers=cfg["n_layers"], n_heads=cfg.get("n_heads", 8),
              max_T=cfg.get("T", 64), mlp=cfg.get("mlp", "gelu"),
              aux_trunk=cfg.get("aux_trunk", 0))
    if cfg.get("clocks"):
        kw["clocks"] = {int(k): int(v) for k, v in cfg["clocks"].items()}
    if cfg.get("gate_init") is not None:
        kw["gate_init"] = cfg["gate_init"]
    return ScanLM(V, **kw, **(cfg.get("scan") or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--blocks", type=int, default=2, help="new zero-init trunk blocks")
    ap.add_argument("--slots", type=float, default=2.0, help="hippocampus slots multiplier")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]); scan = dict(cfg.get("scan") or {})
    sd = dict(ck["model"])
    V, d = sd["embed.weight"].shape
    heads = cfg.get("n_heads", 8)
    n0 = cfg["n_layers"]
    # --- the trunk: appended blocks that do nothing yet
    for i in range(a.blocks):
        b = ScanBlock(d, heads, cfg.get("mlp", "gelu"))
        with torch.no_grad():
            nn.init.zeros_(b.attn.out_proj.weight); nn.init.zeros_(b.attn.out_proj.bias)
            last = [m for m in b.mlp.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        for k, v in b.state_dict().items():
            sd["blocks.%d.%s" % (n0 + i, k)] = v.clone()
    cfg["n_layers"] = n0 + a.blocks
    # --- the hippocampus: more frozen addresses, an empty store
    kd_base0 = scan.get("kd_base", 1); kd_max0 = scan.get("kd_max", 4096)
    scan["kd_base"] = int(round(kd_base0 * a.slots))
    scan["kd_max"] = int(round(kd_max0 * a.slots))
    cfg["scan"] = scan
    fresh = build(cfg, V)                                  # birth init at the new geometry
    fsd = fresh.state_dict()
    for k in list(sd):
        if k.startswith("stores.") and (k.endswith(".proj") or k.endswith(".phase")):
            sd[k] = fsd[k].clone()                          # new addresses (frozen RFF)
    KD = fresh.KD
    # the live state: the store restarts empty at the new size; all else kept
    for key in ("st_live", "st"):
        st = ck.get(key)
        if isinstance(st, dict) and isinstance(st.get("M"), dict):
            for band, M in list(st["M"].items()):
                B = M.shape[0]
                st["M"][band] = torch.zeros(B, d, KD[int(band)], dtype=M.dtype)
    # sanity: the grown geometry loads
    fresh.load_state_dict({k: v for k, v in sd.items() if k in fsd}, strict=False)
    ck["model"] = sd; ck["cfg"] = cfg
    ck.setdefault("grown", []).append({"blocks": a.blocks, "slots": a.slots,
                                       "n_layers": cfg["n_layers"], "KD": {str(k): v for k, v in KD.items()}})
    torch.save(ck, a.dst)
    n_new = sum(v.numel() for k, v in sd.items() if k.startswith("blocks.") and int(k.split(".")[1]) >= n0)
    print("grown: %d -> %d blocks (+%.1fM params, zero-init), hippocampus slots %s -> %s; saved -> %s"
          % (n0, cfg["n_layers"], n_new / 1e6, {k: min(int(kd_max0), 512 * (2 ** i) * kd_base0) for i, k in enumerate(sorted(KD))}, KD, a.dst))


if __name__ == "__main__":
    main()
