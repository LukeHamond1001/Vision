#!/usr/bin/env python3
"""regestate.py — the most a Mac can do: RE-INITIALISE THE ORGANS, KEEP
THE LANGUAGE, and re-gestate the organs locally on the lives diet.

Why not from random weights? Language needs ~1e8-1e9 tokens of
exposure; a caretaker day gives ~1e3. So the trunk (the language: the
scan blocks, the council, the tied embedding) is kept from the base,
and the ORGANS that carry the organism's fragilities are born again at
their default init and trained here on lives built with the face as a
sense (law 14), faces moving mid-utterance (law 13), its own face in
its mouth (law 15) and one-shot recall asked for (episodic asks):

  reward slot + value heads (BG · DA), the hippocampus (stores,
  mem/key/query projections, store_in, alpha, read gates), the goal
  query, the routing head, the council slots.

The trunk adapts gently (a small lr) to hearing faces as senses and to
the four new tokens; the organs learn at full lr. Time-budgeted;
saves a life the serve can raise.

  python3 scripts/regestate.py data/ship_scan16_final.pt data/ship_tok_v17.json \
      --data data/gest_v17 --out data/organism_life_v3.pt --hours 2
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from tokenizers import Tokenizer                                   # noqa: E402
from scripts.scan_infer import load_scan                           # noqa: E402
from iga.lm_scan import ScanLM, press_levels_from_events           # noqa: E402
from iga.lm_diet import LaneConveyor                               # noqa: E402

ORGANS = ("value", "reward_emb", "stores", "mem_proj", "key_proj",
          "query_proj", "store_in", "alpha", "read_gate", "goal_query",
          "route_head", "slot")


def grow_vocab(m, V_new):
    """every vocabulary-sized tensor grows: the tied embedding/head
    small-random (silent until taught), other per-token weights at
    their mean, the press LUT zero."""
    E = m.embed.weight
    V0 = E.shape[0]
    if V_new <= V0:
        return []
    tied = hasattr(m, "head") and m.head.weight.data_ptr() == E.data_ptr()
    grown = []

    def _grow(t, is_embed):
        extra = V_new - t.shape[0]
        if t.dim() >= 2 and is_embed:
            new = torch.empty((extra,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype)
            nn.init.normal_(new, std=float(t.float().std()) * 0.5)
        elif t.dtype.is_floating_point:
            new = t.float().mean(0, keepdim=True).expand((extra,) + tuple(t.shape[1:])).to(t.dtype).clone()
        else:
            new = torch.zeros((extra,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype)
        return torch.cat([t, new], 0)

    with torch.no_grad():
        for name, prm in list(m.named_parameters()):
            if prm.dim() >= 1 and prm.shape[0] == V0:
                if tied and name == "head.weight":
                    continue
                mod = m
                parts = name.split(".")
                for pp in parts[:-1]:
                    mod = getattr(mod, pp)
                setattr(mod, parts[-1], nn.Parameter(_grow(prm.data, name == "embed.weight")))
                grown.append(name)
        for name, buf in list(m.named_buffers()):
            if buf.dim() >= 1 and buf.shape[0] == V0:
                mod = m
                parts = name.split(".")
                for pp in parts[:-1]:
                    mod = getattr(mod, pp)
                setattr(mod, parts[-1], _grow(buf, False))
                grown.append(name)
    if tied:
        m.head.weight = m.embed.weight
    return grown


def fresh_twin(state, V):
    """a ScanLM at birth init with the base's architecture (load_scan's
    own recipe), so the organs can be reborn at their exact default."""
    cfg = dict(state["cfg"])
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
    ap.add_argument("base")
    ap.add_argument("tok")
    ap.add_argument("--data", required=True, help="prepared shard dir (lm_data_life prepare)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--lr-organs", type=float, default=3e-4)
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--lr-embed", type=float, default=5e-5)
    ap.add_argument("--save-every-min", type=float, default=15.0)
    ap.add_argument("--keep-organs", action="store_true", help="do not re-initialise the organs (adaptation only)")
    a = ap.parse_args()
    dev = a.dev if (a.dev != "mps" or torch.backends.mps.is_available()) else "cpu"
    tok = Tokenizer.from_file(a.tok)
    m, state = load_scan(a.base, tok, dev)
    V = tok.get_vocab_size()
    grown = grow_vocab(m, V)
    print("[regestate] vocabulary %s" % ("grew: " + ", ".join(grown) if grown else "unchanged"), file=sys.stderr)
    if not a.keep_organs:
        twin = fresh_twin(state, V).to(dev)
        tsd = twin.state_dict()
        msd = m.state_dict()
        reborn = []
        for k in msd:
            top = k.split(".")[0]
            if top in ORGANS and k in tsd and tsd[k].shape == msd[k].shape:
                msd[k] = tsd[k].clone()
                reborn.append(k)
        m.load_state_dict(msd, strict=False)
        del twin, tsd
        print("[regestate] organs reborn at birth init: %d tensors in %s"
              % (len(reborn), sorted({k.split('.')[0] for k in reborn})), file=sys.stderr)
    # the press LUT / eot ids were set by load_scan on the (grown) model
    conv = LaneConveyor(a.data, n_lanes=a.lanes)
    organ_p, embed_p, trunk_p = [], [], []
    for name, p in m.named_parameters():
        top = name.split(".")[0]
        if top in ORGANS:
            organ_p.append(p)
        elif top in ("embed", "head", "aux_head"):
            embed_p.append(p)
        else:
            trunk_p.append(p)
    opt = torch.optim.AdamW([{"params": organ_p, "lr": a.lr_organs},
                             {"params": embed_p, "lr": a.lr_embed},
                             {"params": trunk_p, "lr": a.lr_trunk}], weight_decay=0.0)
    m.train()
    st = m.init_state(a.lanes, dev)
    t0 = time.time()
    last_save = t0
    step = 0
    toks_seen = 0
    ce_ema = None
    vl_ema = None

    def save(tag):
        life = {"model": m.state_dict(), "cfg": state["cfg"], "step": state.get("step"),
                "nursery_steps": step, "regestate": {"steps": step, "tokens": toks_seen,
                                                     "ce_ema": ce_ema, "hours": (time.time() - t0) / 3600.0,
                                                     "organs_reborn": not a.keep_organs, "data": a.data},
                "life": {"facts": [], "study": [], "progress": {}, "surp_mu": None,
                         "pursuit": None, "pursuit_installment": False, "press_log": [],
                         "notice_peak_dyn": None, "budget_history": [], "day_n": 0,
                         "saliences": {}, "n_human_presses": 0}}
        torch.save(life, a.out)
        print("[regestate] saved (%s) -> %s at step %d, %.2fM tokens, ce %.3f"
              % (tag, a.out, step, toks_seen / 1e6, ce_ema or 0.0), file=sys.stderr)

    while (time.time() - t0) / 3600.0 < a.hours:
        x, y, events = conv.chunk(a.T)
        x, y = x.to(dev), y.to(dev)
        pl = press_levels_from_events(events, x.shape[0], a.T, dev)
        dl = None
        if getattr(m, "store_wipe", None):
            dl = [lane for lane, evs in enumerate(events) for (_p, k_, _d) in evs if k_ == "day"]
        opt.zero_grad(set_to_none=True)
        logits, st, _ = m(x, st, None, press_levels=pl, day_lanes=dl)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss = ce
        vl = m.pop_value_loss() if hasattr(m, "pop_value_loss") else None
        if vl is not None:
            loss = loss + vl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        # the state carries across chunks (a life is one stream); detach it
        def _det(s):
            if torch.is_tensor(s):
                return s.detach()
            if isinstance(s, dict):
                return {k: _det(v) for k, v in s.items()}
            if isinstance(s, (list, tuple)):
                r = [_det(v) for v in s]
                return tuple(r) if isinstance(s, tuple) else r
            return s
        st = _det(st)
        step += 1
        toks_seen += x.numel()
        c = float(ce.detach())
        ce_ema = c if ce_ema is None else 0.98 * ce_ema + 0.02 * c
        if vl is not None:
            v_ = float(vl.detach())
            vl_ema = v_ if vl_ema is None else 0.98 * vl_ema + 0.02 * v_
        if step % 25 == 0:
            el = time.time() - t0
            print("[regestate] step %d | %.2fM tok | %.0f tok/s | ce %.3f | value %.4f | %.1f min"
                  % (step, toks_seen / 1e6, toks_seen / max(1.0, el), ce_ema, vl_ema or 0.0, el / 60.0),
                  file=sys.stderr, flush=True)
        if (time.time() - last_save) / 60.0 >= a.save_every_min:
            save("periodic")
            last_save = time.time()
    save("final")


if __name__ == "__main__":
    main()
