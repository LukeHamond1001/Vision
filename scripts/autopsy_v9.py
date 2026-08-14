"""A54 v9 autopsy — the pre-registered instrument, committed during
the training window (A54b) so every scoring rule exists before any
v9 number does.

TM-v9-clean, the out-of-window subset. Fixed by construction and
IDENTICAL under every serving this file supports:
  nat probes where (a) the answer token does not occur in the 2048
  tokens before the use site, (b) def and use fall in different
  T=2048 chunks of the lane stream (attention cannot reach the def
  even in principle — only a store can carry it), (c) the def lies
  inside the probe's lane segment (the conveyor's answerable rule),
  (d) the use sits past the 6-chunk warmup, before the last fully
  served T=2048 chunk, and not on a 1024 boundary (a local-0
  position has no prior logit at either serving).
The subset hash is printed so the v9 run and the r5_best rescore
can be proven to have scored the same instrument.

Battery (--modes): organs | ce | table | tmclean | completion
  ce          criterion 2 — full-vs-lesion CE, bar >=2.8% rel
              (r5_best pinned blind: 2.6570 / 2.7609 = +3.76%)
  tmclean     full-shard per-probe full-vs-lesion + sign test
  completion  criterion 3 — sub-tokens 2+ of TM-v9-clean
              identifier runs (LCP of def/use token sequences),
              positions found by lane-cursor arithmetic (first
              pass is wrap-free), so no completion position is
              missed; scoring rule unchanged from A53.
  table       200-chunk standard table at serve T (criterion 4)

Comparability (A54b): headline = native serve T on TM-v9-clean
(v9 at --serve-T 2048; r5_best rescored at its native 1024 on the
SAME subset); secondary = v9 at --serve-T 1024. --max-T must match
the checkpoint's training max_T (KD store shapes follow it).
"""
import argparse
import hashlib
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import numpy as np
import torch

from iga.lm_hybrid import HybridLM
from iga.lm_data_ultrachat import load_tokenizer, UltraConveyor

T_CLEAN = 2048
LANES = 2
WARM = 6


def build_subset(toks, events):
    seg = len(toks) // LANES
    hi_clean = (seg // T_CLEAN -
                (1 if seg % T_CLEAN == 0 else 0)) * T_CLEAN
    tm_clean, comp = set(), {}
    for e in events:
        if e.get("kind") != "probe" or not e.get("nat"):
            continue
        pos, gap = e["pos"], e["gap"]
        dpos = pos - gap
        lane = min(pos // seg, LANES - 1)
        lo = lane * seg
        rel = pos - lo
        if dpos < lo:
            continue                      # def outside lane segment
        if rel // T_CLEAN == (dpos - lo) // T_CLEAN:
            continue                      # same T=2048 chunk
        if e["answer"] in toks[max(0, pos - T_CLEAN):pos]:
            continue                      # answer visible in window
        if rel < WARM * T_CLEAN or rel + 8 >= hi_clean:
            continue                      # warmup / unserved tail
        if rel % 1024 == 0:
            continue                      # local-0 at some serving
        tm_clean.add(pos)
        L = 0                             # LCP of def and use runs
        while L < 8 and pos + L < len(toks) and \
                toks[dpos + L] == toks[pos + L]:
            L += 1
        if L >= 2:
            comp[pos] = [pos + j for j in range(1, L)
                         if (pos + j - lo) % 1024 != 0]
    h = hashlib.md5(",".join(map(str, sorted(tm_clean)))
                    .encode()).hexdigest()[:12]
    n_cpos = sum(len(v) for v in comp.values())
    print(f"TM-v9-clean {len(tm_clean)}  completion ids "
          f"{len(comp)} ({n_cpos} positions)  hash {h}", flush=True)
    return tm_clean, comp, seg


def n_full_pass(seg, T):
    return seg // T - (1 if seg % T == 0 else 0)


@torch.no_grad()
def mean_ce(model, shard, T, n_chunks=400):
    conv = UltraConveyor(shard, n_lanes=LANES)
    st = model.init_state(LANES, "cpu")
    tot = 0.0
    for ci in range(n_chunks):
        x, y, _ = conv.chunk(T)
        logits, st, _ = model(x, st, None)
        tot += float(torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            y.reshape(-1)))
        if ci % 100 == 0:
            print(f"  ce chunk {ci}: running {tot/(ci+1):.4f}",
                  flush=True)
    return tot / n_chunks


@torch.no_grad()
def table200(model, shard, T, tm_clean, n_chunks=200):
    conv = UltraConveyor(shard, n_lanes=LANES)
    st = model.init_state(LANES, "cpu")
    stats = {}
    for ci in range(WARM + n_chunks):
        x, y, events = conv.chunk(T)
        logits, st, _ = model(x, st, None)
        if ci < WARM:
            continue
        logp = torch.log_softmax(logits.float(), dim=-1)
        for lane, evl in enumerate(events):
            for p, kind, d in evl:
                if kind != "probe" or p <= 0 or \
                        not d.get("answerable", True):
                    continue
                if d.get("nat"):
                    key = "TM" if d["pos"] in tm_clean else (
                        "nat<1k" if d["gap"] < 1024 else "nat-other")
                else:
                    lo = lane * conv.seg
                    pos, plant = d["pos"], d["pos"] - d["gap"]
                    key = "pl-same" if (pos - lo) // T == \
                        (plant - lo) // T else (
                        "pl-strad" if d["gap"] <= T else "pl-cross")
                s = stats.setdefault(key, [0.0, 0, 0])
                s[0] += float(logp[lane, p - 1, d["answer"]].exp())
                s[1] += int(int(logits[lane, p - 1].argmax())
                            == d["answer"])
                s[2] += 1
    return {k: (round(v[0] / v[2], 3), round(v[1] / v[2], 2), v[2])
            for k, v in sorted(stats.items())}


@torch.no_grad()
def tmclean_fullshard(model, shard, T, tm_clean):
    conv = UltraConveyor(shard, n_lanes=LANES)
    st = model.init_state(LANES, "cpu")
    per = {}
    for ci in range(n_full_pass(conv.seg, T)):
        x, y, events = conv.chunk(T)
        logits, st, _ = model(x, st, None)
        if ci < WARM:
            continue
        lg = None
        for lane, evl in enumerate(events):
            for p, kind, d in evl:
                if kind != "probe" or p <= 0 or \
                        not d.get("answerable", True) or \
                        not d.get("nat") or d["pos"] not in tm_clean:
                    continue
                if lg is None:
                    lg = torch.log_softmax(logits.float(), dim=-1)
                per[d["pos"]] = (
                    float(lg[lane, p - 1, d["answer"]].exp()),
                    int(int(logits[lane, p - 1].argmax())
                        == d["answer"]))
    return per


@torch.no_grad()
def completion(model, shard, T, comp, toks):
    want = {}
    for pos, rng in comp.items():
        for p in rng:
            want[p] = int(toks[p])
    conv = UltraConveyor(shard, n_lanes=LANES)
    st = model.init_state(LANES, "cpu")
    tot = [0.0, 0, 0]
    for ci in range(n_full_pass(conv.seg, T)):
        x, y, events = conv.chunk(T)
        logits, st, _ = model(x, st, None)
        if ci < WARM:
            continue
        for lane in range(LANES):
            base = lane * conv.seg + ci * T   # first pass: wrap-free
            if ci < WARM + 4:                 # layout self-check
                for pl, kind, d in events[lane]:
                    if kind == "probe":
                        assert d["pos"] - pl == base, \
                            f"anchor mismatch {d['pos']-pl}!={base}"
                        break
            for loc in range(1, T):
                ab = base + loc
                if ab in want:
                    lg = logits[lane, loc - 1].float()
                    pr = torch.softmax(lg, -1)[want[ab]]
                    tot[0] += float(pr)
                    tot[1] += int(int(lg.argmax()) == want[ab])
                    tot[2] += 1
    return (round(tot[0] / max(tot[2], 1), 4),
            round(tot[1] / max(tot[2], 1), 4), tot[2])


def sign_test(per_full, per_les):
    deltas = [per_full[k][0] - per_les[k][0]
              for k in per_full if k in per_les]
    pos = sum(d > 1e-9 for d in deltas)
    neg = sum(d < -1e-9 for d in deltas)
    n = pos + neg
    if not n:
        return
    pval = sum(comb(n, i) for i in range(pos, n + 1)) / 2 ** n
    med = sorted(deltas)[len(deltas) // 2]
    print(f"SIGN TEST: +{pos}/-{neg} (n={n}), one-sided "
          f"p={pval:.4f}, median {med:+.5f}", flush=True)


def summarize(per, label):
    if per:
        print(f"{label}: p={sum(v[0] for v in per.values())/len(per):.4f}"
              f" top1={sum(v[1] for v in per.values())/len(per):.4f}"
              f" n={len(per)}", flush=True)


def organs(m, tok):
    if hasattr(m, "alpha"):
        print("ALPHA:", {k: round(float(m.alpha[k].detach()), 4)
                         for k in m.alpha.keys()}, flush=True)
    print("betas:", {k: round(float(torch.sigmoid(
        m.stores[k].beta.detach())), 4) for k in m.stores.keys()},
        flush=True)
    print("qmix softmax:", [round(float(x), 3) for x in
                            torch.softmax(m.qmix.detach(), dim=0)],
          flush=True)
    u = m.tok_u.detach()
    print(f"tok_u: mean {float(u.mean()):.4f} std "
          f"{float(u.std()):.4f} min {float(u.min()):.3f} "
          f"max {float(u.max()):.3f}", flush=True)
    dec = lambda ids: [repr(tok.decode([i])) for i in ids]
    print("tok_u TOP-40:", ", ".join(
        dec(torch.topk(u, 40).indices.tolist())), flush=True)
    print("tok_u BOT-40:", ", ".join(
        dec(torch.topk(-u, 40).indices.tolist())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--max-T", type=int, default=2048)
    ap.add_argument("--serve-T", type=int, default=2048)
    ap.add_argument("--modes", default="organs,ce,table,tmclean,"
                                       "completion")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    assert a.serve_T <= a.max_T
    toks = np.fromfile(os.path.join(a.shard, "tokens.bin"),
                       dtype=np.uint16)
    events = [json.loads(l) for l in
              open(os.path.join(a.shard, "events.jsonl"))]
    tm_clean, comp, _ = build_subset(toks, events)
    if a.modes == "subset":
        return
    tok = load_tokenizer(os.path.join(a.shard, "tokenizer.json"))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = HybridLM(tok.get_vocab_size(), d=a.d, max_T=a.max_T,
                 store="matrix", use_xl=False, gate_init=-2.0,
                 keyed="logit")
    m.load_state_dict(ck["model"])
    m.eval()
    name = a.label or os.path.basename(a.ckpt)
    print(f"== {name} step {ck.get('step')} serve-T {a.serve_T}",
          flush=True)
    modes = set(a.modes.split(","))
    if "organs" in modes:
        organs(m, tok)
    if "ce" in modes:
        m.lesioned = set()
        full = mean_ce(m, a.shard, a.serve_T)
        print(f"CE full      : {full:.4f}", flush=True)
        m.lesioned = {3, 4, 5}
        les = mean_ce(m, a.shard, a.serve_T)
        print(f"CE lesionALL : {les:.4f}", flush=True)
        print(f"CE advantage : {les-full:+.4f} "
              f"({(les-full)/les*100:+.2f}% rel; bar +2.80%)",
              flush=True)
    if "table" in modes:
        m.lesioned = set()
        print("t200 full     :",
              table200(m, a.shard, a.serve_T, tm_clean), flush=True)
        m.lesioned = {3, 4, 5}
        print("t200 lesionALL:",
              table200(m, a.shard, a.serve_T, tm_clean), flush=True)
    if "tmclean" in modes:
        m.lesioned = set()
        per_full = tmclean_fullshard(m, a.shard, a.serve_T, tm_clean)
        summarize(per_full, "TMclean full     ")
        m.lesioned = {3, 4, 5}
        per_les = tmclean_fullshard(m, a.shard, a.serve_T, tm_clean)
        summarize(per_les, "TMclean lesionALL")
        sign_test(per_full, per_les)
    if "completion" in modes:
        m.lesioned = set()
        print("COMP full     :",
              completion(m, a.shard, a.serve_T, comp, toks),
              flush=True)
        m.lesioned = {3, 4, 5}
        print("COMP lesionALL:",
              completion(m, a.shard, a.serve_T, comp, toks),
              flush=True)
    print("AUTOPSY DONE", flush=True)


if __name__ == "__main__":
    main()
