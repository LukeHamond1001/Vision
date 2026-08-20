"""v10 gate driver — the pre-registered debug experiments that decide
what ships in the 500M flash. All CPU, all $0, all on PREPARED
real-corpus mini-shards (the actual v10 data path).

Gates (thresholds pre-registered HERE, before any run):
  G1 binder     — in-ctx closed-set argmax accuracy >= 2x chance on
                  the eval shard (A69-R2's precondition law).
  G2 ordering   — bio beats shuffle-sessions-KEEP-WORLD control on
                  cross-day recall bins (n >= 50 probes).
  G3 pairs      — >= 3 ARM C pairs formed through training, all
                  w1 == tw, audit only_paid, arm C steps > 0.
  QUAD          — hybrid-vs-TransformerLM x bio-vs-shuffled: the
                  organ program's value measured (A50's twin at
                  debug scale). Reported, not gated (evidence row).
  A71 slowheavy — bands 5/6 at 2x width: ships iff eval CE beats
                  uniform AND recall-by-gap not worse (v5.0's
                  slowheavy LOST; the burden is on the organ).
  A73 splice    — splice=0.35: ships iff eval CE better AND
                  cross-day recall not worse than base.
  A74 novelty   — novelty=0.5: same shipping rule as A73.
  A75 tie       — tie_embed: ships iff eval CE better at FEWER
                  params AND G1 binder still passes (store laws
                  live in the binder).

Each arm = same seed, same shard, same steps; the knob is the only
variable. Evidence -> results/evidence/v10_gates.json.
Usage: python3 scripts/life_gate.py [steps] [d] [arms_csv|all]
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_life import (prepare_life, simple_only,          # noqa
                              smoltalk2_source, ultrachat_source)
from iga.lm_data_ultrachat import UltraConveyor, load_tokenizer   # noqa
from iga.lm_sleep import Sleeper                                  # noqa
from iga.lm_train import train                                    # noqa

D = int(sys.argv[2]) if len(sys.argv) > 2 else 128
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
ARMS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["all"])
LANES, T = 4, 256
BUDGET = 6_000_000           # 4 lives x 1.5M
SEED = 7
GATE = {"g1_chance_x": 2.0, "g2_min_probes": 50, "g3_min_pairs": 3}
EVID = "results/evidence/v10_gates.json"
RAW = "data/v10_raw"


def _sources(eval_mode=False):
    uc = ("data/ultrachat_heldout.jsonl" if eval_mode
          else "data/ultrachat_raw.jsonl")
    st2 = sorted(f"{RAW}/{f}" for f in os.listdir(RAW)
                 if f.endswith(".parquet") and "magpie" not in f)
    mag = sorted(f"{RAW}/{f}" for f in os.listdir(RAW)
                 if "magpie" in f)
    skip = 3000 if eval_mode else 0
    return {"default": ultrachat_source(uc),
            "tokenizer": ultrachat_source(uc),
            "infancy": simple_only(ultrachat_source(uc)),
            "childhood": ultrachat_source(uc, 40),
            "adolescence": smoltalk2_source(st2) if not eval_mode
            else ultrachat_source(uc, skip),
            "tail": smoltalk2_source(mag) if not eval_mode
            else ultrachat_source(uc, skip + 800)}


def build_shards():
    tok_ref = "data/life_gate_bio/tokenizer.json"
    if not os.path.exists("data/life_gate_bio/manifest.json"):
        prepare_life("data/life_gate_bio", BUDGET, LANES, seed=SEED,
                     world_seed=99, vocab=16384,
                     sources=_sources())
    if not os.path.exists("data/life_gate_ctrl/manifest.json"):
        prepare_life("data/life_gate_ctrl", BUDGET, LANES, seed=SEED,
                     world_seed=99, vocab=16384,
                     tokenizer_path=tok_ref, sources=_sources(),
                     shuffle_sessions=True)
    if not os.path.exists("data/life_gate_eval/manifest.json"):
        prepare_life("data/life_gate_eval", 1_500_000, 2, seed=101,
                     world_seed=999, vocab=16384,
                     tokenizer_path=tok_ref,
                     sources=_sources(eval_mode=True))


def train_arm(tag, data, arch="hybrid", **kw):
    t0 = time.time()
    sl = None
    if arch == "hybrid":
        sl = Sleeper(arm="C", every=16, block_chunks=2, seed=1,
                     **{k: kw.pop(k) for k in
                        ("splice", "novelty", "homeostasis")
                        if k in kw})
        sl.press_pay = (T, 64)
    model, drive, vocab, ce0, ce1 = train(
        d=D, lanes=LANES, T=T, steps=STEPS, seed=0, device="cpu",
        arch=arch, store="matrix", keyed="logit", norm_mix=True,
        aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
        log_every=max(STEPS // 4, 1), data=data, sleep=sl, **kw)
    meta = {"ce_first": ce0, "ce_last": ce1,
            "secs": round(time.time() - t0),
            "params": sum(p.numel() for p in model.parameters())}
    if sl is not None:
        meta["pairs"] = len(sl.pairs)
        meta["pair_law"] = all(p["w1"] == p["tw"] for p in sl.pairs)
        meta["sleep_steps"] = sl.steps_taken
        meta["only_paid"] = sl.audit()["only_paid"]
    print(f"[{tag}] {meta}", flush=True)
    return model, meta


BINS = [(1, 256, "in-ctx"), (257, 2048, "short"),
        (2049, 8192, "b3"), (8193, 40000, "b4"),
        (40001, 400000, "b5+")]


@torch.no_grad()
def evaluate(model, arch="hybrid"):
    """Stream the eval shard: CE + probe accuracy binned by TRUE
    gap (closed set = answer + distractors)."""
    conv = UltraConveyor("data/life_gate_eval", n_lanes=2)
    model.eval()
    st = model.init_state(2, "cpu")
    ce_sum, ce_n = 0.0, 0
    acc = {name: [0, 0] for _, _, name in BINS}
    chunks = min(2500, len(conv.tokens) // (2 * T) - 2)
    for _ in range(chunks):
        x, y, events = conv.chunk(T)
        out = model(x, st, None)
        logits, st = out[0], out[1]
        if hasattr(model, "pop_write_cost"):
            model.pop_write_cost()
            model.pop_recon()
        if hasattr(model, "detach_state"):
            st = model.detach_state(st)
        ce_sum += float(torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            y.reshape(-1)))
        ce_n += 1
        lp = torch.log_softmax(logits.float(), -1)
        for lane, evs in enumerate(events):
            for p, kind, d in evs:
                if kind != "probe" or p <= 0 \
                        or not d.get("answerable", True):
                    continue
                cand = [d["answer"]] + list(d["distractors"])
                row = lp[lane, p - 1]
                best = max(cand, key=lambda i: float(row[i]))
                for lo, hi, name in BINS:
                    if lo <= d["gap"] <= hi:
                        acc[name][0] += int(best == d["answer"])
                        acc[name][1] += 1
                        break
    return {"ce": round(ce_sum / max(ce_n, 1), 4),
            "recall": {k: {"acc": round(h / n, 3) if n else None,
                           "n": n} for k, (h, n) in acc.items()}}


def main():
    build_shards()
    out = {"steps": STEPS, "d": D, "arms": {}}
    want = (["bio", "ctrl", "a71", "a73", "a74", "a75",
             "tf_bio", "tf_ctrl"] if ARMS == ["all"] else ARMS)

    runs = {
        "bio": dict(data="data/life_gate_bio"),
        "ctrl": dict(data="data/life_gate_ctrl"),
        "a71": dict(data="data/life_gate_bio",
                    band_widths={5: 2 * D}),
        "a73": dict(data="data/life_gate_bio", splice=0.35),
        "a74": dict(data="data/life_gate_bio", novelty=0.5),
        "a75": dict(data="data/life_gate_bio", tie_embed=True),
        "a77": dict(data="data/life_gate_bio",
                    dream={"every_nights": 4, "n": 4, "max_new": 48,
                           "min_q": 0.55}),
        "tf_bio": dict(data="data/life_gate_bio",
                       arch="transformer"),
        "tf_ctrl": dict(data="data/life_gate_ctrl",
                        arch="transformer"),
    }
    for tag in want:
        kw = dict(runs[tag])
        arch = kw.pop("arch", "hybrid")
        model, meta = train_arm(tag, arch=arch, **kw)
        ev = evaluate(model, arch)
        out["arms"][tag] = {"train": meta, "eval": ev}
        print(f"[{tag}/eval] {json.dumps(ev)}", flush=True)
        del model

    a = out["arms"]
    verdicts = {}
    if "bio" in a:
        r = a["bio"]["eval"]["recall"]
        chance = 1 / 5              # closed set: answer + 4 distractors
        ic = r["in-ctx"]
        verdicts["G1_binder"] = bool(
            ic["n"] >= 20 and ic["acc"] is not None
            and ic["acc"] >= GATE["g1_chance_x"] * chance)
        verdicts["G3_pairs"] = bool(
            a["bio"]["train"].get("pairs", 0) >= GATE["g3_min_pairs"]
            and a["bio"]["train"].get("pair_law")
            and a["bio"]["train"].get("only_paid")
            and a["bio"]["train"].get("sleep_steps", 0) > 0)
    if "bio" in a and "ctrl" in a:
        cross = [k for k in ("b3", "b4", "b5+")]
        nb = sum(a["bio"]["eval"]["recall"][k]["n"] for k in cross)
        bio_h = sum((a["bio"]["eval"]["recall"][k]["acc"] or 0)
                    * a["bio"]["eval"]["recall"][k]["n"]
                    for k in cross)
        ctl_h = sum((a["ctrl"]["eval"]["recall"][k]["acc"] or 0)
                    * a["ctrl"]["eval"]["recall"][k]["n"]
                    for k in cross)
        verdicts["G2_ordering"] = bool(
            nb >= GATE["g2_min_probes"] and bio_h > ctl_h)
        verdicts["G2_detail"] = {"bio_hits": round(bio_h, 1),
                                 "ctrl_hits": round(ctl_h, 1),
                                 "n": nb}
    # shipping rules, amended after the first run exposed two
    # holes: (1) CE "better" must clear a 1% NOISE FLOOR — seed
    # variance is unmeasured (A35/A37) and the single-run
    # attribution ban applies to organ deltas too; (2) the
    # cross-day bin (b5+) is IN the regression check — it is the
    # architecture's load-bearing bin, not an optional extra.
    for organ in ("a71", "a73", "a74", "a75"):
        if organ in a and "bio" in a:
            ce_o, ce_b = a[organ]["eval"]["ce"], a["bio"]["eval"]["ce"]
            ce_win = ce_o < ce_b * 0.99
            rec_o = a[organ]["eval"]["recall"]
            rec_b = a["bio"]["eval"]["recall"]
            rec_ok = all(
                (rec_o[k]["acc"] or 0) >= (rec_b[k]["acc"] or 0) - .05
                for k in ("in-ctx", "b3", "b4", "b5+"))
            verdicts[f"{organ.upper()}_ships"] = bool(ce_win and
                                                      rec_ok)
            verdicts[f"{organ.upper()}_detail"] = {
                "ce_delta_pct": round((ce_o / ce_b - 1) * 100, 2),
                "rec_ok": rec_ok}
    if "tf_bio" in a and "bio" in a:
        verdicts["QUAD"] = {
            "hybrid_bio_ce": a["bio"]["eval"]["ce"],
            "tf_bio_ce": a["tf_bio"]["eval"]["ce"],
            "hybrid_ctrl_ce": a.get("ctrl", {}).get("eval",
                                                    {}).get("ce"),
            "tf_ctrl_ce": a.get("tf_ctrl", {}).get("eval",
                                                   {}).get("ce")}
    out["verdicts"] = verdicts
    os.makedirs(os.path.dirname(EVID), exist_ok=True)
    with open(EVID, "w") as f:
        json.dump(out, f, indent=1)
    print("\n=== VERDICTS ===", flush=True)
    for k, v in verdicts.items():
        print(f"  {k}: {v}", flush=True)
    print(f"saved {EVID}", flush=True)


if __name__ == "__main__":
    main()
