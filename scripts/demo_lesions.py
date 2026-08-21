"""The centerpiece demos — one being, two removals (docs/CENTERPIECE.md).

Act 1  the being as lived: facts planted in the served life at three
       gaps (in-ctx, short = one chunk back, long = b4+ chunks back)
       with neutral dialogue between plant and ask.
Demo 1 bands removed   (`lesion bands`): every band's memory token
       zeroed, stores still read — the slow thread off. Expected: the
       being stops carrying the day; exact facts the stores hold survive.
Demo 2 stores removed, bands on (`lesion store`): expected: exact
       recall falls at every gap, the thread of the day holds.
Both   (`lesion both`): tokens AND stores gone — the cortex alone with
       its chunk; the in-the-moment extreme, read beside the two.

Every reading is PAIRED on the same committed state: the probe is a
score-only forward on a state copy, the reply is greedy speech
generated on a copy (never appended), the removal is applied through
ServeSession.lesion_scope for that read only. Headline = speech recall
(the fact's color word in the greedy reply), one-sided paired sign
test base > removed per bin; p_true of the answer token is the
diagnostic beside it. Thread continuity = share of the day's names the
being speaks when asked who it has been talking about.

Usage:
  python3 scripts/demo_lesions.py <ckpt_dir> <outdir>
      [--d 512 --layers 6 --ckpt v94sp.pt --tokenizer tokenizer_press.json]
      [--n 20] [--bins in-ctx,short,b4] [--filler lines.txt] [--device cpu]
ckpt_dir holds the checkpoint (model state dict under "model", optional
"cfg") and the tokenizer. --n is per bin (the protocol's floor is 20).
Probe set and plants are seeded (SEED) and written to the evidence JSON
before any reading, so the set is frozen by the commit that adds it.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_gen import NAMES, OBJECTS, COLORS, ROOMS  # noqa: E402
from iga.lm_hybrid import HybridLM                   # noqa: E402
from iga.lm_serve import ServeSession, LESION_MODES  # noqa: E402
from iga.lm_sleep import state_copy                  # noqa: E402

SEED = 11
CONDITIONS = ("none", "bands", "store", "both")
# gap bins in CHUNKS of filler between plant and ask (T tokens each);
# in-ctx = same pending window, short = one chunk (b3), b4 = 8 chunks
# (16k tokens at T=2048), b5 = 64 chunks
BIN_CHUNKS = {"in-ctx": 0, "short": 1, "b4": 8, "b5": 64}
FILLER = [
    "the morning was quiet and the kettle took a while to boil .",
    "we walked to the market and the bread was still warm .",
    "it rained a little after lunch and then the sun came back .",
    "the neighbour's dog slept on the step all afternoon .",
    "someone left a window open and the curtains kept moving .",
    "the train was late again so we read on the platform .",
    "the garden needs weeding but the tomatoes are doing well .",
    "we fixed the squeaky hinge with a drop of oil .",
    "the library closes early on thursdays now .",
    "a letter came for the old tenant and we sent it on .",
]


# ---------- the probe set ----------
def build_facts(rng, n, bins):
    facts, used = [], set()
    for b in bins:
        for _ in range(n):
            while True:
                nm, ob = rng.choice(NAMES), rng.choice(OBJECTS)
                if (nm, ob) not in used:
                    break
            used.add((nm, ob))
            facts.append({"name": nm, "obj": ob,
                          "col": rng.choice(COLORS),
                          "room": rng.choice(ROOMS), "bin": b})
    return facts


def plant_text(f):
    return (f"by the way {f['name']} kept a {f['col']} {f['obj']} "
            f"in the {f['room']} .")


def ask_ids(tok, eot_h, f):
    """question ids (ending in the human eot) and the answer token."""
    stem = f"the {f['obj']} was"
    full = tok.encode(f"{stem} {f['col']} .").ids
    pre = tok.encode(stem).ids
    ans = full[len(pre)]
    q = tok.encode(f"what color of {f['obj']} was {f['name']} kept ?").ids
    return q + [eot_h], pre, ans


# ---------- paired readings on state copies ----------
@torch.no_grad()
def _tail(s, ids):
    ctx = s.pending[-(s.T - len(ids)):] if s.pending else []
    return ctx + ids


@torch.no_grad()
def p_true(s, ids, ans):
    x = torch.tensor([_tail(s, ids)], dtype=torch.long, device=s.device)
    with s.lesion_scope():
        lg, _, _ = s.m(x, state_copy(s.st), None)
    s.m.pop_write_cost(); s.m.pop_recon()
    return float(torch.softmax(lg[0, -1].float(), -1)[ans])


@torch.no_grad()
def speak_copy(s, ids, max_new=12):
    """Greedy speech on a COPY of the committed state with the question
    appended to the pending tail — nothing is appended to the life.
    The forward sees at most T tokens (the serve window)."""
    st = state_copy(s.st)
    seq = list(_tail(s, ids))
    out = []
    with s.lesion_scope():
        for _ in range(max_new):
            x = torch.tensor([seq[-s.T:]], dtype=torch.long,
                             device=s.device)
            lg, _, _ = s.m(x, state_copy(st), None)
            s.m.pop_write_cost(); s.m.pop_recon()
            t = int(lg[0, -1].float().argmax())
            if t == s.eot_m:
                break
            out.append(t)
            seq.append(t)
    return out


def said(tok, out_ids, word):
    text = tok.decode(out_ids, skip_special_tokens=False) \
        if hasattr(tok, "decode") else " ".join(map(str, out_ids))
    return word in text.split(), text


def read_fact(s, tok, f, max_new):
    q, pre, ans = ask_ids(tok, s.eot_h, f)
    row = {}
    for cond in CONDITIONS:
        s.lesion(cond)
        out = speak_copy(s, q, max_new=max_new)
        ok, text = said(tok, out, f["col"])
        row[cond] = {"said": ok, "text": text,
                     "p_true": round(p_true(s, q + pre, ans), 5)}
    s.lesion("none")
    return row


def read_thread(s, tok, names, max_new=24):
    q = tok.encode("who have we been talking about today ?").ids \
        + [s.eot_h]
    row = {}
    for cond in CONDITIONS:
        s.lesion(cond)
        out = speak_copy(s, q, max_new=max_new)
        text = tok.decode(out, skip_special_tokens=False) \
            if hasattr(tok, "decode") else " ".join(map(str, out))
        words = set(text.split())
        hit = sum(1 for nm in names if nm in words)
        row[cond] = {"names": hit, "of": len(names), "text": text}
    s.lesion("none")
    return row


# ---------- statistics ----------
def sign_test_one_sided(pairs):
    """pairs of (base_correct, removed_correct); H1: base > removed.
    Exact binomial on the discordant pairs; ties dropped."""
    plus = sum(1 for b, r in pairs if b and not r)
    minus = sum(1 for b, r in pairs if r and not b)
    n = plus + minus
    if n == 0:
        return 1.0, plus, minus
    p = sum(math.comb(n, k) for k in range(plus, n + 1)) / 2 ** n
    return p, plus, minus


def summarize(facts, reads):
    bins = sorted({f["bin"] for f in facts}, key=lambda b: BIN_CHUNKS[b])
    out = {}
    for b in bins:
        idx = [i for i, f in enumerate(facts) if f["bin"] == b]
        row = {"n": len(idx)}
        for cond in CONDITIONS:
            row[cond] = {
                "speech_recall": round(sum(
                    reads[i][cond]["said"] for i in idx) / len(idx), 4),
                "p_true_mean": round(sum(
                    reads[i][cond]["p_true"] for i in idx) / len(idx), 5)}
        for cond in ("bands", "store", "both"):
            p, plus, minus = sign_test_one_sided(
                [(reads[i]["none"]["said"], reads[i][cond]["said"])
                 for i in idx])
            row[f"sign_base_gt_{cond}"] = {"p": round(p, 5),
                                           "plus": plus, "minus": minus}
        out[b] = row
    return out


# ---------- the session ----------
def load_being(args):
    tok = load_tokenizer(os.path.join(args.ckpt_dir, args.tokenizer))
    blob = torch.load(os.path.join(args.ckpt_dir, args.ckpt),
                      map_location="cpu", weights_only=False)
    cfg = blob.get("cfg") or {}
    m = HybridLM(tok.get_vocab_size(),
                 d=int(cfg.get("d", args.d)),
                 n_layers=int(cfg.get("n_layers", args.layers)),
                 max_T=int(cfg.get("T", args.T)),
                 store=cfg.get("store", "matrix"),
                 keyed=cfg.get("keyed", "logit"),
                 norm_mix=cfg.get("norm_mix", True),
                 aux_trunk=cfg.get("aux_trunk", 0.2),
                 use_xl=cfg.get("use_xl", False),
                 gate_init=cfg.get("gate_init", -2.0),
                 clocks=cfg.get("clocks"),
                 attn=cfg.get("attn", "abs"),
                 qk_norm=cfg.get("qk_norm", False),
                 mlp=cfg.get("mlp", "gelu"))
    m.load_state_dict(blob["model"])
    return m, tok, blob


def run(args, m=None, tok=None, out_dir=None):
    rng = random.Random(SEED)
    bins = [b for b in args.bins.split(",") if b]
    for b in bins:
        assert b in BIN_CHUNKS, b
    if m is None:
        m, tok, _ = load_being(args)
    T = int(getattr(m, "max_T", args.T) or args.T)
    s = ServeSession(m, tok, T=T, device=args.device, temperature=0.0,
                     max_reply=args.max_new, seed=SEED,
                     log_path=os.path.join(out_dir, "session.jsonl")
                     if out_dir else None)
    filler = FILLER
    if args.filler and os.path.exists(args.filler):
        filler = [l.strip() for l in open(args.filler) if l.strip()]
    facts = build_facts(rng, args.n, bins)
    evidence = {"seed": SEED, "bins": bins, "n_per_bin": args.n,
                "T": T, "conditions": list(CONDITIONS),
                "facts": facts, "filler_lines": len(filler)}
    if out_dir:
        with open(os.path.join(out_dir, "probe_set.json"), "w") as f:
            json.dump(evidence, f, indent=1)   # frozen before readings

    fill_i = [0]

    def fill_until(pos_target):
        """neutral dialogue (filler lines cycle) until the committed
        stream reaches pos_target"""
        while s.n_committed < pos_target:
            s.user(filler[fill_i[0] % len(filler)])
            fill_i[0] += 1

    t0 = time.time()
    # plant in order of decreasing gap so every ask lands at its bin:
    # long-gap facts first, then filler, then shorter ones; the
    # in-ctx facts are asked inside the same pending window
    by_bin = {}
    for i, f in enumerate(facts):
        by_bin.setdefault(f["bin"], []).append(i)
    bins_desc = sorted(by_bin, key=lambda b: -BIN_CHUNKS[b])
    planted_names = []
    for j, b in enumerate(bins_desc):
        if BIN_CHUNKS[b] == 0:
            # in-ctx: the plants and the ask must share one window —
            # open a fresh one if they would spill into a commit
            room = sum(len(tok.encode(plant_text(facts[i])).ids) + 1
                       for i in by_bin[b]) + 64
            if len(s.pending) + room > T:
                s.flush()
        last = s.pos
        for i in by_bin[b]:
            facts[i]["planted_at"] = s.pos
            last = s.pos
            s.user(plant_text(facts[i]))
            if args.reply:
                s.reply()      # the being's own words enter its life
            planted_names.append(facts[i]["name"])
        nxt = BIN_CHUNKS[bins_desc[j + 1]] if j + 1 < len(bins_desc) else 0
        # the ask must find this bin's LAST plant >= its chunks back;
        # later bins add at least nxt chunks more, so fill only the rest
        need_pos = last + (BIN_CHUNKS[b] - nxt) * T
        if BIN_CHUNKS[b] > nxt:
            fill_until(need_pos)
    # every bin's gap is now satisfied: read all facts on the same state
    asked_at = {"pos": s.pos, "committed": s.n_committed}
    reads = [read_fact(s, tok, f, args.max_new) for f in facts]
    thread = read_thread(s, tok, sorted(set(planted_names)))
    summary = summarize(facts, reads)
    res = {"probe_set": evidence, "summary": summary, "thread": thread,
           "reads": [{**facts[i], **reads[i]} for i in range(len(facts))],
           "asked_at": asked_at, "pos": s.pos, "committed": s.n_committed,
           "elapsed_s": round(time.time() - t0, 1)}
    if out_dir:
        with open(os.path.join(out_dir, "demo_lesions.json"), "w") as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(out_dir, "transcripts.md"), "w") as f:
            f.write("# demo transcripts (greedy speech on state copies)\n\n")
            for r in res["reads"]:
                f.write(f"## {r['bin']}: what color of {r['obj']} was "
                        f"{r['name']} kept? (planted: {r['col']})\n")
                for cond in CONDITIONS:
                    f.write(f"- **{cond}**: {r[cond]['text']}  "
                            f"(p_true {r[cond]['p_true']})\n")
                f.write("\n")
            f.write("## who have we been talking about today?\n")
            for cond in CONDITIONS:
                f.write(f"- **{cond}** ({thread[cond]['names']}/"
                        f"{thread[cond]['of']} names): "
                        f"{thread[cond]['text']}\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ckpt", default="v94sp.pt")
    ap.add_argument("--tokenizer", default="tokenizer_press.json")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--bins", default="in-ctx,short,b4")
    ap.add_argument("--filler", default="")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reply", action="store_true",
                    help="the being replies to each plant (appended)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = run(a, out_dir=a.out_dir)
    print(json.dumps({"summary": res["summary"],
                      "thread": {c: (v["names"], v["of"])
                                 for c, v in res["thread"].items()},
                      "elapsed_s": res["elapsed_s"]}, indent=1))


if __name__ == "__main__":
    main()
