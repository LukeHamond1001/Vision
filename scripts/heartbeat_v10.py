"""v10 heartbeat pack — the in-flight instruments and KILL criteria
(spec section 6). Runs against a checkpoint while the flash trains;
one invocation appends one row set to the output jsonl. The pod
wrapper greps for the KILL sentinel: kill, fix, relaunch — a caught
disease costs hours, not the run.

KILL constants are PRE-REGISTERED here, in code, before token one.
Growth-chart milestones are soft (warn, never kill).

Usage:
  python3 scripts/heartbeat_v10.py --ckpt CKPT.pt --data SHARD \
      --eval-data EVAL_SHARD [--step N] [--out results/hb_v10.jsonl]
Model config is read from the checkpoint's "cfg" dict when present,
else from CLI flags mirroring train()'s.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import UltraConveyor, load_tokenizer  # noqa
from iga.lm_hybrid import BAND6_CLOCKS, HybridLM                 # noqa
from iga import lm_judge as J                                    # noqa

KILL = {
    # CE vs its own running best: 3 consecutive heartbeat rises of
    # >= this relative margin = divergence (late-run watch: v9.4
    # peaked at 266k/488k — banking saves the artifact, this saves
    # the compute)
    "ce_rise_margin": 0.03, "ce_rise_rows": 3,
    # the band-education vital sign: cross-day recall advantage
    # must exist after this many tokens
    "gap_flat_after_tok": 1.5e9, "gap_min_acc_over_chance": 1.25,
    # value function forming: prophet AUC floor after half the run
    "prophet_auc_floor": 0.55, "prophet_after_frac": 0.5,
    # conviction disease: max false-belief mass on cast facts
    "incumbent_mass": 0.90, "incumbent_rows": 2,
    "confident_wrong_frac": 0.90,   # prevalence, not max (2026-08-21 #2)
    # babble: distinct-3gram ratio floor on fixed prompts
    "collapse_distinct3_floor": 0.35,
    # correction collateral (the A68-T S3 lesson): allies of
    # corrected facts must hold >= this fraction of the belief of
    # unrelated same-class facts
    "collateral_floor": 0.70,
    # the judge/plumbing integrity check on the tail
    "tail_audit_mismatch": 0.01,
    # ledger pruning must never outrun the sleeper
    "pruned_unharvested": 1,
}
GROWTH = {   # soft milestones (warn) keyed by flash fraction
    # fracs follow STAGES_V10_FLASH (.08/.27/.38/.27): infancy ends at
    # .08, childhood at .35, adolescence at .73 (2026-08-21 — the old
    # .10/.50/.90 keys were the pre-ratification stage table)
    "infancy_end": {"frac": 0.10, "distinct3": 0.5},
    "childhood_end": {"frac": 0.35, "binder_x_chance": 2.0},
    "adolescence_end": {"frac": 0.73, "prophet_auc": 0.55},
}
BINS = [(1, 256, "in-ctx"), (257, 2048, "short"),
        (2049, 8192, "b3"), (8193, 40000, "b4"),
        (40001, 400000, "b5"), (400001, 10 ** 12, "b6")]
PROMPTS = [
    "good morning .", "tell me about the harbor town .",
    "what should i cook tonight ?", "one morning later .",
    "explain how rivers form .", "can you help me plan a trip ?",
    "what color of coin was nedra kept ?", "the town waited .",
    "why is the sky blue ?", "how do i sort a list in python ?",
    "that day was done .", "tell me a short story .",
    "what did we talk about before ?", "give me three fruits .",
    "how are you today ?", "still . the wind moved that day .",
]


def load_model(a):
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    cfg = blob.get("cfg", {}) if isinstance(blob, dict) else {}
    tok = load_tokenizer(os.path.join(a.data, "tokenizer.json"))
    m = HybridLM(
        tok.get_vocab_size(), d=cfg.get("d", a.d),
        n_layers=cfg.get("n_layers", a.n_layers),
        n_heads=cfg.get("n_heads", 8), max_T=cfg.get("T", a.T),
        store="matrix", keyed=cfg.get("keyed", "logit"), norm_mix=True, aux_trunk=0.2,
        use_xl=False, gate_init=-2.0,
        clocks=cfg.get("clocks", BAND6_CLOCKS),
        band_widths=cfg.get("band_widths"),
        tie_embed=cfg.get("tie_embed", False),
        attn=cfg.get("attn", "abs"), qk_norm=cfg.get("qk_norm", False),
        mlp=cfg.get("mlp", "gelu"))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, tok, blob


@torch.no_grad()
def probe_ce_recall(m, eval_dir, T, chunks, lesion=None):
    conv = UltraConveyor(eval_dir, n_lanes=2)
    dev = next(m.parameters()).device
    m.lesioned = set(lesion or ())
    st = m.init_state(2, dev)
    ce_sum, ce_n = 0.0, 0
    acc = {name: [0, 0] for _, _, name in BINS}
    n_chunks = min(chunks, len(conv.tokens) // (2 * T) - 2)
    for _ in range(n_chunks):
        x, y, events = conv.chunk(T)
        x, y = x.to(dev), y.to(dev)
        lg, st, _ = m(x, st, None)
        m.pop_write_cost()
        m.pop_recon()
        st = m.detach_state(st)
        ce_sum += float(torch.nn.functional.cross_entropy(
            lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1)))
        ce_n += 1
        lp = torch.log_softmax(lg.float(), -1)
        for lane, evs in enumerate(events):
            for p, kind, d in evs:
                if kind != "probe" or p <= 0 \
                        or not d.get("answerable", True):
                    continue
                cand = [d["answer"]] + list(d["distractors"])
                best = max(cand, key=lambda i: float(lp[lane, p - 1, i]))
                for lo, hi, name in BINS:
                    if lo <= d["gap"] <= hi:
                        acc[name][0] += int(best == d["answer"])
                        acc[name][1] += 1
                        break
    m.lesioned = set()
    return (round(ce_sum / max(ce_n, 1), 4),
            {k: {"acc": round(h / n, 3) if n else None, "n": n}
             for k, (h, n) in acc.items()})


@torch.no_grad()
def probe_collapse(m, tok, T):
    """Greedy 64-token continuations of fixed prompts; the collapse
    signature is distinct-3gram contraction. 36000 AMENDMENT
    (2026-08-21): greedy distinct3 alone cannot tell entropy
    collapse (the disease) from greedy degeneration (a decoding
    artifact every undertrained LM shows — Holtzman 2019); the row
    now also carries distinct3 under temperature-1 sampling (fixed
    seed) and the mean next-token entropy along the greedy path.
    Collapse = sampled diversity AND entropy contracting together;
    greedy loops with healthy entropy are the decoder, not the
    model. Verdict logic unchanged (greedy floor, WARN)."""
    dev = next(m.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(0)

    def run(ptxt, sample):
        ids = tok.encode(ptxt).ids[-(T - 70):]
        st = m.init_state(1, dev)
        out, ents = [], []
        x = torch.tensor([ids], device=dev)
        for _ in range(64):
            lg, st, _ = m(x, st, None)
            m.pop_write_cost()
            m.pop_recon()
            st = m.detach_state(st)
            logits = lg[0, -1].float()
            if sample:
                pr = torch.softmax(logits, -1).cpu()
                nxt = int(torch.multinomial(pr, 1, generator=gen))
            else:
                lp = torch.log_softmax(logits, -1)
                ents.append(float(-(lp.exp() * lp).sum()))
                nxt = int(logits.argmax())
            out.append(nxt)
            x = torch.tensor([[nxt]], device=dev)
        grams = [tuple(out[i:i + 3]) for i in range(len(out) - 2)]
        return len(set(grams)) / len(grams), ents

    greedy, sampled, ents = [], [], []
    for ptxt in PROMPTS:
        r, e = run(ptxt, False)
        greedy.append(r)
        ents.extend(e)
        r, _ = run(ptxt, True)
        sampled.append(r)
    return {"distinct3": round(sum(greedy) / len(greedy), 4),
            "distinct3_sampled": round(sum(sampled) / len(sampled), 4),
            "entropy": round(sum(ents) / len(ents), 4)}


BOUNDARY_BUCKETS = ((0, 16), (16, 64), (64, 256), (256, 1024))


@torch.no_grad()
def probe_boundary(m, eval_dir, T, chunks=64):
    """The NECESSITY meter (2026-08-21, memory math): CE by position in
    the chunk, state carried across chunks, under base / thread off /
    stores off / both off. Attention sees one chunk; at its first
    tokens the cortex is blind to everything before the boundary unless
    an organ carries it. On the 78M raised life the deficit was 1.5
    nats on the first 16 tokens (5.06 vs 3.51 late in the chunk, ~0.18
    nats/token averaged = 5% of CE) and the organs recovered 0.4% of it.
    Reported as the deficit (early minus late CE) and each removal's
    delta at the early buckets: the organs are load-bearing at the
    boundary exactly when thread_off/store_off RAISE the early CE."""
    dev = next(m.parameters()).device
    conv = UltraConveyor(eval_dir, n_lanes=2)
    n_chunks = max(1, min(chunks, len(conv.tokens) // (2 * T) - 2))
    xs = []
    for _ in range(n_chunks):
        x, y, _ = conv.chunk(T)
        xs.append((x, y))
    out = {"n_chunks": n_chunks}
    conds = {"base": {}, "thread_off": {"mem_off": True},
             "store_off": {"store_read_off": True},
             "both_off": {"lesioned": set(m.bands)}}
    for name, flags in conds.items():
        m.lesioned, m.store_read_off, m.mem_off = set(), False, False
        for k, v in flags.items():
            setattr(m, k, v)
        st = m.init_state(2, dev)
        sums = torch.zeros(T, device=dev)
        try:
            for x, y in xs:
                x, y = x.to(dev), y.to(dev)
                lg, st, _ = m(x, st, None)
                m.pop_write_cost(); m.pop_recon()
                st = m.detach_state(st)
                ce = torch.nn.functional.cross_entropy(
                    lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1),
                    reduction="none").view(x.shape[0], T).mean(0)
                sums += ce
        finally:
            m.lesioned, m.store_read_off, m.mem_off = set(), False, False
        per = sums / n_chunks
        row = {f"{a}-{b}": round(float(per[a:min(b, T)].mean()), 4)
               for a, b in BOUNDARY_BUCKETS if a < T}
        row["late"] = round(float(per[T // 2:].mean()), 4)
        out[name] = row
    base = out["base"]
    out["deficit_0_16"] = round(base["0-16"] - base["late"], 4)
    out["deficit_16_64"] = round(base["16-64"] - base["late"], 4)
    out["deltas_0_64"] = {c: round((out[c]["0-16"] + out[c]["16-64"]) / 2
                                   - (base["0-16"] + base["16-64"]) / 2, 4)
                          for c in ("thread_off", "store_off", "both_off")}
    return out


@torch.no_grad()
def probe_store_health(m, tok):
    """Is the contextual memory ABLE to carry a fact? The v9.4 final
    battery found the store's learned key mix collapsed onto the
    immediately preceding token (qmix softmax [0.999, ...]) — a bigram
    cache that cannot retrieve "what colour was Mira's key". Read the
    key mix and the read path every beat: qmix softmax (top offsets,
    entropy), tok_u of the cast's entity words vs colours vs the
    corpus mean (are entities key-worthy?), read scale alpha and the
    read gate per band. Parameter reads only — no forward."""
    from iga.lm_gen import NAMES, OBJECTS, COLORS
    from iga.lm_data_life import EPISODIC_NAMES, EPISODIC_OBJECTS
    out = {}
    qm = getattr(m, "qmix", None)
    if qm is not None:
        w = torch.softmax(qm.detach().float(), 0)
        top = torch.topk(w, min(3, w.numel()))
        out["qmix_top"] = [[int(i), round(float(v), 4)]
                           for v, i in zip(top.values, top.indices)]
        out["qmix_entropy"] = round(float(-(w * (w + 1e-12).log()).sum()), 4)
        out["qmix_len"] = int(w.numel())
    tu = getattr(m, "tok_u", None)
    if tu is not None:
        tu = tu.detach().float()
        def mean_u(words):
            ids = []
            for wd in words:
                for form in (wd, " " + wd):
                    i = tok.token_to_id(form)
                    if i is not None and i < tu.numel():
                        ids.append(i); break
            return round(float(tu[ids].mean()), 4) if ids else None
        out["tok_u"] = {"mean": round(float(tu.mean()), 4),
                        "std": round(float(tu.std()), 4),
                        "names": mean_u(list(NAMES) + list(EPISODIC_NAMES)),
                        "objects": mean_u(list(OBJECTS) + list(EPISODIC_OBJECTS)),
                        "colors": mean_u(list(COLORS))}
    al = getattr(m, "alpha", None)
    if al is not None:
        out["alpha"] = {k: round(float(v.detach()), 4) for k, v in al.items()}
    rg = getattr(m, "read_gate", None)
    if rg is not None:
        out["read_gate"] = {k: round(float(torch.sigmoid(v.detach())), 4)
                            for k, v in rg.items()}
    return out


def probe_cast(m, tok, manifest):
    """Fresh-state bare asks over the manifest's cast facts:
    incumbent mass (max false-color prob where false beats true),
    class means (selectivity), and the correction-collateral guard
    (allies of corrected facts vs unrelated same-class facts)."""
    from iga.lm_data_ultrachat import COLORS
    cids = {c: tok.encode(" " + c).ids[0] for c in COLORS}
    rows = []
    for life in manifest["lives"][:4]:
        for f in life["cast"]:
            if not f["asks"]:
                continue
            q = (f"what color of {f['obj']} was {f['name']} kept ? "
                 f"<eot_human> the {f['obj']} was")
            dev = next(m.parameters()).device
            x = torch.tensor([tok.encode(q).ids], device=dev)
            lg, _, _ = m(x, m.init_state(1, dev), None)
            m.pop_write_cost()
            m.pop_recon()
            pr = torch.softmax(lg[0, -1].float(), -1)
            probs = {c: float(pr[i]) for c, i in cids.items()}
            rows.append({**f, "p_true": probs[f["col"]],
                         "p_max_false": max(v for c, v in probs.items()
                                            if c != f["col"])})
    if not rows:
        return {}
    incumbent = max((r["p_max_false"] for r in rows
                     if r["p_max_false"] > r["p_true"]), default=0.0)
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r["p_true"])
    cls_mean = {k: round(sum(v) / len(v), 4)
                for k, v in by_cls.items()}
    # A67's disease at population level: the FRACTION of cast facts
    # where a false color holds >= KILL["incumbent_mass"] AND beats
    # the truth. The max over ~96 facts saturates at ~1.0 for any
    # softmax model (the 30000 row: 0.992 with p_true class means of
    # .20-.36) — only prevalence can fall as corrections land.
    cw = sum(1 for r in rows
             if r["p_max_false"] > r["p_true"]
             and r["p_max_false"] >= KILL["incumbent_mass"])
    return {"n_facts": len(rows),
            "incumbent_mass": round(incumbent, 4),
            "confident_wrong_frac": round(cw / len(rows), 4),
            "class_mean_p_true": cls_mean}


def probe_tail_audit(data_dir, manifest, tok, sample=120):
    """Re-grade sampled tail press events with the manifest's FROZEN
    judge copy; press <-> threshold mismatches = grading/plumbing
    bug = kill."""
    import numpy as np
    toks = np.fromfile(os.path.join(data_dir, "tokens.bin"),
                       dtype=np.uint16)
    ev = [json.loads(l) for l in
          open(os.path.join(data_dir, "events.jsonl"))]
    life_len = manifest["life_len"]
    jm = manifest["judge"]
    eot_h = tok.token_to_id("<eot_human>")
    eot_m = tok.token_to_id("<eot_model>")
    marks = {tok.token_to_id(s): int(s[1:3].replace(">", ""))
             for s in ("<+1>", "<+2>")}
    # judge presses carry the stage that graded them (no boundary
    # guessing); the audit re-grades tail-stage presses only
    tail_btns = [e for e in ev if e["kind"] == "button"
                 and e["v"] > 0 and e.get("stage") == "tail"]
    step = max(1, len(tail_btns) // sample)
    mark_ids = set(marks) | {tok.token_to_id(s)
                             for s in ("<-1>", "<-2>")}
    bad = n = clipped = 0
    for e in tail_btns[::step]:
        pos = e["pos"]
        if int(toks[pos]) not in marks:
            bad += 1
            n += 1
            continue
        # walk back: press turn <- model turn <- human turn. The
        # window must out-reach the longest exchange; a turn whose
        # start lies beyond it is UNVERIFIABLE (clipped text is not
        # the text the builder graded), never a mismatch.
        seg = list(toks[max(0, pos - 2400):pos])
        try:
            m_end = len(seg) - 1 - seg[::-1].index(eot_m)
            h_end = m_end - 1 - seg[:m_end][::-1].index(eot_h)
            h_start = None
            for j in range(h_end - 1, -1, -1):
                if seg[j] in (eot_m, eot_h):
                    h_start = j + 1
                    break
            if h_start is None and pos - 2400 > 0:
                clipped += 1
                continue
            h_start = h_start or 0
            # a prior exchange's press mark can sit just inside the
            # turn boundary — the builder graded raw text, so strip
            while h_start < h_end and seg[h_start] in mark_ids:
                h_start += 1
            h_txt = tok.decode(seg[h_start:h_end])
            m_txt = tok.decode(seg[h_end + 1:m_end])
        except ValueError:
            continue
        q = J.grade_dialogue(h_txt, m_txt)
        stage = e.get("stage", "tail")
        want = 2 if q >= jm["q2"][stage] else (
            1 if q >= jm["q1"][stage] else 0)
        n += 1
        if want != e["v"]:
            bad += 1
    return {"n": n, "clipped": clipped,
            "mismatch": round(bad / n, 4) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--out", default="results/heartbeat_v10.jsonl")
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--tokens", type=int, default=-1)
    ap.add_argument("--total-tokens", type=int, default=0)
    ap.add_argument("--d", type=int, default=1280)
    ap.add_argument("--n-layers", type=int, default=20)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--skip-lesions", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    m, tok, blob = load_model(a)
    m.to(a.device)
    manifest = json.load(open(os.path.join(a.data, "manifest.json")))
    rows = []

    ce, recall = probe_ce_recall(m, a.eval_data, a.T, a.chunks)
    rows.append({"probe": "ce_recall", "ce": ce, "recall": recall})
    col = probe_collapse(m, tok, a.T)
    d3 = col["distinct3"]
    rows.append({"probe": "collapse", **col})
    try:
        rows.append({"probe": "store_health", **probe_store_health(m, tok)})
    except Exception as e:            # a diagnostic must never kill a beat
        rows.append({"probe": "store_health", "error": repr(e)[:120]})
    try:
        rows.append({"probe": "boundary",
                     **probe_boundary(m, a.eval_data, a.T,
                                      chunks=min(64, a.chunks))})
    except Exception as e:
        rows.append({"probe": "boundary", "error": repr(e)[:120]})
    cast = probe_cast(m, tok, manifest)
    # A67's incumbent DISEASE is conviction that resists CORRECTION —
    # it cannot exist before corrections have been experienced. Stamp
    # how many the stream has delivered (per-lane position vs the
    # manifest's correction_pos registry); the kill arms on this.
    lane_toks = a.tokens // max(manifest.get("n_lives", 1), 1)
    cast["n_corr_seen"] = sum(
        1 for p in manifest.get("correction_pos", []) if p <= lane_toks)
    rows.append({"probe": "cast", **cast})
    rows.append({"probe": "tail_audit",
                 **probe_tail_audit(a.data, manifest, tok)})
    if not a.skip_lesions:
        # like-for-like base: the first live batteries compared a
        # quarter-sample lesioned CE against the FULL-sample base —
        # the fixed walk-length offset (0.1804) swamped the real
        # deltas identically for all four bands
        small = max(a.chunks // 4, 50)
        ce_b, rec_b = probe_ce_recall(m, a.eval_data, a.T, small)

        def _racc(rec):
            return {k: v["acc"] for k, v in rec.items()}
        for k in sorted(m.bands):
            ce_l, rec_l = probe_ce_recall(m, a.eval_data, a.T, small,
                                          lesion=(k,))
            rows.append({"probe": f"lesion_b{k}",
                         "ce_delta": round(ce_l - ce_b, 4),
                         "recall": _racc(rec_l)})
        # the DEMO's organs (2026-08-21, user's three acts): Act 2 =
        # all bands off (in-the-moment), Act 3 = the STORE off with the
        # bands on (the slow thread without the facts). Both measured
        # on every lesion beat, CE and recall, against the same base.
        ce_l, rec_l = probe_ce_recall(m, a.eval_data, a.T, small,
                                      lesion=tuple(m.bands))
        rows.append({"probe": "lesion_bands_all",
                     "ce_delta": round(ce_l - ce_b, 4),
                     "recall": _racc(rec_l)})
        m.store_read_off = True
        try:
            ce_l, rec_l = probe_ce_recall(m, a.eval_data, a.T, small)
        finally:
            m.store_read_off = False
        rows.append({"probe": "lesion_store",
                     "ce_delta": round(ce_l - ce_b, 4),
                     "recall": _racc(rec_l)})
        # the slow THREAD alone (memory tokens zeroed, stores still
        # read): lesion_bands_all minus this row is what the band
        # STATES carry beyond their stores — Demo 1's own organ
        m.mem_off = True
        try:
            ce_l, rec_l = probe_ce_recall(m, a.eval_data, a.T, small)
        finally:
            m.mem_off = False
        rows.append({"probe": "lesion_thread",
                     "ce_delta": round(ce_l - ce_b, 4),
                     "recall": _racc(rec_l)})
        rows.append({"probe": "lesion_base", "ce": round(ce_b, 4),
                     "recall": _racc(rec_b)})
    if isinstance(blob, dict) and "sleeper" in blob:
        rows.append({"probe": "pruned_unharvested",
                     "n": blob["sleeper"].get("pruned_unharvested",
                                              0)})

    # verdicts
    hist = []
    if os.path.exists(a.out):
        hist = [json.loads(l) for l in open(a.out)]
    prev_ce = [r["ce"] for h in hist for r in h.get("rows", [])
               if r.get("probe") == "ce_recall"]
    verdict, why = "ok", []
    frac = (a.tokens / a.total_tokens) if (a.total_tokens > 0
                                           and a.tokens > 0) else 1.0
    # collapse = distinct-3gram CONTRACTION (the docstring's own
    # definition): below the floor AND lower than both previous rows.
    # Amendment #3 (2026-08-21, ledgered): greedy argmax decoding of a
    # healthy small model loops (0.165 -> 0.207 while eval CE FELL
    # 1.55 -> 1.25 — expansion, not collapse); a level-only floor at
    # frac>=0.10 would have killed the 36000 battery on an artifact.
    prev_d3 = [r["distinct3"] for h in hist[-2:] for r in h.get("rows", [])
               if r.get("probe") == "collapse"]
    contracting = len(prev_d3) >= 2 and all(d3 < x for x in prev_d3)
    # 2026-08-21 DEMOTION (user decision after three instrument false
    # positives, zero true diseases): judgment criteria — collapse,
    # conviction prevalence, CE divergence — are WARNS the watcher acts
    # on manually. Automatic stops remain only where minutes matter
    # and nothing is arguable: non-finite loss (trainer), tail-audit
    # mismatch (plumbing), dead instruments (blind).
    if d3 < KILL["collapse_distinct3_floor"] and frac >= 0.10 \
            and contracting:
        why = why + [f"WARN collapse: distinct3 {d3} contracting"]
    elif d3 < KILL["collapse_distinct3_floor"] and frac >= 0.10:
        why = why + [f"warn: distinct3 {d3} below floor, not contracting"]
    # growth chart (soft, 2026-08-21 wired): the childhood-end binder
    # milestone — in-ctx recall >= 2x chance (closed set of 5 -> 40%)
    # once the flash passes childhood's end. A missed milestone is
    # investigated, never killed on.
    g = GROWTH["childhood_end"]
    inctx = recall.get("in-ctx", {}).get("acc")
    if frac >= g["frac"] and inctx is not None \
            and inctx < g["binder_x_chance"] * 0.20:
        why = why + [f"growth: binder unarmed past childhood end "
                     f"(in-ctx {inctx} < {g['binder_x_chance'] * 0.20:.2f})"]
    # incumbent kill ARMS only once >=16 corrections have been lived
    # (2026-08-21 amendment, ledgered: the raw max-over-facts mass is
    # ~1.0 from birth for any softmax model — the 07:45 KILL fired on
    # a meter that saturates before its disease can exist; history
    # rows lacking n_corr_seen never count toward the kill)
    # amendment #2 (ledgered): the statistic is PREVALENCE of
    # confidently-wrong cast facts — >= 90% of the cast confidently
    # wrong after >= 16 lived corrections, two armed rows running, is
    # a broken correction pathway. The max-mass field stays as
    # telemetry; rows lacking confident_wrong_frac never count.
    cwf = cast.get("confident_wrong_frac")
    if cast.get("n_corr_seen", 0) >= 16 and cwf is not None and \
            cwf >= KILL["confident_wrong_frac"]:
        n_bad = 1 + sum(1 for h in hist[-KILL["incumbent_rows"]:]
                        for r in h.get("rows", [])
                        if r.get("probe") == "cast" and
                        r.get("n_corr_seen", 0) >= 16 and
                        r.get("confident_wrong_frac") is not None and
                        r["confident_wrong_frac"]
                        >= KILL["confident_wrong_frac"])
        if n_bad >= KILL["incumbent_rows"]:
            why = why + [f"WARN conviction: confident-wrong prevalence {cwf}"]
    ta = rows[3]
    if ta.get("mismatch") is not None and \
            ta["mismatch"] > KILL["tail_audit_mismatch"]:
        verdict, why = "KILL", why + [f"tail audit {ta['mismatch']}"]
    if prev_ce:
        best = min(prev_ce)
        rises = sum(1 for c in (prev_ce + [ce])[-KILL["ce_rise_rows"]:]
                    if c > best * (1 + KILL["ce_rise_margin"]))
        if rises >= KILL["ce_rise_rows"]:
            why = why + ["WARN ce divergence"]

    row = {"step": a.step, "tokens": a.tokens, "rows": rows,
           "verdict": verdict, "why": why}
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    if verdict == "KILL":
        print("KILL", flush=True)      # the pod wrapper's sentinel
        sys.exit(3)


if __name__ == "__main__":
    main()
