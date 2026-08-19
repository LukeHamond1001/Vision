"""v5.0 trainer — conveyor -> bands -> drive, one loop, checkpointed.

Loss = next-token CE
     + FID_W * mean(1 - band next-window fidelity)   (predictor training)
     + drive pays (in-graph, negative: arriving is rewarded)

v0 scope notes (honest): episode recall holds settle at scene end
rather than <eot_model> (the scope machinery exists; turn settlement
is a later wiring). Band-fidelity maintain-holds pay ledger-only —
their gradient pressure already lives in the fidelity loss, and
double-paying the same quantity would be minting.

Usage:  python -m iga.lm_train smoke
        python -m iga.lm_train run --d 256 --steps 20000 --chunk 512
"""

import argparse
import math
import os
import time
import torch


def atomic_save(obj, path):
    """A54 audit (C2): checkpoints are the crash-recovery artifact —
    a kill mid-write must never leave a truncated file where the
    last good one was."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

from .lm_conveyor import Vocab, Conveyor, splits
from .lm_bands import BandLM
from .lm_drive import Drive

FID_W = 0.1
WRITE_W = 0.01   # A24: gentle pressure to keep band writes sparse
RECON_W = 0.05   # A28: write-fidelity — read back what was just
                 # written; the in-chunk gradient path for the write
                 # head (cross-chunk detachment blocks the other one)


def cap_gate_norms(model, max_norm=1.0):
    """A42: bound the per-position gate heads. v6.2's only
    monotonically growing quantity was these weight norms
    (0->1.72), and held-out binding collapsed late as they
    sharpened — the crowding-out law's late-onset clause. The cap
    keeps position-selectivity while bounding how strong the read
    path can ever get."""
    if not hasattr(model, "read_gate_pos"):
        return
    with torch.no_grad():
        for lin in model.read_gate_pos.values():
            n = float(lin.weight.norm())
            if n > max_norm:
                lin.weight.mul_(max_norm / n)


def process_chunk(model, drive, conveyor, T, device, opt=None,
                  bf16=False):
    x, y, events = conveyor.chunk(T)
    x, y = x.to(device), y.to(device)
    # A49: bf16 autocast covers forward + loss build; backward and
    # opt run outside; states are re-anchored to fp32 at the chunk
    # boundary so precision-sensitive accumulators (band vectors,
    # matrix decay products) never compound in bf16 storage
    dev_type = "cuda" if "cuda" in str(device) else "cpu"
    import contextlib
    ac = (torch.autocast(dev_type, dtype=torch.bfloat16)
          if bf16 else contextlib.nullcontext())
    ce_blind = None
    if getattr(model, "gate_mode", "") == "entropy" and \
            model.training:
        # A51 R2 metamemory: blind pass on a THROWAWAY state copy
        # (bands tick once per chunk — on the real pass only). Blind
        # CE trains every chunk so reads can never hollow the base;
        # its entropy decides where reads flow on the real pass.
        def _st_copy(st):
            out = {}
            for k, v in st.items():
                if isinstance(v, dict):
                    out[k] = {kk: (vv.detach().clone()
                                   if torch.is_tensor(vv) else vv)
                              for kk, vv in v.items()}
                elif torch.is_tensor(v):
                    out[k] = v.detach().clone()
                elif isinstance(v, list):
                    out[k] = [t.detach().clone()
                              if torch.is_tensor(t) else t for t in v]
                else:
                    out[k] = v
            return out
        real_st = model._st
        model.entropy_gate = None
        model._st = _st_copy(real_st)   # detached throwaway (A38's
        # write-credit graph on M must not leak into the blind pass)
        with ac:
            logits_a, _, _ = model(x, model._st, None)
        model.pop_write_cost(); model.pop_recon()   # discard pass-A aux
        ce_blind = torch.nn.functional.cross_entropy(
            logits_a.float().reshape(-1, model.vocab_size),
            y.reshape(-1))
        lp = torch.log_softmax(logits_a.float().detach(), dim=-1)
        H = -(lp.exp() * lp).sum(-1)                # [B, T]
        del lp
        model.entropy_gate = torch.sigmoid(
            model.ent_a * (H - model.ent_tau))
        if opt is not None:
            # free the blind graph BEFORE the real pass (16GB cards):
            # gradients accumulate; the opt block must not re-zero
            opt.zero_grad()
            ce_blind.backward()
        ce_blind = ce_blind.detach()
        del logits_a
        model._st = real_st
    with ac:
        logits, model._st, ticks = model(x, model._st, None)
    if getattr(model, "gate_mode", "") == "entropy":
        model.entropy_gate = None
    ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, model.vocab_size), y.reshape(-1))
    losses = [ce]
    # A58b (R8b): pay-the-trunk — aux CE through the SEPARATE aux
    # head on the final hidden, so trunk blocks keep earning
    # gradient on store-covered chunks while the production head
    # stays uncompromised (only set when the bonus actually fired)
    aux_h = getattr(model, "_aux_hidden", None)
    if aux_h is not None and getattr(model, "aux_trunk", 0.0) > 0:
        aux_lg = model.aux_head(model.lnf(aux_h))
        losses.append(model.aux_trunk *
                      torch.nn.functional.cross_entropy(
                          aux_lg.float().reshape(-1,
                                                 model.vocab_size),
                          y.reshape(-1)))
        model._aux_hidden = None
    fid_terms = []
    for k in range(1, len(ticks)):
        for _, fid in ticks[k]:
            if fid.requires_grad:
                fid_terms.append((1 - fid).mean())
            drive.tick_fid(k, fid)
    if fid_terms:
        losses.append(FID_W * torch.stack(fid_terms).mean())
    if hasattr(model, "pop_write_cost"):
        wc = model.pop_write_cost()
        if wc is not None:
            losses.append(WRITE_W * wc)
    if hasattr(model, "pop_recon"):
        rc = model.pop_recon()
        if rc is not None:
            losses.append(RECON_W * rc)
    logp = torch.log_softmax(logits.float(), dim=-1)
    for lane, evs in enumerate(events):
        for p, kind, d in sorted(evs, key=lambda e: e[0]):
            if kind == "probe" and p > 0:
                prob = logp[lane, p - 1, d["answer"]].exp()
                if d.get("distractors"):
                    # binding-margin channel (v5.1): pay only for
                    # beating the other colors in play — prior- and
                    # recency-tracking are worth exactly zero
                    pd = torch.stack([logp[lane, p - 1, i].exp()
                                      for i in d["distractors"]]).max()
                    reading = torch.clamp(prob - pd, min=0.0)
                else:
                    reading = prob
                drive.probe(lane, reading, d["gap"])
            elif kind == "earned":
                drive.earned(lane, d["ok"])
    drive.step_t += T
    drive.sweep(losses)
    loss = torch.stack([(l if l.dim() == 0 else l.mean()).float()
                        for l in losses]).sum()
    if opt is not None:
        if ce_blind is None:
            opt.zero_grad()          # entropy branch already zeroed
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cap_gate_norms(model)
    model._st = model.detach_state(model._st)
    if bf16:
        st = model._st
        for key in ("h", "acc", "M"):
            if key in st:
                st[key] = {k: v.float() for k, v in st[key].items()}
        st["pend"] = {k: (v.float() if v is not None else None)
                      for k, v in st.get("pend", {}).items()}
    # reading tensors retained across the boundary leave the graph here:
    # pay gradients flow through settlement-chunk readings only
    drive.detach_readings()
    return float(ce.detach()), float(loss.detach())


@torch.no_grad()
def holdout_probe(model, pe, T, device, warm=12, score=8):
    """A30/A32: live held-out mini-eval. v5.8 redesign: FRESH state
    per call + fixed warmup — v5.7's persistent probe state (written
    by an ensemble of past weights) reported binding that no fixed-
    weight offline measurement could reproduce at any state depth;
    the probe now mirrors the registered eval's conditions."""
    model.eval()
    seg = pe["conv"].seg
    st = model.init_state(pe["conv"].n_lanes, device)
    for ci in range(warm + score):
        x, y, events = pe["conv"].chunk(T)
        x = x.to(device)
        logits, st, _ = model(x, st, None)
        if ci < warm:
            continue
        logp = torch.log_softmax(logits, dim=-1)
        for lane, evs in enumerate(events):
            lo = lane * seg
            for p, kind, dd in evs:
                if kind != "probe" or p <= 0 or \
                        not dd.get("answerable", True):
                    continue
                pos, plant = dd["pos"], dd["pos"] - dd["gap"]
                if (pos - lo) // T == (plant - lo) // T:
                    key = "same"
                elif dd["gap"] <= T:
                    key = "straddle"
                else:
                    key = "cross"
                s = pe["agg"].setdefault(key, [0.0, 0, 0])
                s[0] += float(logp[lane, p - 1, dd["answer"]].exp())
                s[1] += int(int(logits[lane, p - 1].argmax())
                            == dd["answer"])
                s[2] += 1
    model.train()
    # A54 audit (H3): UNROUNDED means — the banking channel
    # reconstitutes sums from these, and rounding error scales with
    # cumulative n (±0.1-0.2 by 366k steps, incl. the >1.0
    # overshoots seen at R2). Round at print sites only.
    return {k: [v[0] / v[2], v[1] / v[2], v[2]]
            for k, v in sorted(pe["agg"].items()) if v[2]}


def train(d=64, lanes=4, T=256, steps=40, seed=0, device="cpu",
          ckpt=None, log_every=10, data=None, talk="dense", widths=None,
          compile_model=False, constants=None, arch="bands",
          resume=None, offset_frac=0.0, store="vector", eval_data=None,
          use_xl=True, gate_init=-4.0, read_drop=0.5,
          read_drop_end=None, gate_mode="scalar", lr=3e-4,
          bf16=False, lam=0.25, keyed=None, lr_decay="none",
          lr_total_steps=None, norm_mix=False, aux_trunk=0.0,
          hold_cap=None, sleep=None):
    """resume (A26): path to a checkpoint — model + optimizer + drive
    EMAs/records/minted/vetoes continue; step numbering continues.
    sleep (A62): a lm_sleep.Sleeper — wake/sleep alternation; the
    wake loop is untouched and sleep=None is the certified v9.4
    trainer bit-exactly (L2).
    offset_frac: start each conveyor lane this far into its segment
    (continuation rides the unseen tail; one-epoch law holds). Open
    holds and band states are NOT in checkpoints — holds re-propose
    within a sweep, slow states rebuild within a few clocks
    (ledgered)."""
    if "cuda" in str(device):
        torch.set_float32_matmul_precision("high")  # TF32 (A12)
    torch.manual_seed(seed)  # reproducible init (A14)
    drive = Drive(n_lanes=lanes, lam=lam, seed=seed, constants=constants,
                  hold_cap=hold_cap,
                  imagination_absent=(arch == "transformer"),
                  absent_bands={1, 2} if arch == "hybrid" else ())
    if data:  # prepared real-data shard (A8): UltraChat conveyor
        from .lm_data_ultrachat import UltraConveyor, load_tokenizer
        import os
        conveyor = UltraConveyor(data, n_lanes=lanes,
                                 offset_frac=offset_frac)
        tok = load_tokenizer(os.path.join(data, "tokenizer.json"))
        vocab, vocab_size = tok, tok.get_vocab_size()
    else:
        vocab = Vocab()
        vocab_size = len(vocab)
        conveyor = Conveyor(vocab, n_lanes=lanes, seed=splits(seed)["train"],
                            bias_fn=drive.bin_weights)
    if sleep is not None:
        conveyor = sleep.tap(conveyor)   # A62: record wake tokens
    if arch == "transformer":
        from .lm_transformer import TransformerLM
        model = TransformerLM(vocab_size, d=d, max_T=T).to(device)
    elif arch == "hybrid":
        from .lm_hybrid import HybridLM
        model = HybridLM(vocab_size, d=d, max_T=T, store=store,
                         norm_mix=norm_mix, aux_trunk=aux_trunk,
                         use_xl=use_xl, gate_init=gate_init,
                         read_drop=read_drop, gate_mode=gate_mode,
                         keyed=keyed).to(device)
        drive.bin_band = {0: 3, 1: 3, 2: 4, 3: 5}  # carry bands (A19)
    else:
        model = BandLM(vocab_size, d=d, talk=talk,
                       widths=widths).to(device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile unavailable ({e}); running eager")
    model._st = model.init_state(lanes, device)
    # A45: lr is width-sensitive — d=128's 3e-4 carried into
    # d=384 (4.4x params) is the lead suspect for v7.1's
    # gates-shut held-out bleed (circuit churn under updates
    # the loss barely notices)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step0 = 0
    if resume:
        state = torch.load(resume, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step0 = state.get("step", 0)
        dsnap = state.get("drive", {})
        drive.ema = dict(dsnap.get("ema", {}))
        drive.records = dict(dsnap.get("records", {}))
        drive.minted = set(dsnap.get("minted", []))
        drive.vetoes = dsnap.get("vetoes", 0)
        drive.step_t = step0 * T
        print(f"resumed {resume} at step {step0} "
              f"(ema keys: {sorted(drive.ema)})", flush=True)
    peval = None
    if data and eval_data:
        from .lm_data_ultrachat import UltraConveyor as _UC
        peval = {"conv": _UC(eval_data, n_lanes=2), "agg": {}}
        if resume:
            # A54e (F4): the banking baseline must survive resume —
            # with best reset to -1.0, the first pooled window after
            # a crash re-banked best.pt unconditionally, letting a
            # worse model overwrite the banked peak. prev_same is
            # deliberately NOT restored: it tracks the in-process
            # cumulative window and must restart with the fresh
            # conveyor. Legacy ckpts (no peval_best) seed the
            # baseline from the first pooled window instead of
            # banking on it.
            if state.get("peval_best") is not None:
                peval["best"] = state["peval_best"]
            else:
                peval["seed_baseline"] = True
    if sleep is not None:
        sleep.bind(drive)   # A62: after resume, so the buffer's
                            # absolute offset matches drive.step_t
    t0 = time.time()
    ce_first = None
    trace = (ckpt + ".trace.jsonl") if ckpt else None
    for step in range(step0 + 1, step0 + steps + 1):
        if lr_decay == "cosine":
            # A54 audit (C3): v8.0 at this width/duration peaked at
            # 10% and bled for 90% on constant lr — the lr x
            # DURATION confound. Cosine to 10% (the ledgered v8.1
            # candidate) removes it from the scale gate. The frac
            # uses the GLOBAL step over the run's total so a resume
            # continues the schedule instead of restarting it
            # (A54d: a late-resume lr jump would be the bleed).
            tot = lr_total_steps or (step0 + steps)
            frac = min(step / max(tot, 1), 1.0)
            f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr * f
        if read_drop_end is not None and hasattr(model, "read_drop"):
            # A39 bootstrap knob: linear read-dropout anneal — early
            # protection for induction, late oxygen for the store
            frac = (step - step0) / max(steps, 1)
            model.read_drop = read_drop + (read_drop_end - read_drop) * frac
        ce, loss = process_chunk(model, drive, conveyor, T, device,
                                 opt, bf16=bf16)
        if sleep is not None:
            sleep.maybe_sleep(model, opt, drive, step)
        ce_first = ce_first or ce
        if step % log_every == 0 or step == 1:
            tok_s = lanes * T * (step - step0) / (time.time() - t0)
            # A24 L2: channel EMAs ride every log line — run 4's
            # cross-window transient lived and died between snapshots
            emas = " ".join(
                f"{k.replace('recall:', '')}={drive.ema[k]:+.3f}"
                for k in sorted(drive.ema))
            print(f"step {step:5d}  ce {ce:.3f}  loss {loss:.3f}  "
                  f"holds {len(drive.ledger):4d}  {tok_s:,.0f} tok/s  "
                  f"[{emas}]", flush=True)
            row = {"step": step, "ce": round(ce, 4),
                   "ema": {k: round(float(v), 5)
                           for k, v in drive.ema.items()},
                   "vetoes": drive.vetoes,
                   "holds": len(drive.ledger)}
            if peval and step % 2000 < log_every:
                hp = holdout_probe(model, peval, T, device)
                row["holdout"] = hp
                print("    holdout(cum): "
                      + str({k: [round(v[0], 3), round(v[1], 2),
                                 v[2]] for k, v in hp.items()}),
                      flush=True)
                # A42: recent-window same-chunk + best-ckpt banking.
                # v6.2's cumulative average hid an 8k-step held-out
                # collapse; the peak model existed only as a lucky
                # rolling snapshot. Track the delta since the last
                # probe and bank the best model seen.
                if "same" in hp:
                    s_now, _, n_now = hp["same"]
                    p_sum, p_n = peval.get("prev_same", (0.0, 0))
                    dn = n_now - p_n
                    # A54 audit (H3): sample floor via ACCUMULATION —
                    # windows pool until >=10 fresh probes, then the
                    # pooled recent mean is evaluated and the window
                    # resets. Tiny windows can no longer bank on
                    # noise, and banking cadence survives sparse
                    # probe draws (R5 saw dn of 1-7).
                    if dn >= 10:
                        recent = (s_now * n_now - p_sum) / dn
                        row["same_recent"] = round(recent, 4)
                        print(f"    same(recent {dn} probes): "
                              f"{recent:.3f}", flush=True)
                        if peval.pop("seed_baseline", False):
                            peval["best"] = max(recent,
                                                peval.get("best", -1.0))
                            print(f"    banking baseline seeded "
                                  f"({recent:.3f})", flush=True)
                        elif ckpt and recent > peval.get("best", -1.0) \
                                and n_now >= 20:
                            peval["best"] = recent
                            atomic_save({"model": model.state_dict(),
                                         "step": step,
                                         "same_recent": recent},
                                        ckpt + ".best.pt")
                            print(f"    best banked @ {step} "
                                  f"({recent:.3f})", flush=True)
                        peval["prev_same"] = (s_now * n_now, n_now)
            if trace:
                import json as _json
                with open(trace, "a") as f:
                    f.write(_json.dumps(row) + "\n")
        if ckpt and step % 500 == 0:
            atomic_save({"model": model.state_dict(),
                         "opt": opt.state_dict(), "step": step,
                         "peval_best": (peval or {}).get("best"),
                         "drive": {"ema": drive.ema,
                                   "records": drive.records,
                                   "minted": sorted(drive.minted),
                                   "holds_settled": len(drive.ledger),
                                   "vetoes": drive.vetoes}}, ckpt)
    audit = drive.audit()
    print("audit:", audit)
    print("panel:\n" + drive.panel())
    return model, drive, vocab, ce_first, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--ckpt", default="lm_ladder.pt")
    ap.add_argument("--data", default=None,
                    help="prepared shard dir (lm_data_ultrachat prepare)")
    ap.add_argument("--talk", default="tick")
    ap.add_argument("--constants", default=None,
                    help="calibrated constants json (lm_calibrate)")
    ap.add_argument("--arch", default="bands",
                    choices=["bands", "transformer", "hybrid"])
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from (A26)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="conveyor lane offset fraction (A26)")
    ap.add_argument("--store", default="vector",
                    choices=["vector", "matrix"],
                    help="hybrid band storage substrate (A28)")
    ap.add_argument("--eval-data", default=None, dest="eval_data",
                    help="held-out shard for live circuit probes (A30)")
    ap.add_argument("--xl", default="on", choices=["on", "off"],
                    help="Transformer-XL chunk carry (A36: benched)")
    ap.add_argument("--gate-init", type=float, default=-4.0,
                    dest="gate_init",
                    help="read-gate init logit (A39 bootstrap knob)")
    ap.add_argument("--read-drop", type=float, default=0.5,
                    dest="read_drop",
                    help="matrix read-dropout p (A39 bootstrap knob)")
    ap.add_argument("--read-drop-end", type=float, default=None,
                    dest="read_drop_end",
                    help="linear anneal target for read-dropout (A39)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--hold-cap", type=int, default=None,
                    help="A60: max new drive holds per sweep (density throttle; None=uncapped)")
    ap.add_argument("--lam", type=float, default=0.25,
                    help="drive pay weight (A51 ablation: 0)")
    ap.add_argument("--gate-mode", default="scalar", dest="gate_mode",
                    choices=["scalar", "position", "entropy"],
                    help="matrix read gate: scalar per band, or "
                         "per-position learned head (A41 candidate)")
    ap.add_argument("--lr-decay", default="none", dest="lr_decay",
                    choices=["none", "cosine"],
                    help="cosine: decay lr to 10% over the run "
                         "(A54: the lr x duration guard)")
    ap.add_argument("--lr-total-steps", type=int, default=None,
                    dest="lr_total_steps",
                    help="global schedule length for lr decay; "
                         "keeps a resume on the same curve")
    ap.add_argument("--norm-mix", action="store_true",
                    help="A55 (R6): unit-normalize key mixes before "
                         "the RFF lift (fixes the flat-kernel key "
                         "collision, A54e F2)")
    ap.add_argument("--aux-trunk", type=float, default=0.0,
                    help="A58 (R8): pay-the-trunk aux CE weight on "
                         "pre-bonus logits (anti-starvation)")
    ap.add_argument("--keyed", default=None,
                    choices=["token", "logit"],
                    help="A52 (R4) token: per-position writes keyed "
                         "by the token's own embedding. A53 (R5) "
                         "logit: decode-free capacity-sized stores — "
                         "identity values matched straight into the "
                         "logits")
    a = ap.parse_args()
    if a.mode == "smoke":
        model, drive, vocab, ce0, ce1 = train(d=64, lanes=4, T=256, steps=40,
                                              device="cpu", data=a.data)
        assert ce1 < ce0, "smoke: CE did not fall"
        assert drive.audit()["telescoping_exact"], "smoke: ledger not exact"
        print(f"SMOKE PASS  ce {ce0:.3f} -> {ce1:.3f}")
    else:
        train(d=a.d, lanes=a.lanes, T=a.chunk, steps=a.steps, seed=a.seed,
              device=a.device, ckpt=a.ckpt, log_every=50, data=a.data,
              talk=a.talk, constants=a.constants, arch=a.arch,
              resume=a.resume, offset_frac=a.offset, store=a.store,
              eval_data=a.eval_data, use_xl=(a.xl == "on"),
              gate_init=a.gate_init, read_drop=a.read_drop,
              read_drop_end=a.read_drop_end, gate_mode=a.gate_mode,
              lr=a.lr, bf16=a.bf16, lam=a.lam, keyed=a.keyed,
              lr_decay=a.lr_decay, lr_total_steps=a.lr_total_steps,
              norm_mix=a.norm_mix, aux_trunk=a.aux_trunk,
              hold_cap=a.hold_cap)


if __name__ == "__main__":
    main()
