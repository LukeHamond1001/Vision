"""v10 flash driver — the 500M lifetime flash (pod-side).

The run is a chain of train() segments over ONE prepared staged-life
shard (one life per lane; lanes == manifest n_lives asserted). The
segment boundaries are (a) the manifest's stage boundaries, where
the SLEEP DOSE LADDER flips (A64-R3: a fresh trunk under sleep
collapses — infancy is sleepless; the dose ramps with age and lands
on the gate-tested arm C cadence), and (b) heartbeat marks, where
the heartbeat pack (scripts/heartbeat_v10.py) probes the banked
checkpoint and may return the KILL sentinel (exit 3 -> the pod
wrapper banks everything and stops: kill, fix, relaunch — a caught
disease costs hours, not the run).

Everything stateful rides the certified trainer: banking
(best-holdout ckpt + peval_best, A54e F4), atomic saves, resume
(model+opt+drive; offset_frac recomputed from the checkpoint step so
a crash resumes the SAME one-epoch stream position). The Sleeper and
PressProphet live in this process; on a crash relaunch they restart
empty (spans re-accumulate within one buffer horizon; the prophet is
a spectator, B5) — ledgered, matches v9.4 practice.

SLEEP LADDER (pre-registered before token one):
  infancy     every=0   sleepless (A64-R3 fresh-trunk law; the
                        pruned-unharvested infancy ledger is
                        infantile amnesia by design)
  childhood   every=32  half the gate dose — the ramp-in
  adolescence every=16  the life-gate-tested arm C dose (G3 PASS)
  tail        every=16  hold — no dose spike on the cosine tail
  block_chunks=2, press_pay=(T, T//8), homeostasis=1e-3 (A76 SHIP
  dose; window 3e-4..3e-3, 3e-3 overdoses memory 0.69x), splice=0
  novelty=0 dream=None (A73/A74/A77 OUT by gate verdicts).

One-epoch law (A12): total steps = life_len // T rounded DOWN to
the 500-step checkpoint cadence — the ≤499 dropped steps land in
the builder's filler-day padding at each life's end, never in real
material. The driver refuses to run past one epoch.
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_hybrid import BAND6_CLOCKS          # noqa: E402
from iga.lm_press import PressProphet           # noqa: E402
from iga.lm_sleep import Sleeper                # noqa: E402
from iga.lm_train import train                  # noqa: E402

# 2026-08-21 (user): childhood at 1:16 — the certified dose (A62 1:16,
# A76 H=1e-3 at every=16) on a trunk ~400M tokens old; A64-R3's
# collapse was a FRESH trunk at 1:4. Sleep replays press-paid spans =
# binder practice. Infancy stays sleepless (A64-R3).
LADDER = {"infancy": 0, "childhood": 16, "adolescence": 16,
          "tail": 16}
CKPT_CADENCE = 500          # lm_train saves at step % 500 == 0
HB_EVERY = 6000             # heartbeat cadence in steps (~2.7h)
LESION_EVERY = 2            # full band-lesion pass every Nth beat
HB_CRASH_TOL = 3            # consecutive dead heartbeats = abort


def stage_step_bounds(manifest, T):
    """Stage boundaries in TRAINING STEPS (per-lane tokens / T),
    from the manifest's stage fracs over life_len."""
    life_len = manifest["life_len"]
    total = life_len // T // CKPT_CADENCE * CKPT_CADENCE
    bounds, acc = [], 0
    for s in manifest["stages"]:
        acc += s["frac"]
        b = int(total * acc) // CKPT_CADENCE * CKPT_CADENCE
        bounds.append((s["name"], min(b, total)))
    bounds[-1] = (bounds[-1][0], total)
    return bounds, total


def stage_at(bounds, step):
    for name, b in bounds:
        if step < b:
            return name
    return bounds[-1][0]


def segment_ends(bounds, total, hb_every):
    ends = {b for _, b in bounds} | \
           set(range(hb_every, total, hb_every)) | {total}
    return sorted(e for e in ends if 0 < e <= total)


def run_heartbeat(a, step, tokens, total_tokens, beat_i):
    cmd = [sys.executable,
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "heartbeat_v10.py"),
           "--ckpt", a.ckpt, "--data", a.data,
           "--eval-data", a.eval_data, "--out", a.hb_out,
           "--step", str(step), "--tokens", str(tokens),
           "--total-tokens", str(total_tokens),
           "--d", str(a.d), "--n-layers", str(a.n_layers),
           "--T", str(a.T), "--device", a.device,
           "--chunks", str(a.hb_chunks)]
    if beat_i % a.lesion_every:
        cmd.append("--skip-lesions")
    r = subprocess.run(cmd)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--ckpt", default="v10.pt")
    ap.add_argument("--smoke", default=None,
                    help="smoke.json from the paid smoke (lam); "
                         "required unless --lam is given")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--d", type=int, default=1280)
    ap.add_argument("--n-layers", type=int, default=20)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--lr-warmup", type=int, default=2000)
    ap.add_argument("--ledger-cap", type=int, default=200_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hb-every", type=int, default=HB_EVERY)
    ap.add_argument("--lesion-every", type=int, default=LESION_EVERY)
    ap.add_argument("--hb-chunks", type=int, default=2500)   # 2026-08-21: 400 = 819k tok/lane = eval-infancy only; bins b3-b6 structurally empty
    # v10.1 (2026-08-21): precision + gated architecture candidates ride
    # the CLI so the pod env decides; defaults = the certified v10 shape
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--attn", default="abs", choices=["abs", "rope"])
    ap.add_argument("--qk-norm", default="0")
    ap.add_argument("--band-lr-mult", type=float, default=1.0)
    ap.add_argument("--mlp", default="gelu", choices=["gelu", "swiglu"])
    # the CONVEYOR arm (2026-08-21): a shorter window with the clocks
    # multiplied so every band keeps its TOKEN horizon — attention holds
    # less, the bands must carry more. --T 1024 --clock-mult 2 = half
    # window, same horizons; 1 = the certified clocks bit-exactly.
    ap.add_argument("--clock-mult", type=int, default=1)
    # a re-based ladder: "3:1,4:8,5:64,6:512,7:4096,8:32768" (clocks in
    # chunks; horizons = clock x T). Overrides --clock-mult.
    ap.add_argument("--clocks", default="")
    # content-keyed store (2026-08-21, docs/MEMORY_MATH.md): "hidden" keys
    # the store on the trunk's hidden; "logit" = the certified token mix
    ap.add_argument("--keyed", default="logit", choices=["logit", "hidden"])
    # THE ONE-TOKEN ORGANISM (2026-08-21, iga/lm_scan.py): --arch scan
    # with --clocks in TOKENS; --scan-order cortex_first|pfc_first;
    # --lanes overrides one-life-per-lane (the per-token scan needs
    # batch: 32 lanes x T=64 tokens per step) — each life is then cut
    # into lanes/n_lives contiguous pieces and the stage bounds scale
    ap.add_argument("--arch", default="hybrid", choices=["hybrid", "scan"])
    ap.add_argument("--scan-order", default="cortex_first",
                    choices=["cortex_first", "pfc_first"])
    # extra ScanLM kwargs as JSON, e.g. '{"n_council": 2, "slot_every": 8,
    # "write_every": 4}' (docs/ONE_TOKEN_PLAN.md: one knob per iteration)
    ap.add_argument("--scan-opts", default="")
    ap.add_argument("--lanes", type=int, default=0)
    # band repair (docs/MEMORY_MATH.md 5): credit routing, centred
    # fidelity, the tail memory token; 0/0/0 = certified bit-exactly
    ap.add_argument("--band-credit", type=int, default=0)
    ap.add_argument("--band-center", type=int, default=0)
    ap.add_argument("--tail-tokens", type=int, default=0)
    ap.add_argument("--hb-out", default="results/hb_v10.jsonl")
    ap.add_argument("--trace", default="results/v10_driver.jsonl")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="SHAKEDOWN ONLY: cap total steps")
    ap.add_argument("--dry", action="store_true",
                    help="print the segment plan and exit")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.data, "manifest.json")))
    lanes = manifest["n_lives"]
    life_len = manifest["life_len"]
    bounds, total = stage_step_bounds(manifest, a.T)
    if a.lanes and a.lanes != lanes:
        # lanes/n_lives pieces per life: a lane holds life_len x
        # n_lives / lanes tokens; every bound scales the same way
        f = lanes / a.lanes
        life_len = int(life_len * f)
        total = int(total * f) // CKPT_CADENCE * CKPT_CADENCE
        bounds = [(n, min(int(b * f) // CKPT_CADENCE * CKPT_CADENCE, total))
                  for n, b in bounds]
        bounds[-1] = (bounds[-1][0], total)
        lanes = a.lanes
        print(f"LANES override: {lanes} lanes ({manifest['n_lives']} lives), "
              f"{life_len:,} tokens per lane, {total} steps")
    true_total = total          # probe arming uses the REAL life
    if a.max_steps:             # length even under a shakedown cap
        total = min(total, a.max_steps // CKPT_CADENCE
                    * CKPT_CADENCE) or a.max_steps
        bounds = [(n, min(b, total)) for n, b in bounds]
        bounds[-1] = (bounds[-1][0], total)
        print(f"SHAKEDOWN: capped at {total} steps")
    assert total * a.T <= life_len, "one-epoch law (A12)"
    lam = a.lam
    if lam is None:
        assert a.smoke, "need --smoke smoke.json or --lam"
        lam = json.load(open(a.smoke))["lam"]
    ends = segment_ends(bounds, total, a.hb_every)

    clocks = {k: v * a.clock_mult for k, v in BAND6_CLOCKS.items()}
    if a.clocks:
        clocks = {int(kv.split(":")[0]): int(kv.split(":")[1])
                  for kv in a.clocks.split(",") if kv}
    scan_opts = None
    if a.arch == "scan":
        scan_opts = {"order": a.scan_order}
        if a.scan_opts:
            scan_opts.update(json.loads(a.scan_opts))
    plan = {"lanes": lanes, "life_len": life_len,
            "total_steps": total,
            "total_tokens": total * a.T * lanes, "lam": lam,
            "bounds": bounds, "ladder": LADDER,
            "segments": len(ends), "precision": a.precision,
            "attn": a.attn, "qk_norm": str(a.qk_norm) == "1",
            "band_lr_mult": a.band_lr_mult, "mlp": a.mlp,
            "clock_mult": a.clock_mult, "clocks": clocks, "keyed": a.keyed,
            "arch": a.arch, "scan_order": a.scan_order, "scan": scan_opts,
            "band_credit": a.band_credit, "band_center": a.band_center,
            "tail_tokens": a.tail_tokens,
            "hb_every": a.hb_every,
            "hb_chunks": a.hb_chunks, "lesion_every": a.lesion_every}
    print("PLAN " + json.dumps(plan), flush=True)
    if a.dry:
        return

    sl = Sleeper(arm="C", every=0, block_chunks=2, seed=1,
                 homeostasis=1e-3)
    sl.press_pay = (a.T, a.T // 8)
    # the prophet samples band states by CHUNK clocks; the scan's
    # clocks are in tokens
    pclocks = ({k: max(1, c // a.T) for k, c in clocks.items()}
               if a.arch == "scan" else clocks)
    prophet = PressProphet(d=a.d, clocks=pclocks,
                           holdout_frac=0.1, device=a.device)

    cur = 0
    if os.path.exists(a.ckpt):
        cur = torch.load(a.ckpt, map_location="cpu",
                         weights_only=False).get("step", 0)
        print(f"resuming at step {cur}", flush=True)
    os.makedirs(os.path.dirname(a.trace) or ".", exist_ok=True)
    hb_dead = 0
    seg_i = 0
    carry = None      # live band states threaded across segments
    for end in ends:
        if end <= cur:
            continue
        seg_i += 1
        stage = stage_at(bounds, cur)
        sl.every = LADDER[stage]
        t0 = time.time()
        model, drive, vocab, ce0, ce1 = train(
            d=a.d, n_layers=a.n_layers, lanes=lanes, T=a.T,
            steps=end - cur, seed=1000 + seg_i, device=a.device,
            arch=a.arch, store="matrix", keyed=a.keyed,
            scan=scan_opts,
            norm_mix=True, aux_trunk=0.2, use_xl=False,
            gate_init=-2.0, clocks=clocks,
            data=a.data, eval_data=a.eval_data, ckpt=a.ckpt,
            resume=(a.ckpt if cur else None),
            offset_frac=cur * a.T / life_len,
            lr=a.lr, lr_warmup=a.lr_warmup, lr_decay="cosine",
            lr_total_steps=total, lam=lam,
            ledger_cap=a.ledger_cap, sleep=sl, prophet=prophet,
            log_every=a.log_every, carry_state=carry,
            precision=a.precision, attn=a.attn,
            qk_norm=(str(a.qk_norm) == "1"),
            band_lr_mult=a.band_lr_mult, mlp=a.mlp,
            band_credit=bool(a.band_credit), band_center=bool(a.band_center),
            tail_tokens=a.tail_tokens,
            horizon_rule=("clock" if a.clocks else "fixed"))
        carry = model._st
        aucs = {}
        for k in sorted(prophet.clocks):
            try:
                v = prophet.auc(k)
                if v is not None:
                    aucs[str(k)] = round(float(v), 4)
            except Exception:
                pass
        row = {"seg": seg_i, "from": cur, "to": end, "stage": stage,
               "every": sl.every, "ce_first": round(ce0 or 0, 4),
               "ce_last": round(ce1 or 0, 4),
               "pairs": len(sl.pairs), "sleep_steps": sl.steps_taken,
               # "everything contributing" vitals: the pair law, the
               # amygdala tag actually replaying, the economy alive
               "pair_law_ok": all(p.get("w1") == p.get("tw")
                                  for p in sl.pairs),
               "hot_pairs": sum(1 for p in sl.pairs
                                if p.get("hot")),
               "minted": len(drive.minted),
               "vetoes": drive.vetoes,
               "ledger": len(drive.ledger),
               "prophet_auc": aucs,
               "secs": round(time.time() - t0)}
        with open(a.trace, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("SEG " + json.dumps(row), flush=True)
        if ce1 is None or not math.isfinite(ce1):
            print("ABORT non-finite CE — banked ckpt stands",
                  flush=True)
            sys.exit(4)
        del model, drive, vocab
        gc.collect()
        if "cuda" in a.device:
            torch.cuda.empty_cache()
        cur = end
        rc = run_heartbeat(a, cur, cur * a.T * lanes,
                           true_total * a.T * lanes, seg_i)
        if rc == 3:
            print("KILL", flush=True)   # pod wrapper's sentinel
            sys.exit(3)
        if rc != 0:
            hb_dead += 1
            print(f"heartbeat crashed rc={rc} "
                  f"({hb_dead}/{HB_CRASH_TOL})", flush=True)
            if hb_dead >= HB_CRASH_TOL:
                print("ABORT instruments dead — flying blind is "
                      "not permitted", flush=True)
                sys.exit(4)
        else:
            hb_dead = 0
    print(f"DONE {cur} steps ({cur * a.T * lanes:,} tokens). "
          f"Banked best: {a.ckpt}.best.pt", flush=True)


if __name__ == "__main__":
    main()
