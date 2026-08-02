"""v4.0: the complete-architecture sequencing campaign (docs/SEQUENCING.md).

    python -m iga.experiments_v40 tiny                     # local smoke
    python -m iga.experiments_v40 rehearsal <arm> [steps]  # pre-flight D'/E
    python -m iga.experiments_v40 full <arm> <seed>        # one fleet arm

Arms (same PPO core, same worlds — ppo_pixel with v4.0 hooks):
  full         goal machine (proposer + prospective evaluation + G5 +
               one-shot frontier curiosity + exact claims) + the C4
               flinch veto IFF pre-flight F passed both gates
  no-flinch    the machine without the acting-time veto
  no-proposer  same machinery, proposer replaced by a STATIC serialized
               restore list (the v1.2-class homeostat) — the ablation
               that isolates the LADDER. Disclosed difference from
               v1.2: serialized per-channel ramps, not a simultaneous
               composite; the machinery is otherwise identical to the
               proposer arms.
  native       Crafter achievement reward (the incumbent)

G-seq (pre-registered): full-arm distinct achievements per episode
(median of the last third) >= 3; no-proposer < 1.5; native reported,
not gated. The proposal trace is the paper figure.

All drive-layer components are FROZEN before RL: instrument heads
(results/v40_instrument.pt, pre-flight B), forward model
(results/v40_forward.pt, pre-flight F), machine constants (GMConfig).
The policy sees pixels only.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from .crafter_support import HUD_ROWS, SLOW_EMA_TAU, slow_stats
from .experiments import RESULTS
from .goal_machine import CHANNELS, GMConfig, GoalMachine, channel_windows
from .ppo_pixel import run_ppo_arm

FIXED_TARGETS = ((1, 8.0), (2, 8.0), (3, 8.0))   # food, drink, energy
ARMS = ("full", "no-flinch", "no-proposer", "native")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_instrument():
    blob = torch.load(RESULTS / "v40_instrument.pt", weights_only=False)
    inst = blob["inst"]
    spec = channel_windows(blob["stock_windows"])
    stds = tuple(inst["channels"][k]["std"] for k in CHANNELS)
    return inst, spec, stds


def make_readout(inst, spec):
    """Frozen batched readout. measure_x takes float frames in [0,1]
    (N,3,64,64); measure_np takes uint8 HWC numpy (N,64,64,3)."""
    chans = [(spec[k], inst["channels"][k]["w"],
              float(inst["channels"][k]["b"]),
              float(inst["channels"][k]["a"]),
              float(inst["channels"][k]["c"])) for k in CHANNELS]

    def measure_x(x: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(x.shape[0], len(CHANNELS))
        for i, (sp, w, b, a, c) in enumerate(chans):
            kind, (r0, r1), (c0, c1) = sp
            xx = x[:, :, HUD_ROWS, :] if kind == "hud" else x
            m[:, i] = a * (xx[:, :, r0:r1, c0:c1].flatten(1) @ w + b) + c
        return m

    def measure_np(obs_np: np.ndarray) -> torch.Tensor:
        return measure_x(torch.from_numpy(obs_np).permute(0, 3, 1, 2)
                         .float().div(255.0))

    return measure_x, measure_np


def machine_cfg(stds, arm: str) -> GMConfig:
    base = GMConfig(stds=stds)
    w = (1.0, base.holds[1] / base.holds[0])     # spec hold-ratio law
    if arm == "no-proposer":
        return GMConfig(stds=stds, w_bands=w, use_proposer=False,
                        fixed_targets=FIXED_TARGETS)
    return GMConfig(stds=stds, w_bands=w)


def make_drive(n_envs: int, stds, arm: str, inst, spec):
    """The v4.0 reward path: per-env goal machines over the frozen
    readout. Episode boundaries follow the v1.2 rule: the dying
    transition pays nothing; claims settle at the last-seen state."""
    _, measure_np = make_readout(inst, spec)
    cfg = machine_cfg(stds, arm)
    machines = [GoalMachine(cfg) for _ in range(n_envs)]
    state = {"started": False}

    def reward_fn(obs2, r_native, dones):
        m2 = measure_np(obs2)
        r = np.zeros(len(machines), dtype=np.float32)
        if not state["started"]:
            for k, gm in enumerate(machines):
                gm.reset(m2[k])
            state["started"] = True
            return r
        for k, gm in enumerate(machines):
            if dones[k]:
                gm.step(gm.m.clone(), done=True)
                gm.reset(m2[k])
            else:
                r[k], _ = gm.step(m2[k])
        return r

    return reward_fn, machines


def make_flinch(inst, spec):
    """C4 acting-time veto from the frozen forward model. Returns None
    (with the gate record) when pre-flight F failed — the documented
    SPEC C4 deferral."""
    fwd = torch.load(RESULTS / "v40_forward.pt", weights_only=False)
    if not all(fwd["gates"].values()):
        return None, fwd["gates"]
    from .crafter_support import EncoderCrafter
    enc = EncoderCrafter(band_dims=(16, 4, 2))
    enc.load_state_dict(torch.load(RESULTS / "v09_encoder.pt",
                                   map_location="cpu"))
    enc.eval()
    model = torch.nn.Sequential(
        torch.nn.Linear(fwd["in_dim"], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
    model.load_state_dict(fwd["state_dict"])
    model.eval()
    measure_x, _ = make_readout(inst, spec)
    n_act = fwd["n_act"]
    alpha = 1.0 / SLOW_EMA_TAU
    es = {"v": None}   # EMA across boundaries — the v1.2 encode_phi
    #                    precedent (tau=10 washes resets out in ~30 steps)

    def hook(x_float, a, dist):
        s = slow_stats(x_float)
        es["v"] = s if es["v"] is None else (1 - alpha) * es["v"] + alpha * s
        z = enc(x_float, slow_feats=es["v"])
        m = measure_x(x_float)
        feats = torch.cat([z, m], dim=-1)
        health = m[:, 0]
        for _ in range(3):
            oh = torch.nn.functional.one_hot(a, n_act).float()
            pred = model(torch.cat([feats, oh], dim=-1))[:, 0]
            flinch = (pred <= -0.5) & (health >= 4.5)
            if not bool(flinch.any()):
                return a
            a = torch.where(flinch, dist.sample(), a)
        oh = torch.nn.functional.one_hot(a, n_act).float()
        pred = model(torch.cat([feats, oh], dim=-1))[:, 0]
        still = (pred <= -0.5) & (health >= 4.5)
        return torch.where(still, torch.zeros_like(a), a)   # noop

    return hook, fwd["gates"]


def _machine_summary(machines) -> dict:
    led = [h for gm in machines for h in gm.ledger]
    arr = [h for h in led if h["closed_by"] == "arrive"]
    fr = [h for h in arr if h["frontier"]]
    cells = {}
    for h in fr:
        key = f"{h['channel']}>={h['target']:.0f}"
        cells[key] = cells.get(key, 0) + 1
    return {"holds": len(led), "arrivals": len(arr),
            "frontier_arrivals": len(fr),
            "timeouts": sum(1 for h in led if h["closed_by"] == "timeout"),
            "bonus_total": float(sum(gm.total_bonus for gm in machines)),
            "cells": cells,
            "rejects": {k: sum(gm.rejects[k] for gm in machines)
                        for k in machines[0].rejects} if machines else {}}


def run_arm(arm: str, seed: int, steps: int, n_envs: int = 12,
            log_every: int = 100_000, tag: str = "v40") -> dict:
    assert arm in ARMS, arm
    t0 = time.time()
    inst, spec, stds = load_instrument()
    reward_fn, machines, hook, mode = None, [], None, "custom"
    if arm == "native":
        mode = "native"
    else:
        reward_fn, machines = make_drive(n_envs, stds, arm, inst, spec)
    if arm == "full":
        hook, fgates = make_flinch(inst, spec)
        print(f"[{tag}] flinch gates {fgates} -> "
              f"{'ARMED' if hook else 'DEFERRED (SPEC C4)'}", flush=True)

    last = {"n": 0}

    def log_cb(step_total, stats):
        if step_total - last["n"] < log_every:
            return
        last["n"] = step_total
        k = max(1, len(stats["survival"]) // 3)
        ms = _machine_summary(machines) if machines else {}
        print(f"[{tag}] {arm} s{seed} {step_total/1e6:.2f}M: "
              f"eps {len(stats['survival'])} "
              f"surv {np.median(stats['survival'][-k:]) if stats['survival'] else 0:.0f} "
              f"achv {np.median(stats['achv_n'][-k:]) if stats['achv_n'] else 0:.1f} "
              f"arriv {ms.get('frontier_arrivals', '-')} "
              f"bonus {ms.get('bonus_total', '-')} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[{tag}] {arm} s{seed} device={DEVICE} steps={steps}", flush=True)
    out = run_ppo_arm(None, None, None, mode, seed, total_steps=steps,
                      n_envs=n_envs, reward_fn=reward_fn, sample_hook=hook,
                      log_cb=log_cb, device=DEVICE)
    k = max(1, len(out["survival"]) // 3)
    ach_counts: dict = {}
    for d in out["achv"][-k:]:
        for name in d:
            ach_counts[name] = ach_counts.get(name, 0) + 1
    row = {
        "arm": arm, "seed": seed, "steps": steps,
        "episodes": len(out["survival"]),
        "survival": float(np.median(out["survival"][-k:])),
        "achv_median": float(np.median(out["achv_n"][-k:]))
        if out["achv_n"] else 0.0,
        "achv_mean": float(np.mean(out["achv_n"][-k:]))
        if out["achv_n"] else 0.0,
        "achv_table": ach_counts,
        "drink": float(np.mean(out["drink"][-k:])),
        "food": float(np.mean(out["food"][-k:])),
        "energy": float(np.mean(out["energy"][-k:])),
        "slept": float(np.mean(out["slept"][-k:])),
        "machine": _machine_summary(machines) if machines else {},
        "elapsed_s": time.time() - t0,
    }
    # bank the row (resume pattern), replacing any prior (arm, seed) row
    path = RESULTS / f"{tag}_sequencing.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [r for r in rows if not (r["arm"] == arm and r["seed"] == seed)]
    rows.append(row)
    path.write_text(json.dumps(rows, indent=2))
    torch.save(out["policy_state"], RESULTS / f"{tag}_policy_{arm}_s{seed}.pt")
    if machines:
        trace = [f"t={t} {'S' if band else 'V'} {evn} {ch}>={lvl:.0f}"
                 for (t, evn, band, ch, lvl) in machines[0].trace[-600:]]
        (RESULTS / f"{tag}_trace_{arm}_s{seed}.json").write_text(
            json.dumps(trace, indent=0))
    print(f"[{tag}] {arm} s{seed} DONE: surv {row['survival']:.0f} "
          f"achv-med {row['achv_median']:.1f} table {ach_counts} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return row


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    if mode == "tiny":
        row = run_arm("full", seed=1, steps=16_384, n_envs=4,
                      log_every=8_192, tag="v40tiny")
        assert row["machine"]["holds"] > 0, "machine never held a goal"
        print("[v40tiny] smoke OK", flush=True)
    elif mode == "rehearsal":
        arm = sys.argv[2]
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
        run_arm(arm, seed=1, steps=steps, n_envs=12, tag="v40r")
    elif mode == "full":
        arm, seed = sys.argv[2], int(sys.argv[3])
        run_arm(arm, seed=seed, steps=3_000_000, n_envs=12, tag="v40")
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
