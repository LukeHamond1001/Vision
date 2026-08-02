"""v2.1: the battery-homeostasis behavioral campaign. Run:

    python -m iga.experiments_v21 tiny
    python -m iga.experiments_v21 full     # local: ~3h, all arms x seeds

Pre-registered (2026-08-02, before any full run). Four arms, same
Gaussian-PPO learner, same worlds:

  task          forward-velocity reward only (the baseline that runs
                until it browns out)
  task+wired    task + register progress on the resource lines
                (battery high, temp low, wear slowed), per-band weights
                w ∝ τ (the spec law), channels normalized by walk std.
                Potential-based -> non-farmable.
  task+penalty  task + a good-faith hand-engineered penalty (the
                incumbent): -0.5*[battery<0.2] -0.3*temp -50*dwear
                (lightly tuned by us, disclosed as such)
  wired         registers only (the pure-drives control)

Gates:
  G-uptime   brownout fraction: task+wired <= task-only / 5   (CI-clean)
  G-task     task return: task+wired >= 0.8 x task-only
  G-parity   task+wired matches-or-beats task+penalty on the
             (uptime, task) pair — the zero-coefficients claim
  Mechanism  docked% up, temp down, wear-rate down in wired arms.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from .experiments import RESULTS
from .ppo_proprio import run_battery_arm

# w ∝ τ over walk-std-normalized channels (phase-1 measurements)
TAUS = {"battery": 92.0, "temp": 54.0, "wear": 28856.0}
STDS = {"battery": 0.25, "temp": 0.05, "wear": 0.02}   # walk stds, v20 data
WSUM = sum(TAUS.values())
W = {k: TAUS[k] / WSUM for k in TAUS}


def phi_of(info) -> float:
    b = (1.0 - info["battery"]) / STDS["battery"]
    t = info["temp"] / STDS["temp"]
    w = info["wear"] / STDS["wear"]
    return (W["battery"] * b ** 2 + W["temp"] * t ** 2
            + W["wear"] * w ** 2) ** 0.5


def make_reward(arm: str):
    if arm == "task":
        return lambda rt, i, p: rt
    if arm == "task+penalty":
        def rf(rt, i, p):
            dw = (i["wear"] - p["wear"]) if p else 0.0
            return rt - 0.5 * (i["battery"] < 0.2) - 0.3 * i["temp"] - 50 * dw
        return rf
    scale = 1.0 if arm == "wired" else 0.5

    def rf(rt, i, p):
        prog = (phi_of(p) - phi_of(i)) if p else 0.0
        return (rt if arm == "task+wired" else 0.0) + scale * prog
    return rf


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    steps = 16384 if tiny else 2_000_000
    seeds = (1,) if tiny else (1, 2, 3, 4, 5)
    rows = []
    for arm in ("task", "task+wired", "task+penalty", "wired"):
        rf = make_reward(arm)
        for s in seeds:
            out = run_battery_arm(rf, seed=s, total_steps=steps)
            k = max(1, len(out["task"]) // 3)
            row = {"arm": arm, "seed": s,
                   "task": float(np.mean(out["task"][-k:])),
                   "brownout": float(np.mean(out["brownout"][-k:])),
                   "docked": float(np.mean(out["docked"][-k:])),
                   "wear_rate": float(np.mean(out["wear_rate"][-k:])),
                   "temp": float(np.mean(out["temp_mean"][-k:]))}
            rows.append(row)
            (RESULTS / "v21_battery.json").write_text(json.dumps(rows, indent=2))
            print(f"[v2.1] {arm:12s} s{s}: task {row['task']:7.0f} "
                  f"brownout {row['brownout']:.3f} docked {row['docked']:.3f} "
                  f"wear {row['wear_rate']:.2e} temp {row['temp']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    m = {a: {k: float(np.mean([r[k] for r in v]))
             for k in ("task", "brownout", "docked", "wear_rate", "temp")}
         for a, v in by.items()}
    g_up = m["task+wired"]["brownout"] <= m["task"]["brownout"] / 5 \
        if m["task"]["brownout"] > 0 else m["task+wired"]["brownout"] < 0.01
    g_task = m["task+wired"]["task"] >= 0.8 * m["task"]["task"]
    g_par = (m["task+wired"]["brownout"] <= m["task+penalty"]["brownout"] * 1.1
             and m["task+wired"]["task"] >= 0.9 * m["task+penalty"]["task"])
    print(f"[v2.1] means: " + json.dumps(m), flush=True)
    print(f"[v2.1] G-uptime {'PASS' if g_up else 'FAIL'}  "
          f"G-task {'PASS' if g_task else 'FAIL'}  "
          f"G-parity {'PASS' if g_par else 'FAIL'}", flush=True)
    print(f"[v2.1] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
