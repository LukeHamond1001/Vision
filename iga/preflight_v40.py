"""v4.0 pre-flight (ALL LOCAL — pods only for the final fleet).

Step A (this file, mode 'counters'): inventory-counter routing at full
scale, locally — closed-form heads need no pods.
  A1  extended walk recording inventory truth (wood, sapling, stone)
  A2  coverage audit: which counters VARY under random play (stone
      needs a pickaxe chain -> expected unroutable; documented, not
      hidden)
  A3  exclusive-change differencing -> counter digit windows
  A4  closed-form heads + held-out routing gates (>= 0.9 each varying
      counter)

Later steps (separate modes): 'harness' scripted-chain mechanics audit,
'audit' proposal-agenda inspection, 'reach' fixed-target reachability,
'rehearsal' one-seed full dress run.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

from .crafter_support import HUD_ROWS, closed_form_head, measure_rho
from .experiments import RESULTS

ITEMS = ("wood", "sapling", "stone")


def collect_inventory_walk(steps: int, seed: int, phase_random: bool = True):
    import crafter
    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    obs = env.reset()
    if phase_random:
        env._step = int(rng.integers(0, 300))
    frames, ep_ids = [], []
    truth = {k: [] for k in ITEMS + ("food", "drink", "daylight")}
    ep = 0
    for t in range(steps):
        # collection-biased walk (discovery protocol, truth-free): 35%
        # 'do' presses, rest random — random play barely chops (wood
        # changed 125x in 40k steps; counters need events to be
        # discoverable). Same legitimacy class as phase randomization.
        u = rng.random()
        if u < 0.40:
            a = 5                                   # 'do' (chop/collect)
        elif u < 0.70 and t > 0:
            pass                                    # repeat last move —
            # persistence reaches trees that pure jitter never does
        else:
            a = int(rng.integers(1, 5))             # a movement action
        obs, r, done, info = env.step(a)
        frames.append(torch.from_numpy(obs.copy()).permute(2, 0, 1))
        inv = info["inventory"]
        for k in ITEMS:
            truth[k].append(float(inv.get(k, 0)))
        truth["food"].append(float(inv.get("food", 0)))
        truth["drink"].append(float(inv.get("drink", 0)))
        truth["daylight"].append(float(env._world.daylight))
        ep_ids.append(ep)
        if done:
            ep += 1
            obs = env.reset()
            if phase_random:
                env._step = int(rng.integers(0, 300))
    return (torch.stack(frames), {k: torch.tensor(v) for k, v in truth.items()},
            torch.tensor(ep_ids))


def main() -> None:
    t0 = time.time()
    frames, truth, ep_ids = collect_inventory_walk(60_000, seed=0)
    frames_t, truth_t, ep_t = collect_inventory_walk(15_000, seed=77)
    print(f"[pf-A] walk {frames.shape} ({time.time()-t0:.0f}s)", flush=True)

    # A2 coverage audit
    varying = []
    for k in ITEMS:
        mx, changes = float(truth[k].max()), int((truth[k][1:] != truth[k][:-1]).sum())
        print(f"[pf-A] coverage {k}: max {mx:.0f}, changes {changes}", flush=True)
        if changes >= 200:
            varying.append(k)
    print(f"[pf-A] routable candidates: {varying} "
          f"(others documented as coverage-excluded)", flush=True)

    # A3 exclusive-change differencing over the full frame (counters may
    # render outside the vitals strip)
    d = {k: (truth[k][1:] - truth[k][:-1]) for k in varying}
    others = {k: (truth[k][1:] - truth[k][:-1])
              for k in ("food", "drink")}
    same = (ep_ids[1:] == ep_ids[:-1])
    windows = {}
    for k in varying:
        m = same & (d[k].abs() >= 1)
        for j, dv in d.items():
            if j != k:
                m = m & (dv == 0)
        for j, dv in others.items():
            m = m & (dv == 0)
        if int(m.sum()) < 15:
            print(f"[pf-A] {k}: too few exclusive events ({int(m.sum())})",
                  flush=True)
            continue
        idx = m.nonzero().squeeze(1)
        diff = (frames[idx + 1].float() - frames[idx].float()).abs().mean(dim=(0, 1))
        hot = (diff > diff.max() * 0.35).nonzero()
        r0, r1 = int(hot[:, 0].min()), int(hot[:, 0].max()) + 1
        c0, c1 = int(hot[:, 1].min()), int(hot[:, 1].max()) + 1
        windows[k] = ((r0, r1), (c0, c1))
        print(f"[pf-A] {k}: n={int(m.sum())} window rows[{r0},{r1}) "
              f"cols[{c0},{c1})", flush=True)

    # A4 closed-form routing on held-out
    rho = measure_rho(truth["wood"], ep_ids, 20) if "wood" in varying else 0.9
    tau = -20 / math.log(max(min(float(rho), 0.995), 0.05))
    gates = {}
    for k, ((r0, r1), (c0, c1)) in windows.items():
        X = frames[:, :, r0:r1, c0:c1].float().div(255.0).flatten(1)
        Xt = frames_t[:, :, r0:r1, c0:c1].float().div(255.0).flatten(1)
        w, b = closed_form_head(X, ep_ids, 20, tau)
        z = Xt @ w + b
        c = abs(float(torch.corrcoef(torch.stack([z, truth_t[k]]))[0, 1])) \
            if float(truth_t[k].std()) > 1e-6 else float("nan")
        gates[k] = c
        print(f"[pf-A] routing {k}: {c:.3f} "
              f"({'PASS' if c >= 0.9 else 'FAIL'} vs 0.9)", flush=True)
    (RESULTS / "v40_preflight_A.json").write_text(json.dumps(
        {"varying": varying, "windows": {k: list(map(list, v))
                                         for k, v in windows.items()},
         "routing": gates}, indent=2))
    print(f"[pf-A] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
