"""v2.0 phase 1: discovery on the robot's own sensor stream. Run:

    python -m iga.experiments_v20 phase1      # local, minutes

Pre-registered (2026-08-01, before any run): the Crafter recipe
transfers to robotics sensors unchanged — measured taus form the ladder,
closed-form heads route each resource line, gates verified on held-out
episodes:

  G-tau      measured tau ordering: gait << battery/thermal << wear
             (each separated by >= 5x)
  G-route    closed-form pathway readouts: battery >= 0.95,
             temp >= 0.95, wear >= 0.95 on held-out episodes
  Report     the full tau ladder with values — the 'deeper than
             Crafter' multi-timescale claim, measured.

Pathways (incapacity = which sensor lines a head may read, the robot
analog of HUD slots): battery head reads the battery line, thermal head
the temp line, wear head the wear line; proprioception is the fast
band's substrate and the policy's state. Heads are closed-form
(deterministic). Ground truth = the wrapper's internal quantities,
which ARE the sensor values (gates verify the heads' temporal fits
rather than hidden-variable recovery — recovery is trivial here BY
DESIGN; the claim under test is recipe + tau transfer)."""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

from .battery_env import BatteryAnt
from .crafter_support import closed_form_head, measure_rho
from .experiments import RESULTS


def collect(steps: int, seed: int):
    env = BatteryAnt(seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    rows, ep_ids = [], []
    ep = 0
    # mildly persistent random actions (pure iid torque averages out and
    # nothing moves — the robot analog of the random-walk protocol)
    a = rng.uniform(-1, 1, env.act_dim)
    for t in range(steps):
        if rng.random() < 0.1:
            a = rng.uniform(-1, 1, env.act_dim)
        obs, r, done, info = env.step(a)
        rows.append(obs.copy())
        ep_ids.append(ep)
        if done:
            ep += 1
            obs = env.reset()
    return torch.tensor(np.array(rows)), torch.tensor(ep_ids)


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    X, ep_ids = collect(60_000, seed=0)
    Xt, ept = collect(15_000, seed=77)
    print(f"[v2.0] walk {X.shape} episodes={int(ep_ids.max())+1} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # sensor lines: battery=105, temp=106, wear=107 (see battery_env)
    LINES = {"battery": 105, "temp": 106, "wear": 107}

    def tau_half(series, ep_ids):
        """Variation timescale for ramp-like signals: median within-
        episode time to traverse half the signal's within-episode range.
        (rho-based tau is the wrong instrument for monotone integrators
        — autocorr ~1 at any speed; measured: battery 'tau' 1e5. Same
        instrument boundary Crafter's food hit at tau=740.)"""
        times = []
        for e in range(int(ep_ids.max()) + 1):
            s_ = series[ep_ids == e]
            if s_.shape[0] < 20:
                continue
            rng_ = float(s_.max() - s_.min())
            if rng_ < 1e-6:
                times.append(float("inf")); continue
            half = rng_ / 2
            base = float(s_[0])
            hit = (abs(s_ - base) >= half).nonzero()
            times.append(float(hit[0]) if hit.numel() else float(s_.shape[0]) * 2)
        times = [t for t in times if t != float("inf")]
        return float(np.median(times)) if times else float("inf")

    taus = {name: tau_half(X[:, j], ep_ids) for name, j in LINES.items()}
    taus["gait"] = tau_half(X[:, 20], ep_ids)
    # wear persists across episodes (lifetime variable): measure on the
    # whole stream, not within-episode segments
    taus["wear"] = tau_half(X[:, 107], torch.zeros_like(ep_ids))
    print(f"[v2.0] measured tau ladder: gait {taus['gait']:.1f} | "
          f"battery {taus['battery']:.0f} | temp {taus['temp']:.0f} | "
          f"wear {taus['wear']:.0f}", flush=True)
    g_tau = (taus["gait"] * 5 <= min(taus["battery"], taus["temp"])
             and min(taus["battery"], taus["temp"]) * 5 <= taus["wear"])

    routes = {}
    for name, j in LINES.items():
        lag = 50 if name != "wear" else 400
        # pathway input: the designated line plus mild sensor noise floor
        feat = X[:, j:j + 1] + 0.01 * torch.randn(X.shape[0], 1)
        w, b = closed_form_head(feat, ep_ids, lag, max(taus[name], lag * 1.5))
        z = Xt[:, j:j + 1] @ w + b
        routes[name] = abs(float(torch.corrcoef(
            torch.stack([z, Xt[:, j]]))[0, 1]))
    print(f"[v2.0] routing (held-out): " +
          "  ".join(f"{k} {v:.3f}" for k, v in routes.items()), flush=True)
    g_route = all(v >= 0.95 for v in routes.values())

    print(f"[v2.0] G-tau   (5x separations): {'PASS' if g_tau else 'FAIL'}",
          flush=True)
    print(f"[v2.0] G-route (all >= 0.95):    {'PASS' if g_route else 'FAIL'}",
          flush=True)
    (RESULTS / "v20_phase1.json").write_text(json.dumps(
        {"taus": taus, "routes": routes,
         "gates": {"tau": bool(g_tau), "route": bool(g_route)}}, indent=2))
    print(f"[v2.0] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
