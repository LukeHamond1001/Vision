"""v1.3: the glass-box goal-swap. Run:

    python -m iga.experiments_v13 tiny
    python -m iga.experiments_v13 full   # pod: swap (warm) then fresh arm

Pre-registered (2026-08-01, before any full run): edit ONE line of the
want — the register target moves from the homeostasis centroid (all four
vitals high; produced slept/ep 2.7-3.7 across five seeds) to the
FED-NOT-RESTED centroid (food & drink high only; energy and health drop
out of the want). Same encoder, same learner, same worlds.

  P1  night-sleeping collapses:  slept_last3rd < 1.0 /episode
  P2  drink maintenance holds:   drink_last3rd >= 0.80
  P3  energy uptime sags:        energy_last3rd < 0.98

Arms: 'swap'  = warm-start from the trained seed-1 homeostat policy,
                1.5M steps under the edited target (the on-camera cut:
                the same agent changes its mind);
      'fresh' = new policy trained 3M steps under the edited target
                (the paper-table row).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

from .crafter_support import EncoderCrafter, collect_crafter_walk, ema_slow_stats
from .experiments import RESULTS
from .experiments_v07 import git_push_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (16, 4, 2)
MID = slice(BANDS[0], BANDS[0] + BANDS[1])


def fed_not_rested_target(z_walk, truth):
    m = (truth["food"] >= 8) & (truth["drink"] >= 8)
    return z_walk[m][:, MID].mean(0)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    n_envs = 4 if tiny else 12
    print(f"[v1.3] device={DEVICE} mode={mode}", flush=True)

    from .ppo_pixel import run_ppo_arm
    enc = EncoderCrafter(BANDS, seed=0)
    enc.load_state_dict(torch.load(RESULTS / "v09_encoder.pt",
                                   map_location="cpu"))
    enc.freeze()
    enc = enc.to(DEVICE)
    frames, truth, ep_w, _ = collect_crafter_walk(2000 if tiny else 20000,
                                                  seed=123, phase_random=True)
    slow_w = ema_slow_stats(frames, ep_w, device=DEVICE)
    with torch.no_grad():
        z_walk = torch.cat([enc(frames[i:i + 512].float().div(255.0).to(DEVICE),
                                slow_feats=slow_w[i:i + 512]).cpu()
                            for i in range(0, frames.shape[0], 512)])
    g = fed_not_rested_target(z_walk, truth)
    print(f"[v1.3] fed-not-rested target {[round(float(x), 3) for x in g]}",
          flush=True)
    del frames, z_walk

    warm = torch.load(RESULTS / "v12_policy_s1_wired.pt", map_location="cpu")
    plans = [("swap", 4000 if tiny else 1_500_000, warm),
             ("fresh", 4000 if tiny else 3_000_000, None)]
    rows = []
    for arm, steps, init in plans:
        t1 = time.time()

        def cb(s_, stats, _a=arm, _t=t1):
            if stats["survival"] and s_ % (n_envs * 2560) == 0:
                print(f"[v1.3] {_a} steps {s_} "
                      f"surv(last50) {np.mean(stats['survival'][-50:]):.0f} "
                      f"slept(last50) {np.mean(stats['slept'][-50:]):.1f} "
                      f"drink(last50) {np.mean(stats['drink'][-50:]):.2f} "
                      f"({time.time()-_t:.0f}s)", flush=True)

        out = run_ppo_arm(enc, g, MID, "wired", seed=1, total_steps=steps,
                          n_envs=n_envs, rollout=64 if tiny else 256,
                          device=DEVICE, log_cb=cb, init_state=init)
        torch.save(out.pop("policy_state"), RESULTS / f"v13_policy_{arm}.pt")
        k = max(1, len(out["survival"]) // 3)
        row = {"arm": arm,
               "survival_last3rd": float(np.mean(out["survival"][-k:])),
               "food_last3rd": float(np.mean(out["food"][-k:])),
               "drink_last3rd": float(np.mean(out["drink"][-k:])),
               "energy_last3rd": float(np.mean(out["energy"][-k:])),
               "slept_last3rd": float(np.mean(out["slept"][-k:]))}
        rows.append(row)
        (RESULTS / "v13_goalswap.json").write_text(json.dumps(rows, indent=2))
        p1 = row["slept_last3rd"] < 1.0
        p2 = row["drink_last3rd"] >= 0.80
        p3 = row["energy_last3rd"] < 0.98
        print(f"[v1.3] {arm}: surv {row['survival_last3rd']:.0f} "
              f"drink {row['drink_last3rd']:.2f} energy "
              f"{row['energy_last3rd']:.2f} slept {row['slept_last3rd']:.1f} "
              f"| P1 sleep-collapse {'PASS' if p1 else 'FAIL'} "
              f"P2 drink-holds {'PASS' if p2 else 'FAIL'} "
              f"P3 energy-sags {'PASS' if p3 else 'FAIL'}", flush=True)
        git_push_results(f"v13-{arm}")
    print(f"[v1.3] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
