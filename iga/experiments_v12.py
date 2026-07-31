"""v1.2: homeostasis three-arm campaign under vectorized PPO. Run:

    python -m iga.experiments_v12 tiny          # CPU smoke
    python -m iga.experiments_v12 full <seed>   # one seed, three arms (pod)

Pre-registered (2026-07-31, before any full run):
  Arms     wired = register progress toward the HOMEOSTASIS target (all
           four vitals held high) in the frozen round-8 mid band;
           native = Crafter reward; zero = nothing. Same PPO learner,
           same worlds, same nets everywhere.
  G-wired-zero    survival IQM(wired)/IQM(zero) >= 1.25, paired-diff
                  bootstrap 95% CI > 0  [primary — "survives longer
                  BECAUSE of the wired reward"]
  G-wired-native  reported both directions; ratio >= 1.0 with CIs =
                  "matches engineered reward", >= 1.15 CI>0 = "beats it"
  Mechanism       wired arm separates on vital uptime (food/drink/
                  energy) and sleeps-at-night rate — the register's
                  fingerprint; measured, not assumed.

Lineage note: v1.1 (single-env DQN, 1M steps) measured native = zero =
175 — that learner regime teaches nothing from ANY reward; this is the
published-regime retry (parallel PPO, 3M steps), with the register
upgraded from food/drink to the full vital state (round 8).
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
ARMS = ("wired", "native", "zero")


def homeostasis_target(z_walk: torch.Tensor, truth: dict) -> torch.Tensor:
    """Centroid of the mid band over frames with ALL vitals high — the
    register's 'healthy state'. Falls back to top-decile vitality sum."""
    m = ((truth["food"] >= 8) & (truth["drink"] >= 8)
         & (truth["energy"] >= 8) & (truth["health"] >= 8))
    if int(m.sum()) < 64:
        vit = (truth["food"] + truth["drink"] + truth["energy"]
               + truth["health"]).float()
        m = vit >= torch.quantile(vit, 0.9)
    return z_walk[m][:, MID].mean(0)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    total_steps = 6144 if tiny else 3_000_000
    n_envs = 4 if tiny else 12
    print(f"[v1.2] device={DEVICE} mode={mode} seed={seed} "
          f"steps/arm={total_steps} envs={n_envs}", flush=True)

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
    g = homeostasis_target(z_walk, truth)
    print(f"[v1.2] homeostasis target {[round(float(x), 3) for x in g]}",
          flush=True)
    del frames, z_walk

    rows = []
    for arm in ARMS:
        t1 = time.time()

        pushed = [0]

        def cb(steps, stats, _arm=arm, _t1=t1):
            if stats["survival"] and steps % (n_envs * 2560) == 0:
                print(f"[v1.2] {_arm} steps {steps} eps "
                      f"{len(stats['survival'])} "
                      f"surv(last50) {np.mean(stats['survival'][-50:]):.0f} "
                      f"slept(last50) {np.mean(stats['slept'][-50:]):.1f} "
                      f"({time.time()-_t1:.0f}s)", flush=True)
            if steps - pushed[0] >= 400_000:   # mid-arm heartbeat for the
                pushed[0] = steps              # pilot's tripwires
                import subprocess
                subprocess.run(["git", "add", "-A", "run.log"],
                               capture_output=True)
                subprocess.run(["git", "commit", "-qm",
                                f"hb-{_arm}-{steps}"], capture_output=True)
                subprocess.run(["git", "push", "-qf", "origin", "HEAD"],
                               capture_output=True)

        out = run_ppo_arm(enc, g, MID, arm, seed=seed,
                          total_steps=total_steps, n_envs=n_envs,
                          rollout=64 if tiny else 256,
                          device=DEVICE, log_cb=cb)
        torch.save(out.pop("policy_state"),
                   RESULTS / f"v12_policy_s{seed}_{arm}.pt")
        k = max(1, len(out["survival"]) // 3)
        row = {"seed": seed, "arm": arm,
               "survival_last3rd": float(np.mean(out["survival"][-k:])),
               "food_last3rd": float(np.mean(out["food"][-k:])),
               "drink_last3rd": float(np.mean(out["drink"][-k:])),
               "energy_last3rd": float(np.mean(out["energy"][-k:])),
               "slept_last3rd": float(np.mean(out["slept"][-k:])),
               "survival_all": out["survival"],
               "slept_all": out["slept"]}
        rows.append(row)
        (RESULTS / f"v12_seed{seed}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        print(f"[v1.2] seed {seed} {arm}: last3rd surv "
              f"{row['survival_last3rd']:.0f} food {row['food_last3rd']:.2f} "
              f"drink {row['drink_last3rd']:.2f} energy "
              f"{row['energy_last3rd']:.2f} slept {row['slept_last3rd']:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        git_push_results(f"v12-s{seed}-{arm}")
    print(f"[v1.2] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
