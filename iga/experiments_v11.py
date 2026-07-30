"""v1.1: the three-arm wired-reward experiment. Run:

    python -m iga.experiments_v11 tiny          # CPU smoke, all arms
    python -m iga.experiments_v11 full <seed>   # one seed, three arms (pod)

Design (2026-07-30, pre-registered before any full run):
The policy is a standard pixel Q-net in EVERY arm; the discovered latent
is the REWARD SUBSTRATE only (registers/wiring) — the architecture's
actual claim, disentangled from state sufficiency (runs 1-3 measured the
cost of conflating them). Arms differ ONLY in the reward stream:

  wired   register progress in the frozen mid band (no env reward, no
          labels, no reward engineering — wire it, don't train it)
  native  Crafter's built-in reward (strong baseline: health deltas +
          achievements)
  zero    no reward (isolates 'BECAUSE of the wired reward')

Gates (primary first):
  G-wired-zero    survival IQM(wired) / IQM(zero) >= 1.25 across seeds,
                  paired-diff bootstrap 95% CI > 0  [the user's sentence:
                  "the agent survives longer because of the wired reward"]
  G-wired-native  survival IQM(wired) / IQM(native) >= 1.0 with
                  overlapping-or-better CIs = "matches the engineered
                  reward"; ratio >= 1.15 with CI > 0 = "beats it"
  Mechanism       drink%/food% separate in wired arm (the register
                  points at meters; the signature must too)

Each arm pushes partial results every ~100 episodes (jsonl per seed).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

from .crafter_agent import estimate_register_target, iqm, train_arm_pixel_dqn
from .crafter_support import EncoderCrafter, collect_crafter_walk, ema_slow_stats
from .experiments import RESULTS
from .experiments_v07 import git_push_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (16, 2, 2)
MID = slice(BANDS[0], BANDS[0] + BANDS[1])
ARMS = ("wired", "native", "zero")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    total_steps = 3000 if tiny else 1_000_000
    print(f"[v1.1] device={DEVICE} mode={mode} seed={seed} "
          f"steps/arm={total_steps}", flush=True)

    enc = EncoderCrafter(BANDS, seed=0)
    enc.load_state_dict(torch.load(RESULTS / "v09_encoder.pt", map_location="cpu"))
    enc.freeze()
    enc = enc.to(DEVICE)
    frames, truth, ep_w, _ = collect_crafter_walk(2000 if tiny else 20000,
                                                  seed=123, phase_random=True)
    slow_w = ema_slow_stats(frames, ep_w, device=DEVICE)
    with torch.no_grad():
        z_walk = torch.cat([enc(frames[i:i + 512].float().div(255.0).to(DEVICE),
                                slow_feats=slow_w[i:i + 512]).cpu()
                            for i in range(0, frames.shape[0], 512)])
    g_mid = estimate_register_target(z_walk, truth, MID)
    print(f"[v1.1] register target {[round(float(x), 3) for x in g_mid]}",
          flush=True)
    del frames, z_walk

    out_rows = []
    for arm in ARMS:
        out = train_arm_pixel_dqn(enc, g_mid, MID, arm, seed=seed,
                                  total_steps=total_steps,
                                  max_steps=200 if tiny else 1000,
                                  device=DEVICE,
                                  log_every_eps=5 if tiny else 50)
        torch.save(out.pop("policy_state"),
                   RESULTS / f"v11_policy_s{seed}_{arm}.pt")
        k = max(1, len(out["survival"]) // 3)
        row = {"seed": seed, "arm": arm,
               "survival_last3rd": float(np.mean(out["survival"][-k:])),
               "food_frac_last3rd": float(np.mean(out["food_frac"][-k:])),
               "drink_frac_last3rd": float(np.mean(out["drink_frac"][-k:])),
               "survival_all": out["survival"],
               "food_frac_all": [round(f, 3) for f in out["food_frac"]],
               "drink_frac_all": [round(f, 3) for f in out["drink_frac"]]}
        out_rows.append(row)
        (RESULTS / f"v11_seed{seed}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in out_rows))
        print(f"[v1.1] seed {seed} arm {arm}: last3rd survival "
              f"{row['survival_last3rd']:.0f} food% {row['food_frac_last3rd']:.2f} "
              f"drink% {row['drink_frac_last3rd']:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
        git_push_results(f"v11-s{seed}-{arm}")
    print(f"[v1.1] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
