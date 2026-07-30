"""v0.9 phase 2: the wired agent on Crafter. Run:

    python -m iga.experiments_v10 tiny    # CPU smoke
    python -m iga.experiments_v10 full    # 10 seeds x 2 arms survival table

Pre-registered gates (written 2026-07-30, before any full-scale run):
  G-behav  IQM over seeds of last-third mean survival, wiring-on vs
           wiring-off: ratio >= 1.15 AND the 95% bootstrap CI of the
           per-seed paired difference excludes 0.
  Report   full per-seed table (survival + food-hold fraction), IQM +
           bootstrap CIs both arms, rliable-style. Both arms share the
           SAME frozen encoder, policy class, learner, and inputs; only
           the reward stream differs (register progress in the frozen
           mid band vs Crafter's native reward).

Protocol: behavior runs STANDARD Crafter (morning spawns — phase
randomization was a phase-1 discovery-protocol tool). The encoder is
the phase-1 artifact (results/v09_encoder.pt), loaded frozen. The
register target is the well-fed centroid of a phase-randomized
calibration walk, expressed in the latent vocabulary (designers point
at goals; discovery only forbids designating variables to the encoder).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from .crafter_agent import estimate_register_target, iqm, train_arm
from .crafter_support import EncoderCrafter, collect_crafter_walk
from .experiments import RESULTS
from .experiments_v07 import git_push_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (4, 2, 2)
MID = slice(BANDS[0], BANDS[0] + BANDS[1])


def bootstrap_ci(xs, n=4000, seed=0, q=(2.5, 97.5)):
    gen = np.random.default_rng(seed)
    xs = np.asarray(xs, dtype=np.float64)
    stats = [iqm(gen.choice(xs, size=len(xs), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(stats, q)
    return float(lo), float(hi)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    seeds = range(1, 3 if tiny else 11)
    episodes = 6 if tiny else 150
    max_steps = 150 if tiny else 1000
    print(f"[v1.0] device={DEVICE} mode={mode} seeds={list(seeds)}", flush=True)

    enc = EncoderCrafter(BANDS, seed=0)
    state = torch.load(RESULTS / "v09_encoder.pt", map_location="cpu")
    enc.load_state_dict(state)
    enc.freeze()
    enc = enc.to(DEVICE)
    print("[v1.0] phase-1 encoder loaded (frozen)", flush=True)

    frames, truth, _ = collect_crafter_walk(3000 if tiny else 20000, seed=123,
                                            phase_random=True)
    with torch.no_grad():
        z_walk = torch.cat([enc(frames[i:i + 512].float().div(255.0).to(DEVICE)).cpu()
                            for i in range(0, frames.shape[0], 512)])
    g_mid = estimate_register_target(z_walk, truth, MID)
    n_wellfed = int(((truth["food"] >= 8) & (truth["drink"] >= 8)).sum())
    print(f"[v1.0] register target {g_mid.tolist()} "
          f"(well-fed frames in walk: {n_wellfed})", flush=True)
    del frames, z_walk

    rows = []
    log = (RESULTS / "v10_behavior.jsonl").open("w")
    for seed in seeds:
        for wiring in (True, False):
            out = train_arm(enc, g_mid, MID, wiring, seed=seed,
                            episodes=episodes, max_steps=max_steps,
                            device=DEVICE)
            k = max(1, len(out["survival"]) // 3)
            row = {"seed": seed, "wiring": wiring,
                   "survival_last3rd": float(np.mean(out["survival"][-k:])),
                   "food_frac_last3rd": float(np.mean(out["food_frac"][-k:])),
                   "survival_all": out["survival"],
                   "food_frac_all": [round(f, 3) for f in out["food_frac"]]}
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[v1.0] seed {seed} wiring={int(wiring)} "
                  f"last3rd survival {row['survival_last3rd']:.0f} "
                  f"food_frac {row['food_frac_last3rd']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        git_push_results(f"v10-seed{seed}")
    log.close()

    on = [r["survival_last3rd"] for r in rows if r["wiring"]]
    off = [r["survival_last3rd"] for r in rows if not r["wiring"]]
    diff = [a - b for a, b in zip(on, off)]
    iqm_on, iqm_off = iqm(on), iqm(off)
    ci_on, ci_off = bootstrap_ci(on), bootstrap_ci(off)
    d_lo, d_hi = bootstrap_ci(diff)
    ratio = iqm_on / max(iqm_off, 1e-6)
    g_behav = ratio >= 1.15 and d_lo > 0
    print(f"[v1.0] IQM survival wiring-on {iqm_on:.0f} CI [{ci_on[0]:.0f},"
          f"{ci_on[1]:.0f}]  wiring-off {iqm_off:.0f} CI [{ci_off[0]:.0f},"
          f"{ci_off[1]:.0f}]", flush=True)
    print(f"[v1.0] ratio {ratio:.2f}  paired-diff CI [{d_lo:.0f},{d_hi:.0f}]",
          flush=True)
    print(f"[v1.0] G-behav (ratio>=1.15, diff CI>0): "
          f"{'PASS' if g_behav else 'FAIL'}", flush=True)

    summary = {"iqm_on": iqm_on, "iqm_off": iqm_off, "ci_on": ci_on,
               "ci_off": ci_off, "diff_ci": [d_lo, d_hi], "ratio": ratio,
               "gate_behav": bool(g_behav),
               "register_target": g_mid.tolist(),
               "food_frac_on": [r["food_frac_last3rd"] for r in rows if r["wiring"]],
               "food_frac_off": [r["food_frac_last3rd"] for r in rows if not r["wiring"]]}
    (RESULTS / "v10_summary.json").write_text(json.dumps(summary, indent=2))
    git_push_results("v10-verdicts")
    print(f"[v1.0] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
