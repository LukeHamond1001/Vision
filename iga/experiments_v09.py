"""v0.9 phase 1: three-band discovery on Crafter. Run:

    python -m iga.experiments_v09 tiny    # CPU smoke
    python -m iga.experiments_v09 full    # GPU: measured taus + pretraining + routing

Pre-registered gates (written before any full-scale run):
  G-slow   slow band |corr| with daylight >= 0.8 on held-out episodes,
           and exceeding the PCA control's best daylight dim by >= 0.1
  G-mid    mid band's best meter (food or drink) >= 0.8
  Report   full 3-band x 5-variable routing matrix; cross-band leaks
           reported, not hidden (Crafter renders deterministically, so
           HUD micro-leaks into global stats are expected per the
           nuisance-ladder law — the claim is routing, not leak-freedom)

Timescale priors are MEASURED (R-taus): empirical within-episode
autocorrelations of the ground-truth variables set tau per band.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from .crafter_support import (collect_crafter_walk, ema_slow_stats,
                              measure_rho, partial_corr,
                              pretrain_crafter_encoder, routing_matrix)
from .experiments import RESULTS
from .experiments_v07 import PixelPCA, encode_all, git_push_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (16, 2, 2)   # round 6: fast band 4->16. Behavioral runs 1-2
# (PPO and replay-DQN, both flat at up-to-250k steps with a verified
# -0.984-aligned dense reward) localized the ceiling to PERCEPTION: the
# policy cannot navigate-to/face water from a direction-blind state; 4
# fast dims cannot carry egocentric geometry. The spec's fast band is
# capacity-unconstrained by design — (4,2,2) was frugality, not doctrine.
# Slow/mid pathways and their gates are untouched.


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[v0.9] device={DEVICE} mode={mode}", flush=True)

    frames, truth, ep_ids = collect_crafter_walk(2000 if tiny else 40000, seed=0,
                                                 phase_random=True)
    frames_t, truth_t, ep_t = collect_crafter_walk(500 if tiny else 5000, seed=77,
                                                   phase_random=True)
    tt = float(torch.corrcoef(torch.stack([truth_t["daylight"],
                                           truth_t["food"]]))[0, 1])
    print(f"[v0.9] truth collinearity daylight-food (phase-randomized): "
          f"{tt:+.3f}", flush=True)
    print(f"[v0.9] walk {frames.shape} episodes={int(ep_ids.max())+1} "
          f"({time.time()-t0:.0f}s)", flush=True)
    for var in ("daylight", "food", "drink"):
        print(f"[v0.9] coverage {var}: min={float(truth[var].min()):.2f} "
              f"max={float(truth[var].max()):.2f}", flush=True)

    # R-taus: measure, then set the priors from data
    lag_probe = {"fast": 5, "mid": 20, "slow": 40}
    rho_day = measure_rho(truth["daylight"], ep_ids, lag_probe["slow"])
    rho_food = measure_rho(truth["food"], ep_ids, lag_probe["mid"])
    tau_slow = -lag_probe["slow"] / math.log(max(min(rho_day, 0.995), 0.05))
    tau_mid = -lag_probe["mid"] / math.log(max(min(rho_food, 0.995), 0.05))
    taus = (5.0, float(tau_mid), float(tau_slow))
    lags = (lag_probe["fast"], lag_probe["mid"], lag_probe["slow"])
    print(f"[v0.9] measured rho: daylight@40={rho_day:.3f} food@20={rho_food:.3f} "
          f"-> taus={tuple(round(x,1) for x in taus)}", flush=True)
    git_push_results("v09-collection")

    enc = pretrain_crafter_encoder(frames, ep_ids, band_dims=BANDS,
                                   taus=taus, lags=lags,
                                   epochs=60 if tiny else 1500,
                                   batch=256 if tiny else 512,
                                   device=DEVICE,
                                   log_every=20 if tiny else 100)
    print(f"[v0.9] pretraining done ({time.time()-t0:.0f}s)", flush=True)
    torch.save(enc.state_dict(), RESULTS / "v09_encoder.pt")

    slow_t = ema_slow_stats(frames_t, ep_t, device=DEVICE)
    with torch.no_grad():
        z = torch.cat([enc(frames_t[i:i + 512].float().div(255.0).to(DEVICE),
                           slow_feats=slow_t[i:i + 512]).cpu()
                       for i in range(0, frames_t.shape[0], 512)])
    matrix = routing_matrix(z, truth_t, BANDS)
    pca = PixelPCA(frames)
    zp = encode_all(pca, frames_t)
    pca_day = max(float(torch.corrcoef(torch.stack([zp[:, j],
                                                    truth_t["daylight"]]))[0, 1].abs())
                  for j in range(zp.shape[1]))

    print("[v0.9] routing matrix (band x variable, max |corr|):", flush=True)
    for band in ("fast", "mid", "slow"):
        row = {v: round(matrix[f'{band}_{v}'], 2)
               for v in ("daylight", "food", "drink", "energy", "health")}
        print(f"[v0.9]   {band:5s} {row}", flush=True)
    print(f"[v0.9] pca best daylight dim: {pca_day:.3f}", flush=True)

    # Partial-corr attribution (round 3, pre-registered before the run):
    # collinearity-robust routing — the slow band must carry daylight BEYOND
    # what meters explain, and the mid band meters beyond daylight.
    off = BANDS[0]
    best_meter = truth_t["food"] if matrix["mid_food"] >= matrix["mid_drink"] \
        else truth_t["drink"]
    p_slow = max(abs(partial_corr(z[:, off + BANDS[1] + j], truth_t["daylight"],
                                  best_meter)) for j in range(BANDS[2]))
    p_mid = max(abs(partial_corr(z[:, off + j], best_meter,
                                 truth_t["daylight"])) for j in range(BANDS[1]))
    print(f"[v0.9] partial: slow-daylight|meter {p_slow:.3f}  "
          f"mid-meter|daylight {p_mid:.3f}", flush=True)

    g_slow = matrix["slow_daylight"] >= 0.8 and \
        matrix["slow_daylight"] - pca_day >= 0.1
    g_mid = max(matrix["mid_food"], matrix["mid_drink"]) >= 0.8
    g_part = p_slow >= 0.5 and p_mid >= 0.5
    print(f"[v0.9] G-slow (daylight >=0.8, beats pca): "
          f"{'PASS' if g_slow else 'FAIL'}", flush=True)
    print(f"[v0.9] G-mid  (best meter >=0.8):          "
          f"{'PASS' if g_mid else 'FAIL'}", flush=True)
    print(f"[v0.9] G-part (partials >=0.5):            "
          f"{'PASS' if g_part else 'FAIL'}", flush=True)

    matrix["pca_daylight"] = pca_day
    matrix["taus"] = list(taus)
    matrix["partials"] = {"slow_daylight_given_meter": float(p_slow),
                          "mid_meter_given_daylight": float(p_mid)}
    matrix["gates"] = {"slow": bool(g_slow), "mid": bool(g_mid),
                      "part": bool(g_part)}
    (RESULTS / "v09_routing.json").write_text(json.dumps(matrix, indent=2))
    git_push_results("v09-verdicts")
    print(f"[v0.9] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
