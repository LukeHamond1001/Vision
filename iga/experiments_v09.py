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

from .crafter_support import (collect_crafter_walk, measure_rho,
                              pretrain_crafter_encoder, routing_matrix)
from .experiments import RESULTS
from .experiments_v07 import PixelPCA, encode_all, git_push_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = (4, 2, 2)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[v0.9] device={DEVICE} mode={mode}", flush=True)

    frames, truth, ep_ids = collect_crafter_walk(2000 if tiny else 40000, seed=0)
    frames_t, truth_t, ep_t = collect_crafter_walk(500 if tiny else 5000, seed=77)
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

    z = encode_all(enc, frames_t)
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

    g_slow = matrix["slow_daylight"] >= 0.8 and \
        matrix["slow_daylight"] - pca_day >= 0.1
    g_mid = max(matrix["mid_food"], matrix["mid_drink"]) >= 0.8
    print(f"[v0.9] G-slow (daylight >=0.8, beats pca): "
          f"{'PASS' if g_slow else 'FAIL'}", flush=True)
    print(f"[v0.9] G-mid  (best meter >=0.8):          "
          f"{'PASS' if g_mid else 'FAIL'}", flush=True)

    matrix["pca_daylight"] = pca_day
    matrix["taus"] = list(taus)
    matrix["gates"] = {"slow": bool(g_slow), "mid": bool(g_mid)}
    (RESULTS / "v09_routing.json").write_text(json.dumps(matrix, indent=2))
    git_push_results("v09-verdicts")
    print(f"[v0.9] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
