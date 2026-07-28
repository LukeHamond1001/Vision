"""v0.5 scale-up, phase 1: slow-structure discovery from high-dimensional
entangled observations. Run:

    python -m iga.experiments_v05

Question: from observation-space random walks alone — where NO input column
is the slow variable (charge modulates 25 positional receptive fields
multiplicatively, plus indirect mixed features) — does the OU-ladder recipe
recover a banded latent whose slow band tracks the true slow variable?
Control: PCA on the same walks with the same band slicing (same data, no
temporal prior).

Pre-registered success criteria (evaluated on a HELD-OUT walk; true c is
used for EVALUATION ONLY, never in training):
  P1  OU slow band's max |corr| with c >= 0.9
  P2  OU beats PCA on slow-band c-tracking by a clear margin
  P3  OU fast band's c-corr < slow band's (routing, not just capture)
"""

from __future__ import annotations

import json

import torch

from .envs.chargeworld import ChargeWorld
from .envs.hdcharge import HDSensor, ObservationLatent
from .experiments import RESULTS
from .latent import BandedLatent
from .pretrain import pretrain_ou_ladder


def collect_hd_walk(sensor: HDSensor, steps: int, coverage_reset_every: int = 100,
                    seed: int = 0):
    """Random-policy walk; returns (observations, true_c) — c logged for
    evaluation only."""
    env = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=seed)
    gen = torch.Generator().manual_seed(seed)
    env.reset()
    obs, cs = [], []
    for t in range(steps):
        a = (torch.rand(2, generator=gen) * 2 - 1) * 0.1
        env.step(a)
        obs.append(sensor.observe(env.pos, env.c))
        cs.append(env.c)
        if (t + 1) % coverage_reset_every == 0:
            env.reset()
            env.pos = torch.rand(2, generator=gen)
            env.c = float(torch.rand((), generator=gen))
    return torch.stack(obs), torch.tensor(cs)


def band_c_corr(W: torch.Tensor, obs: torch.Tensor, c: torch.Tensor,
                band_dims=(6, 2)) -> dict:
    z = obs @ W.T
    out, off = {}, 0
    for k, bd in enumerate(band_dims):
        corr = torch.zeros(bd)
        for j in range(bd):
            corr[j] = torch.corrcoef(torch.stack([z[:, off + j], c]))[0, 1].abs()
        out[f"band{k}_max_c_corr"] = float(corr.max())
        off += bd
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    sensor = HDSensor()
    obs, c = collect_hd_walk(sensor, steps=20000, seed=0)
    obs_test, c_test = collect_hd_walk(sensor, steps=4000, seed=77)

    # OU-ladder recipe, 'band' context mode (no designated slow input)
    W_ou = pretrain_ou_ladder(obs, band_dims=[6, 2], taus=[10.0, 300.0],
                              lags=[15, 15], segment_len=100,
                              context_amp=0.3, context_mode="band", seed=0)

    # PCA control: top-8 components of the same (centered) walk, same slicing
    mu = obs.mean(0)
    _, _, V = torch.pca_lowrank(obs - mu, q=8)
    W_pca = V.T.contiguous()

    report = {
        "ou": band_c_corr(W_ou, obs_test, c_test),
        "pca": band_c_corr(W_pca, obs_test, c_test),
    }
    for name, r in report.items():
        print(f"[v0.5:p1] {name:4s} slow-band c-corr={r['band1_max_c_corr']:.3f} "
              f"fast-band c-corr={r['band0_max_c_corr']:.3f}", flush=True)

    p1 = report["ou"]["band1_max_c_corr"] >= 0.9
    p2 = report["ou"]["band1_max_c_corr"] - report["pca"]["band1_max_c_corr"] >= 0.1
    p3 = report["ou"]["band0_max_c_corr"] < report["ou"]["band1_max_c_corr"]
    print(f"[v0.5:p1] P1 (slow band tracks c >=0.9): {'PASS' if p1 else 'FAIL'}")
    print(f"[v0.5:p1] P2 (beats PCA by >=0.1):       {'PASS' if p2 else 'FAIL'}")
    print(f"[v0.5:p1] P3 (routing, not just capture):{'PASS' if p3 else 'FAIL'}")
    report["criteria"] = {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3)}
    (RESULTS / "v05_phase1_discovery.json").write_text(json.dumps(report, indent=2))
    print("wrote results/v05_phase1_discovery.json")


if __name__ == "__main__":
    main()
