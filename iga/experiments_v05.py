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


def band_c_corr_lat(lat, obs: torch.Tensor, c: torch.Tensor,
                    band_dims=(6, 2)) -> dict:
    """band_c_corr for any latent (matrix or frozen module)."""
    with torch.no_grad():
        z = torch.stack([lat.embed_obs(o) for o in obs])
    out, off = {}, 0
    for k, bd in enumerate(band_dims):
        corr = torch.zeros(bd)
        for j in range(bd):
            corr[j] = torch.corrcoef(torch.stack([z[:, off + j], c]))[0, 1].abs()
        out[f"band{k}_max_c_corr"] = float(corr.max())
        off += bd
    return out


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


def make_encoders(sensor: HDSensor, include_iso: bool = False):
    """Train the OU encoder(s) and the PCA control on the same walk; normalize
    all so fast-band step-scale ≈ 1 (identical calibration — the comparison
    isolates ROUTING quality, nothing else). `include_iso` adds the v0.6
    isometry-regularized variant."""
    from .envs.hdcharge import ObservationLatent

    obs, _ = collect_hd_walk(sensor, steps=20000, seed=0)
    variants = {"ou": dict()}
    if include_iso == "iso":
        variants["ou_iso"] = dict(lam_iso=5.0)
    elif include_iso == "geo":
        from .pretrain import geodesic_pairs
        variants["ou_geo"] = dict(geo=geodesic_pairs(obs), lam_geo=5.0)
    Ws = {name: pretrain_ou_ladder(obs, band_dims=[6, 2], taus=[10.0, 300.0],
                                   lags=[15, 15], segment_len=100,
                                   context_amp=0.3, context_mode="band",
                                   seed=0, **kw)
          for name, kw in variants.items()}
    mu = obs.mean(0)
    _, _, V = torch.pca_lowrank(obs - mu, q=8)
    Ws["pca"] = V.T.contiguous()
    out = {}
    for name, W in Ws.items():
        lat = ObservationLatent(W, [6, 2], sensor)
        out[name] = ObservationLatent(W / lat.step_scale(0), [6, 2], sensor)
    if include_iso == "mlp":
        from .envs.hdcharge import NonlinearObservationLatent
        from .pretrain import geodesic_pairs, pretrain_mlp_encoder
        enc = pretrain_mlp_encoder(obs, band_dims=[6, 2], taus=[10.0, 300.0],
                                   lags=[15, 15], segment_len=100,
                                   context_amp=0.3, geo=geodesic_pairs(obs),
                                   lam_geo=5.0, seed=0)
        lat = NonlinearObservationLatent(enc, [6, 2], sensor)
        lat._post_scale(lat.step_scale(0))
        out["ou_mlp"] = lat
    return out


def build_phase2(seed: int, which: str, encoders) -> tuple:
    from .envs.hdcharge import HDChargeWorld
    from .heads import FixedRewardHead, SumHead
    from .ladder import LadderAgent, LadderConfig

    lat = encoders[which]
    W0 = getattr(lat, "_W", torch.zeros(8, lat.world_dim))   # MLP latents have no matrix
    env = HDChargeWorld(W0, [6, 2], sensor=lat.sensor, seed=seed)
    env.latent = lat            # share the calibrated instance
    gen = torch.Generator().manual_seed(seed + 4000)
    world = torch.cat([torch.rand(384, 2, generator=gen),
                       torch.rand(384, 1, generator=gen)], dim=1)
    support = torch.stack([lat.embed(w) for w in world])
    # Long-range geometry calibration (identical procedure both arms):
    # learned latents warp far distances (RBF saturation compresses them —
    # measured far/step ratio 2.8 in OU vs ~7.6 rigid), so every
    # distance-valued parameter is scaled by the latent's measured
    # start->pad distance relative to the rigid-world 0.76.
    with torch.no_grad():
        rho = float(torch.linalg.vector_norm(
            env.embed_world(env.start, 0.0) - env.embed_world(env.pad, 0.0))) / 0.76
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.door, 1.0), sigma=0.30 * rho, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.1), sigma=0.20 * rho, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.6), sigma=0.20 * rho, proxy_samples=support)],
        weights=[1.0, 0.3, 0.3])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.12 * rho,
                               proxy_samples=support)
    # arrive-eps calibrated to the MEASURED slow-band charge amplitude
    # (round-10 requirement: radii scaled to band metrics, applied
    # prospectively; identical procedure for both encoders)
    with torch.no_grad():
        amp_slow = float(torch.linalg.vector_norm(
            (env.embed_world((0.5, 0.5), 1.0)
             - env.embed_world((0.5, 0.5), 0.0))[lat.band_slices[1]]))
    eps_slow = max(0.05 * max(amp_slow, 1e-3), 0.01)
    agent = LadderAgent(
        LadderConfig(seed=seed, veto_threshold=0.5, holds=(12, 12),
                     arrive_eps=(0.08 * rho, eps_slow), w_prog_bands=(1.0, 1.0),
                     leash_radius=0.15 * rho, cap_radius=max(0.1 * rho, 0.02),
                     grad_proposals=True, use_flinch=False),
        head_pos, head_neg, lat)
    return agent, env


def phase2(conds=("ou", "pca"), include_iso: bool = False,
           outfile: str = "v05_phase2_behavior.json") -> None:
    import numpy as np
    from .experiments import bootstrap_ci, iqm

    sensor = HDSensor()
    encoders = make_encoders(sensor, include_iso=include_iso)
    # geometry + routing report per encoder (pre-registered gates)
    obs_test, c_test = collect_hd_walk(sensor, steps=4000, seed=77)
    for name, lat in encoders.items():
        r = band_c_corr_lat(lat, obs_test, c_test)
        far = float(torch.linalg.vector_norm(
            lat.embed(torch.tensor([0.5, 0.1, 0.0]))
            - lat.embed(torch.tensor([0.2, 0.8, 0.0]))))
        print(f"[v0.5:geom] {name:6s} slow-corr={r['band1_max_c_corr']:.3f} "
              f"far/step={far:.2f} (rigid ~0.76)", flush=True)
    results = []
    for which in conds:
        per_seed = {"return": [], "max_c": []}
        for s in range(12):
            agent, env = build_phase2(s, which, encoders)
            ret, mc, scored = 0.0, 0.0, 0
            for ep in range(150):
                stats = agent.run_episode(env, max_steps=160)
                if ep >= 75:
                    ret += stats["return"]
                    mc += env.max_c
                    scored += 1
            per_seed["return"].append(ret / scored)
            per_seed["max_c"].append(mc / scored)
        row = {"which": which, "per_seed": per_seed}
        for m in ("return", "max_c"):
            v = np.array(per_seed[m])
            lo, hi = bootstrap_ci(v)
            row[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
        results.append(row)
        print(f"[v0.5:p2] {which:4s} max_c={row['max_c']['iqm']:.3f} "
              f"CI={[round(x, 3) for x in row['max_c']['ci95']]} "
              f"return={row['return']['iqm']:+.3f} "
              f"CI={[round(x, 3) for x in row['return']['ci95']]}", flush=True)
    (RESULTS / outfile).write_text(json.dumps(results, indent=2))
    print(f"wrote results/{outfile}")


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "phase2":
        RESULTS.mkdir(exist_ok=True)
        phase2()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "iso":
        # v0.6 isometry experiment: does homogenizing the step scale convert
        # routing into competence? Pre-registered gates printed by the
        # geometry report; behavioral comparison ou_iso vs ou.
        RESULTS.mkdir(exist_ok=True)
        phase2(conds=("ou_iso", "ou"), include_iso=True,
               outfile="v06_isometry.json")
        return
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
