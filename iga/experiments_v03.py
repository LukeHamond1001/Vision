"""v0.3 experiment: does the pretrained multi-timescale latent supply, from
data, what round 8 hand-tuned? Run:

    python -m iga.experiments_v03 [quick]

Conditions (all ladder_short, round-9 spec-true mechanics, ChargeWorld):
  random_handw   random-block latent + hand-set w_prog_bands=(1,3)
                 [the round-8-style configuration under current mechanics]
  random_now     random-block latent + w_prog_bands=(1,1)
                 [how much the hand weights carry]
  pretrained_now pretrained OU-ladder latent + w_prog_bands=(1,1)
                 [the thesis: whitening amplifies the slow band, so the
                 REPRESENTATION supplies the scale — no hand weights]

Pre-registered predictions:
1. Band discovery: charge routes into the slow band (max corr with c higher
   in band 1 than band 0) from the OU objective alone.
2. pretrained_now ≥ random_handw > random_now on max charge. If
   pretrained_now only matches random_now, whitening did not supply the
   scale and the representation thesis fails this test.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .envs.chargeworld import ChargeWorld
from .experiments import RESULTS, bootstrap_ci, iqm
from .experiments_e5c import build as build_e5c
from .heads import FixedRewardHead, SumHead
from .ladder import LadderAgent, LadderConfig
from .latent import BandedLatent
from .pretrain import (PretrainedBandedLatent, band_routing_report,
                       collect_random_walk, pretrain_ou_ladder)


_PRETRAINED_CACHE: dict = {}


def make_pretrained_latent(seed: int = 0) -> PretrainedBandedLatent:
    if seed in _PRETRAINED_CACHE:
        return _PRETRAINED_CACHE[seed]
    scratch_env = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=seed)
    traj = collect_random_walk(scratch_env, steps=20000, seed=seed)
    # Timescales matched to the walk's real mixing: position (step 0.1,
    # reflecting box) decorrelates over ~10-30 steps; charge between coverage
    # resets persists (decay 0.002 -> timescale ~500). Same lag for both bands
    # so the objective, not the sampling, does the sorting.
    W = pretrain_ou_ladder(traj, band_dims=[6, 2], taus=[10.0, 300.0],
                           lags=[15, 15], segment_len=100, seed=seed)
    lat = PretrainedBandedLatent(W, part_dims=[2, 1], band_dims=[6, 2])
    # Scale-normalize so leash/cap/arrival radii (latent units) mean the same
    # world-scale thing as in the random-block latent: fast-band position
    # step_scale -> 1. Whitening amplified everything ~1/std; without this the
    # pretrained condition would run with an unfairly tight leash.
    lat = PretrainedBandedLatent(W / lat.step_scale(0), part_dims=[2, 1], band_dims=[6, 2])
    _PRETRAINED_CACHE[seed] = lat
    return lat


def make_leakfree_latent() -> PretrainedBandedLatent:
    """Necessity ablation: the pretrained latent with the fast band made
    c-blind (zero the charge column of the fast-band rows). If the v0.3 win
    vanishes here, the cross-band leak is the active ingredient."""
    base = make_pretrained_latent(seed=0)
    W = base._W.clone()
    W[base.band_slices[0], -1] = 0.0
    lat = PretrainedBandedLatent(W, part_dims=[2, 1], band_dims=[6, 2])
    return PretrainedBandedLatent(W / lat.step_scale(0), part_dims=[2, 1], band_dims=[6, 2])


def make_leakin_latent(amp: float = 0.57, seed: int = 0) -> PretrainedBandedLatent:
    """Sufficiency ablation: the RANDOM block latent with a synthetic charge
    leak injected into the fast band at the pretrained latent's measured
    amplitude. If the win appears here, the leak alone (no learned geometry)
    suffices."""
    rb = BandedLatent([2, 1], [6, 2], seed=0)
    W = torch.zeros(8, 3)
    W[0:6, 0:2] = rb._blocks[0]
    W[6:8, 2:3] = rb._blocks[1]
    gen = torch.Generator().manual_seed(seed + 11)
    v = torch.randn(6, generator=gen)
    W[0:6, 2] = v / torch.linalg.vector_norm(v) * amp
    lat = PretrainedBandedLatent(W, part_dims=[2, 1], band_dims=[6, 2])
    return PretrainedBandedLatent(W / lat.step_scale(0), part_dims=[2, 1], band_dims=[6, 2])


def make_pretrained_v04_latent() -> PretrainedBandedLatent:
    """v0.4 objective: within-band whitening + deliberate slow->fast context
    coupling at the round-2-measured amplitude. The question it answers:
    does LEARNING the winning ingredient match hand-injecting it?"""
    if "v04" in _PRETRAINED_CACHE:
        return _PRETRAINED_CACHE["v04"]
    scratch_env = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=0)
    traj = collect_random_walk(scratch_env, steps=20000, seed=0)
    W = pretrain_ou_ladder(traj, band_dims=[6, 2], taus=[10.0, 300.0],
                           lags=[15, 15], segment_len=100,
                           context_amp=0.6, seed=0)
    lat = PretrainedBandedLatent(W, part_dims=[2, 1], band_dims=[6, 2])
    lat = PretrainedBandedLatent(W / lat.step_scale(0), part_dims=[2, 1], band_dims=[6, 2])
    _PRETRAINED_CACHE["v04"] = lat
    return lat


def build(seed: int, cond: str):
    if cond == "pretrained_leakfree":
        latent = make_leakfree_latent()
    elif cond == "random_leakin":
        latent = make_leakin_latent()
    elif cond == "pretrained_v04":
        latent = make_pretrained_v04_latent()
    elif cond.startswith("pretrained"):
        latent = make_pretrained_latent(seed=0)   # one shared pretraining
    else:
        latent = BandedLatent([2, 1], [6, 2], seed=0)
    env = ChargeWorld(latent, seed=seed)
    gen = torch.Generator().manual_seed(seed + 4000)
    world = torch.cat([torch.rand(384, 2, generator=gen),
                       torch.rand(384, 1, generator=gen)], dim=1)
    support = torch.stack([latent.embed(w) for w in world])
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.door, 1.0), sigma=0.30, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.1), sigma=0.20, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.6), sigma=0.20, proxy_samples=support),
         FixedRewardHead(env.embed_world((0.15, 0.15), 0.0), sigma=0.08, proxy_samples=support),
         FixedRewardHead(env.embed_world((0.85, 0.15), 0.0), sigma=0.08, proxy_samples=support)],
        weights=[1.0, 0.3, 0.3, 0.15, 0.15])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.12,
                               proxy_samples=support)
    w_bands = (1.0, 3.0) if cond == "random_handw" else (1.0, 1.0)
    agent = LadderAgent(LadderConfig(seed=seed, veto_threshold=0.5, holds=(12, 12),
                                     w_prog_bands=w_bands),
                        head_pos, head_neg, latent)
    return agent, env


def run_cond(cond: str, seeds: int, episodes: int, max_steps: int = 160) -> dict:
    per_seed = {"return": [], "max_c": []}
    for s in range(seeds):
        agent, env = build(s, cond)
        ret, mc, scored = 0.0, 0.0, 0
        for ep in range(episodes):
            stats = agent.run_episode(env, max_steps=max_steps)
            if ep >= episodes // 2:
                ret += stats["return"]
                mc += env.max_c
                scored += 1
        per_seed["return"].append(ret / scored)
        per_seed["max_c"].append(mc / scored)
    out = {"cond": cond, "seeds": seeds, "episodes": episodes, "per_seed": per_seed}
    for m in ("return", "max_c"):
        v = np.array(per_seed[m])
        lo, hi = bootstrap_ci(v)
        out[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "ablate":
        # Mechanism identification (pre-registered): leak NECESSARY if
        # pretrained_leakfree falls to ~random_now; leak SUFFICIENT if
        # random_leakin rises to ~pretrained_now.
        from .pretrain import band_routing_report
        from .latent import BandedLatent as _BL
        scratch = ChargeWorld(_BL([2, 1], [6, 2], seed=0), seed=99)
        traj = collect_random_walk(scratch, 4000, seed=99)
        for name, lat in (("leakfree", make_leakfree_latent()),
                          ("leakin", make_leakin_latent())):
            print(f"[v0.3:ablate] routing {name}: {band_routing_report(lat, traj)}", flush=True)
        results = []
        for cond in ("random_now", "random_leakin", "pretrained_now", "pretrained_leakfree"):
            r = run_cond(cond, seeds=12, episodes=150)
            results.append(r)
            print(f"[v0.3:ablate] {cond:19s} max_c={r['max_c']['iqm']:.3f} "
                  f"CI={[round(x, 3) for x in r['max_c']['ci95']]} "
                  f"return={r['return']['iqm']:+.3f}", flush=True)
        (RESULTS / "v03_leak_ablation.json").write_text(json.dumps(results, indent=2))
        print("wrote results/v03_leak_ablation.json")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "v04":
        # Pre-registered: pretrained_v04 >= random_leakin means the deliberate
        # objective learns the coupling at least as well as hand-injection;
        # pretrained_v04 < random_leakin means hand-design wins at this scale.
        from .pretrain import band_routing_report
        from .latent import BandedLatent as _BL
        scratch = ChargeWorld(_BL([2, 1], [6, 2], seed=0), seed=99)
        traj = collect_random_walk(scratch, 4000, seed=99)
        print(f"[v0.4] routing: {band_routing_report(make_pretrained_v04_latent(), traj)}",
              flush=True)
        results = []
        for cond in ("random_now", "random_leakin", "pretrained_v04"):
            r = run_cond(cond, seeds=12, episodes=150)
            results.append(r)
            print(f"[v0.4] {cond:16s} max_c={r['max_c']['iqm']:.3f} "
                  f"CI={[round(x, 3) for x in r['max_c']['ci95']]} "
                  f"return={r['return']['iqm']:+.3f}", flush=True)
        (RESULTS / "v04_objective.json").write_text(json.dumps(results, indent=2))
        print("wrote results/v04_objective.json")
        return
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    seeds, episodes = (3, 12) if quick else (12, 150)
    RESULTS.mkdir(exist_ok=True)

    # Prediction 1 first: band routing, reported before any behavior runs.
    latent = make_pretrained_latent(seed=0)
    scratch_env = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=99)
    traj = collect_random_walk(scratch_env, steps=4000, seed=99)
    report = band_routing_report(latent, traj)
    print(f"[v0.3] band routing: {report}", flush=True)

    results = {"routing": report, "conditions": []}
    for cond in ("random_handw", "random_now", "pretrained_now"):
        r = run_cond(cond, seeds, episodes)
        results["conditions"].append(r)
        print(f"[v0.3] {cond:14s} max_c={r['max_c']['iqm']:.3f} "
              f"CI={[round(x, 3) for x in r['max_c']['ci95']]} "
              f"return={r['return']['iqm']:+.3f}", flush=True)
    (RESULTS / "v03_representation.json").write_text(json.dumps(results, indent=2))
    print("wrote results/v03_representation.json")


if __name__ == "__main__":
    main()
