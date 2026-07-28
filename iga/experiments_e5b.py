"""E5b (SPEC §10): ladder-vs-flat with deeper timescale separation, plus a
higher-n replication of E5-easy. Run:

    python -m iga.experiments_e5b [quick]

Part 1 — ThreeZoneWorld (two slow transitions per episode): flat vs ladder,
tracking return and gates-crossed rate. Part 2 — TwoZone easy variant at
n=12 to tighten the round-6 CIs.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .agent import Agent, AgentConfig
from .audits import calibrate_threshold
from .envs.threezone import ThreeZoneWorld
from .experiments import RESULTS, bootstrap_ci, iqm
from .experiments_ladder import run as run_e5
from .heads import FixedRewardHead, SumHead
from .ladder import LadderAgent, LadderConfig
from .latent import BandedLatent


def build(seed: int, kind: str):
    latent = BandedLatent(part_dims=[2, 1], band_dims=[6, 2], seed=0)
    env = ThreeZoneWorld(latent)
    gen = torch.Generator().manual_seed(seed + 3000)
    world = torch.cat([torch.rand(384, 2, generator=gen),
                       torch.randint(0, 3, (384, 1), generator=gen).float()], dim=1)
    support = torch.stack([latent.embed(w) for w in world])
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.reward_site, 2.0), sigma=0.25, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.gates[0], 0.0), sigma=0.2, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.gates[1], 1.0), sigma=0.2, proxy_samples=support)],
        weights=[1.0, 0.25, 0.3])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.15,
                               proxy_samples=support)
    tau = 0.5  # f-scale veto (prospective evaluation, round 9)
    if kind == "ladder":
        agent = LadderAgent(LadderConfig(seed=seed, veto_threshold=tau, holds=(12, 120)),
                            head_pos, head_neg, latent)
    else:
        agent = Agent(AgentConfig(seed=seed, veto_threshold=tau, value_bar=0.02),
                      head_pos, head_neg, latent)
    return agent, env


def run_kind(kind: str, seeds: int, episodes: int, max_steps: int = 140) -> dict:
    per_seed = {"return": [], "gates_rate": []}
    for s in range(seeds):
        agent, env = build(s, kind)
        ret, gates, scored = 0.0, 0.0, 0
        for ep in range(episodes):
            stats = agent.run_episode(env, max_steps=max_steps)
            if ep >= episodes // 2:
                ret += stats["return"]
                gates += env.flips_this_episode / 2.0     # fraction of the 2 gates
                scored += 1
        per_seed["return"].append(ret / scored)
        per_seed["gates_rate"].append(gates / scored)
    out = {"kind": kind, "seeds": seeds, "episodes": episodes, "per_seed": per_seed}
    for m in ("return", "gates_rate"):
        v = np.array(per_seed[m])
        lo, hi = bootstrap_ci(v)
        out[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
    return out


def main() -> None:
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    seeds, episodes = (3, 10) if quick else (12, 60)
    RESULTS.mkdir(exist_ok=True)

    results = {"threezone": [], "e5_easy_n12": []}
    for kind in ("flat", "ladder"):
        r = run_kind(kind, seeds, episodes)
        results["threezone"].append(r)
        print(f"[E5b:3zone] {kind:7s} return={r['return']['iqm']:+.3f} "
              f"CI={[round(x, 3) for x in r['return']['ci95']]} "
              f"gates={r['gates_rate']['iqm']:.2f} "
              f"CI={[round(x, 3) for x in r['gates_rate']['ci95']]}", flush=True)

    if not quick:
        easy_kwargs = dict(gate=(0.45, 0.25), reward_site=(0.75, 0.75), site_radius=0.1)
        for kind in ("flat", "ladder"):
            r = run_e5(kind, seeds=12, episodes=100, env_kwargs=easy_kwargs, max_steps=120)
            results["e5_easy_n12"].append(r)
            print(f"[E5b:easy12] {kind:7s} return={r['return']['iqm']:+.3f} "
                  f"CI={[round(x, 3) for x in r['return']['ci95']]} "
                  f"flip_rate={r['flip_rate']['iqm']:.2f}", flush=True)

    (RESULTS / "e5b_threezone.json").write_text(json.dumps(results, indent=2))
    print("wrote results/e5b_threezone.json")


if __name__ == "__main__":
    main()
