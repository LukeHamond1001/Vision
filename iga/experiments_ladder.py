"""E5 (SPEC §10): flat v0.1 agent vs register ladder on the two-timescale
world. Run:

    python -m iga.experiments_ladder          # 10 seeds x 60 episodes
    python -m iga.experiments_ladder quick    # 3 x 10

Metrics: return and zone-flip rate (slow-transition achievement), IQM over
seeds with bootstrap CIs, scored on the trained half. Exploratory at scaffold
scale — the structural tests in tests/test_ladder.py are the deliverable; this
comparison is evidence-gathering (SPEC §10, E5).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .agent import Agent, AgentConfig
from .audits import calibrate_threshold
from .envs.twozone import TwoZoneWorld
from .experiments import RESULTS, bootstrap_ci, iqm
from .heads import FixedRewardHead, SumHead
from .ladder import LadderAgent, LadderConfig
from .latent import BandedLatent


def build(seed: int, kind: str, env_kwargs: dict | None = None):
    latent = BandedLatent(part_dims=[2, 1], band_dims=[6, 2], seed=0)
    env = TwoZoneWorld(latent, **(env_kwargs or {}))
    gen = torch.Generator().manual_seed(seed + 500)
    world = torch.cat([torch.rand(256, 2, generator=gen),
                       (torch.rand(256, 1, generator=gen) > 0.5).float()], dim=1)
    support = torch.stack([latent.embed(w) for w in world])
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.reward_site, 1.0), sigma=0.25, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.gate, 0.0), sigma=0.2, proxy_samples=support)],
        weights=[1.0, 0.3])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.15,
                               proxy_samples=support)
    tau = calibrate_threshold(head_neg, support)
    if kind == "ladder":
        agent = LadderAgent(LadderConfig(seed=seed, veto_threshold=tau), head_pos, head_neg, latent)
    else:
        agent = Agent(AgentConfig(seed=seed, veto_threshold=tau, value_bar=0.02),
                      head_pos, head_neg, latent)
    return agent, env


def run(kind: str, seeds: int, episodes: int, env_kwargs: dict | None = None,
        max_steps: int = 100) -> dict:
    per_seed = {"return": [], "flip_rate": []}
    for s in range(seeds):
        agent, env = build(s, kind, env_kwargs)
        ret, flips, scored = 0.0, 0, 0
        for ep in range(episodes):
            stats = agent.run_episode(env, max_steps=max_steps)
            if ep >= episodes // 2:
                ret += stats["return"]
                flips += min(env.flips_this_episode, 1)
                scored += 1
        per_seed["return"].append(ret / scored)
        per_seed["flip_rate"].append(flips / scored)
    out = {"kind": kind, "seeds": seeds, "episodes": episodes, "per_seed": per_seed}
    for m in ("return", "flip_rate"):
        v = np.array(per_seed[m])
        lo, hi = bootstrap_ci(v)
        out[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "quick":
        seeds, episodes, env_kwargs, max_steps, out = 3, 10, None, 100, "e5_ladder.json"
    elif mode == "easy":
        # Eased geometry + longer budget: isolates whether the E5 null at
        # default scale is the ladder failing or the REINFORCE learner failing.
        seeds, episodes, max_steps, out = 8, 150, 120, "e5_ladder_easy.json"
        env_kwargs = dict(gate=(0.45, 0.25), reward_site=(0.75, 0.75), site_radius=0.1)
    else:
        seeds, episodes, env_kwargs, max_steps, out = 10, 60, None, 100, "e5_ladder.json"
    RESULTS.mkdir(exist_ok=True)
    results = []
    for kind in ("flat", "ladder"):
        r = run(kind, seeds, episodes, env_kwargs, max_steps)
        results.append(r)
        print(f"[E5:{mode}] {kind:7s} return={r['return']['iqm']:+.3f} "
              f"CI={[round(x, 3) for x in r['return']['ci95']]} "
              f"flip_rate={r['flip_rate']['iqm']:.2f} "
              f"CI={[round(x, 3) for x in r['flip_rate']['ci95']]}", flush=True)
    (RESULTS / out).write_text(json.dumps(results, indent=2))
    print(f"wrote results/{out}")


if __name__ == "__main__":
    main()
