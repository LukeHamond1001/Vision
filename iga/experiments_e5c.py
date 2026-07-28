"""E5c (SPEC §10, round 8): the charge-world discrimination experiment. Run:

    python -m iga.experiments_e5c [quick]

Three conditions, predictions stated in advance:
  flat          re-proposal drift at window boundaries -> rarely sustains a
                charge; shiny claims win re-rankings
  ladder        slow register holds the c-target across fast windows; dense
                slow-band progress trains the policy onto the pad -> charges
                complete; best return
  ladder_short  τ_slow cut to τ_fast — the internal ablation isolating
                COMMITMENT PERSISTENCE as the active ingredient; predicted to
                collapse toward flat

If ladder does not separate from flat here, register-holding is not the
active ingredient of multi-timescale competence at this scale, and v0.3
pivots from multi-timescale control to multi-timescale representation.

Metrics: return (door completions) and max charge reached (sensitive early
signal), IQM over seeds with bootstrap CIs, scored on the trained half.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .agent import Agent, AgentConfig
from .audits import calibrate_threshold
from .envs.chargeworld import ChargeWorld
from .experiments import RESULTS, bootstrap_ci, iqm
from .heads import FixedRewardHead, SumHead
from .ladder import LadderAgent, LadderConfig
from .latent import BandedLatent


def build(seed: int, kind: str):
    latent = BandedLatent(part_dims=[2, 1], band_dims=[6, 2], seed=0)
    env = ChargeWorld(latent)
    gen = torch.Generator().manual_seed(seed + 4000)
    world = torch.cat([torch.rand(384, 2, generator=gen),
                       torch.rand(384, 1, generator=gen)], dim=1)
    support = torch.stack([latent.embed(w) for w in world])
    # Designed evaluator: door-at-full-charge is the prize; the pad is salient
    # ALONG THE CHARGE PATH (bumps at low and mid charge — a bump only at
    # c=0.5 is invisible at c=0 and breaks discovery: the round-8a all-zero
    # result); two shiny sites are the evaluator's imperfections — worthless
    # but claim-attractive, the temptation at every re-proposal. All three
    # conditions share this evaluator, so its tuning favors none of them.
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.door, 1.0), sigma=0.30, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.1), sigma=0.20, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.6), sigma=0.20, proxy_samples=support),
         FixedRewardHead(env.embed_world((0.15, 0.15), 0.0), sigma=0.08, proxy_samples=support),
         FixedRewardHead(env.embed_world((0.85, 0.15), 0.0), sigma=0.08, proxy_samples=support)],
        weights=[1.0, 0.3, 0.3, 0.15, 0.15])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.12,
                               proxy_samples=support)
    tau = calibrate_threshold(head_neg, support)
    if kind == "flat":
        agent = Agent(AgentConfig(seed=seed, veto_threshold=tau, value_bar=0.02),
                      head_pos, head_neg, latent)
    else:
        holds = (12, 40) if kind == "ladder" else (12, 12)
        agent = LadderAgent(LadderConfig(seed=seed, veto_threshold=tau, holds=holds),
                            head_pos, head_neg, latent)
    return agent, env


def run_kind(kind: str, seeds: int, episodes: int, max_steps: int = 160) -> dict:
    per_seed = {"return": [], "max_c": []}
    for s in range(seeds):
        agent, env = build(s, kind)
        ret, mc, scored = 0.0, 0.0, 0
        for ep in range(episodes):
            stats = agent.run_episode(env, max_steps=max_steps)
            if ep >= episodes // 2:
                ret += stats["return"]
                mc += env.max_c
                scored += 1
        per_seed["return"].append(ret / scored)
        per_seed["max_c"].append(mc / scored)
    out = {"kind": kind, "seeds": seeds, "episodes": episodes, "per_seed": per_seed}
    for m in ("return", "max_c"):
        v = np.array(per_seed[m])
        lo, hi = bootstrap_ci(v)
        out[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
    return out


def main() -> None:
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    seeds, episodes = (3, 12) if quick else (12, 80)
    RESULTS.mkdir(exist_ok=True)
    results = []
    for kind in ("flat", "ladder", "ladder_short"):
        r = run_kind(kind, seeds, episodes)
        results.append(r)
        print(f"[E5c] {kind:12s} return={r['return']['iqm']:+.3f} "
              f"CI={[round(x, 3) for x in r['return']['ci95']]} "
              f"max_c={r['max_c']['iqm']:.2f} "
              f"CI={[round(x, 3) for x in r['max_c']['ci95']]}", flush=True)
    (RESULTS / "e5c_charge.json").write_text(json.dumps(results, indent=2))
    print("wrote results/e5c_charge.json")


if __name__ == "__main__":
    main()
