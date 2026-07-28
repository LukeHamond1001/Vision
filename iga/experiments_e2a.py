"""E2a (SPEC §9): trust-vs-verify, finally isolated. Run:

    python -m iga.experiments_e2a [quick]

Three cells:
  full        trap active, veto on  -> catastrophes should be ~0; some detour cost
  no_veto     trap active, veto off -> the chain routes straight; catastrophes
  paranoia    trap ABSENT, veto on  -> pure false-alarm cost (forgone reward /
              extra steps) of a misspecified pre-mapped alarm

Metrics per cell: return, catastrophes, mean episode length (detour proxy).
The three cells together fill the confusion-matrix row the original ablation
table reported one cell of.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .agent import Agent, AgentConfig
from .audits import calibrate_threshold
from .envs.trapcorridor import TrapCorridor
from .experiments import RESULTS, bootstrap_ci, iqm
from .heads import FixedRewardHead
from .latent import PremappedLatent


def build(seed: int, trap_active: bool, use_veto: bool):
    latent = PremappedLatent(world_dim=2, latent_dim=8, seed=0)
    env = TrapCorridor(latent, trap_active=trap_active, seed=seed)
    gen = torch.Generator().manual_seed(seed + 2000)
    support = torch.stack([latent.embed(w) for w in torch.rand(256, 2, generator=gen)])
    head_pos = FixedRewardHead(env.embed_site(env.reward_site), sigma=0.35, proxy_samples=support)
    # Narrow aversive field: fires on in-strip targets, silent at the gaps and
    # along traversal outside the strip — the "cliff edge gives no warning".
    head_neg = FixedRewardHead(env.embed_site(env.trap_center), sigma=0.10, proxy_samples=support)
    tau = calibrate_threshold(head_neg, support)
    cfg = AgentConfig(seed=seed, veto_threshold=tau, use_veto=use_veto, value_bar=0.02)
    return Agent(cfg, head_pos, head_neg, latent), env


CELLS = [
    ("full", dict(trap_active=True, use_veto=True)),
    ("no_veto", dict(trap_active=True, use_veto=False)),
    ("paranoia", dict(trap_active=False, use_veto=True)),
]


def main() -> None:
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    seeds, episodes, max_steps = (3, 10, 80) if quick else (10, 60, 80)
    RESULTS.mkdir(exist_ok=True)
    results = []
    for name, kw in CELLS:
        per_seed = {"return": [], "catastrophes": [], "ep_len": []}
        for s in range(seeds):
            agent, env = build(s, kw["trap_active"], kw["use_veto"])
            ret, cat, lens, scored = 0.0, 0, [], 0
            for ep in range(episodes):
                steps_before = 0
                stats = agent.run_episode(env, max_steps=max_steps)
                if ep >= episodes // 2:
                    ret += stats["return"]
                    cat += stats["catastrophes"]
                    scored += 1
            per_seed["return"].append(ret / scored)
            per_seed["catastrophes"].append(cat)
        row = {"cell": name, "per_seed": per_seed}
        for m in ("return", "catastrophes"):
            v = np.array(per_seed[m])
            lo, hi = bootstrap_ci(v)
            row[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
        results.append(row)
        print(f"[E2a] {name:9s} return={row['return']['iqm']:+.3f} "
              f"CI={[round(x, 3) for x in row['return']['ci95']]} "
              f"catastrophes={row['catastrophes']['iqm']:.1f} "
              f"CI={[round(x, 3) for x in row['catastrophes']['ci95']]}", flush=True)
    (RESULTS / "e2a_trust.json").write_text(json.dumps(results, indent=2))
    print("wrote results/e2a_trust.json")


if __name__ == "__main__":
    main()
