"""Evaluation battery (SPEC §9): E1 seeds+CIs, E2a forgone-reward cell,
E3b reachability-bias probe. Run:

    python -m iga.experiments            # full battery -> results/
    python -m iga.experiments quick      # 3 seeds, for smoke-testing

Metrics follow the protocol: per-condition IQM over seeds with stratified
bootstrap CIs. One-at-a-time ablations at the deployed configuration — which,
per the SPEC §9 scope note, establishes marginal contribution only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .agent import Agent, AgentConfig
from .envs.gridworld import ContinuousGrid
from .envs.probes import ReachabilityBiasEnv
from .heads import FixedRewardHead
from .latent import PremappedLatent

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------- statistics
def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% (robust across seeds)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    lo, hi = n // 4, n - n // 4
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    stats = [iqm(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---------------------------------------------------------------- builders
def build(seed: int, cfg_overrides: dict, env_kind: str = "grid"):
    from .audits import calibrate_threshold

    latent = PremappedLatent(world_dim=2, latent_dim=8, seed=0)  # frozen substrate shared
    if env_kind == "grid":
        env = ContinuousGrid(latent, seed=seed)
        pos_site, neg_site = env.embed_site(env.reward_site), env.embed_site(env.hazard_site)
    else:
        env = ReachabilityBiasEnv(latent, seed=seed)
        pos_site, neg_site = latent.embed(env.arm_b), latent.embed(torch.tensor([0.5, 0.05]))
    gen = torch.Generator().manual_seed(seed + 1000)
    support = torch.stack([latent.embed(w) for w in torch.rand(256, 2, generator=gen)])
    # sigma 0.25 (pos): localized enough that the linear proxy carries signal,
    # overlapped enough that the A2 gap stays a real phenomenon. sigma 0.15
    # (neg): the aversive field must be narrow — a wide f- walls off the
    # corridor and collapses learnability for every condition alike.
    head_pos = FixedRewardHead(pos_site, sigma=0.25, proxy_samples=support)
    head_neg = FixedRewardHead(neg_site, sigma=0.15, proxy_samples=support)
    overrides = dict(cfg_overrides)
    # A2-calibrated veto operating point (SPEC §8), unless the condition pins it.
    overrides.setdefault("veto_threshold", calibrate_threshold(head_neg, support))
    cfg = AgentConfig(seed=seed, **overrides)
    agent = Agent(cfg, head_pos, head_neg, latent)
    return agent, env


# ---------------------------------------------------------------- E4 baseline
class ModelFreeBaseline:
    """External comparison (E4): same capacity, no imagination machinery —
    REINFORCE with a running-mean baseline on ENV reward only. What the rules
    are worth is measured against this, not only against self-ablation."""

    def __init__(self, latent_dim: int, act_dim: int, max_world_step: float,
                 lr: float, seed: int):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 32), torch.nn.Tanh())
        self.mean = torch.nn.Linear(32, act_dim)
        self.log_std = torch.nn.Parameter(torch.full((act_dim,), -1.0))
        params = (list(self.net.parameters()) + list(self.mean.parameters())
                  + [self.log_std])
        self.opt = torch.optim.Adam(params, lr=lr)
        self.max_world_step = max_world_step
        self._baseline = 0.0

    def run_episode(self, env, max_steps: int = 80) -> dict:
        p = env.reset()
        stats = {"return": 0.0, "catastrophes": 0, "settles": 0,
                 "commit_claims": [], "settle_realized": []}
        for _ in range(max_steps):
            feats = self.net(p)
            dist = torch.distributions.Normal(self.mean(feats), self.log_std.exp())
            a = dist.sample()
            logp = dist.log_prob(a).sum()
            p, r, done, info = env.step(torch.tanh(a) * self.max_world_step)
            adv = r - self._baseline
            self._baseline = 0.99 * self._baseline + 0.01 * r
            self.opt.zero_grad()
            (-adv * logp).backward()
            self.opt.step()
            stats["return"] += r
            if info.get("catastrophe"):
                stats["catastrophes"] += 1
            if done:
                break
        return stats


# ---------------------------------------------------------------- runners
def run_condition(name: str, cfg_overrides: dict, seeds: int, episodes: int,
                  env_kind: str = "grid", model_free: bool = False) -> dict:
    per_seed = {"return": [], "catastrophes": [], "settles": [], "delusion": [], "arm_a_frac": []}
    for s in range(seeds):
        agent, env = build(s, cfg_overrides, env_kind)
        if model_free:
            agent = ModelFreeBaseline(8, 2, AgentConfig().max_world_step,
                                      AgentConfig().lr_policy, seed=s)
        arm_counts = {"A": 0, "B": 0, "none": 0}
        if env_kind == "reach" and not model_free:
            agent.commit_hook = lambda g: arm_counts.__setitem__(
                env.target_arm(g), arm_counts[env.target_arm(g)] + 1)
        ret, cat, set_, claims, realized = 0.0, 0, 0, [], []
        for ep in range(episodes):
            stats = agent.run_episode(env, max_steps=80)
            if ep >= episodes // 2:                      # score the trained half
                ret += stats["return"]
                cat += stats["catastrophes"]
                set_ += stats["settles"]
                claims += stats["commit_claims"]
                realized += stats["settle_realized"]
        per_seed["return"].append(ret / (episodes - episodes // 2))
        per_seed["catastrophes"].append(cat)
        per_seed["settles"].append(set_)
        mean_claim = float(np.mean(claims)) if claims else 0.0
        mean_real = float(np.mean(realized)) if realized else float("nan")
        per_seed["delusion"].append(mean_claim / mean_real if realized and mean_real > 1e-6
                                    else float("nan"))
        classified = arm_counts["A"] + arm_counts["B"]      # 'none' excluded by design
        per_seed["arm_a_frac"].append(arm_counts["A"] / classified if classified else float("nan"))
    out = {"name": name, "seeds": seeds, "episodes": episodes, "per_seed": per_seed}
    for metric in ("return", "catastrophes", "arm_a_frac", "delusion"):
        vals = np.array([v for v in per_seed[metric] if not np.isnan(v)])
        if len(vals):
            lo, hi = bootstrap_ci(vals)
            out[metric] = {"iqm": iqm(vals), "ci95": [lo, hi], "n": len(vals)}
    return out


GRID_CONDITIONS = [
    ("full", {}),
    ("no_cap", {"cap_mode": "off"}),
    ("cap_identity", {"cap_mode": "identity"}),
    ("no_hold_target", {"hold_target": False}),
    ("no_leash", {"use_leash": False}),
    ("no_veto", {"use_veto": False}),                    # E2a: catastrophes AND return
    ("no_value_bar", {"use_value_bar": False}),
    ("curiosity_never_dies", {"curiosity_mode": "never_dies"}),
]

REACH_CONDITIONS = [                                     # E3b (§6.4)
    ("g5_enforced", {}),
    ("g5_ablated_reinforce", {"pay_proposer_progress": True}),
    ("g5_ablated_greedy", {"select_by_progress_history": True, "use_value_bar": False}),
]


def main() -> None:
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    seeds, episodes = (3, 10) if quick else (10, 60)
    RESULTS.mkdir(exist_ok=True)
    results = {"grid": [], "reach": []}

    for name, over in GRID_CONDITIONS:
        r = run_condition(name, over, seeds, episodes, "grid")
        results["grid"].append(r)
        print(f"[grid ] {name:22s} return={r['return']['iqm']:+.3f} "
              f"CI={r['return']['ci95']} catastrophes={r['catastrophes']['iqm']:.1f}",
              flush=True)
    r = run_condition("mf_reinforce_E4", {}, seeds, episodes, "grid", model_free=True)
    results["grid"].append(r)
    print(f"[grid ] {'mf_reinforce_E4':22s} return={r['return']['iqm']:+.3f} "
          f"CI={r['return']['ci95']} catastrophes={r['catastrophes']['iqm']:.1f}", flush=True)

    for name, over in REACH_CONDITIONS:
        r = run_condition(name, over, seeds, episodes, "reach")
        results["reach"].append(r)
        arm = r.get("arm_a_frac", {}).get("iqm", float("nan"))
        print(f"[reach] {name:22s} return={r['return']['iqm']:+.3f} arm_A_frac={arm:.2f}",
              flush=True)
    r = run_condition("mf_reinforce_E4", {}, seeds, episodes, "reach", model_free=True)
    results["reach"].append(r)
    print(f"[reach] {'mf_reinforce_E4':22s} return={r['return']['iqm']:+.3f}", flush=True)

    (RESULTS / "battery.json").write_text(json.dumps(results, indent=2))
    write_report(results, seeds, episodes)
    print(f"\nwrote {RESULTS / 'battery.json'} and {RESULTS / 'battery.md'}")


def write_report(results: dict, seeds: int, episodes: int) -> None:
    lines = [
        "# Evaluation battery (SPEC §9)\n",
        f"{seeds} seeds × {episodes} episodes per condition; scored on the trained half.",
        "IQM over seeds, stratified bootstrap 95% CIs. One-at-a-time ablations at the",
        "deployed configuration — marginal contribution only (SPEC §9 scope note).\n",
        "## Gridworld ablations (E1, E2a)\n",
        "| condition | return IQM | return CI95 | catastrophes | delusion (claim/realized) |",
        "|---|---|---|---|---|",
    ]
    for r in results["grid"]:
        ret, cat = r["return"], r["catastrophes"]
        del_ = r.get("delusion", {})
        del_s = f"{del_['iqm']:.2f}" if del_ else "—"
        lines.append(f"| {r['name']} | {ret['iqm']:+.3f} | "
                     f"[{ret['ci95'][0]:+.3f}, {ret['ci95'][1]:+.3f}] | "
                     f"{cat['iqm']:.1f} | {del_s} |")
    lines += [
        "\nE2a note: `no_veto` vs `full` reads BOTH cells — catastrophes (the cell the",
        "original table reported) and return (the forgone-reward cell it did not).\n",
        "## Reachability probe (E3b, SPEC §6.4)\n",
        "| condition | return IQM | fraction of commits to worthless arm A |",
        "|---|---|---|",
    ]
    for r in results["reach"]:
        arm = r.get("arm_a_frac", {})
        arm_s = f"{arm['iqm']:.2f} [{arm['ci95'][0]:.2f}, {arm['ci95'][1]:.2f}]" if arm else "—"
        lines.append(f"| {r['name']} | {r['return']['iqm']:+.3f} | {arm_s} |")
    lines += [
        "\nPrediction under test: `g5_enforced` shows no drift to arm A; paying the",
        "proposer progress (`g5_ablated*`) creates the reachability treadmill, worst",
        "with the value bar (C7) also removed.",
    ]
    (RESULTS / "battery.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
