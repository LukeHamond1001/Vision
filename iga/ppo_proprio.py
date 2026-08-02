"""v2.0 phase 2 learner: Gaussian PPO over proprioception for BatteryAnt.
In-process vector env (BatteryAnt is ~7k steps/s single-core; no
multiprocessing needed), MLP actor-critic, continuous torques.

Reward arms are composed by the caller per the campaign card:
  task           forward-velocity reward from the base env
  task+wired     task + register progress on (battery, temp, wear) with
                 per-band weights w ∝ τ (the spec law: slower pays more)
  task+penalty   task + a hand-engineered battery/thermal penalty (the
                 good-faith incumbent an engineer would write)
  wired          register progress only
"""

from __future__ import annotations

import numpy as np
import torch

from .battery_env import BatteryAnt


class ProprioAC(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed + 3100)
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, act_dim))
        self.log_std = torch.nn.Parameter(torch.full((act_dim,), -0.5))
        self.v = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 1))

    def dist(self, x):
        mu = self.pi(x)
        return torch.distributions.Normal(mu, self.log_std.exp())


def run_battery_arm(reward_fn, seed: int, total_steps: int = 2_000_000,
                    n_envs: int = 8, rollout: int = 512,
                    gamma: float = 0.99, lam: float = 0.95,
                    lr: float = 3e-4, ent_coef: float = 0.0,
                    log_cb=None) -> dict:
    """reward_fn(r_task, info, prev_info) -> float, per env step.
    Episode stats: task return, brownout steps, docked steps, wear rate,
    temp mean, lifetime id."""
    envs = [BatteryAnt(seed=seed * 131 + k) for k in range(n_envs)]
    obs = np.stack([e.reset() for e in envs])
    obs_dim, act_dim = envs[0].obs_dim, envs[0].act_dim
    net = ProprioAC(obs_dim, act_dim, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    prev_infos = [None] * n_envs
    ep_task = np.zeros(n_envs)
    ep_temp = np.zeros(n_envs)
    ep_wear0 = np.array([e.wear for e in envs])   # wear persists across
    # episodes (lifetime): rate must be the EPISODE DELTA, not the
    # accumulated value (round-1 metric bug: short episodes inflated)
    stats = {"task": [], "brownout": [], "docked": [], "wear_rate": [],
             "temp_mean": [], "steps": []}
    step_total = 0
    while step_total < total_steps:
        O = torch.zeros(rollout, n_envs, obs_dim)
        A = torch.zeros(rollout, n_envs, act_dim)
        LP = torch.zeros(rollout, n_envs)
        V = torch.zeros(rollout, n_envs)
        R = torch.zeros(rollout, n_envs)
        D = torch.zeros(rollout, n_envs)
        for t in range(rollout):
            xt = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = net.dist(xt)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = net.v(xt).squeeze(-1)
            a_np = torch.tanh(a).numpy()
            nxt, rs, ds = [], [], []
            for k, e in enumerate(envs):
                o2, r_task, done, info = e.step(a_np[k])
                r = reward_fn(r_task, info, prev_infos[k])
                prev_infos[k] = info
                ep_task[k] += r_task
                ep_temp[k] += info["temp"]
                if done:
                    n = max(e.t, 1)
                    stats["task"].append(float(ep_task[k]))
                    stats["brownout"].append(info["brownout_steps"] / n)
                    stats["docked"].append(info["docked_steps"] / n)
                    stats["wear_rate"].append(
                        (info["wear"] - ep_wear0[k]) / n
                        if info["wear"] < 1.0 else 1.0)
                    stats["temp_mean"].append(float(ep_temp[k] / n))
                    stats["steps"].append(n)
                    ep_task[k] = 0.0
                    ep_temp[k] = 0.0
                    o2 = e.reset()
                    ep_wear0[k] = e.wear
                    prev_infos[k] = None
                nxt.append(o2); rs.append(r); ds.append(done)
            obs = np.stack(nxt)
            O[t] = xt; A[t] = a; LP[t] = lp; V[t] = v
            R[t] = torch.tensor(rs, dtype=torch.float32)
            D[t] = torch.tensor(ds, dtype=torch.float32)
            step_total += n_envs
        with torch.no_grad():
            v_last = net.v(torch.tensor(obs, dtype=torch.float32)).squeeze(-1)
        adv = torch.zeros(rollout, n_envs)
        run = torch.zeros(n_envs)
        for t in range(rollout - 1, -1, -1):
            v_next = v_last if t == rollout - 1 else V[t + 1]
            delta = R[t] + gamma * v_next * (1 - D[t]) - V[t]
            run = delta + gamma * lam * (1 - D[t]) * run
            adv[t] = run
        ret = adv + V
        b_o = O.reshape(-1, obs_dim); b_a = A.reshape(-1, act_dim)
        b_lp = LP.reshape(-1); b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_lp.shape[0]; mb = n // 8
        for _ in range(6):
            perm = torch.randperm(n)
            for k in range(8):
                idx = perm[k * mb:(k + 1) * mb]
                d = net.dist(b_o[idx])
                lp = d.log_prob(b_a[idx]).sum(-1)
                ratio = torch.exp(lp - b_lp[idx])
                am = b_adv[idx]
                surr = torch.minimum(ratio * am,
                                     ratio.clamp(0.8, 1.2) * am)
                v = net.v(b_o[idx]).squeeze(-1)
                loss = -surr.mean() + 0.5 * ((v - b_ret[idx]) ** 2).mean() \
                    - ent_coef * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if log_cb:
            log_cb(step_total, stats)
    stats["policy_state"] = {k: v.cpu() for k, v in net.state_dict().items()}
    return stats
