"""v1.2: vectorized-PPO harness for the homeostasis three-arm campaign.

Why this exists (measured): v1.1 proved 1M single-env DQN steps teach
NOTHING on Crafter from ANY reward (native = zero = 175); the learner
regime, not the signal, was binding. This is the published-regime tool:
parallel envs + clipped PPO at millions of steps.

Design:
- CrafterVec: N worker processes, each one Crafter env; batched obs out,
  actions in. Worker auto-resets on done and reports per-episode stats
  (survival, vital uptimes, slept-at-night count).
- Reward arms (identical policy/learner/worlds): 'wired' = register
  progress toward the homeostasis target in the frozen mid band (all
  four vitals); 'native' = Crafter reward; 'zero' = nothing.
- Wired rewards are computed CENTRALLY (one batched encoder forward per
  vector step) so workers stay encoder-free.
- Sleep detection for the mechanism table: action==6 (sleep) taken at
  night (daylight < 0.3), counted per episode.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch


def _worker(remote, seed: int):
    import crafter
    env = crafter.Env(seed=seed)
    obs = env.reset()
    ep_len, food_hi, drink_hi, energy_hi, slept = 0, 0, 0, 0, 0
    while True:
        cmd, a = remote.recv()
        if cmd == "step":
            obs, r_native, done, info = env.step(int(a))
            ep_len += 1
            inv = info["inventory"]
            day = float(env._world.daylight)
            if inv.get("food", 0) >= 6: food_hi += 1
            if inv.get("drink", 0) >= 6: drink_hi += 1
            if inv.get("energy", 0) >= 6: energy_hi += 1
            if int(a) == 6 and day < 0.3: slept += 1
            ep_stat = None
            if done or ep_len >= 1000:
                ep_stat = (ep_len, food_hi / max(ep_len, 1),
                           drink_hi / max(ep_len, 1),
                           energy_hi / max(ep_len, 1), slept)
                obs = env.reset()
                ep_len = food_hi = drink_hi = energy_hi = slept = 0
                done = True
            remote.send((obs, float(r_native), bool(done), ep_stat))
        elif cmd == "obs":
            remote.send(obs)
        elif cmd == "close":
            remote.close()
            break


class CrafterVec:
    """N Crafter workers with batched step. Episode stats stream back."""

    def __init__(self, n: int, seed: int):
        ctx = mp.get_context("spawn")
        self.n = n
        self.remotes, self.workers = [], []
        for k in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, seed * 131 + k),
                            daemon=True)
            p.start()
            self.remotes.append(parent)
            self.workers.append(p)

    def obs(self) -> np.ndarray:
        for r in self.remotes:
            r.send(("obs", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions) -> tuple:
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        obs, rews, dones, stats = [], [], [], []
        for r in self.remotes:
            o, rn, d, st = r.recv()
            obs.append(o); rews.append(rn); dones.append(d)
            if st is not None:
                stats.append(st)
        return (np.stack(obs), np.array(rews, dtype=np.float32),
                np.array(dones), stats)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass


class PixelActorCritic(torch.nn.Module):
    def __init__(self, n_actions: int = 17, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed + 1700)
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 512), torch.nn.ReLU())
        self.pi = torch.nn.Linear(512, n_actions)
        self.v = torch.nn.Linear(512, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def run_ppo_arm(enc, g_mid, mid_slice, reward_mode: str, seed: int,
                total_steps: int = 3_000_000, n_envs: int = 12,
                rollout: int = 256, gamma: float = 0.99, lam: float = 0.95,
                clip: float = 0.2, epochs: int = 4, minibatches: int = 4,
                lr: float = 2.5e-4, ent_coef: float = 0.01,
                vf_coef: float = 0.5, device: str = "cpu",
                log_cb=None, init_state: dict | None = None,
                dim_mask=None) -> dict:
    """One PPO arm. Returns per-episode stats streams + policy state.
    Wired reward: batched phi-progress in the frozen homeostasis register,
    with phi state kept per env and reset (to the fresh obs value) on
    episode boundaries so no cross-life reward leaks."""
    from .crafter_support import SLOW_EMA_TAU, slow_stats
    alpha = 1.0 / SLOW_EMA_TAU
    vec = CrafterVec(n_envs, seed)
    net = PixelActorCritic(seed=seed).to(device)
    if init_state is not None:                 # glass-box warm start: the
        net.load_state_dict(init_state)        # same agent, new wants
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    obs = vec.obs()

    def encode_phi(obs_np, es):
        x = torch.from_numpy(obs_np).permute(0, 3, 1, 2).float().div(255.0).to(device)
        s = slow_stats(x)
        es = s if es is None else (1 - alpha) * es + alpha * s
        with torch.no_grad():
            z = enc(x, slow_feats=es)
        diff = z[:, mid_slice] - g_mid.to(device)
        if dim_mask is not None:               # glass-box edit: DELETE a
            diff = diff * dim_mask.to(device)  # dimension from the want
        phi = torch.linalg.vector_norm(diff, dim=1)
        return phi.cpu().numpy(), es

    es = None
    phi = None
    if reward_mode == "wired":
        phi, es = encode_phi(obs, es)
    stats_all = {"survival": [], "food": [], "drink": [], "energy": [],
                 "slept": []}
    step_total = 0
    while step_total < total_steps:
        O = torch.zeros(rollout, n_envs, 3, 64, 64, dtype=torch.uint8)
        A = torch.zeros(rollout, n_envs, dtype=torch.long)
        LP = torch.zeros(rollout, n_envs)
        V = torch.zeros(rollout, n_envs)
        R = torch.zeros(rollout, n_envs)
        D = torch.zeros(rollout, n_envs)
        for t in range(rollout):
            xt = torch.from_numpy(obs).permute(0, 3, 1, 2)
            with torch.no_grad():
                logits, v = net(xt.float().div(255.0).to(device))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            obs2, r_native, dones, ep_stats = vec.step(a.cpu().numpy())
            if reward_mode == "wired":
                phi2, es = encode_phi(obs2, es)
                r = phi - phi2
                # reset phi baseline on episode boundary (fresh life)
                for k, d in enumerate(dones):
                    if d:
                        r[k] = 0.0
                phi = phi2
            elif reward_mode == "native":
                r = r_native
            else:
                r = np.zeros(n_envs, dtype=np.float32)
            O[t] = xt; A[t] = a.cpu(); LP[t] = lp.cpu(); V[t] = v.cpu()
            R[t] = torch.from_numpy(np.asarray(r, dtype=np.float32))
            D[t] = torch.from_numpy(dones.astype(np.float32))
            obs = obs2
            step_total += n_envs
            for st in ep_stats:
                stats_all["survival"].append(st[0])
                stats_all["food"].append(st[1])
                stats_all["drink"].append(st[2])
                stats_all["energy"].append(st[3])
                stats_all["slept"].append(st[4])
        with torch.no_grad():
            _, v_last = net(torch.from_numpy(obs).permute(0, 3, 1, 2)
                            .float().div(255.0).to(device))
            v_last = v_last.cpu()
        adv = torch.zeros(rollout, n_envs)
        run = torch.zeros(n_envs)
        for t in range(rollout - 1, -1, -1):
            v_next = v_last if t == rollout - 1 else V[t + 1]
            delta = R[t] + gamma * v_next * (1 - D[t]) - V[t]
            run = delta + gamma * lam * (1 - D[t]) * run
            adv[t] = run
        ret = adv + V
        b_obs = O.reshape(-1, 3, 64, 64)
        b_a = A.reshape(-1); b_lp = LP.reshape(-1)
        b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n_batch = b_a.shape[0]
        mb = n_batch // minibatches
        for _ in range(epochs):
            perm = torch.randperm(n_batch)
            for k in range(minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                logits, v = net(b_obs[idx].float().div(255.0).to(device))
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b_a[idx].to(device))
                ratio = torch.exp(lp - b_lp[idx].to(device))
                a_mb = b_adv[idx].to(device)
                surr = torch.minimum(ratio * a_mb,
                                     ratio.clamp(1 - clip, 1 + clip) * a_mb)
                loss = -surr.mean() \
                    + vf_coef * ((v - b_ret[idx].to(device)) ** 2).mean() \
                    - ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if log_cb:
            log_cb(step_total, stats_all)
    vec.close()
    stats_all["policy_state"] = {k: v.cpu() for k, v in net.state_dict().items()}
    return stats_all
