"""v0.9 phase 2: wired agent on the discovered Crafter latent.

The behavioral claim, in its purest form: a policy whose ONLY reward is
potential-based progress toward a held register target in the frozen,
discovered latent (wiring-on) survives longer than the same policy class
trained on Crafter's native reward (wiring-off). No ground truth touches
the wiring-on reward path at run time — reward is computed from the frozen
encoder's mid band alone.

Wiring notes (SPEC lineage):
- W3 telescoping: r_t = phi_{t-1} - phi_t with phi = ||z_mid - g||. Total
  return telescopes to phi_0 - phi_T: letting meters empty and dying is
  the strict minimum, holding them pegged is the supremum, and nothing is
  farmable. Survival is never rewarded directly — it emerges because
  Crafter regenerates health only while meters are high.
- Register target g is DESIGNED, expressed in the latent vocabulary: the
  well-fed centroid of the calibration walk (frames with food and drink
  high). Designers point at goals; discovery only forbids designating
  variables to the ENCODER. Same status as v0.7's env.embed_world targets.
- Learner is policy-side machinery (SPEC §5.4, pluggable): categorical
  PPO-lite with GAE(0.9, 0.8), 4 clipped epochs — the hyperparameters the
  continuous EpisodicLearner validated. The core learner stays untouched.
"""

from __future__ import annotations

import numpy as np
import torch


class CrafterPolicy(torch.nn.Module):
    """Categorical policy + critic over the frozen 8-d latent."""

    def __init__(self, z_dim: int = 8, n_actions: int = 17, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed + 500)
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, n_actions))
        self.v = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1))

    def dist(self, z: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.pi(z))


def estimate_register_target(z_walk: torch.Tensor, truth: dict,
                             mid_slice: slice, hi: float = 8.0) -> torch.Tensor:
    """Well-fed centroid: mean mid-band latent over walk frames with food
    AND drink >= hi. Falls back to the 90th-percentile-food frames if the
    joint set is thin."""
    m = (truth["food"] >= hi) & (truth["drink"] >= hi)
    if int(m.sum()) < 64:
        thr = torch.quantile(truth["food"].float(), 0.9)
        m = truth["food"] >= thr
    return z_walk[m][:, mid_slice].mean(0)


def run_wired_episode(env, enc, policy, g_mid, mid_slice,
                      wiring: bool, max_steps: int = 1000,
                      seed_action: int = 0, device: str = "cpu",
                      seen: set | None = None, nov_bonus: float = 0.0):
    """One episode. Returns (transitions dict, survival_steps, truth_log).
    wiring=True: reward = register progress in the frozen mid band.
    wiring=False: reward = Crafter's native reward. Same everything else.

    seen/nov_bonus: SPEC one-shot curiosity, given to BOTH arms
    identically (phase-2 revision 1). Cell key = coarse discretization of
    the full frozen latent; a cell pays nov_bonus exactly once per arm
    and never again — self-retiring novelty (measured deficit: 0-3
    controllable reward events per episode under raw exploration;
    entropy collapse on noise)."""
    from .crafter_support import SLOW_EMA_TAU, slow_stats
    alpha = 1.0 / SLOW_EMA_TAU
    obs = env.reset()
    zs, acts, logps, rews = [], [], [], []
    food_hi = 0
    frame = torch.from_numpy(obs.copy()).permute(2, 0, 1).float().div(255.0)
    with torch.no_grad():
        es = slow_stats(frame.unsqueeze(0).to(device))   # online EMA state,
        # reset at episode start — mirrors ema_slow_stats exactly
        z = enc(frame.to(device), slow_feats=es).cpu()
    phi = float(torch.linalg.vector_norm(z[mid_slice] - g_mid))
    t = 0
    for t in range(max_steps):
        with torch.no_grad():
            d = policy.dist(z.to(device))
            a = d.sample()
            logp = d.log_prob(a)
        obs, r_native, done, info = env.step(int(a))
        frame = torch.from_numpy(obs.copy()).permute(2, 0, 1).float().div(255.0)
        with torch.no_grad():
            es = (1 - alpha) * es + alpha * slow_stats(frame.unsqueeze(0).to(device))
            z_next = enc(frame.to(device), slow_feats=es).cpu()
        phi_next = float(torch.linalg.vector_norm(z_next[mid_slice] - g_mid))
        r = (phi - phi_next) if wiring else float(r_native)
        if seen is not None and nov_bonus > 0:
            cell = tuple(torch.round(z_next * 2.0).to(torch.int64).tolist())
            if cell not in seen:
                seen.add(cell)
                r += nov_bonus
        zs.append(z); acts.append(a.cpu()); logps.append(logp.cpu())
        rews.append(r)
        if float(info["inventory"].get("food", 0)) >= 6:
            food_hi += 1
        z, phi = z_next, phi_next
        if done:
            break
    batch = {"z": torch.stack(zs), "a": torch.stack(acts),
             "logp": torch.stack(logps),
             "r": torch.tensor(rews, dtype=torch.float32)}
    return batch, t + 1, food_hi / max(t + 1, 1)


def ppo_update(policy: CrafterPolicy, opt: torch.optim.Optimizer, batch: dict,
               gamma: float = 0.9, lam: float = 0.8, epochs: int = 4,
               clip: float = 0.2, value_coef: float = 0.5,
               entropy_coef: float = 5e-3, device: str = "cpu") -> None:
    """Categorical PPO-lite mirror of EpisodicLearner.finish: GAE(γ,λ) on
    the undiscounted reward stream, K clipped epochs, advantage-normalized."""
    z = batch["z"].to(device)
    a = batch["a"].to(device)
    logp_old = batch["logp"].to(device)
    r = batch["r"].to(device)
    with torch.no_grad():
        v = policy.v(z).squeeze(-1)
        v_next = torch.cat([v[1:], torch.zeros(1, device=device)])
        delta = r + gamma * v_next - v
        adv = torch.zeros_like(r)
        run = 0.0
        for k in range(r.shape[0] - 1, -1, -1):
            run = float(delta[k]) + gamma * lam * run
            adv[k] = run
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    for _ in range(epochs):
        d = policy.dist(z)
        ratio = torch.exp(d.log_prob(a) - logp_old)
        surr = torch.minimum(ratio * adv,
                             ratio.clamp(1 - clip, 1 + clip) * adv)
        loss = -surr.mean() \
            + value_coef * ((policy.v(z).squeeze(-1) - ret) ** 2).mean() \
            - entropy_coef * d.entropy().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()


def train_arm(enc, g_mid, mid_slice, wiring: bool, seed: int,
              episodes: int, max_steps: int = 1000,
              device: str = "cpu", nov_bonus: float = 0.5) -> dict:
    """One seed, one arm. Returns survival + meter-hold trajectories."""
    import crafter
    env = crafter.Env(seed=seed * 31)   # SAME world sequence for both arms:
    # the paired difference isolates the reward wiring, nothing else
    policy = CrafterPolicy(seed=seed).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    survivals, food_fracs = [], []
    seen: set = set()                   # one-shot curiosity cells, per arm
    for ep in range(episodes):
        batch, alive, ffrac = run_wired_episode(
            env, enc, policy, g_mid, mid_slice, wiring,
            max_steps=max_steps, device=device,
            seen=seen, nov_bonus=nov_bonus)
        ppo_update(policy, opt, batch, device=device)
        survivals.append(alive)
        food_fracs.append(ffrac)
    return {"survival": survivals, "food_frac": food_fracs,
            "policy_state": {k: v.cpu() for k, v in policy.state_dict().items()}}


def iqm(xs) -> float:
    """Interquartile mean — the rliable-style aggregate."""
    v = np.sort(np.asarray(xs, dtype=np.float64))
    k = len(v) // 4
    core = v[k:len(v) - k] if len(v) >= 4 else v
    return float(core.mean())


def record_rollout(enc, policy, g_mid, mid_slice, seed: int = 0,
                   max_steps: int = 1000, device: str = "cpu") -> dict:
    """Run one greedy-ish episode and record everything the demo video
    needs: frames, latent trace, register distance, truth meters. Returns
    dict of tensors/lists; save with torch.save for the render script."""
    import crafter
    from .crafter_support import SLOW_EMA_TAU, slow_stats
    alpha = 1.0 / SLOW_EMA_TAU
    env = crafter.Env(seed=seed)
    obs = env.reset()
    frame = torch.from_numpy(obs.copy()).permute(2, 0, 1).float().div(255.0)
    with torch.no_grad():
        es = slow_stats(frame.unsqueeze(0).to(device))
        z = enc(frame.to(device), slow_feats=es).cpu()
    rec = {"frames": [obs.copy()], "z": [z.clone()],
           "phi": [float(torch.linalg.vector_norm(z[mid_slice] - g_mid))],
           "food": [], "drink": [], "daylight": [], "health": []}
    for t in range(max_steps):
        with torch.no_grad():
            a = int(policy.dist(z.to(device)).sample())
        obs, r, done, info = env.step(a)
        frame = torch.from_numpy(obs.copy()).permute(2, 0, 1).float().div(255.0)
        with torch.no_grad():
            es = (1 - alpha) * es + alpha * slow_stats(frame.unsqueeze(0).to(device))
            z = enc(frame.to(device), slow_feats=es).cpu()
        inv = info["inventory"]
        rec["frames"].append(obs.copy())
        rec["z"].append(z.clone())
        rec["phi"].append(float(torch.linalg.vector_norm(z[mid_slice] - g_mid)))
        rec["food"].append(float(inv.get("food", 0)))
        rec["drink"].append(float(inv.get("drink", 0)))
        rec["daylight"].append(float(env._world.daylight))
        rec["health"].append(float(inv.get("health", 0)))
        if done:
            break
    rec["survival"] = len(rec["food"])
    return rec


class QNet(torch.nn.Module):
    def __init__(self, z_dim: int = 8, n_actions: int = 17, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed + 900)
        self.f = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions))

    def forward(self, z):
        return self.f(z)


def train_arm_dqn(enc, g_mid, mid_slice, wiring: bool, seed: int,
                  total_steps: int = 300_000, max_steps: int = 1000,
                  gamma: float = 0.97, lr: float = 1e-3,
                  eps_decay_steps: int = 100_000,
                  device: str = "cpu", log=None) -> dict:
    """Replay double-DQN arm (SPEC §5.4 pluggable learner, swap #4).
    Measured motivation: on-policy episodic PPO was flat in BOTH arms at
    25-85k steps — controllable meter events are 0-3 per episode, and
    on-policy learning discards each one after a single update. Replay
    reuses them; epsilon-greedy explores without auxiliary bonuses.
    Same learner both arms; only the reward stream differs."""
    import crafter
    from .crafter_support import SLOW_EMA_TAU, slow_stats
    alpha = 1.0 / SLOW_EMA_TAU
    env = crafter.Env(seed=seed * 31)
    q, qt = QNet(seed=seed).to(device), QNet(seed=seed).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 123)
    rng = torch.Generator().manual_seed(seed + 321)
    cap = 100_000
    Z = torch.zeros(cap, 8); A = torch.zeros(cap, dtype=torch.long)
    R = torch.zeros(cap); Z2 = torch.zeros(cap, 8); D = torch.zeros(cap)
    ptr, filled = 0, 0
    survivals, food_fracs = [], []
    step_total = 0
    while step_total < total_steps:
        obs = env.reset()
        frame = torch.from_numpy(obs.copy()).permute(2, 0, 1).float().div(255.0)
        with torch.no_grad():
            es = slow_stats(frame.unsqueeze(0).to(device))
            z = enc(frame.to(device), slow_feats=es).cpu()
        phi = float(torch.linalg.vector_norm(z[mid_slice] - g_mid))
        alive, food_hi = 0, 0
        for t in range(max_steps):
            eps = max(0.1, 1.0 - 0.9 * step_total / eps_decay_steps)
            if float(torch.rand((), generator=rng)) < eps:
                a = int(torch.randint(0, 17, (), generator=rng))
            else:
                with torch.no_grad():
                    a = int(q(z.to(device)).argmax())
            obs, r_native, done, info = env.step(a)
            frame = torch.from_numpy(obs.copy()).permute(2, 0, 1).float().div(255.0)
            with torch.no_grad():
                es = (1 - alpha) * es + alpha * slow_stats(frame.unsqueeze(0).to(device))
                z2 = enc(frame.to(device), slow_feats=es).cpu()
            phi2 = float(torch.linalg.vector_norm(z2[mid_slice] - g_mid))
            r = (phi - phi2) if wiring else float(r_native)
            Z[ptr], A[ptr], R[ptr], Z2[ptr], D[ptr] = z, a, r, z2, float(done)
            ptr = (ptr + 1) % cap
            filled = min(filled + 1, cap)
            z, phi = z2, phi2
            alive += 1
            step_total += 1
            if float(info["inventory"].get("food", 0)) >= 6:
                food_hi += 1
            if filled >= 2000 and step_total % 4 == 0:
                idx = torch.randint(0, filled, (128,), generator=gen)
                zb, ab, rb = Z[idx].to(device), A[idx].to(device), R[idx].to(device)
                z2b, db = Z2[idx].to(device), D[idx].to(device)
                with torch.no_grad():
                    a2 = q(z2b).argmax(1)
                    tgt = rb + gamma * (1 - db) * qt(z2b).gather(1, a2.unsqueeze(1)).squeeze(1)
                pred = q(zb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(pred, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
            if step_total % 1000 == 0:
                qt.load_state_dict(q.state_dict())
            if done:
                break
        survivals.append(alive)
        food_fracs.append(food_hi / max(alive, 1))
        if log and len(survivals) % 100 == 0:
            import numpy as _np
            print(f"[dqn] {'ON ' if wiring else 'OFF'} ep {len(survivals)} "
                  f"steps {step_total} eps {eps:.2f} "
                  f"surv(last50) {_np.mean(survivals[-50:]):.0f} "
                  f"food%(last50) {_np.mean(food_fracs[-50:]):.2f}", flush=True)
    return {"survival": survivals, "food_frac": food_fracs,
            "policy_state": {k: v.cpu() for k, v in q.state_dict().items()}}
