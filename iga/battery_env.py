"""v2.0: the resource layer — battery / thermal / wear over a STANDARD
MuJoCo Ant. The base env is untouched (benchmark integrity); this wrapper
adds the self-maintenance physics and exposes them as plain sensor
channels appended to the observation, the way a real robot's power and
thermal lines sit beside its proprioception.

Resource dynamics (per control step):
  battery  -= k_b * mean|torque|          (drains with effort)
  battery  += k_c  while inside the dock  (charge pad at a fixed site)
  temp     += k_t * mean(torque^2) - k_cool * temp     (heats, cools)
  wear     += k_w * mean(relu(|torque| - spike_thr))   (integrates abuse)
  brownout: battery <= 0 -> torques scaled to 10% until recharged
  episode ends on wear >= 1 (mechanical failure) or time limit

Timescales by construction (the tau-ladder, deeper than Crafter):
  gait dynamics tau ~ 5 · battery/thermal tau ~ hundreds ·
  wear tau ~ tens of thousands.

Sensor channels appended to obs: [battery, temp, wear, dock_dist].
Ground truth for gates = these same quantities (they ARE the state, so
phase-1 routing is verified against the wrapper's internals directly).
"""

from __future__ import annotations

import numpy as np


class BatteryAnt:
    """Standard Ant-v5 + resource layer. Gym-like API (reset/step)."""

    DOCK = np.array([3.0, 3.0])       # charge pad site (x, y)
    DOCK_R = 1.0

    def __init__(self, seed: int = 0, k_b: float = 0.0025, k_c: float = 0.02,
                 k_t: float = 0.004, k_cool: float = 0.01, k_w: float = 2e-5,
                 spike_thr: float = 0.6, max_steps: int = 4000):
        import gymnasium as gym
        self.env = gym.make("Ant-v5")
        self.seed = seed
        self.k_b, self.k_c, self.k_t = k_b, k_c, k_t
        self.k_cool, self.k_w, self.spike_thr = k_cool, k_w, spike_thr
        self.max_steps = max_steps
        self.obs_dim = self.env.observation_space.shape[0] + 4
        self.act_dim = self.env.action_space.shape[0]

    def _xy(self) -> np.ndarray:
        return self.env.unwrapped.data.qpos[:2].copy()

    def _aug(self, obs: np.ndarray) -> np.ndarray:
        dock_dist = float(np.linalg.norm(self._xy() - self.DOCK))
        return np.concatenate([obs, [self.battery, self.temp, self.wear,
                                     dock_dist]]).astype(np.float32)

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed + np.random.randint(10**6))
        self.battery, self.temp, self.wear = 1.0, 0.0, 0.0
        self.t = 0
        self.brownout_steps = 0
        self.docked_steps = 0
        return self._aug(obs)

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32)
        if self.battery <= 0.0:                     # brownout: crippled
            a = a * 0.1
            self.brownout_steps += 1
        obs, r_task, term, trunc, info = self.env.step(a)
        eff = float(np.mean(np.abs(a)))
        self.battery = min(1.0, self.battery - self.k_b * eff)
        in_dock = float(np.linalg.norm(self._xy() - self.DOCK)) < self.DOCK_R
        if in_dock:
            self.battery = min(1.0, self.battery + self.k_c)
            self.docked_steps += 1
        self.temp = max(0.0, self.temp + self.k_t * float(np.mean(a ** 2))
                        - self.k_cool * self.temp)
        self.wear += self.k_w * float(np.mean(np.maximum(
            np.abs(a) - self.spike_thr, 0.0)))
        self.t += 1
        done = bool(term) or self.wear >= 1.0 or self.t >= self.max_steps
        info = {"task_reward": float(r_task), "battery": self.battery,
                "temp": self.temp, "wear": self.wear, "in_dock": in_dock,
                "brownout_steps": self.brownout_steps,
                "docked_steps": self.docked_steps}
        return self._aug(obs), float(r_task), done, info
