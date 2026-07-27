"""Adversarial probes E3a and E3b (SPEC §9).

E3a is fully implemented (it needs no training loop — it is a direct attack on
the cap's keying). E3b ships as an environment plus a measurement harness; the
full multi-seed experiment belongs to the evaluation battery, not the scaffold.
"""

from __future__ import annotations

import torch

from ..constraints import CoverageCap
from ..latent import PremappedLatent


def epsilon_ball_attack(latent: PremappedLatent, cap: CoverageCap, center: torch.Tensor,
                        eps: float, n: int = 64, seed: int = 0) -> dict:
    """E3a: emit n distinct-identity near-duplicates inside an eps-ball and
    count how many the cap admits.

    Prediction (SPEC §C1): an identity-keyed cap admits ~n (collapse regime);
    the neighborhood-keyed cap admits at most max_updates per covered
    neighborhood — with eps at or below the cap radius, ~max_updates total.
    """
    gen = torch.Generator().manual_seed(seed)
    admitted = 0
    keys = set()
    for _ in range(n):
        z = center + torch.randn(center.shape[-1], generator=gen) * (eps / center.shape[-1] ** 0.5)
        keys.add(cap.key(z))
        if cap.allow(z):
            cap.record(z)
            admitted += 1
    return {"emitted": n, "admitted": admitted, "distinct_identities": n, "distinct_keys": len(keys)}


class ReachabilityBiasEnv:
    """E3b: two-armed world for the §6.4 residual.

    Arm A ("treadmill"): always reachable, realized value 0.
    Arm B ("prize"): realized value R, but the traversal succeeds only with
    probability p_success (< 1) — failure returns the agent to start.

    The measurement: over training, the frequency with which the proposer's
    committed target lies in arm A's region. Without C7 (value bar), §6.4
    predicts drift toward arm A whenever H·w_p·(1−p) > p·w_r·R.
    """

    def __init__(self, latent: PremappedLatent, p_success: float = 0.4, prize: float = 1.0,
                 arm_a=(0.9, 0.1), arm_b=(0.1, 0.9), seed: int = 0):
        self.latent = latent
        self.p_success = p_success
        self.prize = prize
        self.arm_a = torch.tensor(arm_a)
        self.arm_b = torch.tensor(arm_b)
        self.start = torch.tensor((0.5, 0.5))
        self.pos = self.start.clone()
        self._gen = torch.Generator().manual_seed(seed)

    def reset(self) -> torch.Tensor:
        self.pos = self.start.clone()
        return self.latent.embed(self.pos)

    def step(self, action: torch.Tensor):
        self.pos = torch.clamp(self.pos + action.detach(), 0.0, 1.0)
        reward, done, info = 0.0, False, {}
        if torch.linalg.vector_norm(self.pos - self.arm_a) < 0.08:
            reward, done, info["arm"] = 0.0, True, "A"          # reachable, worthless
        elif torch.linalg.vector_norm(self.pos - self.arm_b) < 0.08:
            if torch.rand((), generator=self._gen) < self.p_success:
                reward, done, info["arm"] = self.prize, True, "B"
            else:
                self.pos = self.start.clone()                    # risky traversal failed
                info["slipped"] = True
        return self.latent.embed(self.pos), reward, done, info

    def target_arm(self, g_latent: torch.Tensor) -> str:
        """Classify a committed target by nearest arm (for the drift metric)."""
        da = float(self.latent.distance(g_latent, self.latent.embed(self.arm_a)))
        db = float(self.latent.distance(g_latent, self.latent.embed(self.arm_b)))
        return "A" if da < db else "B"
