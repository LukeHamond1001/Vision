"""Pluggable policy learner (SPEC §5.4): episodic advantage actor-critic.

Learner-independence of the commitments: the critic is policy-side machinery.
It reads trunk features (learned, fine), trains in the POLICY optimizer
(disjoint from every proposer — G5), predicts the gated signal (claims already
excluded — G1), and never touches the reward pathway (W1). Returns-to-go are
undiscounted, matching the γ=1 setting (W3).
"""

from __future__ import annotations

import torch


class A2CCore:
    """Per-episode batched advantage actor-critic update."""

    def __init__(self, value_coef: float = 0.5, entropy_coef: float = 1e-3,
                 grad_clip: float = 1.0):
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self._logps: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._entropies: list[torch.Tensor] = []
        self._totals: list[float] = []

    def record(self, logp: torch.Tensor, value: torch.Tensor,
               entropy: torch.Tensor, total: float) -> None:
        self._logps.append(logp)
        self._values.append(value.reshape(()))
        self._entropies.append(entropy)
        self._totals.append(float(total))

    def finish(self, optimizer: torch.optim.Optimizer, params) -> None:
        """One update from the finished episode; buffer cleared either way."""
        if not self._logps:
            return
        logps = torch.stack(self._logps)
        values = torch.stack(self._values)
        entropies = torch.stack(self._entropies)
        # Undiscounted returns-to-go of the gated signal (W3: γ = 1).
        rtg = torch.tensor(self._totals).flip(0).cumsum(0).flip(0)
        adv = rtg - values.detach()
        if len(adv) > 1 and float(adv.std()) > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = (-(adv * logps).mean()
                + self.value_coef * ((values - rtg) ** 2).mean()
                - self.entropy_coef * entropies.mean())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        optimizer.step()
        self._logps, self._values, self._entropies, self._totals = [], [], [], []
