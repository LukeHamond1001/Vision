"""Pluggable policy learner (SPEC §5.4): episodic clipped policy optimization.

Learner-independence of the commitments: everything here is policy-side
machinery. The critic reads trunk features (learned, fine), trains in the
POLICY optimizer (disjoint from every proposer — G5), predicts the gated
signal (claims already excluded — G1), and never touches the reward pathway
(W1). Returns-to-go are undiscounted, matching the γ=1 setting (W3).

History note (recorded in results/INTERPRETATION.md): a plain one-update-per-
episode advantage actor-critic regressed the reach environment badly — it cut
gradient throughput ~80× vs per-step REINFORCE. This version keeps episodic
batching but runs K clipped epochs over stored transitions (PPO-lite),
recovering throughput with the critic's stability.
"""

from __future__ import annotations

import torch


class EpisodicLearner:
    """Store the episode's transitions; K clipped-surrogate epochs at the end."""

    def __init__(self, epochs: int = 4, clip: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 1e-3, grad_clip: float = 1.0):
        self.epochs = epochs
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self._p: list[torch.Tensor] = []
        self._i: list[torch.Tensor] = []
        self._a: list[torch.Tensor] = []      # raw (pre-squash) sampled actions
        self._totals: list[float] = []

    def record(self, p: torch.Tensor, i: torch.Tensor, a_raw: torch.Tensor,
               total: float) -> None:
        self._p.append(p.detach().clone())
        self._i.append(i.detach().clone())
        self._a.append(a_raw.detach().clone())
        self._totals.append(float(total))

    def finish(self, agent, optimizer: torch.optim.Optimizer, params) -> None:
        """K clipped epochs over the episode. `agent` supplies trunk /
        action_head / critic; nothing else of it is touched."""
        if not self._p:
            return
        p = torch.stack(self._p)
        i = torch.stack(self._i)
        a = torch.stack(self._a)
        rtg = torch.tensor(self._totals).flip(0).cumsum(0).flip(0)   # W3: γ = 1
        with torch.no_grad():
            dist0 = agent.action_head.dist(agent.trunk(p, i))
            logp_old = dist0.log_prob(a).sum(-1)
        for _ in range(self.epochs):
            feats = agent.trunk(p, i)
            dist = agent.action_head.dist(feats)
            logp = dist.log_prob(a).sum(-1)
            entropy = dist.entropy().sum(-1)
            values = agent.critic(feats).squeeze(-1)
            adv = rtg - values.detach()
            if len(adv) > 1 and float(adv.std()) > 1e-6:
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            ratio = torch.exp(logp - logp_old)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv)
            loss = (-surr.mean()
                    + self.value_coef * ((values - rtg) ** 2).mean()
                    - self.entropy_coef * entropy.mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            optimizer.step()
        self._p, self._i, self._a, self._totals = [], [], [], []
