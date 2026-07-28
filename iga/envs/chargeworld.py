"""Charge world (E5c): the ladder's make-or-break task.

Design rationale (SPEC §10, round 8): the zone worlds could never
discriminate ladder from flat because their slow variable is DISCRETE — slow-
band progress toward a held target is a step function that pays only at the
flip instant. Here the slow variable is CONTINUOUS with sustain/decay
dynamics, so a held slow target converts into a dense pull every step:

- charge c ∈ [0,1]: rises while the agent sits on the charging pad, decays
  everywhere else — sustained pursuit is NECESSARY, not merely available;
- the door pays +1 only when c ≥ threshold;
- the pre-mapped evaluator carries mild "shiny" bumps: worthless sites whose
  claims tempt every window-boundary re-proposal. Flat re-chooses its whole
  composite at each boundary (drift pressure); the ladder's slow slice is
  locked for τ_slow (commitment persistence).

Band 0 = position (fast), band 1 = charge (slow).
"""

from __future__ import annotations

import torch

from ..latent import BandedLatent


class ChargeWorld:
    def __init__(self, latent: BandedLatent, pad=(0.2, 0.8), pad_radius: float = 0.12,
                 door=(0.8, 0.8), door_radius: float = 0.09, start=(0.5, 0.1),
                 charge_rate: float = 0.02, decay: float = 0.002,
                 threshold: float = 0.8, warm_start_prob: float = 0.0, seed: int = 0):
        # decay 0.002 (round 9): at 0.005 the pad->door trip (~12 steps) cost
        # 0.06 charge, requiring departure at c >= 0.86 — beyond what ANY
        # tested agent sustains, making completion near-infeasible across the
        # board. At 0.002 the trip costs ~0.024 (depart >= 0.83 suffices);
        # sustained charging remains necessary and temptation still costs.
        # Condition-neutral: shared by every agent in every comparison.
        assert latent.part_dims == [2, 1]
        self.latent = latent
        self.pad = torch.tensor(pad)
        self.pad_radius = pad_radius
        self.door = torch.tensor(door)
        self.door_radius = door_radius
        self.start = torch.tensor(start)
        self.charge_rate = charge_rate
        self.decay = decay
        self.threshold = threshold
        self.warm_start_prob = warm_start_prob
        self._gen = torch.Generator().manual_seed(seed + 7000)
        self.pos = self.start.clone()
        self.c = 0.0
        self.warm = False

    def _world(self) -> torch.Tensor:
        return torch.cat([self.pos, torch.tensor([self.c])])

    def reset(self) -> torch.Tensor:
        # Warm-start curriculum (round 9): with probability warm_start_prob
        # the episode begins charged at the pad, rehearsing the charge->door
        # phase directly. Condition-neutral (shared by every agent); the
        # discriminating comparison scores COLD episodes only.
        self.warm = bool(torch.rand((), generator=self._gen) < self.warm_start_prob)
        if self.warm:
            self.pos = self.pad.clone()
            self.c = 0.85
        else:
            self.pos = self.start.clone()
            self.c = 0.0
        self.max_c = self.c
        return self.latent.embed(self._world())

    def step(self, action: torch.Tensor):
        self.pos = torch.clamp(self.pos + action.detach(), 0.0, 1.0)
        on_pad = torch.linalg.vector_norm(self.pos - self.pad) < self.pad_radius
        self.c = min(1.0, self.c + self.charge_rate) if on_pad else max(0.0, self.c - self.decay)
        self.max_c = max(self.max_c, self.c)
        reward, done, info = 0.0, False, {"c": self.c}
        if self.c >= self.threshold and \
                torch.linalg.vector_norm(self.pos - self.door) < self.door_radius:
            reward, done = 1.0, True
        return self.latent.embed(self._world()), reward, done, info

    def embed_world(self, pos, c: float) -> torch.Tensor:
        return self.latent.embed(torch.cat([torch.as_tensor(pos, dtype=torch.float32).reshape(2),
                                            torch.tensor([float(c)])]))
