"""Two-timescale testbed for the register ladder (SPEC §10, E5).

World state = [position (fast) | zone (slow)]. The zone flips 0→1 only when
the agent reaches the gate while in zone 0; reward pays only in zone 1 at the
reward site. So the task has a slow decision (achieve zone 1) that gates the
fast one (reach reward) — the minimal structure where holding a slow goal
across many fast windows can matter.

Emits a BandedLatent embedding: band 0 = position, band 1 = zone.
"""

from __future__ import annotations

import torch

from ..latent import BandedLatent


class TwoZoneWorld:
    def __init__(self, latent: BandedLatent, gate=(0.8, 0.2), reward_site=(0.8, 0.8),
                 site_radius: float = 0.08, start=(0.1, 0.1), seed: int = 0):
        assert latent.part_dims == [2, 1], "TwoZoneWorld needs bands [position(2), zone(1)]"
        self.latent = latent
        self.gate = torch.tensor(gate)
        self.reward_site = torch.tensor(reward_site)
        self.site_radius = site_radius
        self.start = torch.tensor(start)
        self.pos = self.start.clone()
        self.zone = 0.0

    def _world(self) -> torch.Tensor:
        return torch.cat([self.pos, torch.tensor([self.zone])])

    def reset(self) -> torch.Tensor:
        self.pos = self.start.clone()
        self.zone = 0.0
        self.flips_this_episode = 0
        return self.latent.embed(self._world())

    def step(self, action: torch.Tensor):
        self.pos = torch.clamp(self.pos + action.detach(), 0.0, 1.0)
        reward, done, info = 0.0, False, {}
        if self.zone == 0.0 and torch.linalg.vector_norm(self.pos - self.gate) < self.site_radius:
            self.zone = 1.0
            self.flips_this_episode += 1
            info["zone_flip"] = True
        if self.zone == 1.0 and torch.linalg.vector_norm(self.pos - self.reward_site) < self.site_radius:
            reward, done = 1.0, True
        return self.latent.embed(self._world()), reward, done, info

    def embed_world(self, pos, zone: float) -> torch.Tensor:
        return self.latent.embed(torch.cat([torch.as_tensor(pos, dtype=torch.float32).reshape(2),
                                            torch.tensor([zone])]))
