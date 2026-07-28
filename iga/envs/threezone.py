"""Three-zone chain (E5b): deeper timescale separation for ladder-vs-flat.

zone 0 --gate1--> zone 1 --gate2--> zone 2, reward only in zone 2. Two slow
transitions per episode instead of one; the slow band must be held across two
distinct fast subgoal chains. `info['zone_flip']` fires on each gate;
`flips_this_episode` counts gates crossed (0-2).
"""

from __future__ import annotations

import torch

from ..latent import BandedLatent


class ThreeZoneWorld:
    def __init__(self, latent: BandedLatent, gates=((0.8, 0.2), (0.2, 0.8)),
                 reward_site=(0.8, 0.8), site_radius: float = 0.09,
                 start=(0.1, 0.1), seed: int = 0):
        assert latent.part_dims == [2, 1]
        self.latent = latent
        self.gates = [torch.tensor(g) for g in gates]
        self.reward_site = torch.tensor(reward_site)
        self.site_radius = site_radius
        self.start = torch.tensor(start)
        self.n_zones = len(gates) + 1
        self.pos = self.start.clone()
        self.zone = 0

    def _world(self) -> torch.Tensor:
        return torch.cat([self.pos, torch.tensor([float(self.zone)])])

    def reset(self) -> torch.Tensor:
        self.pos = self.start.clone()
        self.zone = 0
        self.flips_this_episode = 0
        return self.latent.embed(self._world())

    def step(self, action: torch.Tensor):
        self.pos = torch.clamp(self.pos + action.detach(), 0.0, 1.0)
        reward, done, info = 0.0, False, {}
        if self.zone < len(self.gates) and \
                torch.linalg.vector_norm(self.pos - self.gates[self.zone]) < self.site_radius:
            self.zone += 1
            self.flips_this_episode += 1
            info["zone_flip"] = True
        if self.zone == self.n_zones - 1 and \
                torch.linalg.vector_norm(self.pos - self.reward_site) < self.site_radius:
            reward, done = 1.0, True
        return self.latent.embed(self._world()), reward, done, info

    def embed_world(self, pos, zone: float) -> torch.Tensor:
        return self.latent.embed(torch.cat([torch.as_tensor(pos, dtype=torch.float32).reshape(2),
                                            torch.tensor([float(zone)])]))
