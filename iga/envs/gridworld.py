"""Minimal continuous gridworld testbed.

World state: position in [0,1]^2. One reward site, one hazard. The env emits
the pre-mapped latent of the position (the env owns the frozen encoder — the
agent never sees raw world state, matching the pre-mapped-latent story).
"""

from __future__ import annotations

import torch

from ..latent import PremappedLatent


class ContinuousGrid:
    def __init__(self, latent: PremappedLatent, reward_site=(0.8, 0.8), hazard_site=(0.45, 0.45),
                 site_radius: float = 0.08, hazard_radius: float = 0.08, start=(0.1, 0.1),
                 seed: int = 0):
        # Hazard sits ON the start->reward diagonal so E2a's trust-vs-verify
        # cells are both live: naive paths meet it, detours exist around it.
        self.latent = latent
        self.reward_site = torch.tensor(reward_site)
        self.hazard_site = torch.tensor(hazard_site)
        self.site_radius = site_radius
        self.hazard_radius = hazard_radius
        self.start = torch.tensor(start)
        self.pos = self.start.clone()
        self._gen = torch.Generator().manual_seed(seed)

    def reset(self) -> torch.Tensor:
        self.pos = self.start.clone()
        return self.latent.embed(self.pos)

    def step(self, action: torch.Tensor):
        self.pos = torch.clamp(self.pos + action.detach(), 0.0, 1.0)
        reward, done, info = 0.0, False, {}
        if torch.linalg.vector_norm(self.pos - self.reward_site) < self.site_radius:
            reward, done = 1.0, True
        if torch.linalg.vector_norm(self.pos - self.hazard_site) < self.hazard_radius:
            reward, done = -1.0, True
            info["catastrophe"] = True
        return self.latent.embed(self.pos), reward, done, info

    def embed_site(self, site: torch.Tensor) -> torch.Tensor:
        return self.latent.embed(site)
