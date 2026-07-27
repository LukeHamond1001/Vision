"""The plastic half of the system: shared trunk, action head, imagination head.

The trunk reads BOTH channel groups (that is its job — action must condition on
the goal, imagination on current perception). It serves ONLY the action and
imagination heads; by W1 nothing here may sit on a path into a reward head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedTrunk(nn.Module):
    def __init__(self, p_dim: int, i_dim: int, hidden: int = 64, feat: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p_dim + i_dim, hidden), nn.Tanh(), nn.Linear(hidden, feat), nn.Tanh()
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([p, i], dim=-1))


class ActionHead(nn.Module):
    """Gaussian policy over world actions (REINFORCE in the scaffold)."""

    def __init__(self, feat: int, act_dim: int, init_log_std: float = -1.0):
        super().__init__()
        self.mean = nn.Linear(feat, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, features: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(features), self.log_std.exp())


class ImaginationHead(nn.Module):
    """Emits target states in latent coordinates (desires, not predictions)."""

    def __init__(self, feat: int, latent_dim: int):
        super().__init__()
        self.out = nn.Linear(feat, latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # G5: features arrive detached from the caller so proposer losses can
        # never leak into the trunk through this path.
        return self.out(features)

    def propose(self, features: torch.Tensor, k: int, noise: float, gen: torch.Generator) -> torch.Tensor:
        base = self.forward(features)
        eps = torch.randn(k, base.shape[-1], generator=gen) * noise
        return base.unsqueeze(0) + eps
