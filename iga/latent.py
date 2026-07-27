"""Pre-mapped latent: frozen geometry shared by every fixed component.

Implements the frozen substrate required by W2 (frozen progress geometry) and
the neighborhood keys required by C1/C6 (cap and curiosity are keyed here, not
by state identity). Nothing in this module may ever appear in an optimizer.
"""

from __future__ import annotations

import torch


class PremappedLatent:
    """Frozen embedding of world observations plus the frozen metric over it.

    The embedding stands in for whatever pretraining produces the pre-mapped
    latent (SPEC §2, §10). It is created once, seeded, and never trained.
    """

    def __init__(self, world_dim: int, latent_dim: int, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        raw = torch.randn(latent_dim, world_dim, generator=gen)
        # Orthonormal columns: distances in latent are a rigid rescale of world
        # distances, so the frozen metric is well-conditioned by construction.
        q, _ = torch.linalg.qr(raw)
        self._embed = q[:, :world_dim].contiguous()
        self._embed.requires_grad_(False)
        self.world_dim = world_dim
        self.latent_dim = latent_dim

    def embed(self, world_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (self._embed @ world_state.reshape(-1, self.world_dim).T).T.squeeze(0)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Frozen metric (W2). Operands must already live in latent coordinates."""
        return torch.linalg.vector_norm(a - b, dim=-1)

    def neighborhood_key(self, z: torch.Tensor, radius: float) -> tuple:
        """Grid hash of the frozen latent at the given radius (C1, C6).

        Contains no learned parameters, therefore ungameable by the proposer.
        """
        with torch.no_grad():
            return tuple(torch.floor(z / radius).to(torch.int64).tolist())

    def frozen_tensors(self) -> list[torch.Tensor]:
        return [self._embed]

    def step_scale(self) -> float:
        """Latent distance moved per unit world distance (for C5 horizon math)."""
        return float(torch.linalg.matrix_norm(self._embed, ord=2))
