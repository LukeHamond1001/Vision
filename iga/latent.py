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


class BandedLatent(PremappedLatent):
    """L1: frozen latent with disjoint bands (block-diagonal embedding).

    World state arrives as concatenated parts (e.g. [position | context]);
    part j embeds into band j via its own frozen orthonormal block. Bands are
    the per-timescale subspaces the register ladder holds targets in.
    """

    def __init__(self, part_dims: list[int], band_dims: list[int], seed: int = 0):
        assert len(part_dims) == len(band_dims)
        self.part_dims = list(part_dims)
        self.band_dims = list(band_dims)
        self.world_dim = sum(part_dims)
        self.latent_dim = sum(band_dims)
        self._blocks: list[torch.Tensor] = []
        for j, (pd, bd) in enumerate(zip(part_dims, band_dims)):
            gen = torch.Generator().manual_seed(seed + 101 * j)
            q, _ = torch.linalg.qr(torch.randn(bd, pd, generator=gen))
            block = q[:, :pd].contiguous()
            block.requires_grad_(False)
            self._blocks.append(block)
        # band slice boundaries in the latent
        self.band_slices: list[slice] = []
        off = 0
        for bd in band_dims:
            self.band_slices.append(slice(off, off + bd))
            off += bd

    def embed(self, world_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts, off = [], 0
            w = world_state.reshape(-1)
            for block, pd in zip(self._blocks, self.part_dims):
                parts.append(block @ w[off:off + pd])
                off += pd
            return torch.cat(parts)

    def band(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z[..., self.band_slices[k]]

    def band_distance(self, a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
        """Frozen per-band metric d_k (L1). Accepts full-latent vectors or
        already-band-local vectors (register targets are band-local)."""
        av = self.band(a, k) if a.shape[-1] == self.latent_dim else a
        bv = self.band(b, k) if b.shape[-1] == self.latent_dim else b
        return torch.linalg.vector_norm(av - bv, dim=-1)

    def compose(self, band_targets: list[torch.Tensor | None]) -> torch.Tensor:
        """L3: composite target g_1 ⊕ ... ⊕ g_K (unset bands are zero)."""
        out = torch.zeros(self.latent_dim)
        for k, g in enumerate(band_targets):
            if g is not None:
                out[self.band_slices[k]] = g.detach()
        return out

    def frozen_tensors(self) -> list[torch.Tensor]:
        return list(self._blocks)

    def step_scale(self, k: int | None = None) -> float:
        blocks = self._blocks if k is None else [self._blocks[k]]
        return max(float(torch.linalg.matrix_norm(b, ord=2)) for b in blocks)
