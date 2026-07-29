"""Pixel ChargeWorld (v0.7): the same dynamics observed as rendered frames.

64×64×3 images: pad disc (brightness tracks standing-on-pad), agent dot,
door bar whose color tracks openness, and GLOBAL ILLUMINATION proportional
to charge — the v0.5 multiplicative entanglement, now in pixels. No channel,
pixel, or feature IS the slow variable; charge is smeared across every lit
pixel. This is the freebies-law gauntlet at its hardest scaffold form:
thousands of observation dims, a conv encoder, and the full R2 recipe.

Rendering is pure torch (gaussian discs on a fixed meshgrid) — fast enough
to be the data generator for pretraining walks and RL episodes alike.
"""

from __future__ import annotations

import torch

from .chargeworld import ChargeWorld
from ..latent import BandedLatent


class PixelRenderer:
    def __init__(self, size: int = 64):
        self.size = size
        ax = torch.linspace(0.0, 1.0, size)
        self.gy, self.gx = torch.meshgrid(ax, ax, indexing="ij")

    def _disc(self, center, sigma: float) -> torch.Tensor:
        d2 = (self.gx - float(center[0])) ** 2 + (self.gy - 1.0 + float(center[1])) ** 2
        return torch.exp(-d2 / (2 * sigma ** 2))

    def render(self, pos: torch.Tensor, c: float, pad, door, threshold: float) -> torch.Tensor:
        """Returns [3, size, size] in [0,1]."""
        img = torch.zeros(3, self.size, self.size)
        img[2] += 0.15                                   # night-blue floor
        pad_glow = self._disc(pad, 0.10)
        img[1] += 0.55 * pad_glow                        # pad: green disc
        img[2] += 0.25 * pad_glow
        door_open = 1.0 if c >= threshold else 0.0
        door_glow = self._disc(door, 0.07)
        img[0] += (0.6 - 0.4 * door_open) * door_glow    # door: red -> gold
        img[1] += (0.2 + 0.5 * door_open) * door_glow
        agent_dot = self._disc(pos, 0.035)
        img[0] += 0.9 * agent_dot
        img[1] += 0.9 * agent_dot
        img[2] += 0.9 * agent_dot
        illum = 0.55 + 0.45 * float(c)                   # charge = global light
        return (img * illum).clamp(0.0, 1.0)


class PixelObservationLatent(BandedLatent):
    """Frozen encoder over rendered frames (any module frame->latent). Same
    contracts as the v0.6 nonlinear latent: frozen metric, band slices,
    empirical step scale, embed_delta raises (C4 forward-model deferral)."""

    def __init__(self, encoder, band_dims: list[int], renderer: PixelRenderer,
                 pad, door, threshold: float):
        super().__init__([2, 1], band_dims, seed=0)
        self.encoder = encoder
        self.renderer = renderer
        self._pad, self._door, self._thr = pad, door, threshold
        self.world_dim = 3

    def embed_obs(self, frame: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(frame)

    def embed(self, world_state: torch.Tensor) -> torch.Tensor:
        w = world_state.reshape(3)
        f = self.renderer.render(w[:2], float(w[2]), self._pad, self._door, self._thr)
        return self.embed_obs(f)

    def frozen_tensors(self) -> list[torch.Tensor]:
        return list(self.encoder.parameters()) + list(self.encoder.buffers())

    def embed_delta(self, world_delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("pixel encoder: C4 lookahead needs a frozen "
                                  "forward model (SPEC §11 R1 deferral)")

    def _post_scale(self, s: float) -> None:
        with torch.no_grad():
            self.encoder.out_scale.copy_(self.encoder.out_scale / s)

    def step_scale(self, k: int | None = None) -> float:
        gen = torch.Generator().manual_seed(123)
        pts = torch.rand(48, 2, generator=gen)
        cs = torch.rand(48, generator=gen)
        norms = []
        for p, c in zip(pts, cs):
            a = torch.randn(2, generator=gen)
            a = a / torch.linalg.vector_norm(a) * 0.1
            z0 = self.embed(torch.cat([p, c.reshape(1)]))
            z1 = self.embed(torch.cat([torch.clamp(p + a, 0, 1), c.reshape(1)]))
            d = z1 - z0
            norms.append(torch.linalg.vector_norm(
                d if k is None else d[self.band_slices[k]]))
        return float(torch.stack(norms).median() / 0.1)


class PixelChargeWorld:
    """ChargeWorld dynamics; emits ENCODED latents of rendered frames through
    a frozen conv encoder. embed_world goes world -> frame -> encoder."""

    def __init__(self, latent, seed: int = 0, size: int = 64, **charge_kwargs):
        self.renderer = PixelRenderer(size)
        self.inner = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=seed,
                                 **charge_kwargs)
        self.latent = latent                             # PixelObservationLatent
        self.pad, self.door = self.inner.pad, self.inner.door
        self.start = self.inner.start

    @property
    def c(self):
        return self.inner.c

    @property
    def pos(self):
        return self.inner.pos

    @property
    def max_c(self):
        return self.inner.max_c

    def frame(self) -> torch.Tensor:
        return self.renderer.render(self.inner.pos, self.inner.c, self.inner.pad,
                                    self.inner.door, self.inner.threshold)

    def reset(self) -> torch.Tensor:
        self.inner.reset()
        return self.latent.embed_obs(self.frame())

    def step(self, action: torch.Tensor):
        _, r, done, info = self.inner.step(action)
        return self.latent.embed_obs(self.frame()), r, done, info

    def embed_world(self, pos, c: float) -> torch.Tensor:
        f = self.renderer.render(torch.as_tensor(pos, dtype=torch.float32).reshape(2),
                                 float(c), self.inner.pad, self.inner.door,
                                 self.inner.threshold)
        return self.latent.embed_obs(f)
