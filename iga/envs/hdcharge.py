"""HD ChargeWorld (v0.5 scale-up, phase 1): the same dynamics seen through a
fixed nonlinear sensor — high-dimensional, entangled observations where no
input column IS the slow variable.

Sensor (32-d): a 5×5 grid of positional radial-basis units whose gains are
multiplicatively modulated by charge (illumination-style entanglement:
o_i = rbf_i(pos) · (0.7 + 0.3·c)), plus 3 indirect charge-driven features
(c, c², 1−c passed through fixed affine+tanh mixing) and 4 fixed linear
position features. The pretraining question: can the OU-ladder recipe
recover a banded latent whose slow band tracks c — from observation
trajectories alone, with c never designated? The control: PCA on the same
walks (same data, no temporal prior).

Deliberate scope cut, documented: no hazard sites, hence no flinch — the
C4 lookahead needs a frozen one-step model once the sensor is nonlinear
(embed_delta exactness is a linear-encoder privilege). That question is
DEFERRED, not waived.
"""

from __future__ import annotations

import torch

from .chargeworld import ChargeWorld
from ..latent import BandedLatent


class HDSensor:
    """Fixed nonlinear observation map world(3) -> o(32). Never trained."""

    def __init__(self, seed: int = 0):
        gen = torch.Generator().manual_seed(seed + 9000)
        xs = torch.linspace(0.1, 0.9, 5)
        self.centers = torch.cartesian_prod(xs, xs)          # 25 RBF centers
        self.rbf_sigma = 0.15
        self.A_c = torch.randn(3, 3, generator=gen) * 0.8    # mixes (c, c^2, 1-c)
        self.A_p = torch.randn(4, 2, generator=gen) * 0.8    # 4 linear pos features
        self.obs_dim = 25 + 3 + 4

    def observe(self, pos: torch.Tensor, c: float) -> torch.Tensor:
        with torch.no_grad():
            d2 = ((self.centers - pos.unsqueeze(0)) ** 2).sum(-1)
            rbf = torch.exp(-d2 / (2 * self.rbf_sigma ** 2)) * (0.7 + 0.3 * c)
            cf = torch.tanh(self.A_c @ torch.tensor([c, c * c, 1.0 - c]))
            pf = torch.tanh(self.A_p @ pos)
            return torch.cat([rbf, cf, pf])


class HDChargeWorld:
    """ChargeWorld dynamics; emits ENCODED latents through a frozen encoder
    over the sensor. `embed_world` (for head sites) goes world -> sensor ->
    encoder. The encoder arrives from pretraining (or PCA for the control)."""

    def __init__(self, encoder_W: torch.Tensor, band_dims: list[int],
                 sensor: HDSensor | None = None, seed: int = 0, **charge_kwargs):
        self.sensor = sensor or HDSensor()
        self.inner = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=seed,
                                 **charge_kwargs)
        self.latent = ObservationLatent(encoder_W, band_dims, self.sensor)
        # convenience passthroughs used by builders/metrics
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

    def reset(self) -> torch.Tensor:
        self.inner.reset()
        return self.latent.embed_obs(self.sensor.observe(self.inner.pos, self.inner.c))

    def step(self, action: torch.Tensor):
        _, r, done, info = self.inner.step(action)
        o = self.sensor.observe(self.inner.pos, self.inner.c)
        return self.latent.embed_obs(o), r, done, info

    def embed_world(self, pos, c: float) -> torch.Tensor:
        o = self.sensor.observe(torch.as_tensor(pos, dtype=torch.float32).reshape(2), float(c))
        return self.latent.embed_obs(o)


class ObservationLatent(BandedLatent):
    """Frozen linear encoder over OBSERVATIONS (o -> latent). Bands are output
    slices as always; the frozen-metric contract (W2) holds over o-space.
    embed_delta is NOT available (nonlinear sensor) — the flinch's exactness
    was a linear-world privilege; deferred with the hazard-free testbed."""

    def __init__(self, W: torch.Tensor, band_dims: list[int], sensor: HDSensor):
        super().__init__([2, 1], band_dims, seed=0)   # slice bookkeeping only
        self.world_dim = sensor.obs_dim
        self._W = W.detach().clone()
        self._W.requires_grad_(False)
        self.sensor = sensor

    def embed_obs(self, o: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._W @ o.reshape(self.world_dim)

    def embed(self, world_state: torch.Tensor) -> torch.Tensor:  # world -> o -> z
        w = world_state.reshape(3)
        return self.embed_obs(self.sensor.observe(w[:2], float(w[2])))

    def frozen_tensors(self) -> list[torch.Tensor]:
        return [self._W]

    def embed_delta(self, world_delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "nonlinear sensor: one-step lookahead needs a frozen forward model "
            "(SPEC C4 deferral — hazard-free testbed does not exercise it)")

    def _post_scale(self, s: float) -> None:
        self._W = self._W / s
        self._ss_cache = {}

    def step_scale(self, k: int | None = None) -> float:
        # empirical: median latent displacement per max-size world step,
        # probed on random states (frozen quantities only)
        gen = torch.Generator().manual_seed(123)
        pts = torch.rand(64, 2, generator=gen)
        cs = torch.rand(64, generator=gen)
        deltas = []
        for p, c in zip(pts, cs):
            a = torch.randn(2, generator=gen)
            a = a / torch.linalg.vector_norm(a) * 0.1
            z0 = self.embed_obs(self.sensor.observe(p, float(c)))
            z1 = self.embed_obs(self.sensor.observe(torch.clamp(p + a, 0, 1), float(c)))
            d = z1 - z0
            deltas.append(d if k is None else d[self.band_slices[k]])
        return float(torch.stack([torch.linalg.vector_norm(d) for d in deltas]).median() / 0.1)


class NonlinearObservationLatent(ObservationLatent):
    """v0.6: frozen MLP encoder over observations. Same contracts as
    ObservationLatent (frozen metric, band slices, empirical step-scale,
    embed_delta raises); the encoder is a frozen module rather than a
    matrix — which is exactly what escapes the linear-chord ceiling."""

    def __init__(self, encoder, band_dims: list[int], sensor: HDSensor):
        BandedLatent.__init__(self, [2, 1], band_dims, seed=0)
        self.world_dim = sensor.obs_dim
        self.encoder = encoder            # frozen (EncoderMLP.freeze())
        self.sensor = sensor

    def embed_obs(self, o: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(o.reshape(self.world_dim))

    def frozen_tensors(self) -> list[torch.Tensor]:
        return list(self.encoder.parameters()) + list(self.encoder.buffers())

    def _post_scale(self, s: float) -> None:
        with torch.no_grad():
            self.encoder.out_scale.copy_(self.encoder.out_scale / s)
        self._ss_cache = {}
