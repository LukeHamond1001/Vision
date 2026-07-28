"""v0.3: multi-timescale latent pretraining — the sketched-OU ladder (weak form).

This is the session's opening proposal made executable at scaffold scale:
constrain the latent to a stationary process whose bands carry prescribed
timescales. Weak-SIGReg-style marginal (covariance whitening toward identity)
plus per-band OU innovation losses:

    L = Σ_k  E‖ z_k(t+Δ) − ρ_k(Δ) z_k(t) ‖²  /  (1 − ρ_k(Δ)²)
        + λ_cov ‖ Cov(z) − I ‖²_F

with ρ_k(Δ) = exp(−Δ/τ_k): the fast band must decorrelate over short lags,
the slow band must persist. Training data: random-walk exploration in the
environment — no rewards, no evaluator, no controller.

Two properties this buys, pre-registered in experiments_v03.py:
1. BAND DISCOVERY: the slowest world variable (charge) should be routed into
   the slow band by the objective alone — no hand assignment.
2. EMERGENT SCALE: whitening gives the slow variable unit variance, which
   amplifies slow-band distances relative to a raw embedding — supplying,
   from data, the per-band progress weighting that round 8 hand-tuned
   (w ∝ τ). The pretrained condition therefore runs with w_prog_bands=(1,1).

The encoder is LINEAR and frozen after pretraining: embed_delta stays exact
(the C4 flinch's lookahead), W2's frozen-metric contract holds verbatim, and
every wiring assertion extends unchanged.
"""

from __future__ import annotations

import torch

from .latent import BandedLatent


def collect_random_walk(env, steps: int, max_world_step: float = 0.1,
                        coverage_reset_every: int = 100, seed: int = 0) -> torch.Tensor:
    """Random-policy exploration; returns raw WORLD states [steps, world_dim].

    Coverage resets randomize BOTH position and charge periodically: a random
    policy almost never charges, so without them c has near-zero variance in
    the walk and whitening amplifies its noise into every band (measured:
    charge leaked into the fast band at 12× before this fix). Between resets
    the dynamics are the environment's own, so per-band autocorrelations are
    genuine."""
    gen = torch.Generator().manual_seed(seed)
    env.reset()
    out = []
    for t in range(steps):
        a = (torch.rand(2, generator=gen) * 2 - 1) * max_world_step
        env.step(a)
        out.append(torch.cat([env.pos.clone(), torch.tensor([env.c])]))
        if (t + 1) % coverage_reset_every == 0:
            env.reset()
            env.pos = torch.rand(2, generator=gen)
            env.c = float(torch.rand((), generator=gen))
    return torch.stack(out)


def pretrain_ou_ladder(world_traj: torch.Tensor, band_dims: list[int],
                       taus: list[float], lags: list[int], epochs: int = 400,
                       lr: float = 1e-2, lam_cov: float = 1.0,
                       segment_len: int | None = None, seed: int = 0,
                       ) -> torch.Tensor:
    """Learn a linear map W: world → latent under the OU-ladder objective.
    Returns W (latent_dim × world_dim), to be frozen by the caller.

    `segment_len`: coverage-reset period of the collection. Innovation pairs
    that span a reset are MASKED OUT (episode-boundary masking) — teleports
    are coverage devices, not dynamics, and scoring across them poisons the
    slow prior (measured: charge's effective lag-15 autocorrelation fell
    below the slow band's target and routing inverted)."""
    torch.manual_seed(seed)
    world_dim = world_traj.shape[1]
    latent_dim = sum(band_dims)
    # standardize inputs so no raw variable starts privileged
    mu, sd = world_traj.mean(0), world_traj.std(0).clamp_min(1e-6)
    x = (world_traj - mu) / sd
    W = torch.nn.Parameter(torch.randn(latent_dim, world_dim) * 0.5)
    opt = torch.optim.Adam([W], lr=lr)
    slices = []
    off = 0
    for bd in band_dims:
        slices.append(slice(off, off + bd))
        off += bd
    T = x.shape[0]
    max_lag = max(lags)
    idx = torch.arange(T - max_lag)
    masks = {}
    for lag in set(lags):
        if segment_len is None:
            masks[lag] = torch.ones(T - max_lag, dtype=torch.bool)
        else:
            masks[lag] = (idx // segment_len) == ((idx + lag) // segment_len)
    for _ in range(epochs):
        z = x @ W.T
        loss = torch.tensor(0.0)
        for sl, tau, lag in zip(slices, taus, lags):
            rho = float(torch.exp(torch.tensor(-lag / tau)))
            m = masks[lag]
            z0, z1 = z[:T - max_lag][m][:, sl], z[lag:T - max_lag + lag][m][:, sl]
            innov = ((z1 - rho * z0) ** 2).mean() / max(1 - rho ** 2, 1e-3)
            loss = loss + innov
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (T - 1)
        loss = loss + lam_cov * ((cov - torch.eye(latent_dim)) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # fold input standardization into the returned map: z = W (x−μ)/σ = W' x + b;
    # drop the constant offset (translations don't affect any distance the
    # architecture uses) and return the linear part.
    with torch.no_grad():
        W_final = (W / sd.unsqueeze(0)).detach().clone()
    return W_final


class PretrainedBandedLatent(BandedLatent):
    """BandedLatent whose embedding is a learned (then frozen) linear map
    rather than random orthonormal blocks. Bands are output slices; the
    matrix may mix world variables freely — band routing is the OBJECTIVE's
    job, not the constructor's."""

    def __init__(self, W: torch.Tensor, part_dims: list[int], band_dims: list[int]):
        # initialize the parent for slice bookkeeping, then override the map
        super().__init__(part_dims, band_dims, seed=0)
        assert W.shape == (sum(band_dims), sum(part_dims))
        self._W = W.detach().clone()
        self._W.requires_grad_(False)

    def embed(self, world_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._W @ world_state.reshape(self.world_dim)

    def embed_delta(self, world_delta: torch.Tensor) -> torch.Tensor:
        """Actions move only the position part; exact by linearity."""
        with torch.no_grad():
            d = torch.zeros(self.world_dim)
            d[:self.part_dims[0]] = world_delta.reshape(self.part_dims[0])
            return self._W @ d

    def frozen_tensors(self) -> list[torch.Tensor]:
        return [self._W]

    def step_scale(self, k: int | None = None) -> float:
        Wp = self._W[:, :self.part_dims[0]]
        if k is None:
            return float(torch.linalg.matrix_norm(Wp, ord=2))
        return float(torch.linalg.matrix_norm(Wp[self.band_slices[k], :], ord=2))


def band_routing_report(latent: PretrainedBandedLatent, world_traj: torch.Tensor) -> dict:
    """Interpretability check: how much of each band's variance is driven by
    charge vs position? Returns per-band correlation of band-norm with c."""
    with torch.no_grad():
        z = torch.stack([latent.embed(w) for w in world_traj])
        c = world_traj[:, -1]
        out = {}
        for k, sl in enumerate(latent.band_slices):
            zb = z[:, sl]
            # max abs correlation of any band coordinate with charge
            corr = torch.zeros(zb.shape[1])
            for j in range(zb.shape[1]):
                zj = zb[:, j]
                corr[j] = torch.corrcoef(torch.stack([zj, c]))[0, 1].abs()
            # distance amplification: latent distance per unit Δc, at fixed pos
            probe0 = torch.zeros(latent.world_dim); probe0[-1] = 0.0
            probe1 = torch.zeros(latent.world_dim); probe1[-1] = 1.0
            amp = float(torch.linalg.vector_norm(
                (latent.embed(probe1) - latent.embed(probe0))[sl]))
            out[f"band{k}"] = {"max_corr_with_c": float(corr.max()), "c_amplification": amp}
    return out
