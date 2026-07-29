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
                       segment_len: int | None = None,
                       context_amp: float | None = None, lam_ctx: float = 5.0,
                       context_mode: str = "column", lam_iso: float = 0.0,
                       geo: tuple | None = None, lam_geo: float = 0.0,
                       seed: int = 0) -> torch.Tensor:
    """Learn a linear map W: world → latent under the OU-ladder objective.
    Returns W (latent_dim × world_dim), to be frozen by the caller.

    `segment_len`: coverage-reset period of the collection. Innovation pairs
    that span a reset are MASKED OUT (episode-boundary masking) — teleports
    are coverage devices, not dynamics, and scoring across them poisons the
    slow prior (measured: charge's effective lag-15 autocorrelation fell
    below the slow band's target and routing inverted).

    `context_amp` (v0.4, from the round-2 leak ablation): if set, the
    objective DELIBERATELY couples the slowest world variable (last input
    column) into the FAST band at this world-unit amplitude, and covariance
    whitening applies WITHIN bands only — full whitening penalizes exactly
    the cross-band coupling that the ablation proved necessary and
    sufficient for the representation win. Cross-band value coupling belongs
    in the metric (SPEC L1, v0.3 revision)."""
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
    for lag in set(lags) | ({1} if lam_iso > 0 else set()):
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
        if lam_iso > 0:
            # v0.6: homogeneous step-scale regularization. Learned encoders
            # warp long-range geometry LOCALLY (measured: far/step 2.8 vs
            # rigid 7.6 — v0.5 phase 2's routing-without-competence result);
            # penalize the squared coefficient of variation of per-step
            # displacement norms — scale-invariant, so it shapes UNIFORMITY
            # without fighting whitening's scale.
            m1 = masks[1]
            d = torch.linalg.vector_norm(z[1:T - max_lag + 1][m1] - z[:T - max_lag][m1],
                                         dim=-1)
            loss = loss + lam_iso * (d.var() / d.mean().clamp_min(1e-6) ** 2)
        if lam_geo > 0 and geo is not None:
            # v0.6: geodesic matching — normalized latent chords toward
            # normalized graph geodesics (scale-free on both sides).
            node_idx, gi, gj, dgeo = geo
            zn = z[node_idx]
            chord = torch.linalg.vector_norm(zn[gi] - zn[gj], dim=-1)
            loss = loss + lam_geo * ((chord / chord.mean().clamp_min(1e-6)
                                      - dgeo / dgeo.mean()) ** 2).mean()
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (T - 1)
        if context_amp is None:
            loss = loss + lam_cov * ((cov - torch.eye(latent_dim)) ** 2).sum()
        else:
            for sl in slices:                                # within-band whitening only
                bd = sl.stop - sl.start
                loss = loss + lam_cov * ((cov[sl, sl] - torch.eye(bd)) ** 2).sum()
            if context_mode == "column":
                # v0.4: couple to a DESIGNATED slow input column (last) —
                # usable only when the slow variable is a known input. Ratio
                # form is invariant to post-training scale normalization.
                amp_c = torch.linalg.vector_norm(W[slices[0], -1]) / sd[-1]
                amp_pos = torch.linalg.matrix_norm(W[slices[0], :-1] / sd[:-1], ord=2)
                loss = loss + lam_ctx * (amp_c / amp_pos.clamp_min(1e-6) - context_amp) ** 2
            else:
                # v0.5 'band': couple the fast band to the LEARNED slow band —
                # definable with no knowledge of which input carries the slow
                # variable (the scale-up setting). Normalized cross-band
                # coupling driven to the prescribed amplitude.
                zf, zs = zc[:, slices[0]], zc[:, slices[1]]
                cross = (zf.T @ zs) / (T - 1)
                fro = torch.linalg.matrix_norm(cross) / (cross.shape[0] * cross.shape[1]) ** 0.5
                loss = loss + lam_ctx * (fro - context_amp) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    # fold input standardization into the returned map: z = W (x−μ)/σ = W' x + b;
    # drop the constant offset (translations don't affect any distance the
    # architecture uses) and return the linear part.
    with torch.no_grad():
        W_final = (W / sd.unsqueeze(0)).detach().clone()
    return W_final


def geodesic_pairs(obs: torch.Tensor, n_nodes: int = 400, k: int = 8,
                   n_pairs: int = 4000, seed: int = 0):
    """v0.6: geodesic distance targets from walk observations alone.

    Subsample nodes, build a kNN graph with observation-space edge lengths
    (local o-distances are trustworthy where sensors are locally faithful),
    run vectorized Floyd–Warshall, and sample finite-distance pairs. Returns
    (node_indices_into_obs, pair_i, pair_j, d_geo). Used by the geodesic-
    matching objective term: latent CHORDS should match graph GEODESICS —
    the curvature fix that step-norm homogenization (refuted) could not be."""
    gen = torch.Generator().manual_seed(seed + 5000)
    node_idx = torch.randperm(obs.shape[0], generator=gen)[:n_nodes]
    X = obs[node_idx]
    D = torch.cdist(X, X)
    knn = D.topk(k + 1, largest=False).indices[:, 1:]
    A = torch.full_like(D, float("inf"))
    A[torch.arange(n_nodes)] = float("inf")
    rows = torch.arange(n_nodes).unsqueeze(1).expand_as(knn)
    A[rows.reshape(-1), knn.reshape(-1)] = D[rows.reshape(-1), knn.reshape(-1)]
    A = torch.minimum(A, A.T)
    A.fill_diagonal_(0.0)
    for m in range(n_nodes):                       # min-plus Floyd–Warshall
        A = torch.minimum(A, A[:, m:m + 1] + A[m:m + 1, :])
    ii = torch.randint(0, n_nodes, (n_pairs,), generator=gen)
    jj = torch.randint(0, n_nodes, (n_pairs,), generator=gen)
    d = A[ii, jj]
    ok = torch.isfinite(d) & (ii != jj)
    return node_idx, ii[ok], jj[ok], d[ok]


class EncoderMLP(torch.nn.Module):
    """Small nonlinear encoder, FROZEN after pretraining (v0.6). Frozen ≠
    linear: W2's frozen metric and every frozen-latent constraint survive;
    what linearity alone provided (exact embed_delta, chord-structure
    inheritance from observation space) is either deferred or — for the
    curvature ceiling — exactly what this class exists to escape."""

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, latent_dim))
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sd) * self.out_scale

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


def pretrain_mlp_encoder(world_traj: torch.Tensor, band_dims: list[int],
                         taus: list[float], lags: list[int],
                         segment_len: int | None = None,
                         context_amp: float | None = 0.3, lam_ctx: float = 5.0,
                         lam_cov: float = 1.0, geo: tuple | None = None,
                         lam_geo: float = 5.0, epochs: int = 800,
                         lr: float = 1e-3, seed: int = 0) -> EncoderMLP:
    """v0.6: the full recipe over a nonlinear encoder — OU-ladder innovations
    (boundary-masked), within-band whitening, band-mode context coupling, and
    geodesic matching (feasible only now: linear maps cannot send equal
    observation chords to unequal latent chords). Returns the FROZEN encoder."""
    enc = EncoderMLP(world_traj.shape[1], sum(band_dims), seed=seed)
    enc.mu.copy_(world_traj.mean(0))
    enc.sd.copy_(world_traj.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(enc.net.parameters(), lr=lr)
    slices, off = [], 0
    for bd in band_dims:
        slices.append(slice(off, off + bd))
        off += bd
    T = world_traj.shape[0]
    max_lag = max(lags)
    idx = torch.arange(T - max_lag)
    masks = {}
    for lag in set(lags):
        masks[lag] = (torch.ones(T - max_lag, dtype=torch.bool) if segment_len is None
                      else (idx // segment_len) == ((idx + lag) // segment_len))
    for _ in range(epochs):
        z = enc(world_traj)
        loss = torch.tensor(0.0)
        for sl, tau, lag in zip(slices, taus, lags):
            rho = float(torch.exp(torch.tensor(-lag / tau)))
            m = masks[lag]
            z0, z1 = z[:T - max_lag][m][:, sl], z[lag:T - max_lag + lag][m][:, sl]
            loss = loss + ((z1 - rho * z0) ** 2).mean() / max(1 - rho ** 2, 1e-3)
        if geo is not None and lam_geo > 0:
            node_idx, gi, gj, dgeo = geo
            zn = z[node_idx]
            chord = torch.linalg.vector_norm(zn[gi] - zn[gj], dim=-1)
            loss = loss + lam_geo * ((chord / chord.mean().clamp_min(1e-6)
                                      - dgeo / dgeo.mean()) ** 2).mean()
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (T - 1)
        for sl in slices:
            bd = sl.stop - sl.start
            loss = loss + lam_cov * ((cov[sl, sl] - torch.eye(bd)) ** 2).sum()
        if context_amp is not None:
            zf, zs = zc[:, slices[0]], zc[:, slices[1]]
            cross = (zf.T @ zs) / (T - 1)
            fro = torch.linalg.matrix_norm(cross) / (cross.shape[0] * cross.shape[1]) ** 0.5
            loss = loss + lam_ctx * (fro - context_amp) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.freeze()
    return enc


class EncoderCNN(torch.nn.Module):
    """Conv encoder for pixel observations (v0.7). Frozen after pretraining;
    same R1 contract as EncoderMLP.

    Band-structured heads (round 5 — the identifiability fix): the slow band
    reads ONLY global-average-pooled channel statistics (translation-
    invariant global features — where slow variables like illumination live,
    and where coarse position largely cancels); the fast band keeps the
    spatial flatten. Measured motivation: with an unstructured head, full
    training satisfies the temporal objective with arbitrary timescale-
    matched nonlinear mixtures uncorrelated with EVERY generator (probe
    table, run r4) — encoder capacity was an implicit identifiability prior
    at sensor scale, and pixels revoke it. Architecture must re-state the
    factorization the objective alone cannot identify."""

    def __init__(self, band_dims=(5, 3), size: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 4, stride=2, padding=1), torch.nn.ReLU(),   # 32
            torch.nn.Conv2d(16, 32, 4, stride=2, padding=1), torch.nn.ReLU(),  # 16
            torch.nn.Conv2d(32, 32, 4, stride=2, padding=1), torch.nn.ReLU())  # 8
        feat_spatial = 32 * (size // 8) ** 2
        self.fast_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feat_spatial, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, band_dims[0]))
        self.slow_head = torch.nn.Linear(32, band_dims[1])      # GAP features only
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        h = self.trunk(x)
        z_fast = self.fast_head(h)
        z_slow = self.slow_head(h.mean(dim=(2, 3)))             # global average pool
        z = torch.cat([z_fast, z_slow], dim=-1) * self.out_scale
        return z.squeeze(0) if single else z

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


def pretrain_cnn_encoder(frames: torch.Tensor, band_dims: list[int],
                         taus: list[float], lags: list[int],
                         segment_len: int | None = None,
                         context_amp: float | None = 0.3, lam_ctx: float = 5.0,
                         lam_cov: float = 1.0, geo: tuple | None = None,
                         lam_geo: float = 5.0, epochs: int = 300,
                         batch: int = 1024, lr: float = 3e-4, seed: int = 0,
                         device: str = "cpu", log_every: int = 50) -> EncoderCNN:
    """The R2 recipe over a conv encoder on rendered frames. Minibatched over
    contiguous windows so the temporal terms stay valid; geodesic pairs are
    computed by the CALLER (kNN on downsampled frames) and passed in."""
    enc = EncoderCNN(band_dims=tuple(band_dims), size=frames.shape[-1], seed=seed).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    slices, off = [], 0
    for bd in band_dims:
        slices.append(slice(off, off + bd))
        off += bd
    T = frames.shape[0]
    max_lag = max(lags)
    gen = torch.Generator().manual_seed(seed + 77)
    node_idx = gi = gj = dgeo = None
    if geo is not None:
        node_idx, gi, gj, dgeo = geo
        node_frames = frames[node_idx].to(device)
        dgeo = dgeo.to(device)
    # Batch composition IS objective design at scale (v0.7 freebies-law
    # entry): full-batch statistics were a small-data freebie. Contiguous
    # windows make within-batch samples near-duplicates (effective n ~ the
    # handful of coverage segments spanned) and the distributional terms
    # train on noise — measured on GPU as routing collapse (0.274 vs 0.9+).
    # Temporal terms draw RANDOM within-segment (t, t+lag) pairs across the
    # whole walk; distributional terms draw an INDEPENDENT uniform subset.
    valid = {}
    for lag in set(lags):
        idx = torch.arange(T - max_lag)
        if segment_len is not None:
            keep = (idx // segment_len) == ((idx + lag) // segment_len)
            valid[lag] = idx[keep]
        else:
            valid[lag] = idx
    for ep in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for sl, tau, lag in zip(slices, taus, lags):
            rho = float(torch.exp(torch.tensor(-lag / tau)))
            pick = valid[lag][torch.randint(0, valid[lag].shape[0], (batch,), generator=gen)]
            z0 = enc(frames[pick].to(device))
            z1 = enc(frames[pick + lag].to(device))
            innov = ((z1[:, sl] - rho * z0[:, sl]) ** 2).mean() / max(1 - rho ** 2, 1e-3)
            loss = loss + innov
        stat = torch.randint(0, T, (batch,), generator=gen)
        z0_all = enc(frames[stat].to(device))
        zc = z0_all - z0_all.mean(0)
        cov = (zc.T @ zc) / (batch - 1)
        for sl in slices:
            bd = sl.stop - sl.start
            loss = loss + lam_cov * ((cov[sl, sl] - torch.eye(bd, device=device)) ** 2).sum()
        if context_amp is not None:
            zf, zs = zc[:, slices[0]], zc[:, slices[1]]
            cross = (zf.T @ zs) / (batch - 1)
            fro = torch.linalg.matrix_norm(cross) / (cross.shape[0] * cross.shape[1]) ** 0.5
            loss = loss + lam_ctx * (fro - context_amp) ** 2
        if geo is not None:
            zn = enc(node_frames)
            chord = torch.linalg.vector_norm(zn[gi] - zn[gj], dim=-1)
            loss = loss + lam_geo * ((chord / chord.mean().clamp_min(1e-6)
                                      - dgeo / dgeo.mean()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and ep % log_every == 0:
            print(f"[pretrain-cnn] epoch {ep} loss {float(loss):.4f}", flush=True)
    enc.to("cpu")
    enc.freeze()
    return enc


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
