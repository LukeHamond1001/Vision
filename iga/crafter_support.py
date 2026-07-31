"""v0.9: Crafter phase 1 — three-band discovery on a recognized benchmark.

Design honors the program's own laws:
- R-taus: band timescales are MEASUREMENTS — empirical autocorrelations of
  the ground-truth variables set the priors (the v0.7 round-10 law,
  operationalized).
- Episode-boundary masking: deaths are Crafter's native coverage resets;
  temporal pairs never span them.
- Incapacity library per band: slow head reads global photometric stats
  (daylight IS global illumination — library entry one, on a real game);
  mid head reads a fixed HUD-strip region grid (rows ~47-56, where Crafter
  renders the meters — spatially pinned, the region pathway's native home);
  fast head keeps conv capacity.
Ground truth (evaluation-only, never in training): info['inventory'] meters
and env._world.daylight.
"""

from __future__ import annotations

import numpy as np
import torch


def collect_crafter_walk(steps: int, seed: int = 0,
                         phase_random: bool = False):
    """Random-policy Crafter play. Returns (frames uint8 [T,3,64,64],
    truth dict of [T] tensors, episode_ids [T]).

    phase_random: start each life at a uniform-random time of day (offset
    env._step; daylight is a pure function of it, 300-step cycle). Crafter
    always spawns at morning, so in random play EVERY truth variable is a
    monotone function of life-age — measured truth-truth correlations:
    food-drink 0.98, daylight-meters 0.85. Under that collinearity the
    routing matrix cannot attribute a band to a variable (off-diagonals are
    bounded below by diagonal x truth-corr). Randomizing the spawn phase
    breaks daylight out of the lockstep with zero policy change. Protocol
    note: phase 1 (discovery) owns its data protocol; phase 2 (behavior)
    runs standard Crafter."""
    import crafter
    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    frames, ep_ids, acts = [], [], []
    truth = {k: [] for k in ("daylight", "food", "drink", "energy", "health")}
    ep = 0
    obs = env.reset()
    if phase_random:
        env._step = int(rng.integers(0, 300))
    for t in range(steps):
        a = int(rng.integers(0, env.action_space.n))
        obs, r, done, info = env.step(a)
        acts.append(a)
        frames.append(torch.from_numpy(obs.copy()).permute(2, 0, 1))
        inv = info["inventory"]
        truth["daylight"].append(float(env._world.daylight))
        truth["food"].append(float(inv.get("food", 0)))
        truth["drink"].append(float(inv.get("drink", 0)))
        truth["energy"].append(float(inv.get("energy", 0)))
        truth["health"].append(float(inv.get("health", 0)))
        ep_ids.append(ep)
        if done:
            ep += 1
            obs = env.reset()
            if phase_random:
                env._step = int(rng.integers(0, 300))
    return (torch.stack(frames), {k: torch.tensor(v) for k, v in truth.items()},
            torch.tensor(ep_ids), torch.tensor(acts))


def measure_rho(series: torch.Tensor, ep_ids: torch.Tensor, lag: int) -> float:
    """Within-episode autocorrelation at `lag` (the tau-from-data law)."""
    T = series.shape[0] - lag
    m = ep_ids[:T] == ep_ids[lag:lag + T]
    a, b = series[:T][m], series[lag:lag + T][m]
    if a.numel() < 32 or float(a.std()) < 1e-6:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


HUD_ROWS = slice(47, 56)
MID_WINDOWS = (((2, 9), (1, 6)), ((2, 9), (8, 14)),
               ((2, 9), (16, 21)), ((3, 9), (23, 29)))   # round 10:
# per-gauge DIGIT windows (health, food, drink, energy), instrument-
# located (exclusive-change differencing). Round-10 recipe: slot heads
# are solved in CLOSED FORM — the 1-dim linear OU objective is a
# generalized eigenproblem (innovation cov vs feature cov, rank-filtered
# at 1e-3 relative variance); the head takes the smallest-innovation
# eigenvector that is NON-REDUNDANT with the slow band (|corr| < 0.5 on
# the walk) — cross-band deflation, the architecture's own anti-
# duplication rule. Deterministic: no optimizer, no seeds, no basins.
# Discovered en route: Adam never reached the objective's optima (the
# "0.79 slow asymptote" was an OPTIMIZATION artifact — closed-form slow
# reads daylight 0.945); and raw eigen-ORDER is unstable under near-ties
# (darkness vs gauge), which deflation-by-content resolves.
MID_SLOTS = ((0, 7), (7, 15), (15, 22), (22, 31))   # round 9 (superseded)
# column slots (health, food, drink, energy), located by exclusive-change
# pixel differencing (correlation cannot separate slots — vitals are too
# inter-correlated; the pixels know). One gauge, one pathway: each
# sub-head CANNOT see the other meters, so routing holds by
# construction instead of by objective preference. Round-8 lesson: a
# single 4-dim head over the whole strip let food/drink capture the
# capacity (energy 0.36).
SLOW_EMA_TAU = 10.0   # round 4: frozen exponential smoother in the slow
# pathway. Round-3 dissection: the trained slow band was the green-blue
# composition axis (|corr| 0.94-0.99 with blueness, own rho@40 only
# 0.33-0.39 — FASTER than daylight's 0.715): whitening's variance
# preference beat the innovation term. Measured cure: after EMA(tau~10),
# per-step composition flicker dies and the stat space's top-variance
# direction BECOMES daylight (PC1 corr 0.71 -> 0.83). The EMA is the OU
# prior expressed architecturally — a slow band that cannot represent
# fast content. Parameter-free, frozen, episode-masked.


def slow_stats(x: torch.Tensor) -> torch.Tensor:
    """Raw slow-pathway features: per-channel mean/std + p10/p50/p90 over
    WORLD rows only (rows 0:47 — the HUD is a dashboard, not photometry)."""
    world = x[:, :, :HUD_ROWS.start, :]
    flat = world.flatten(2)
    qs = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=x.device),
                        dim=2)
    return torch.cat([world.mean(dim=(2, 3)), world.std(dim=(2, 3)),
                      qs.permute(1, 0, 2).flatten(1)], dim=-1)


def ema_slow_stats(frames_u8: torch.Tensor, ep_ids: torch.Tensor,
                   tau: float = SLOW_EMA_TAU, batch: int = 512,
                   device: str = "cpu") -> torch.Tensor:
    """EMA(tau) of slow_stats over the walk, resetting at episode
    boundaries. Returns [T, 15] on CPU — precomputed once, indexed by
    training pairs and eval alike."""
    raw = torch.cat([slow_stats(frames_u8[i:i + batch].float().div(255.0)
                                .to(device)).cpu()
                     for i in range(0, frames_u8.shape[0], batch)])
    alpha = 1.0 / tau
    out = torch.empty_like(raw)
    prev = raw[0]
    for t in range(raw.shape[0]):
        if t and int(ep_ids[t]) != int(ep_ids[t - 1]):
            prev = raw[t]
        prev = (1 - alpha) * prev + alpha * raw[t]
        out[t] = prev
    return out


class EncoderCrafter(torch.nn.Module):
    """Three-band encoder, one incapacity pathway per band.
    band_dims = (fast, mid, slow); input [B,3,64,64] in [0,1]."""

    def __init__(self, band_dims=(4, 2, 2), seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.band_dims = tuple(band_dims)
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 8 * 8, 128), torch.nn.Tanh())
        self.fast_head = torch.nn.Linear(128, band_dims[0])
        # mid (round 2): RAW HUD-strip pixels -> linear. The 2x8 pooled grid
        # destroyed the meter NUMERALS (4x5px digits average to near-equal
        # luminance; mid-band meter corr collapsed to 0.22). Incapacity that
        # matters = spatial confinement to the HUD; linear template matching
        # over the raw strip reads digits while still seeing no world.
        assert band_dims[1] == len(MID_WINDOWS), "mid band = one dim per gauge"
        self.mid_heads = torch.nn.ModuleList(
            [torch.nn.Linear(3 * (r1 - r0) * (c1 - c0), 1)
             for (r0, r1), (c0, c1) in MID_WINDOWS])
        # slow (round 2): global photometric stats enriched with per-channel
        # luminance percentiles (p10/p50/p90). Crafter's viewport SCROLLS, so
        # mean/std alone is a terrain-composition signal (daylight corr stuck
        # at 0.54); illumination shifts all percentiles together while
        # composition changes their spread — linear contrasts can isolate the
        # common shift. Still global, still position-incapable.
        # Round 3: stats over WORLD ROWS ONLY (0:47). Full-frame stats
        # included the HUD, whose meters are the SLOWEST readable signal
        # (innovation loss 0.30 vs daylight-direction 2.26 at the trained
        # rho) — so innovation minimization dutifully read the meters
        # through the contamination (slow-food 0.53 > slow-daylight 0.44).
        # A pathway is only as incapable as its input: 'global world
        # photometry' must not include a dashboard.
        self.slow_head = torch.nn.Linear(15, band_dims[2])
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, x: torch.Tensor,
                slow_feats: torch.Tensor | None = None) -> torch.Tensor:
        """slow_feats: precomputed EMA'd slow-pathway features ([B,15] or
        [15]). When omitted, raw per-frame stats are used (stateless
        fallback) — training, eval, and online callers all pass the EMA
        so every path sees the same slow pathway (round 4)."""
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
            if slow_feats is not None and slow_feats.dim() == 1:
                slow_feats = slow_feats.unsqueeze(0)
        z_fast = self.fast_head(self.trunk(x))
        strip = x[:, :, HUD_ROWS, :]
        z_mid = torch.cat(
            [h(strip[:, :, r0:r1, c0:c1].flatten(1))
             for h, ((r0, r1), (c0, c1)) in zip(self.mid_heads, MID_WINDOWS)],
            dim=-1)
        stats = slow_stats(x) if slow_feats is None else slow_feats.to(x.device)
        z_slow = self.slow_head(stats)
        z = torch.cat([z_fast, z_mid, z_slow], dim=-1) * self.out_scale
        return z.squeeze(0) if single else z

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


def pretrain_crafter_encoder(frames_u8: torch.Tensor, ep_ids: torch.Tensor,
                             band_dims=(4, 2, 2),
                             taus=(5.0, 60.0, 150.0), lags=(5, 20, 40),
                             lam_cov: float = 1.0, epochs: int = 1200,
                             batch: int = 512, lr: float = 3e-4, seed: int = 0,
                             device: str = "cpu", log_every: int = 100,
                             actions: torch.Tensor | None = None,
                             lam_fwd: float = 1.0, n_actions: int = 17
                             ) -> EncoderCrafter:
    """OU-ladder innovations per band (episode-masked random pairs) +
    within-band whitening. No geodesic term in phase 1 (routing only);
    no cross-band context term (routing separation is the claim under test).
    Slow-pathway features are EMA'd once up front (round 4).

    Round 7: ACTION-CONDITIONED one-step forward model on the fast band
    (the component deferred since v0.5 — C4 flinch's frozen model). The
    trunk autopsy showed the smoothness objective learns zero object
    features anywhere (trunk probes 0.11-0.18); predicting z_fast(t+1)
    from (z_fast(t), a_t) is the architecture's own forcing function for
    action-relevant perception (approach geometry, facing, obstacles).
    The predictor is auxiliary (kept for the future flinch); action
    sensitivity is INSTRUMENTED — the term must beat shuffled actions or
    it has degenerated to smoothness."""
    enc = EncoderCrafter(band_dims, seed=seed).to(device)
    inv = None
    params = list(enc.parameters())
    if actions is not None and lam_fwd > 0:
        torch.manual_seed(seed + 300)
        inv = torch.nn.Sequential(
            torch.nn.Linear(2 * band_dims[0], 64), torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions)).to(device)
        params += list(inv.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    slow_feats = ema_slow_stats(frames_u8, ep_ids, device=device)
    slices, off = [], 0
    for bd in band_dims:
        slices.append(slice(off, off + bd))
        off += bd
    T = frames_u8.shape[0]
    max_lag = max(lags)
    gen = torch.Generator().manual_seed(seed + 77)
    valid = {}
    for lag in set(lags) | {1}:
        idx = torch.arange(T - max(max_lag, 1))
        keep = ep_ids[idx] == ep_ids[idx + lag]
        valid[lag] = idx[keep]
    for ep in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for sl, tau, lag in zip(slices, taus, lags):
            pick = valid[lag][torch.randint(0, valid[lag].shape[0], (batch,),
                                            generator=gen)]
            z0 = enc(frames_u8[pick].float().div(255.0).to(device),
                     slow_feats=slow_feats[pick])
            z1 = enc(frames_u8[pick + lag].float().div(255.0).to(device),
                     slow_feats=slow_feats[pick + lag])
            rho = float(torch.exp(torch.tensor(-lag / tau)))
            loss = loss + ((z1[:, sl] - rho * z0[:, sl]) ** 2).mean() \
                / max(1 - rho ** 2, 1e-3)
        if inv is not None:
            p1 = valid[1][torch.randint(0, valid[1].shape[0], (batch,),
                                        generator=gen)]
            zf0 = enc(frames_u8[p1].float().div(255.0).to(device),
                      slow_feats=slow_feats[p1])[:, :band_dims[0]]
            zf1 = enc(frames_u8[p1 + 1].float().div(255.0).to(device),
                      slow_feats=slow_feats[p1 + 1])[:, :band_dims[0]]
            logits = inv(torch.cat([zf0, zf1], dim=1))
            a_true = actions[p1].to(device)
            loss = loss + lam_fwd * torch.nn.functional.cross_entropy(logits,
                                                                      a_true)
        stat = torch.randint(0, T, (batch,), generator=gen)
        z = enc(frames_u8[stat].float().div(255.0).to(device),
                slow_feats=slow_feats[stat])
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (batch - 1)
        for sl in slices:
            bd = sl.stop - sl.start
            loss = loss + lam_cov * ((cov[sl, sl] - torch.eye(bd, device=device)) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and ep % log_every == 0:
            sens = ""
            if inv is not None:
                with torch.no_grad():
                    acc = float((logits.argmax(1) == a_true).float().mean())
                sens = f" inv-acc {acc:.3f} (chance {1/n_actions:.3f})"
            print(f"[v0.9:pretrain] epoch {ep} loss {float(loss):.4f}{sens}",
                  flush=True)
    # Round 10: slow and mid heads are solved in CLOSED FORM (the 1-dim
    # linear OU objective is a generalized eigenproblem; Adam provably
    # never reached its optima — the 0.79 slow "asymptote" was an
    # optimization artifact; closed form reads daylight 0.945 locally).
    # Slow: first two eigvecs of the EMA-stat pathway. Mid: per-gauge
    # digit windows, each deflated against the slow output (cross-band
    # non-redundancy). Deterministic — no restarts, no seeds.
    sl_lag, sl_tau = lags[-1], taus[-1]
    F = slow_feats.cpu()
    w0, b0 = closed_form_head(F, ep_ids, sl_lag, sl_tau)
    F0 = F - F.mean(0)
    z0 = (F0 @ w0).unsqueeze(1)
    w1, b1 = closed_form_head(F, ep_ids, sl_lag, sl_tau, deflate=z0)
    with torch.no_grad():
        enc.slow_head.weight.copy_(torch.stack([w0, w1]).to(device))
        enc.slow_head.bias.copy_(torch.tensor([b0, b1], device=device))
    z_slow_walk = torch.stack([F0 @ w0, F0 @ w1], 1)
    md_lag, md_tau = lags[1], taus[1]
    strip_w = frames_u8[:, :, HUD_ROWS, :].float().div(255.0)
    for h, ((r0, r1), (c0, c1)) in zip(enc.mid_heads, MID_WINDOWS):
        Xw = strip_w[:, :, r0:r1, c0:c1].flatten(1)
        wm, bm = closed_form_head(Xw, ep_ids, md_lag, md_tau,
                                  deflate=z_slow_walk)
        with torch.no_grad():
            h.weight.copy_(wm.unsqueeze(0).to(device))
            h.bias.copy_(torch.tensor([bm], device=device))
    print("[v0.9:pretrain] slow+mid heads <- closed form (deterministic)",
          flush=True)
    enc.freeze()
    return enc


def closed_form_head(X: torch.Tensor, ep_ids: torch.Tensor, lag: int,
                     tau: float, deflate: torch.Tensor | None = None,
                     var_floor: float = 1e-3):
    """Exact optimum of the 1-dim linear OU head: generalized eigvecs of
    (innovation cov, feature cov), rank-filtered; picks the smallest-
    innovation direction with |corr| < 0.5 against every column of
    `deflate` (cross-band non-redundancy). Returns (weight, bias) with
    unit output variance on the walk."""
    mu = X.mean(0)
    Xc = X - mu
    rho = float(torch.exp(torch.tensor(-lag / tau)))
    idx = torch.arange(X.shape[0] - lag)
    v = idx[ep_ids[idx] == ep_ids[idx + lag]]
    d = Xc[v + lag] - rho * Xc[v]
    A = (d.T @ d) / d.shape[0]
    B = (Xc.T @ Xc) / Xc.shape[0]
    eb, Vb = torch.linalg.eigh(B)
    keep = eb > var_floor * eb.max()
    P = Vb[:, keep]
    Bih = torch.diag(eb[keep].rsqrt())
    _, Va = torch.linalg.eigh(Bih @ (P.T @ A @ P) @ Bih)
    for k in range(Va.shape[1]):
        w = P @ (Bih @ Va[:, k])
        z = Xc @ w
        if deflate is not None and deflate.numel():
            red = max(abs(float(torch.corrcoef(
                torch.stack([z, deflate[:, j]]))[0, 1]))
                for j in range(deflate.shape[1]))
            if red >= 0.5:
                continue
        sd = z.std().clamp_min(1e-8)
        w = w / sd
        return w, float(-(mu @ w))
    w = P @ (Bih @ Va[:, 0])
    sd = (Xc @ w).std().clamp_min(1e-8)
    w = w / sd
    return w, float(-(mu @ w))


def partial_corr(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    """corr(a, b | c): residualize both against c (linear), correlate.
    The collinearity-robust attribution instrument."""
    c1 = torch.stack([c, torch.ones_like(c)], 1)
    ra = a - c1 @ torch.linalg.lstsq(c1, a.unsqueeze(1)).solution.squeeze(1)
    rb = b - c1 @ torch.linalg.lstsq(c1, b.unsqueeze(1)).solution.squeeze(1)
    if float(ra.std()) < 1e-6 or float(rb.std()) < 1e-6:
        return float("nan")
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def routing_matrix(z: torch.Tensor, truth: dict, band_dims=(4, 2, 2)) -> dict:
    """Per-band max |corr| with each ground-truth variable."""
    out, off = {}, 0
    names = ["fast", "mid", "slow"]
    for k, bd in enumerate(band_dims):
        for var, series in truth.items():
            if float(series.std()) < 1e-6:
                out[f"{names[k]}_{var}"] = float("nan")
                continue
            corr = torch.zeros(bd)
            for j in range(bd):
                corr[j] = torch.corrcoef(torch.stack([z[:, off + j],
                                                      series]))[0, 1].abs()
            out[f"{names[k]}_{var}"] = float(corr.max())
        off += bd
    return out
