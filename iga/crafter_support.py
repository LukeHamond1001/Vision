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


def collect_crafter_walk(steps: int, seed: int = 0):
    """Random-policy Crafter play. Returns (frames uint8 [T,3,64,64],
    truth dict of [T] tensors, episode_ids [T])."""
    import crafter
    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    frames, ep_ids = [], []
    truth = {k: [] for k in ("daylight", "food", "drink", "energy", "health")}
    ep = 0
    obs = env.reset()
    for t in range(steps):
        obs, r, done, info = env.step(int(rng.integers(0, env.action_space.n)))
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
    return (torch.stack(frames), {k: torch.tensor(v) for k, v in truth.items()},
            torch.tensor(ep_ids))


def measure_rho(series: torch.Tensor, ep_ids: torch.Tensor, lag: int) -> float:
    """Within-episode autocorrelation at `lag` (the tau-from-data law)."""
    T = series.shape[0] - lag
    m = ep_ids[:T] == ep_ids[lag:lag + T]
    a, b = series[:T][m], series[lag:lag + T][m]
    if a.numel() < 32 or float(a.std()) < 1e-6:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


HUD_ROWS = slice(47, 56)


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
        self.mid_head = torch.nn.Linear(9 * 64 * 3, band_dims[1])
        # slow (round 2): global photometric stats enriched with per-channel
        # luminance percentiles (p10/p50/p90). Crafter's viewport SCROLLS, so
        # mean/std alone is a terrain-composition signal (daylight corr stuck
        # at 0.54); illumination shifts all percentiles together while
        # composition changes their spread — linear contrasts can isolate the
        # common shift. Still global, still position-incapable.
        self.slow_head = torch.nn.Linear(15, band_dims[2])
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        z_fast = self.fast_head(self.trunk(x))
        strip = x[:, :, HUD_ROWS, :]
        z_mid = self.mid_head(strip.flatten(1))
        flat = x.flatten(2)
        qs = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=x.device),
                            dim=2)                       # [3, B, C]
        stats = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3)),
                           qs.permute(1, 0, 2).flatten(1)], dim=-1)
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
                             device: str = "cpu", log_every: int = 100
                             ) -> EncoderCrafter:
    """OU-ladder innovations per band (episode-masked random pairs) +
    within-band whitening. No geodesic term in phase 1 (routing only);
    no cross-band context term (routing separation is the claim under test)."""
    enc = EncoderCrafter(band_dims, seed=seed).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    slices, off = [], 0
    for bd in band_dims:
        slices.append(slice(off, off + bd))
        off += bd
    T = frames_u8.shape[0]
    max_lag = max(lags)
    gen = torch.Generator().manual_seed(seed + 77)
    valid = {}
    for lag in set(lags):
        idx = torch.arange(T - max_lag)
        keep = ep_ids[idx] == ep_ids[idx + lag]
        valid[lag] = idx[keep]
    for ep in range(epochs):
        loss = torch.tensor(0.0, device=device)
        for sl, tau, lag in zip(slices, taus, lags):
            pick = valid[lag][torch.randint(0, valid[lag].shape[0], (batch,),
                                            generator=gen)]
            z0 = enc(frames_u8[pick].float().div(255.0).to(device))
            z1 = enc(frames_u8[pick + lag].float().div(255.0).to(device))
            rho = float(torch.exp(torch.tensor(-lag / tau)))
            loss = loss + ((z1[:, sl] - rho * z0[:, sl]) ** 2).mean() \
                / max(1 - rho ** 2, 1e-3)
        stat = torch.randint(0, T, (batch,), generator=gen)
        z = enc(frames_u8[stat].float().div(255.0).to(device))
        zc = z - z.mean(0)
        cov = (zc.T @ zc) / (batch - 1)
        for sl in slices:
            bd = sl.stop - sl.start
            loss = loss + lam_cov * ((cov[sl, sl] - torch.eye(bd, device=device)) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and ep % log_every == 0:
            print(f"[v0.9:pretrain] epoch {ep} loss {float(loss):.4f}", flush=True)
    enc.freeze()
    return enc


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
