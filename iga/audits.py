"""Offline audits A1–A3 (SPEC §8). Pure computation over fixed functions —
no environment interaction, no training runs. Runnable as:

    python -m iga.audits
"""

from __future__ import annotations

import torch

from .heads import FixedRewardHead
from .latent import PremappedLatent


def audit_trigger_set(head_neg: FixedRewardHead, support: torch.Tensor, margin: float,
                      threshold: float, n: int = 2048, seed: int = 0) -> dict:
    """A1: where does f− fire over support ⊕ margin? Returns the fraction of
    probed points above threshold and the firing points themselves."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, support.shape[0], (n,), generator=gen)
    probes = support[idx] + torch.randn(n, support.shape[1], generator=gen) * margin
    vals = head_neg._f(probes)
    firing = probes[vals > threshold]
    return {"fraction_firing": float((vals > threshold).float().mean()), "firing_points": firing}


def audit_proxy_gap(head: FixedRewardHead, candidates: torch.Tensor) -> dict:
    """A2: |w·g − f(g)| over candidate targets — the linear-proxy error that
    prospective vetoes and value bars act on (SPEC §C4, §C7)."""
    with torch.no_grad():
        lin = head.claim(candidates)
        true = head._f(candidates)
        gap = (lin - true).abs()
    return {
        "mean_gap": float(gap.mean()),
        "max_gap": float(gap.max()),
        "false_block_rate_at_0.5": float(((lin > 0.5) & (true <= 0.5)).float().mean()),
        "false_pass_rate_at_0.5": float(((lin <= 0.5) & (true > 0.5)).float().mean()),
    }


def calibrate_threshold(head: FixedRewardHead, support: torch.Tensor,
                        danger_level: float = 0.5) -> float:
    """A2 corollary: choose the veto/bar operating point ON THE PROXY SCALE.

    The linear proxy w·g and the true evaluator f(g) live on different scales
    and are imperfectly correlated (the audited gap); an absolute threshold on
    the proxy is therefore meaningless until calibrated. Returns the median
    proxy value over support states whose TRUE evaluation exceeds
    `danger_level` — the audited operating point, computed offline from fixed
    functions only."""
    with torch.no_grad():
        f = head._f(support)
        c = head.claim(support)
        dangerous = c[f > danger_level]
        if dangerous.numel() == 0:
            return float(c.max()) + 1.0          # nothing dangerous on-manifold
        return float(dangerous.median())


def audit_leash_lipschitz(head: FixedRewardHead, support: torch.Tensor, radius: float,
                          n_pairs: int = 4096, seed: int = 0) -> dict:
    """A3: finite-difference Lipschitz estimate of f± within the leash margin.
    Bounds worst-case reward error at the leash boundary (SPEC §C3 residual):
    |f(x) − f(anchor)| ≤ L · radius for x within the margin."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, support.shape[0], (n_pairs,), generator=gen)
    a = support[idx]
    b = a + torch.randn(n_pairs, support.shape[1], generator=gen) * radius
    with torch.no_grad():
        num = (head._f(a) - head._f(b)).abs()
        den = torch.linalg.vector_norm(a - b, dim=-1).clamp_min(1e-9)
        lip = float((num / den).max())
    return {"lipschitz_est": lip, "max_boundary_error_bound": lip * radius}


def _demo() -> None:
    from .envs.gridworld import ContinuousGrid

    latent = PremappedLatent(world_dim=2, latent_dim=8)
    env = ContinuousGrid(latent)
    gen = torch.Generator().manual_seed(1)
    world_samples = torch.rand(512, 2, generator=gen)
    support = torch.stack([latent.embed(w) for w in world_samples])
    head_pos = FixedRewardHead(env.embed_site(env.reward_site), sigma=0.4, proxy_samples=support)
    head_neg = FixedRewardHead(env.embed_site(env.hazard_site), sigma=0.4, proxy_samples=support)

    a1 = audit_trigger_set(head_neg, support, margin=0.1, threshold=0.5)
    a2 = audit_proxy_gap(head_neg, support)
    a3 = audit_leash_lipschitz(head_pos, support, radius=0.1)
    print("A1 trigger set:", {k: v for k, v in a1.items() if k != "firing_points"},
          f"({a1['firing_points'].shape[0]} firing points)")
    print("A2 proxy gap (negative head):", a2)
    print("A3 leash Lipschitz (positive head):", a3)


if __name__ == "__main__":
    _demo()
