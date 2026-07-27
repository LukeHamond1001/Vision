"""Fixed reward heads (W1, W4).

R±(p, i) = f±(p) + w±·i

- f±: fixed nonlinear evaluator over perception channels. Here: a designed
  radial evaluator around a pre-mapped site (the "pre-mapped" story taken
  literally); in a richer system, any frozen network works, provided it is
  frozen *and* reads pre-mapped channels, not trunk features.
- w±: fixed linear proxy over imagination channels — fit once at construction
  (least squares of f± over on-manifold samples), then frozen. This literally
  instantiates SPEC §C4's caveat: prospective judgments on imagined targets are
  linear-proxy judgments; audit A2 measures the gap.

No tensor in this module ever requires grad. `assert_parameter_free` is called
by the Agent at construction (W1) and by the test suite.
"""

from __future__ import annotations

import torch


class FixedRewardHead:
    def __init__(self, site: torch.Tensor, sigma: float, proxy_samples: torch.Tensor):
        self.site = site.detach().clone()
        self.site.requires_grad_(False)
        self.sigma = float(sigma)
        # Fit w once: least-squares linear proxy of f over on-manifold samples.
        with torch.no_grad():
            y = self._f(proxy_samples).reshape(-1, 1)
            sol = torch.linalg.lstsq(proxy_samples, y).solution
        self.w = sol.reshape(-1).detach().clone()
        self.w.requires_grad_(False)

    def _f(self, p: torch.Tensor) -> torch.Tensor:
        d2 = ((p - self.site) ** 2).sum(dim=-1)
        return torch.exp(-d2 / (2.0 * self.sigma**2))

    def realized(self, p: torch.Tensor) -> torch.Tensor:
        """f±(p): the evaluator on reality. The only value learning may credit."""
        return self._f(p.reshape(-1, p.shape[-1])).squeeze(0)

    def claim(self, i: torch.Tensor) -> torch.Tensor:
        """w±·i: the quote (G1). Ranks targets; never enters learning."""
        return i @ self.w

    def forward(self, p: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Full head output. Provided so the G1 identity is testable:
        forward(p, i) − claim(i) == realized(p), exactly, per sample."""
        return self.realized(p) + self.claim(i)

    def frozen_tensors(self) -> list[torch.Tensor]:
        return [self.site, self.w]

    def assert_parameter_free(self) -> None:
        for t in self.frozen_tensors():
            assert not t.requires_grad, "W1 violated: reward-head tensor requires grad"
