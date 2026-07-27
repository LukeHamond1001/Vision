"""Hard constraints: leash (C3), coverage cap (C1), curiosity (C6).

All three are keyed to the frozen latent (no learned parameters anywhere in
this module), which is what makes them ungameable by the proposer.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .latent import PremappedLatent


class Leash:
    """C3: hard projection of emitted targets onto the empirical support ball.

    Non-learned enforcer: fixed metric + fixed radius + empirical anchor set of
    actually-perceived states. MUST NOT be replaced by a penalty or a learned
    density model (SPEC §C3).
    """

    def __init__(self, latent: PremappedLatent, radius: float, max_support: int = 4096):
        self.latent = latent
        self.radius = float(radius)
        self._support: list[torch.Tensor] = []
        self.max_support = max_support

    def anchor(self, p: torch.Tensor) -> None:
        """Record an actually-perceived state into the support set."""
        with torch.no_grad():
            self._support.append(p.detach().clone())
            if len(self._support) > self.max_support:
                self._support.pop(0)

    def support(self) -> torch.Tensor:
        assert self._support, "leash has no anchors yet"
        return torch.stack(self._support)

    def project(self, g: torch.Tensor) -> torch.Tensor:
        """Project g to within `radius` of the nearest support point."""
        with torch.no_grad():
            s = self.support()
            d = self.latent.distance(s, g.unsqueeze(0).expand_as(s))
            j = int(torch.argmin(d))
            nearest, dist = s[j], float(d[j])
            if dist <= self.radius:
                return g.detach().clone()
            direction = (g.detach() - nearest) / dist
            return nearest + direction * self.radius


class CoverageCap:
    """C1: at most `max_updates` learning updates per frozen-latent
    neighborhood per round. Keyed by neighborhood, NOT state identity —
    identity keys are defeated by ε-ball near-duplicates (probe E3a).

    `mode` exists for the ablation battery only: 'identity' is the defeatable
    variant, 'off' disables the cap. Deployed configuration is 'neighborhood'.
    """

    def __init__(self, latent: PremappedLatent, radius: float, max_updates: int = 2,
                 mode: str = "neighborhood"):
        assert mode in ("neighborhood", "identity", "off")
        self.latent = latent
        self.radius = float(radius)
        self.max_updates = int(max_updates)
        self.mode = mode
        self._counts: dict[tuple, int] = defaultdict(int)

    def key(self, z: torch.Tensor) -> tuple:
        if self.mode == "identity":
            return tuple(z.detach().tolist())
        return self.latent.neighborhood_key(z, self.radius)

    def allow(self, z: torch.Tensor) -> bool:
        if self.mode == "off":
            return True
        return self._counts[self.key(z)] < self.max_updates

    def record(self, z: torch.Tensor) -> None:
        if self.mode != "off":
            self._counts[self.key(z)] += 1

    def reset_round(self) -> None:
        self._counts.clear()


class Curiosity:
    """C6: one-shot novelty, neighborhood-keyed, extinguished only by an
    ACTUAL visit — never by proposal or veto. Permanent within a run.

    `mode` exists for the ablation battery: 'never_dies' pays the bonus on
    every visit (the farming-prone variant), 'off' disables it. Deployed
    configuration is 'oneshot'.
    """

    def __init__(self, latent: PremappedLatent, radius: float, bonus: float = 0.1,
                 mode: str = "oneshot"):
        assert mode in ("oneshot", "never_dies", "off")
        self.latent = latent
        self.radius = float(radius)
        self.bonus_value = float(bonus)
        self.mode = mode
        self._extinguished: set[tuple] = set()

    def bonus(self, z: torch.Tensor) -> float:
        if self.mode == "off":
            return 0.0
        if self.mode == "never_dies":
            return self.bonus_value
        key = self.latent.neighborhood_key(z, self.radius)
        return 0.0 if key in self._extinguished else self.bonus_value

    def visit(self, z: torch.Tensor) -> None:
        self._extinguished.add(self.latent.neighborhood_key(z, self.radius))
