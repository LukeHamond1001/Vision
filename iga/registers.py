"""Goal register (C2): one persistent slot, immutable while its window is open."""

from __future__ import annotations

import torch


class RegisterHeldError(RuntimeError):
    """Raised on any attempt to re-choose the target mid-window (C2)."""


class GoalRegister:
    def __init__(self):
        self._g: torch.Tensor | None = None
        self._remaining = 0

    @property
    def open(self) -> bool:
        return self._remaining > 0

    @property
    def target(self) -> torch.Tensor:
        assert self._g is not None, "no committed target"
        return self._g

    @property
    def remaining(self) -> int:
        return self._remaining

    def commit(self, g: torch.Tensor, window: int) -> None:
        if self.open:
            raise RegisterHeldError("C2: goal register may not be re-chosen mid-window")
        # detach+clone: the committed target is a CONSTANT. Progress computed
        # against it can never carry gradient back to the proposer (G5).
        self._g = g.detach().clone()
        self._remaining = int(window)

    def tick(self) -> None:
        if self._remaining > 0:
            self._remaining -= 1

    def close(self) -> None:
        """Close the window (arrival or timeout). Only then may a new commit occur."""
        self._remaining = 0
