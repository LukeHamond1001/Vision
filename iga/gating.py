"""Learning-signal gating (SPEC §5): the three signals, the exact subtraction,
the IOU ledger, and the proposer's two legitimate currencies.

The invariant this module exists to uphold (§6.3): there is no path from the
imagination head into the learning signal that bypasses the world. Claims rank;
progress pays the policy for real traversal; realized value pays on arrival.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .heads import FixedRewardHead
from .latent import PremappedLatent


def realized_from_total(total: torch.Tensor, claim: torch.Tensor) -> torch.Tensor:
    """G1: the subtraction as an algebraic identity. Under W4 this equals
    f±(p) exactly, per sample, with no backward pass and no estimator."""
    return total - claim


@dataclass
class IOU:
    """A positive claim logged at commit time (G3). Settled on arrival, then
    discarded — its only afterlife is the calibration error it returns."""

    target: torch.Tensor          # committed target (constant, detached)
    propose_features: torch.Tensor  # trunk features at proposal time (detached)
    claim_pos: float
    claim_neg: float


@dataclass
class Signals:
    realized_pos: float
    realized_neg: float
    progress: float
    curiosity: float

    def total(self, w_prog: float = 1.0) -> float:
        """The learning signal. Note what is absent: any claim term (G1)."""
        return (self.realized_pos - self.realized_neg) + w_prog * self.progress + self.curiosity


class LearningGate:
    def __init__(self, latent: PremappedLatent, head_pos: FixedRewardHead, head_neg: FixedRewardHead):
        self.latent = latent
        self.head_pos = head_pos
        self.head_neg = head_neg
        self._ledger: list[IOU] = []

    # --- signal assembly (per step) -------------------------------------
    def progress(self, p_t: torch.Tensor, p_t1: torch.Tensor, g: torch.Tensor) -> float:
        """Gap-closing progress in the frozen metric (W2). Both state operands
        are perception-side; g is a detached constant (G5)."""
        with torch.no_grad():
            return float(self.latent.distance(p_t, g) - self.latent.distance(p_t1, g))

    def signals(self, p_t: torch.Tensor, p_t1: torch.Tensor, g: torch.Tensor | None,
                curiosity: float) -> Signals:
        """g=None means no committed target (goal-less wandering): the progress
        term is zero and the signal is realized outcome + curiosity only."""
        with torch.no_grad():
            return Signals(
                realized_pos=float(self.head_pos.realized(p_t1)),
                realized_neg=float(self.head_neg.realized(p_t1)),
                progress=0.0 if g is None else self.progress(p_t, p_t1, g),
                curiosity=float(curiosity),
            )

    # --- the IOU ledger (G3) --------------------------------------------
    def log_iou(self, target: torch.Tensor, propose_features: torch.Tensor) -> None:
        with torch.no_grad():
            self._ledger.append(
                IOU(
                    target=target.detach().clone(),
                    propose_features=propose_features.detach().clone(),
                    claim_pos=float(self.head_pos.claim(target.detach())),
                    claim_neg=float(self.head_neg.claim(target.detach())),
                )
            )

    @property
    def open_ious(self) -> int:
        return len(self._ledger)

    def settle(self, p_arrival: torch.Tensor) -> tuple[IOU, float, float]:
        """Arrival reconciliation (G3): realized value enters learning (via
        `signals`, already realized-only); the IOU is popped — discarded, not
        averaged in. Returns the IOU and the calibration errors, which are the
        proposer's training targets (G5), nothing else's."""
        iou = self._ledger.pop()
        with torch.no_grad():
            err_pos = float(self.head_pos.realized(p_arrival)) - iou.claim_pos
            err_neg = float(self.head_neg.realized(p_arrival)) - iou.claim_neg
        return iou, err_pos, err_neg
