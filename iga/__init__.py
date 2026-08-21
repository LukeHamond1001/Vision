"""Imagination-Gated Agent — reference scaffold for docs/drive-layer/SPEC.md (the drive-layer program; the language being lives in iga/lm_*).

Structural correctness over performance: this package exists to make the
wiring commitments (SPEC §3) and gating rules (SPEC §5) executable and
testable, not to post benchmark numbers.
"""

from .agent import Agent, AgentConfig
from .constraints import CoverageCap, Curiosity, Leash
from .gating import LearningGate, Signals, realized_from_total
from .heads import FixedRewardHead
from .latent import PremappedLatent
from .registers import GoalRegister, RegisterHeldError

__all__ = [
    "Agent", "AgentConfig", "CoverageCap", "Curiosity", "Leash",
    "LearningGate", "Signals", "realized_from_total", "FixedRewardHead",
    "PremappedLatent", "GoalRegister", "RegisterHeldError",
]
