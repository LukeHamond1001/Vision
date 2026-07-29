"""BankWorld (v0.8): a slow variable that photometric statistics cannot see.

Six luminance-identical resource dots. Touching a field dot RELOCATES it to
the bank strip (top row) — same color, same brightness, different position.
The slow variable is the banked count b ∈ {0..6}: pure spatial
configuration; global per-channel means and stds are invariant to it by
construction. The door pays when b >= 4 and the agent reaches it.

This is the incapacity-library discriminator (SPEC-to-be §12): v0.7's
raw-photometric slow pathway MUST fail routing here (its input cannot vary
with b); a fixed region-pooled pathway admits b — but also re-admits coarse
agent position, so the temporal prior must sort them. Pre-registered in
experiments_v08.
"""

from __future__ import annotations

import torch

from .pixelcharge import PixelRenderer


class BankWorld:
    N_DOTS = 6

    def __init__(self, seed: int = 0, size: int = 64, need: int = 4):
        self.size = size
        self.need = need
        self.renderer = PixelRenderer(size)
        self.door = torch.tensor([0.9, 0.5])
        self.start = torch.tensor([0.1, 0.5])
        self._gen = torch.Generator().manual_seed(seed + 12000)
        self.reset()

    def _bank_slot(self, j: int) -> torch.Tensor:
        return torch.tensor([0.15 + 0.12 * j, 0.85])  # full disc INSIDE the
        # frame: slots at 0.95 clipped half the gaussian mass off-image,
        # making global brightness decrease with b — a photometric leak
        # the negative control caught (photo-head b-corr 0.958)

    def _scatter(self, gen) -> torch.Tensor:
        return torch.rand(self.N_DOTS, 2, generator=gen) * torch.tensor([0.8, 0.55]) \
            + torch.tensor([0.1, 0.12])   # field region ends below the bank row

    def reset(self, banked_init: int = 0) -> dict:
        self.pos = self.start.clone()
        self.field_pos = self._scatter(self._gen)
        self.banked = [False] * self.N_DOTS
        for j in range(min(banked_init, self.N_DOTS)):
            self.banked[j] = True
        self.max_b = self.b
        return {}

    @property
    def b(self) -> int:
        return sum(self.banked)

    def frame(self) -> torch.Tensor:
        img = torch.zeros(3, self.size, self.size)
        img[2] += 0.15
        door_open = 1.0 if self.b >= self.need else 0.0
        door_glow = self.renderer._disc(self.door, 0.06)
        img[0] += (0.6 - 0.4 * door_open) * door_glow
        img[1] += (0.2 + 0.5 * door_open) * door_glow
        for j in range(self.N_DOTS):
            center = self._bank_slot(j) if self.banked[j] else self.field_pos[j]
            dot = self.renderer._disc(center, 0.03)
            img[1] += 0.7 * dot                      # identical color everywhere
            img[2] += 0.4 * dot
        agent = self.renderer._disc(self.pos, 0.035)
        img[0] += 0.9 * agent
        img[1] += 0.9 * agent
        img[2] += 0.9 * agent
        # Photometric NUISANCE (v0.8 round 3): memoryless per-frame
        # illumination jitter. Exact photometric invariance is a mirage in
        # rendered scenes (clipping, then saturation, each leaked b at the
        # ~0.1-1% level and whitened readouts amplified either to corr 0.96);
        # the realistic discriminator is signal-to-nuisance ratio: +-15%
        # jitter drowns micro-leaks (and, being memoryless, cannot satisfy
        # the slow prior) while bank-row occupancy swings its region ~100%.
        # Round 4: PER-CHANNEL independent gains. Common-mode jitter is
        # defeated by null-space contrasts (a linear head cancels the shared
        # constant direction exactly and keeps the micro-leak, measured at
        # corr 0.968 through +-15% common jitter). Independent channel gains
        # admit no fixed cancelling contrast in channel space; spatial
        # contrasts within the region pathway still cancel their common
        # component. This ladder of control-defeats is itself the finding.
        # Round 5: gain AND offset per channel (the textbook sensor model).
        # Per-channel gain alone still left one invariant contrast per
        # channel (mean and std share the gain; corr 0.879 through it).
        # An additive offset moves means but not stds — the per-channel
        # constant subspace becomes 2-D, the null space closes, and no
        # linear invariant survives. Spatial (region) contrasts cancel
        # additive offsets exactly and shrug off common gain.
        gains = 0.85 + 0.30 * torch.rand(3, 1, 1, generator=self._gen)
        offsets = 0.06 * torch.rand(3, 1, 1, generator=self._gen)
        return (img * gains + offsets).clamp(0.0, 1.0)

    def step(self, action: torch.Tensor):
        self.pos = torch.clamp(self.pos + action.detach(), 0.0, 1.0)
        reward, done, info = 0.0, False, {}
        for j in range(self.N_DOTS):
            if not self.banked[j] and \
                    torch.linalg.vector_norm(self.pos - self.field_pos[j]) < 0.05:
                self.banked[j] = True                # relocate: field -> bank
                info["banked"] = True
        self.max_b = max(self.max_b, self.b)
        if self.b >= self.need and \
                torch.linalg.vector_norm(self.pos - self.door) < 0.07:
            reward, done = 1.0, True
        return self.frame(), reward, done, info
