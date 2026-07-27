"""Register ladder (SPEC §10, L1–L4): the multi-timescale agent.

One goal register per latent band, geometrically spaced hold-lengths. All
v0.1 commitments hold per level: the reward pathway stays parameter-free
(W1) with claims linear over the COMPOSITE imagination channels (L3/W4),
per-band progress is potential-based within each hold window (L2/§6.1),
progress pays the policy and never any proposer (L4/G5), and the value bar
is mandatory at the slowest level (L4/C7 — where the E3b-confirmed treadmill
concentrates).

Scaffold simplification, documented: leash projection acts on the composite
target, then committed slices of other bands are re-imposed; the projected
slice may sit slightly outside the ball that a per-band projection would
give. Acceptable at scaffold scale; a production leash projects per band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constraints import CoverageCap, Curiosity, Leash
from .latent import BandedLatent
from .registers import GoalRegister
from .trunk import ActionHead, SharedTrunk


@dataclass
class LadderConfig:
    holds: tuple = (12, 96)          # τ_1 < τ_2 (fast, slow), in steps
    arrive_eps: tuple = (0.08, 0.3)  # per-band arrival radii
    leash_radius: float = 0.15
    cap_radius: float = 0.1
    cap_max_updates: int = 2
    curiosity_bonus: float = 0.1
    value_bar: float = 0.02          # C7; ALWAYS enforced at the top level (L4)
    veto_threshold: float = 10.0     # calibrate via A2 for real runs
    proposal_k: int = 12
    proposal_noise: float = 0.25
    w_prog: float = 1.0
    lr_policy: float = 3e-3
    lr_proposer: float = 3e-3
    max_world_step: float = 0.1
    seed: int = 0


@dataclass
class LevelIOU:
    features: torch.Tensor
    claim_pos: float
    claim_neg: float


class LadderAgent:
    def __init__(self, cfg: LadderConfig, head_pos, head_neg, latent: BandedLatent):
        self.cfg = cfg
        self.latent = latent
        self.head_pos, self.head_neg = head_pos, head_neg
        self.K = len(latent.band_dims)
        assert len(cfg.holds) == self.K

        d = latent.latent_dim
        self.trunk = SharedTrunk(d, d)
        self.action_head = ActionHead(feat=32, act_dim=2)
        self.proposers = torch.nn.ModuleList(
            [torch.nn.Linear(32, bd) for bd in latent.band_dims])

        self.registers = [GoalRegister() for _ in range(self.K)]
        self.leash = Leash(latent, cfg.leash_radius)
        self.cap = CoverageCap(latent, cfg.cap_radius, cfg.cap_max_updates)
        self.curiosity = Curiosity(latent, cfg.cap_radius, cfg.curiosity_bonus)

        self.opt_policy = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.action_head.parameters()),
            lr=cfg.lr_policy)
        self.opt_proposer = torch.optim.Adam(self.proposers.parameters(), lr=cfg.lr_proposer)

        self._gen = torch.Generator().manual_seed(cfg.seed)
        self.p = torch.zeros(d)
        self.i = torch.zeros(d)
        self._ious: dict[int, LevelIOU] = {}
        self._baseline = 0.0
        self.assert_wiring()

    # ---------------------------------------------------------------- wiring
    def assert_wiring(self) -> None:
        self.head_pos.assert_parameter_free()
        self.head_neg.assert_parameter_free()
        frozen = {id(t) for t in (self.head_pos.frozen_tensors()
                                  + self.head_neg.frozen_tensors()
                                  + self.latent.frozen_tensors())}
        for opt in (self.opt_policy, self.opt_proposer):
            for group in opt.param_groups:
                for prm in group["params"]:
                    assert id(prm) not in frozen, "W1/W2 violated in ladder"
        pol = {id(p) for g in self.opt_policy.param_groups for p in g["params"]}
        pro = {id(p) for g in self.opt_proposer.param_groups for p in g["params"]}
        assert pol.isdisjoint(pro), "G5 violated: shared params across optimizers"

    # ---------------------------------------------------------------- channels (W5/L3)
    def observe(self, p_new: torch.Tensor) -> None:
        self.p = p_new.detach().clone()
        self.leash.anchor(self.p)

    def _composite(self) -> torch.Tensor:
        return self.latent.compose(
            [r.target if r.open else None for r in self.registers])

    def _write_imagination(self) -> None:
        self.i = self._composite()

    # ---------------------------------------------------------------- propose/commit
    def propose_level(self, k: int) -> bool:
        """Propose and commit for band k (its window must be closed). Slice-k
        candidates are valued as part of the full composite (L3)."""
        feats = self.trunk(self.p, self.i)
        base = self.proposers[k](feats.detach())               # G5: detached input
        noise = torch.randn(self.cfg.proposal_k, base.shape[-1],
                            generator=self._gen) * self.cfg.proposal_noise
        best, best_score = None, None
        top_level = (k == self.K - 1)
        for c in (base.unsqueeze(0) + noise):
            comp = self._composite()
            comp[self.latent.band_slices[k]] = c.detach()
            comp = self.leash.project(comp)                     # C3 (composite; see module note)
            for j, r in enumerate(self.registers):              # re-impose held slices
                if j != k and r.open:
                    comp[self.latent.band_slices[j]] = r.target
            g_k = comp[self.latent.band_slices[k]]
            with torch.no_grad():
                claim_pos = float(self.head_pos.claim(comp))
                claim_neg = float(self.head_neg.claim(comp))
                dist_k = float(self.latent.band_distance(self.p, comp, k))
            if claim_neg > self.cfg.veto_threshold:             # C4 prospective veto
                continue
            h_k = self.cfg.holds[k] * self.cfg.max_world_step * self.latent.step_scale(k)
            if dist_k > h_k:                                    # C5 per-band horizon
                continue
            if (top_level or True) and claim_pos - claim_neg < self.cfg.value_bar:
                # C7: mandatory at top level (L4); scaffold keeps it on for
                # all levels — fast-level relaxation is a battery knob, not
                # a default.
                continue
            score = claim_pos - claim_neg                       # claims rank (G1)
            if best_score is None or score > best_score:
                best, best_score = g_k.clone(), score
        if best is None:
            return False
        self.registers[k].commit(best, window=self.cfg.holds[k])   # L2/C2 per band
        self._write_imagination()
        comp = self._composite()
        self._ious[k] = LevelIOU(features=feats.detach().clone(),
                                 claim_pos=float(self.head_pos.claim(comp)),
                                 claim_neg=float(self.head_neg.claim(comp)))
        return True

    # ---------------------------------------------------------------- traverse
    def act(self):
        feats = self.trunk(self.p, self.i)
        dist = self.action_head.dist(feats)
        a = dist.sample()
        return torch.tanh(a) * self.cfg.max_world_step, dist.log_prob(a).sum()

    def learn_step(self, p_prev: torch.Tensor, p_now: torch.Tensor, logp) -> bool:
        bonus = self.curiosity.bonus(p_now)
        self.curiosity.visit(p_now)                             # C6
        prog = 0.0
        for k, r in enumerate(self.registers):                  # Σ_k per-band progress (L2)
            if r.open:
                with torch.no_grad():
                    prog += float(self.latent.band_distance(p_prev, r.target, k)
                                  - self.latent.band_distance(p_now, r.target, k))
        capped = not self.cap.allow(p_now)                      # C1 gates progress only
        use_prog = 0.0 if capped else prog
        if not capped and any(r.open for r in self.registers):
            self.cap.record(p_now)
        with torch.no_grad():
            realized = float(self.head_pos.realized(p_now)) - float(self.head_neg.realized(p_now))
        total = realized + self.cfg.w_prog * use_prog + bonus
        adv = total - self._baseline
        self._baseline = 0.99 * self._baseline + 0.01 * total
        self.opt_policy.zero_grad()
        (-adv * logp).backward()
        self.opt_policy.step()
        return capped

    # ---------------------------------------------------------------- arrive/calibrate
    def settle_levels(self) -> int:
        """Per-band arrival reconciliation (G3) + per-level calibration (G5).
        Timeout at a boundary discards the IOU — re-choice is then legal (L2)."""
        settled = 0
        for k, r in enumerate(self.registers):
            if not r.open:
                continue
            arrived = float(self.latent.band_distance(self.p, r.target, k)) < self.cfg.arrive_eps[k]
            timeout = r.remaining <= 0
            if not (arrived or timeout):
                continue
            iou = self._ious.pop(k, None)
            if arrived and iou is not None:
                err = float(self.head_pos.realized(self.p)) - iou.claim_pos
                w_slice = self.head_pos.effective_w[self.latent.band_slices[k]]
                g_prop = self.proposers[k](iou.features)
                target = torch.tensor(float((w_slice * r.target).sum()) + err)
                loss = ((g_prop * w_slice).sum() - target) ** 2
                self.opt_proposer.zero_grad()
                loss.backward()
                self.opt_proposer.step()
                settled += 1
            r.close()
            self._write_imagination()
        return settled

    # ---------------------------------------------------------------- episode
    def run_episode(self, env, max_steps: int = 100) -> dict:
        self.observe(env.reset())
        self.cap.reset_round()
        stats = {"return": 0.0, "settles": 0, "zone_flips": 0}
        for _ in range(max_steps):
            for k in reversed(range(self.K)):                   # slow levels propose first
                if not self.registers[k].open:
                    self.propose_level(k)
            p_prev = self.p
            action, logp = self.act()
            p_now, r, done, info = env.step(action)
            self.observe(p_now)
            for reg in self.registers:
                if reg.open:
                    reg.tick()
            self.learn_step(p_prev, self.p, logp)
            stats["return"] += r
            stats["zone_flips"] += int(bool(info.get("zone_flip")))
            stats["settles"] += self.settle_levels()
            if done:
                break
        return stats
