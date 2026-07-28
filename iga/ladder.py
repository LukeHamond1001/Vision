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
from .learner import EpisodicLearner
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
    veto_threshold: float = 0.5      # f-scale (prospective evaluation, round 9)
    flinch_threshold: float = 0.5    # C4 acting-time veto (see agent.py)
    flinch_resamples: int = 4
    grad_proposals: bool = False     # round 10: imagination climbs the frozen
    #   evaluator — candidates stepped along ∇(f+−f−) from the current state.
    #   Parameter-free ambition; fixes proposal coverage in high-dim pools.
    grad_steps: tuple = (0.05, 0.1, 0.15, 0.2)
    proposal_k: int = 12
    proposal_noise: float = 0.25
    w_prog: float = 1.0
    # Per-band progress weights (L2 corollary): slow-band progress is
    # intrinsically small per step (slow variables move slowly) — weighting by
    # hold-length ratio keeps per-window progress budgets comparable across
    # bands. This is a capability ONLY the banded architecture has: a flat
    # agent cannot weight what it cannot separate.
    w_prog_bands: tuple = (1.0, 3.0)
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
        self.critic = torch.nn.Linear(32, 1)     # §5.4: policy-side value head
        self.core = EpisodicLearner()
        self._pending: tuple | None = None
        self.proposers = torch.nn.ModuleList(
            [torch.nn.Linear(32, bd) for bd in latent.band_dims])

        self.registers = [GoalRegister() for _ in range(self.K)]
        self.leash = Leash(latent, cfg.leash_radius)
        self.cap = CoverageCap(latent, cfg.cap_radius, cfg.cap_max_updates)
        self.curiosity = Curiosity(latent, cfg.cap_radius, cfg.curiosity_bonus)

        self.opt_policy = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.action_head.parameters())
            + list(self.critic.parameters()), lr=cfg.lr_policy)
        self.opt_proposer = torch.optim.Adam(self.proposers.parameters(), lr=cfg.lr_proposer)

        self._gen = torch.Generator().manual_seed(cfg.seed)
        self.p = torch.zeros(d)
        self.i = torch.zeros(d)
        self._ious: dict[int, LevelIOU] = {}
        self._baseline = 0.0
        self.commit_hook = None          # probes may observe (level, target)
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
    def _gradient_candidates(self, k: int) -> list[torch.Tensor]:
        """Round 10: candidates stepped along the frozen evaluator's gradient
        in band k, from the CURRENT state under the held composite. Autograd
        through fixed functions only — nothing is trained; every candidate
        still passes leash, veto, bar, and prospective ranking. This is the
        sequencing mechanism: the gradient points padward at low charge and
        doorward at high charge, so phase switching emerges from evaluator
        structure with no new registers."""
        comp = self._composite()
        comp[self.latent.band_slices[k]] = self.latent.band(self.p, k)
        x = comp.detach().clone().requires_grad_(True)
        val = self.head_pos.realized(x) - self.head_neg.realized(x)
        val.backward()
        g = x.grad[self.latent.band_slices[k]]
        n = float(torch.linalg.vector_norm(g))
        if n < 1e-8:
            return []
        ghat = (g / n).detach()
        here = self.latent.band(self.p, k)
        return [here + s * ghat for s in self.cfg.grad_steps]

    def propose_level(self, k: int) -> bool:
        """Propose and commit for band k (its window must be closed). Slice-k
        candidates are valued as part of the full composite (L3)."""
        feats = self.trunk(self.p, self.i)
        base = self.proposers[k](feats.detach())               # G5: detached input
        noise = torch.randn(self.cfg.proposal_k, base.shape[-1],
                            generator=self._gen) * self.cfg.proposal_noise
        pool = [c for c in (base.unsqueeze(0) + noise)]
        if self.cfg.grad_proposals:
            pool.extend(self._gradient_candidates(k))
        best, best_score = None, None
        top_level = (k == self.K - 1)
        for c in pool:
            comp = self._composite()
            # C3 per band (round 9): project ONLY slice k against band-k
            # support; held slices stay as committed. Composite projection let
            # a held slow target's off-support ambition eat the whole leash
            # budget and drag the fast slice with it.
            g_k = self.leash.project_slice(c.detach(), self.latent.band_slices[k])
            comp[self.latent.band_slices[k]] = g_k
            with torch.no_grad():
                # Prospective evaluation (round 9; see agent.py): fixed
                # nonlinear evaluators on the imagined composite. Linear
                # claims rank by a single fixed direction and cannot express
                # phase structure ("pad first, then door").
                f_pos = float(self.head_pos.realized(comp))
                f_neg = float(self.head_neg.realized(comp))
                dist_k = float(self.latent.band_distance(self.p, comp, k))
            if f_neg > self.cfg.veto_threshold:                 # C4 prospective veto
                continue
            h_k = self.cfg.holds[k] * self.cfg.max_world_step * self.latent.step_scale(k)
            if dist_k > h_k:                                    # C5 per-band horizon
                continue
            if (top_level or True) and f_pos - f_neg < self.cfg.value_bar:
                # C7: mandatory at top level (L4); scaffold keeps it on for
                # all levels — fast-level relaxation is a battery knob, not
                # a default.
                continue
            score = f_pos - f_neg                               # prospective rank
            if best_score is None or score > best_score:
                best, best_score = g_k.clone(), score
        if best is None:
            return False
        self.registers[k].commit(best, window=self.cfg.holds[k])   # L2/C2 per band
        self._write_imagination()
        if self.commit_hook is not None:
            self.commit_hook(k, best.detach())
        comp = self._composite()
        self._ious[k] = LevelIOU(features=feats.detach().clone(),
                                 claim_pos=float(self.head_pos.claim(comp)),
                                 claim_neg=float(self.head_neg.claim(comp)))
        return True

    # ---------------------------------------------------------------- traverse
    def _flinches(self, world_action: torch.Tensor) -> bool:
        """C4 acting-time veto via one-step lookahead (see agent.py)."""
        with torch.no_grad():
            p_hat = self.p + self.latent.embed_delta(world_action)
            return float(self.head_neg.realized(p_hat)) > self.cfg.flinch_threshold

    def act(self):
        feats = self.trunk(self.p, self.i)
        dist = self.action_head.dist(feats)
        a = dist.sample()
        for _ in range(self.cfg.flinch_resamples):
            if not self._flinches(torch.tanh(a) * self.cfg.max_world_step):
                break
            a = dist.sample()
        else:
            a = torch.zeros_like(a)
        self._pending = (self.p, self.i, a)
        return (torch.tanh(a) * self.cfg.max_world_step, dist.log_prob(a).sum(),
                dist.entropy().sum(), self.critic(feats).reshape(()))

    def learn_step(self, p_prev: torch.Tensor, p_now: torch.Tensor, logp,
                   entropy=None, value=None) -> bool:
        bonus = self.curiosity.bonus(p_now)
        self.curiosity.visit(p_now)                             # C6
        prog = 0.0
        for k, r in enumerate(self.registers):                  # Σ_k per-band progress (L2)
            if r.open:
                with torch.no_grad():
                    prog += self.cfg.w_prog_bands[k] * float(
                        self.latent.band_distance(p_prev, r.target, k)
                        - self.latent.band_distance(p_now, r.target, k))
        capped = not self.cap.allow(p_now)                      # C1 gates progress only
        use_prog = 0.0 if capped else prog
        if not capped and any(r.open for r in self.registers):
            self.cap.record(p_now)
        with torch.no_grad():
            realized = float(self.head_pos.realized(p_now)) - float(self.head_neg.realized(p_now))
        total = realized + self.cfg.w_prog * use_prog + bonus
        if self._pending is not None:                           # §5.4 episodic path
            self.core.record(*self._pending, total)
            self._pending = None
        else:                                                   # REINFORCE fallback
            adv = total - self._baseline
            self._baseline = 0.99 * self._baseline + 0.01 * total
            self.opt_policy.zero_grad()
            (-adv * logp).backward()
            self.opt_policy.step()
        return capped

    def finish_episode(self) -> None:
        params = [p for g in self.opt_policy.param_groups for p in g["params"]]
        self.core.finish(self, self.opt_policy, params)

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
                # G5 — the proposer's two currencies, both minted by the world:
                # (1) realized value of the REACHED target (REINFORCE on the
                #     proposal log-prob, weighted by f± at arrival — pays
                #     ambition, world-confirmed by construction);
                # (2) calibration accuracy of the claim (pays honesty).
                # Currency (1) was unimplemented until round 9 — proposers
                # learned accuracy but never ambition, leaving proposal pools
                # without valuable candidates to rank.
                with torch.no_grad():
                    realized_net = (float(self.head_pos.realized(self.p))
                                    - float(self.head_neg.realized(self.p)))
                err = float(self.head_pos.realized(self.p)) - iou.claim_pos
                g_prop = self.proposers[k](iou.features)
                dist_prop = torch.distributions.Normal(g_prop, self.cfg.proposal_noise)
                logp = dist_prop.log_prob(r.target).sum()
                w_slice = self.head_pos.effective_w[self.latent.band_slices[k]]
                target = torch.tensor(float((w_slice * r.target).sum()) + err)
                loss = (-realized_net * logp                                 # currency 1
                        + ((g_prop * w_slice).sum() - target) ** 2)          # currency 2
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
            action, logp, entropy, value = self.act()
            p_now, r, done, info = env.step(action)
            self.observe(p_now)
            for reg in self.registers:
                if reg.open:
                    reg.tick()
            self.learn_step(p_prev, self.p, logp, entropy, value)
            stats["return"] += r
            stats["zone_flips"] += int(bool(info.get("zone_flip")))
            stats["settles"] += self.settle_levels()
            if done:
                break
        self.finish_episode()
        return stats
