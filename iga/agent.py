"""Agent assembly and the propose → commit → traverse → arrive → calibrate cycle.

Construction enforces the wiring commitments (SPEC §3) with runtime assertions:
W1 (no reward-head or latent tensor in any optimizer; reward pathway frozen),
W5 (disjoint channel writers). W2/W4 are properties of the frozen modules; the
test suite checks them directly. G5 is enforced structurally: committed targets
are detached constants, proposer losses read detached trunk features, and the
proposer has its own optimizer over imagination-head parameters only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constraints import CoverageCap, Curiosity, Leash
from .gating import LearningGate, Signals
from .learner import EpisodicLearner
from .heads import FixedRewardHead
from .latent import PremappedLatent
from .registers import GoalRegister
from .trunk import ActionHead, ImaginationHead, SharedTrunk


@dataclass
class AgentConfig:
    latent_dim: int = 8
    world_dim: int = 2
    act_dim: int = 2
    leash_radius: float = 0.1
    cap_radius: float = 0.1          # C1: at or below leash radius
    cap_max_updates: int = 2
    curiosity_bonus: float = 0.1
    value_bar: float = 0.05          # C7: v_min
    veto_threshold: float = 0.5      # C4: prospective veto on claim− (battery calibrates via A2)
    flinch_threshold: float = 0.5    # C4 acting-time veto: f− on the imagined next state
    flinch_resamples: int = 4        # resample budget before freezing in place
    proposal_k: int = 16
    proposal_noise: float = 0.2
    window: int = 24                 # commitment window (steps)
    arrive_eps: float = 0.08
    w_prog: float = 1.0
    lr_policy: float = 3e-3
    lr_proposer: float = 3e-3
    max_world_step: float = 0.1
    seed: int = 0

    # --- ablation switches (SPEC §9 battery ONLY; deployed values are the
    # defaults). Each maps to a spec clause it disables or weakens.
    learner: str = "ppo"                 # §5.4: 'ppo' | 'reinforce'
    cap_mode: str = "neighborhood"       # C1: 'neighborhood' | 'identity' | 'off'
    hold_target: bool = True             # C2: False => re-choose every step
    use_leash: bool = True               # C3
    use_veto: bool = True                # C4 (prospective veto)
    use_value_bar: bool = True           # C7
    curiosity_mode: str = "oneshot"      # C6: 'oneshot' | 'never_dies' | 'off'
    pay_proposer_progress: bool = False  # G5 ABLATION: True is the broken variant
    select_by_progress_history: bool = False  # G5 ABLATION (mechanistic): rank
    #   targets by historically achieved window progress instead of claims —
    #   "aim where we reliably make progress" is the treadmill made explicit.


class Agent:
    def __init__(self, cfg: AgentConfig, head_pos: FixedRewardHead, head_neg: FixedRewardHead,
                 latent: PremappedLatent):
        self.cfg = cfg
        self.latent = latent
        self.head_pos, self.head_neg = head_pos, head_neg

        self.trunk = SharedTrunk(cfg.latent_dim, cfg.latent_dim)
        self.action_head = ActionHead(feat=32, act_dim=cfg.act_dim)
        self.imagination_head = ImaginationHead(feat=32, latent_dim=cfg.latent_dim)
        self.critic = torch.nn.Linear(32, 1)     # §5.4: policy-side value head
        self.core = EpisodicLearner()
        self._pending: tuple | None = None       # (p, i, a_raw) stashed by act()

        self.register = GoalRegister()
        self.leash = Leash(latent, cfg.leash_radius)
        self.cap = CoverageCap(latent, cfg.cap_radius, cfg.cap_max_updates, mode=cfg.cap_mode)
        self.curiosity = Curiosity(latent, cfg.cap_radius, cfg.curiosity_bonus,
                                   mode=cfg.curiosity_mode)
        self.gate = LearningGate(latent, head_pos, head_neg)
        self.commit_hook = None          # probes may observe committed targets
        self._pending_logp: torch.Tensor | None = None   # G5-ablation bookkeeping
        self._window_progress = 0.0
        self._baseline = 0.0             # REINFORCE variance baseline (neutral)
        self._prog_history: dict[tuple, tuple[float, int]] = {}  # key -> (mean, n)

        # G5: two optimizers, disjoint parameter sets. Progress/realized train
        # the policy (trunk + action head); calibration trains the proposer.
        self.opt_policy = torch.optim.Adam(
            list(self.trunk.parameters()) + list(self.action_head.parameters())
            + list(self.critic.parameters()), lr=cfg.lr_policy
        )
        self.opt_proposer = torch.optim.Adam(self.imagination_head.parameters(), lr=cfg.lr_proposer)

        self._gen = torch.Generator().manual_seed(cfg.seed)

        # Channel state (W5): p is written only by observe(); i only by _write_imagination().
        self.p = torch.zeros(cfg.latent_dim)
        self.i = torch.zeros(cfg.latent_dim)

        self.assert_wiring()

    # ------------------------------------------------------------------ W1/W5
    def assert_wiring(self) -> None:
        self.head_pos.assert_parameter_free()
        self.head_neg.assert_parameter_free()
        frozen = {id(t) for t in (self.head_pos.frozen_tensors()
                                  + self.head_neg.frozen_tensors()
                                  + self.latent.frozen_tensors())}
        for t in (self.latent.frozen_tensors()):
            assert not t.requires_grad, "W2 violated: latent tensor requires grad"
        for opt in (self.opt_policy, self.opt_proposer):
            for group in opt.param_groups:
                for prm in group["params"]:
                    assert id(prm) not in frozen, "W1/W2 violated: frozen tensor in an optimizer"
        policy_ids = {id(p) for g in self.opt_policy.param_groups for p in g["params"]}
        proposer_ids = {id(p) for g in self.opt_proposer.param_groups for p in g["params"]}
        assert policy_ids.isdisjoint(proposer_ids), "G5 violated: shared params across optimizers"

    # ------------------------------------------------------------------ W5 writers
    def observe(self, p_new: torch.Tensor) -> None:
        """Sole writer of the perception channels: the world."""
        self.p = p_new.detach().clone()
        self.leash.anchor(self.p)

    def _write_imagination(self, g: torch.Tensor) -> None:
        """Sole writer of the imagination channels: the imagination head."""
        self.i = g.detach().clone()

    # ------------------------------------------------------------------ propose/commit
    def propose_and_commit(self, horizon_steps: int) -> dict:
        """SPEC §5.3 propose+commit. Returns diagnostics (candidates, filters)."""
        feats = self.trunk(self.p, self.i)
        # G5: proposer reads detached features; its losses cannot reach the trunk.
        candidates = self.imagination_head.propose(
            feats.detach(), self.cfg.proposal_k, self.cfg.proposal_noise, self._gen
        )
        diag = {"proposed": self.cfg.proposal_k, "vetoed": 0, "off_horizon": 0, "below_bar": 0}
        h_latent = horizon_steps * self.cfg.max_world_step * self.latent.step_scale()

        best, best_score, best_raw = None, None, None
        for k in range(candidates.shape[0]):
            raw = candidates[k]
            g = self.leash.project(raw) if self.cfg.use_leash else raw.detach()  # C3
            with torch.no_grad():
                # Prospective evaluation (round 9): the fixed NONLINEAR
                # evaluators applied to the imagined target — the same move as
                # the C4 flinch, one hop earlier. Linear claims cannot rank:
                # w·g orders every candidate pool by projection onto ONE fixed
                # global direction, state-independent by construction. The
                # claim channel remains solely the exact-subtraction currency
                # (G1); selection and veto read f± directly (no proxy gap).
                f_pos = float(self.head_pos.realized(g))
                f_neg = float(self.head_neg.realized(g))
                dist = float(self.latent.distance(self.p, g))
            if self.cfg.use_veto and f_neg > self.cfg.veto_threshold:  # C4 veto
                diag["vetoed"] += 1
                continue
            if dist > h_latent:                                        # C5 horizon bound
                diag["off_horizon"] += 1
                continue
            if self.cfg.use_value_bar and \
                    f_pos - f_neg < self.cfg.value_bar:                # C7 value bar
                diag["below_bar"] += 1
                continue
            if self.cfg.select_by_progress_history:
                # G5 ABLATION (mechanistic): selection consults achieved
                # progress. Unseen neighborhoods score optimistically at their
                # distance (their maximum progress pot) — so both arms get
                # tried, then history takes over. This is the §6.4 selector.
                mean, n = self._prog_history.get(self.cap.key(g), (dist, 0))
                score = mean
            else:
                score = f_pos - f_neg                                  # prospective rank
            if best_score is None or score > best_score:
                best, best_score, best_raw = g, score, raw

        if best is not None:
            self.register.commit(best, window=min(self.cfg.window, horizon_steps))
            self._write_imagination(best)
            self.gate.log_iou(best, feats)
            self._window_progress = 0.0
            if self.cfg.pay_proposer_progress:
                # G5 ABLATION ONLY. Score-function credit for the committed
                # proposal: exactly the pathway G5 forbids, kept here so probe
                # E3b can demonstrate the treadmill it creates.
                base = self.imagination_head(feats.detach())
                dist = torch.distributions.Normal(base, self.cfg.proposal_noise)
                self._pending_logp = dist.log_prob(best_raw.detach()).sum()
            if self.commit_hook is not None:
                self.commit_hook(best.detach())
            diag["committed"] = True
        else:
            diag["committed"] = False
        return diag

    # ------------------------------------------------------------------ traverse
    def _flinches(self, world_action: torch.Tensor) -> bool:
        """C4 acting-time veto: evaluate the fixed negative head on the
        IMAGINED next state (one-step lookahead in the frozen latent) and act
        on it without world verification. Parameter-free end to end, so the
        flinch inherits the reward pathway's tamper-proofness."""
        with torch.no_grad():
            p_hat = self.p + self.latent.embed_delta(world_action)
            return float(self.head_neg.realized(p_hat)) > self.cfg.flinch_threshold

    def act(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.trunk(self.p, self.i)
        dist = self.action_head.dist(feats)
        a = dist.sample()
        if self.cfg.use_veto:                    # one trust switch covers both vetoes
            for _ in range(self.cfg.flinch_resamples):
                if not self._flinches(torch.tanh(a) * self.cfg.max_world_step):
                    break
                a = dist.sample()
            else:
                a = torch.zeros_like(a)          # freeze in place: the flinch itself
        logp = dist.log_prob(a).sum()
        entropy = dist.entropy().sum()
        value = self.critic(feats).reshape(())
        self._pending = (self.p, self.i, a)      # for the episodic learner
        return torch.tanh(a) * self.cfg.max_world_step, logp, entropy, value

    def learn_step(self, p_prev: torch.Tensor, p_now: torch.Tensor, logp: torch.Tensor,
                   entropy: torch.Tensor | None = None,
                   value: torch.Tensor | None = None) -> tuple[Signals, bool]:
        """One learning step. C1 gates the IMAGINATION-DRIVEN component of the
        signal (progress toward the imagined target); real-outcome learning is
        never capped — SPEC signal 1 is 'used directly'. Under 'a2c' the step
        is recorded and the update happens in finish_episode(); under
        'reinforce' the update is immediate. Returns (signals, progress_capped)."""
        bonus = self.curiosity.bonus(p_now)
        self.curiosity.visit(p_now)                                    # C6: actual visit
        g = self.register.target if self.register.open else None
        sig = self.gate.signals(p_prev, p_now, g, bonus)
        self._window_progress += sig.progress
        capped = not self.cap.allow(p_now)                             # C1
        prog = 0.0 if capped else sig.progress
        if not capped and self.register.open:
            self.cap.record(p_now)
        total = (sig.realized_pos - sig.realized_neg) + self.cfg.w_prog * prog + sig.curiosity
        if self.cfg.learner != "reinforce" and self._pending is not None:
            self.core.record(*self._pending, total)
            self._pending = None
        else:
            advantage = total - self._baseline                         # variance baseline
            self._baseline = 0.99 * self._baseline + 0.01 * total
            loss = -advantage * logp                                   # REINFORCE fallback
            self.opt_policy.zero_grad()
            loss.backward()
            self.opt_policy.step()
        return sig, capped

    def finish_episode(self) -> None:
        """§5.4: per-episode batched policy/critic update (no-op for REINFORCE)."""
        if self.cfg.learner != "reinforce":
            params = [p for g in self.opt_policy.param_groups for p in g["params"]]
            self.core.finish(self, self.opt_policy, params)

    # ------------------------------------------------------------------ arrive/calibrate
    def _pay_proposer_progress(self) -> None:
        """G5 ABLATION ONLY: credit the proposer with the window's progress.
        This is the treadmill-generating pathway; never active by default."""
        if self._pending_logp is None:
            return
        loss = -self._window_progress * self._pending_logp
        self.opt_proposer.zero_grad()
        loss.backward()
        self.opt_proposer.step()
        self._pending_logp = None

    def _record_window_progress(self) -> None:
        """Running mean of achieved window progress per target neighborhood.
        Only consulted under the select_by_progress_history ablation."""
        key = self.cap.key(self.register.target)
        mean, n = self._prog_history.get(key, (0.0, 0))
        self._prog_history[key] = ((mean * n + self._window_progress) / (n + 1), n + 1)

    def abandon(self) -> None:
        """Close the window without arrival (used by the hold-target ablation)."""
        if self.gate.open_ious:
            self.gate._ledger.pop()
        if self.cfg.pay_proposer_progress:
            self._pay_proposer_progress()
        self._record_window_progress()
        self._pending_logp = None
        self.register.close()

    def maybe_settle(self) -> tuple[float, float] | None:
        """Arrival (G3) + calibration (G5). Returns calibration errors if settled."""
        arrived = float(self.latent.distance(self.p, self.register.target)) < self.cfg.arrive_eps
        timeout = self.register.remaining <= 0
        if not (arrived or timeout):
            return None
        if self.cfg.pay_proposer_progress:                             # ablation pathway
            self._pay_proposer_progress()
        self._record_window_progress()
        if not arrived:                                                # timeout: IOU dies unpaid
            self.gate._ledger.pop()
            self.register.close()
            return None
        g_committed = self.register.target
        iou, err_pos, err_neg = self.gate.settle(self.p)
        # G5 — both proposer currencies, world-minted (see ladder.py):
        # (1) realized value of the reached target via REINFORCE on the
        #     proposal log-prob; (2) calibration accuracy of the claim.
        with torch.no_grad():
            realized_net = (float(self.head_pos.realized(self.p))
                            - float(self.head_neg.realized(self.p)))
        g_prop = self.imagination_head(iou.propose_features)
        dist_prop = torch.distributions.Normal(g_prop, self.cfg.proposal_noise)
        logp = dist_prop.log_prob(g_committed).sum()
        target_pos = torch.tensor(iou.claim_pos + err_pos)
        target_neg = torch.tensor(iou.claim_neg + err_neg)
        loss = (-realized_net * logp
                + (self.head_pos.claim(g_prop) - target_pos) ** 2
                + (self.head_neg.claim(g_prop) - target_neg) ** 2)
        self.opt_proposer.zero_grad()
        loss.backward()
        self.opt_proposer.step()
        self.register.close()
        return err_pos, err_neg

    # ------------------------------------------------------------------ episode driver
    def run_episode(self, env, max_steps: int = 120) -> dict:
        p0 = env.reset()
        self.observe(p0)
        self.cap.reset_round()
        stats = {"return": 0.0, "catastrophes": 0, "settles": 0, "progress_capped": 0,
                 "commit_claims": [], "settle_realized": []}
        for t in range(max_steps):
            if not self.cfg.hold_target and self.register.open:        # C2 ablation:
                self.abandon()                                         # re-choose every step
            if not self.register.open:
                self.propose_and_commit(horizon_steps=max_steps - t)
                if self.register.open:
                    stats["commit_claims"].append(
                        float(self.head_pos.claim(self.register.target)))
                # else: goal-less wandering — realized + curiosity still teach
            p_prev = self.p
            action, logp, entropy, value = self.act()
            p_now, real_reward, done, info = env.step(action)
            self.observe(p_now)
            if self.register.open:
                self.register.tick()
            _, capped = self.learn_step(p_prev, self.p, logp, entropy, value)
            if capped:
                stats["progress_capped"] += 1
            stats["return"] += real_reward
            if info.get("catastrophe"):
                stats["catastrophes"] += 1
            if self.register.open and self.maybe_settle() is not None:
                stats["settles"] += 1
                stats["settle_realized"].append(float(self.head_pos.realized(self.p)))
            if done:
                break
        self.finish_episode()
        return stats
