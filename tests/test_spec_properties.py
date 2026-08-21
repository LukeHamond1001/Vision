"""Structural tests keyed to docs/drive-layer/SPEC.md clauses.

These do not test performance. Each test asserts an architectural identity or
enforcement that a spec clause claims — the point is that a refactor which
silently breaks a wiring commitment fails CI, not review.

Run:  python -m unittest discover tests -v
"""

import unittest

import torch

from iga import (Agent, AgentConfig, CoverageCap, Curiosity, FixedRewardHead,
                 GoalRegister, Leash, PremappedLatent, RegisterHeldError)
from iga.envs import ContinuousGrid, epsilon_ball_attack
from iga.gating import realized_from_total


def build_world(seed: int = 0):
    latent = PremappedLatent(world_dim=2, latent_dim=8, seed=seed)
    env = ContinuousGrid(latent, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    support = torch.stack([latent.embed(w) for w in torch.rand(256, 2, generator=gen)])
    head_pos = FixedRewardHead(env.embed_site(env.reward_site), sigma=0.4, proxy_samples=support)
    head_neg = FixedRewardHead(env.embed_site(env.hazard_site), sigma=0.4, proxy_samples=support)
    agent = Agent(AgentConfig(seed=seed), head_pos, head_neg, latent)
    return latent, env, head_pos, head_neg, agent


class TestWiring(unittest.TestCase):
    """SPEC §3 — the five wiring commitments."""

    def test_W1_reward_pathway_parameter_free(self):
        latent, env, head_pos, head_neg, agent = build_world()
        for t in head_pos.frozen_tensors() + head_neg.frozen_tensors():
            self.assertFalse(t.requires_grad)
        frozen = {id(t) for t in head_pos.frozen_tensors() + head_neg.frozen_tensors()
                  + latent.frozen_tensors()}
        for opt in (agent.opt_policy, agent.opt_proposer):
            for group in opt.param_groups:
                for prm in group["params"]:
                    self.assertNotIn(id(prm), frozen)

    def test_W2_frozen_metric(self):
        latent, *_ = build_world()
        for t in latent.frozen_tensors():
            self.assertFalse(t.requires_grad)

    def test_W4_G1_claim_identity_and_exact_attribution(self):
        """forward(p,i) − claim(i) == realized(p) exactly, AND autograd
        gradient×input over i equals the claim — the §6.2 identity."""
        _, _, head_pos, _, _ = build_world()
        gen = torch.Generator().manual_seed(7)
        for _ in range(16):
            p = torch.randn(8, generator=gen)
            i = torch.randn(8, generator=gen) * 10  # wild claims allowed
            total = head_pos.forward(p, i)
            self.assertTrue(torch.allclose(
                realized_from_total(total, head_pos.claim(i)), head_pos.realized(p),
                atol=1e-6))
            i_g = i.clone().requires_grad_(True)
            out = head_pos.realized(p) + head_pos.claim(i_g)
            out.backward()
            self.assertTrue(torch.allclose((i_g.grad * i_g.detach()).sum(),
                                           head_pos.claim(i), atol=1e-5))

    def test_W5_channel_writers_disjoint(self):
        _, env, _, _, agent = build_world()
        agent.observe(env.reset())
        p_before = agent.p.clone()
        agent._write_imagination(torch.randn(8))
        self.assertTrue(torch.equal(agent.p, p_before))       # imagination never touches p
        i_before = agent.i.clone()
        agent.observe(env.latent.embed(torch.tensor([0.3, 0.3])))
        self.assertTrue(torch.equal(agent.i, i_before))       # world never touches i


class TestGating(unittest.TestCase):
    """SPEC §5 and §6.1/§6.3."""

    def test_6_1_telescoping_exact(self):
        latent, _, _, _, agent = build_world()
        gen = torch.Generator().manual_seed(3)
        g = torch.randn(8, generator=gen)
        traj = [torch.randn(8, generator=gen) for _ in range(40)]
        total = sum(agent.gate.progress(traj[t], traj[t + 1], g) for t in range(39))
        expected = float(latent.distance(traj[0], g) - latent.distance(traj[-1], g))
        self.assertAlmostEqual(total, expected, places=4)
        loop = traj + [traj[0]]
        loop_total = sum(agent.gate.progress(loop[t], loop[t + 1], g) for t in range(40))
        self.assertAlmostEqual(loop_total, 0.0, places=4)     # zero over any loop (γ=1)

    def test_G1_signal_invariant_to_claims(self):
        """The learning signal must not move when the claim does (§6.3)."""
        _, env, _, _, agent = build_world()
        agent.observe(env.reset())
        agent.register.commit(torch.randn(8), window=10)
        p_prev, p_now = torch.randn(8), torch.randn(8)
        agent._write_imagination(torch.zeros(8))
        s_small = agent.gate.signals(p_prev, p_now, agent.register.target, 0.0)
        agent._write_imagination(torch.randn(8) * 1e6)        # absurd claim
        s_huge = agent.gate.signals(p_prev, p_now, agent.register.target, 0.0)
        self.assertEqual(s_small.total(), s_huge.total())

    def test_G3_ledger_settles_and_discards(self):
        _, env, _, _, agent = build_world()
        agent.observe(env.reset())
        feats = agent.trunk(agent.p, agent.i).detach()
        g = torch.randn(8)
        agent.gate.log_iou(g, feats)
        self.assertEqual(agent.gate.open_ious, 1)
        _, err_pos, err_neg = agent.gate.settle(torch.randn(8))
        self.assertEqual(agent.gate.open_ious, 0)             # discarded, not averaged
        self.assertIsInstance(err_pos, float)
        self.assertIsInstance(err_neg, float)

    def test_G5_progress_gradients_never_reach_proposer(self):
        _, env, _, _, agent = build_world()
        agent.observe(env.reset())
        agent.propose_and_commit(horizon_steps=100)
        if not agent.register.open:                           # force a target if filters blocked all
            agent.register.commit(agent.p + 0.01, window=10)
        p_prev = agent.p
        action, logp, entropy, value = agent.act()
        p_now, *_ = env.step(action)
        agent.observe(p_now)
        agent.learn_step(p_prev, agent.p, logp, entropy, value)
        agent.finish_episode()                                # a2c update ran (§5.4)
        for prm in agent.imagination_head.parameters():
            self.assertTrue(prm.grad is None or torch.all(prm.grad == 0),
                            "G5 violated: progress gradient reached the proposer")

    def test_5_4_critic_in_policy_optimizer_only(self):
        _, _, _, _, agent = build_world()
        critic_ids = {id(p) for p in agent.critic.parameters()}
        pol = {id(p) for g in agent.opt_policy.param_groups for p in g["params"]}
        pro = {id(p) for g in agent.opt_proposer.param_groups for p in g["params"]}
        self.assertTrue(critic_ids <= pol)
        self.assertTrue(critic_ids.isdisjoint(pro))


class TestConstraints(unittest.TestCase):
    """SPEC §4."""

    def test_C2_register_immutable_mid_window(self):
        r = GoalRegister()
        r.commit(torch.randn(8), window=5)
        with self.assertRaises(RegisterHeldError):
            r.commit(torch.randn(8), window=5)
        r.close()
        r.commit(torch.randn(8), window=5)                    # legal after close

    def test_C3_leash_hard_projection(self):
        latent, env, _, _, _ = build_world()
        leash = Leash(latent, radius=0.1)
        leash.anchor(env.reset())
        far = torch.randn(8) * 50
        g = leash.project(far)
        d = float(latent.distance(g, leash.support()[0]))
        self.assertLessEqual(d, 0.1 + 1e-5)

    def test_C1_cap_is_neighborhood_keyed(self):
        latent, *_ = build_world()
        cap = CoverageCap(latent, radius=0.1, max_updates=2)
        base = torch.zeros(8) + 0.05
        near_dupes = [base + torch.randn(8) * 1e-4 for _ in range(10)]
        admitted = 0
        for z in near_dupes:                                  # distinct identities, one key
            if cap.allow(z):
                cap.record(z)
                admitted += 1
        self.assertEqual(admitted, 2)

    def test_E3a_epsilon_ball_attack_blocked(self):
        latent, *_ = build_world()
        cap = CoverageCap(latent, radius=0.1, max_updates=2)
        out = epsilon_ball_attack(latent, cap, center=torch.zeros(8) + 0.05, eps=0.02)
        self.assertEqual(out["distinct_identities"], 64)
        self.assertLessEqual(out["admitted"], 2 * out["distinct_keys"])
        self.assertLess(out["admitted"], 64)                  # identity keying would admit all

    def test_C6_curiosity_one_shot_neighborhood(self):
        latent, *_ = build_world()
        cur = Curiosity(latent, radius=0.1, bonus=0.5)
        z = torch.zeros(8) + 0.05
        self.assertEqual(cur.bonus(z), 0.5)
        cur.visit(z)
        self.assertEqual(cur.bonus(z), 0.0)                   # dead on contact
        self.assertEqual(cur.bonus(z + 1e-4), 0.0)            # dead for the neighborhood
        self.assertEqual(cur.bonus(z + 10.0), 0.5)            # alive elsewhere


class TestSmoke(unittest.TestCase):
    def test_episode_runs(self):
        _, env, _, _, agent = build_world(seed=2)
        stats = agent.run_episode(env, max_steps=60)
        self.assertIn("return", stats)


if __name__ == "__main__":
    unittest.main()
