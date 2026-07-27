"""Structural tests for the register ladder (SPEC §10, L1-L4).

Run:  python -m unittest discover tests -v
"""

import unittest

import torch

from iga.envs.twozone import TwoZoneWorld
from iga.heads import FixedRewardHead, SumHead
from iga.ladder import LadderAgent, LadderConfig
from iga.latent import BandedLatent
from iga.registers import RegisterHeldError


def build_twozone(seed: int = 0):
    latent = BandedLatent(part_dims=[2, 1], band_dims=[6, 2], seed=0)
    env = TwoZoneWorld(latent)
    gen = torch.Generator().manual_seed(seed + 500)
    world = torch.cat([torch.rand(256, 2, generator=gen),
                       (torch.rand(256, 1, generator=gen) > 0.5).float()], dim=1)
    support = torch.stack([latent.embed(w) for w in world])
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.reward_site, 1.0), sigma=0.25, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.gate, 0.0), sigma=0.2, proxy_samples=support)],
        weights=[1.0, 0.3])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.15,
                               proxy_samples=support)
    agent = LadderAgent(LadderConfig(seed=seed), head_pos, head_neg, latent)
    return latent, env, head_pos, head_neg, agent


class TestLadder(unittest.TestCase):
    def test_L1_banded_frozen_and_band_metric(self):
        latent, *_ = build_twozone()
        for t in latent.frozen_tensors():
            self.assertFalse(t.requires_grad)
        a, b = torch.randn(8), torch.randn(8)
        for k, sl in enumerate(latent.band_slices):
            self.assertTrue(torch.allclose(
                latent.band_distance(a, b, k),
                torch.linalg.vector_norm(a[sl] - b[sl])))

    def test_L2_per_band_telescoping_and_holds(self):
        latent, _, _, _, agent = build_twozone()
        gen = torch.Generator().manual_seed(3)
        g = torch.randn(8, generator=gen)
        traj = [torch.randn(8, generator=gen) for _ in range(30)] + []
        for k in range(2):
            total = sum(float(latent.band_distance(traj[t], g, k)
                              - latent.band_distance(traj[t + 1], g, k))
                        for t in range(29))
            expected = float(latent.band_distance(traj[0], g, k)
                             - latent.band_distance(traj[-1], g, k))
            self.assertAlmostEqual(total, expected, places=4)
            loop = traj + [traj[0]]
            self.assertAlmostEqual(
                sum(float(latent.band_distance(loop[t], g, k)
                          - latent.band_distance(loop[t + 1], g, k)) for t in range(30)),
                0.0, places=4)
        # slow register immutable mid-window while fast may re-choose at boundary
        agent.registers[1].commit(torch.randn(2), window=96)
        with self.assertRaises(RegisterHeldError):
            agent.registers[1].commit(torch.randn(2), window=96)
        agent.registers[0].commit(torch.randn(6), window=2)
        agent.registers[0].close()
        agent.registers[0].commit(torch.randn(6), window=2)   # legal at boundary

    def test_L3_composite_slice_discipline_and_claim_superposition(self):
        latent, env, head_pos, _, agent = build_twozone()
        agent.observe(env.reset())
        g_fast, g_slow = torch.randn(6), torch.randn(2)
        agent.registers[0].commit(g_fast, window=12)
        agent.registers[1].commit(g_slow, window=96)
        agent._write_imagination()
        self.assertTrue(torch.allclose(agent.i[latent.band_slices[0]], g_fast))
        self.assertTrue(torch.allclose(agent.i[latent.band_slices[1]], g_slow))
        # claims superpose over slices: claim(comp) == Σ_k w_slice_k · g_k
        comp = agent.i
        w = head_pos.effective_w
        manual = sum(float((w[latent.band_slices[k]]
                            * comp[latent.band_slices[k]]).sum()) for k in range(2))
        self.assertAlmostEqual(float(head_pos.claim(comp)), manual, places=5)

    def test_L4_W1_wiring(self):
        _, _, head_pos, head_neg, agent = build_twozone()
        head_pos.assert_parameter_free()
        head_neg.assert_parameter_free()
        agent.assert_wiring()

    def test_G5_no_progress_gradient_to_any_proposer(self):
        _, env, _, _, agent = build_twozone()
        agent.observe(env.reset())
        agent.registers[0].commit(agent.p[agent.latent.band_slices[0]] + 0.05, window=12)
        agent._write_imagination()
        p_prev = agent.p
        action, logp = agent.act()
        p_now, *_ = env.step(action)
        agent.observe(p_now)
        agent.learn_step(p_prev, agent.p, logp)
        for prm in agent.proposers.parameters():
            self.assertTrue(prm.grad is None or torch.all(prm.grad == 0))

    def test_smoke_episode(self):
        _, env, _, _, agent = build_twozone(seed=2)
        stats = agent.run_episode(env, max_steps=60)
        self.assertIn("return", stats)


if __name__ == "__main__":
    unittest.main()
