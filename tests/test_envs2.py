"""Smoke tests for the round-7 probe environments."""

import unittest

import torch

from iga.envs.threezone import ThreeZoneWorld
from iga.envs.trapcorridor import TrapCorridor
from iga.latent import BandedLatent, PremappedLatent


class TestProbeEnvs(unittest.TestCase):
    def test_trap_corridor_semantics(self):
        latent = PremappedLatent(2, 8)
        env = TrapCorridor(latent)
        env.reset()
        self.assertTrue(env.in_trap(torch.tensor([0.5, 0.5])))
        self.assertFalse(env.in_trap(torch.tensor([0.5, 0.85])))   # above the trap: safe
        env.pos = torch.tensor([0.32, 0.5])
        _, r, done, info = env.step(torch.tensor([0.05, 0.0]))     # step into trap
        self.assertTrue(info.get("catastrophe"))
        self.assertEqual(r, -1.0)
        env2 = TrapCorridor(latent, trap_active=False)
        env2.reset()
        env2.pos = torch.tensor([0.32, 0.5])
        _, r2, done2, info2 = env2.step(torch.tensor([0.05, 0.0]))
        self.assertFalse(info2.get("catastrophe", False))          # paranoia cell: harmless

    def test_C4_flinch_lookahead(self):
        from iga.experiments_e2a import build
        agent, env = build(0, trap_active=True, use_veto=True)
        agent.observe(env.reset())
        agent.observe(env.latent.embed(torch.tensor([0.38, 0.5])))  # near trap edge
        into_trap = torch.tensor([0.08, 0.0])                       # lands 0.04 from center
        away = torch.tensor([-0.08, 0.0])                           # lands 0.20 from center
        self.assertTrue(agent._flinches(into_trap))
        self.assertFalse(agent._flinches(away))

    def test_three_zone_chain(self):
        latent = BandedLatent([2, 1], [6, 2])
        env = ThreeZoneWorld(latent)
        env.reset()
        env.pos = env.gates[0] - 0.05
        env.step(torch.tensor([0.05, 0.05]))
        self.assertEqual(env.zone, 1)
        env.pos = env.gates[1] - 0.05
        env.step(torch.tensor([0.05, 0.05]))
        self.assertEqual(env.zone, 2)
        env.pos = env.reward_site - 0.05
        _, r, done, _ = env.step(torch.tensor([0.05, 0.05]))
        self.assertEqual(r, 1.0)
        self.assertTrue(done)
        self.assertEqual(env.flips_this_episode, 2)


    def test_charge_world_dynamics(self):
        from iga.envs.chargeworld import ChargeWorld
        latent = BandedLatent([2, 1], [6, 2])
        env = ChargeWorld(latent)
        env.reset()
        env.pos = env.pad.clone()
        for _ in range(10):
            env.step(torch.zeros(2))
        self.assertAlmostEqual(env.c, 0.20, places=5)          # charges on pad
        env.pos = torch.tensor([0.5, 0.5])
        env.step(torch.zeros(2))
        self.assertAlmostEqual(env.c, 0.198, places=5)         # decays off pad
        env.c = 0.79
        env.pos = env.door.clone()
        _, r, done, _ = env.step(torch.zeros(2))
        self.assertEqual(r, 0.0)                               # door gated below threshold
        env.c = 0.85
        env.pos = env.door - 0.01
        _, r, done, _ = env.step(torch.zeros(2))
        self.assertEqual(r, 1.0)                               # opens at threshold
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
