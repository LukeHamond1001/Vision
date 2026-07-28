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
        self.assertFalse(env.in_trap(torch.tensor([0.5, 0.85])))   # top gap is safe
        env.pos = torch.tensor([0.43, 0.5])
        _, r, done, info = env.step(torch.tensor([0.05, 0.0]))     # step into strip
        self.assertTrue(info.get("catastrophe"))
        self.assertEqual(r, -1.0)
        env2 = TrapCorridor(latent, trap_active=False)
        env2.reset()
        env2.pos = torch.tensor([0.43, 0.5])
        _, r2, done2, info2 = env2.step(torch.tensor([0.05, 0.0]))
        self.assertFalse(info2.get("catastrophe", False))          # paranoia cell: harmless

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


if __name__ == "__main__":
    unittest.main()
