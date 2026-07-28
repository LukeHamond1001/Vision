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


    def test_round10_gradient_candidates_phase_switch(self):
        """The evaluator-gradient pool IS the sequencing mechanism: it must
        point padward at low charge and doorward at high charge."""
        from iga.experiments_v03 import build as build_v03
        from iga.ladder import LadderConfig
        agent, env = build_v03(0, "random_now")
        agent.cfg.grad_proposals = True
        lat = env.latent

        def grad_dir_at(pos, c):
            agent.observe(env.embed_world(pos, c))
            agent.registers[1].commit(lat.band(env.embed_world(pos, min(1.0, c + 0.2)), 1),
                                      window=12)
            agent._write_imagination()
            cands = agent._gradient_candidates(0)
            agent.registers[1].close()
            self.assertTrue(cands)
            step = cands[-1] - lat.band(agent.p, 0)
            return step

        pad_l, door_l = lat.band(env.embed_world(env.pad, 0.2), 0), \
            lat.band(env.embed_world(env.door, 0.9), 0)
        mid = (0.5, 0.55)
        # low charge: step should reduce distance to the pad's fast-band coords
        s_low = grad_dir_at(mid, 0.1)
        here = lat.band(env.embed_world(mid, 0.1), 0)
        self.assertLess(float(torch.linalg.vector_norm(here + s_low - pad_l)),
                        float(torch.linalg.vector_norm(here - pad_l)))
        # high charge: step should reduce distance to the door's fast-band coords
        s_high = grad_dir_at(mid, 0.9)
        here_h = lat.band(env.embed_world(mid, 0.9), 0)
        self.assertLess(float(torch.linalg.vector_norm(here_h + s_high - door_l)),
                        float(torch.linalg.vector_norm(here_h - door_l)))

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
