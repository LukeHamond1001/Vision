"""DriveWrapper structural tests: the B-harness laws on a scripted env.
Extractors deliberately read info keys WITHOUT .get defaults — pinning
the reset-seeding path (reset() must not fabricate measurements)."""

import unittest

from iga.wrapper import DriveWrapper


class ScriptedEnv:
    """Deterministic env: 'level' follows a script; 'count' steps up on
    action 1. Episode ends when the script does; reset restarts it."""

    def __init__(self, levels):
        self.levels = list(levels)
        self.t = 0
        self.count = 0.0

    def reset(self):
        self.t = 0
        self.count = 0.0
        return 0                      # obs only; NO info at reset

    def step(self, a):
        if a == 1:
            self.count += 1.0
        done = self.t >= len(self.levels) - 1
        info = {"level": self.levels[self.t], "count": self.count}
        self.t += 1
        return 0, 0.0, done, info


def wrap(levels, **kw):
    return DriveWrapper(
        ScriptedEnv(levels),
        channels={"level": lambda o, i: i["level"],     # no .get: reset
                  "count": lambda o, i: i["count"]},    # must not call
        maintain={"level": (5.5, 8.0)},
        frontier=["count"],
        stds={"level": 1.0, "count": 1.0},
        holds=(6, 50), frontier_bonus=1.0, **kw)


class Gym5TupleEnv(ScriptedEnv):
    """Gymnasium API: reset -> (obs, info); step -> 5-tuple."""

    def reset(self):
        super().reset()
        return 0, {}

    def step(self, a):
        obs, r, done, info = super().step(a)
        return obs, r, done, False, info


class TestDriveWrapper(unittest.TestCase):
    def test_reset_defers_seeding_to_first_step(self):
        # extractors read info[...] with no default: if reset() measured,
        # this would KeyError (the shipped-quickstart bug this pins)
        env = wrap([5, 6, 7, 8, 8, 8])
        env.reset()
        _, r0, _, _ = env.step(0)     # seeding step pays nothing
        self.assertEqual(r0, 0.0)

    def test_telescoping_and_bonus_paid_once(self):
        env = wrap([5, 5, 5, 5, 5, 5, 5, 5])
        env.reset()
        rewards = []
        for a in (0, 1, 0, 0, 0, 0, 0):     # one count climb: 0 -> 1
            _, r, done, _ = env.step(a)
            rewards.append(r)
            if done:
                break
        audit = env.audit()
        self.assertTrue(audit["telescoping_exact"])
        self.assertTrue(audit["mint_bound"])
        arr = [h for h in env.ledger
               if h["closed_by"] == "arrive" and h["channel"] == "count"]
        self.assertEqual(len(arr), 1)
        # count>=1 climb pays w_stock*(1-0) once, plus bonus 1.0 once;
        # the level hold never moves (constant script) so it nets 0:
        w_stock = 50 / 6
        self.assertAlmostEqual(sum(rewards), w_stock * 1.0 + 1.0,
                               places=5)
        self.assertEqual(env.machine.total_bonus, 1.0)

    def test_no_pay_across_reset_two_episodes(self):
        env = wrap([5, 5, 5, 5])
        env.reset()
        ep1 = []
        for _ in range(4):
            _, r, done, _ = env.step(0)
            ep1.append(r)
            if done:
                break
        self.assertEqual(ep1[-1], 0.0)          # dying transition pays 0
        for h in env.ledger:
            self.assertEqual(h["closed_by"], "death")
        env.reset()
        _, r_seed, _, _ = env.step(0)           # new life seeds, pays 0
        self.assertEqual(r_seed, 0.0)
        # bonus one-shot state cleared per life; ledger settled cleanly
        self.assertEqual(env.machine.arrived_cells, set())

    def test_oscillation_nets_zero(self):
        env = wrap([5, 7, 5, 7, 5, 7, 5, 5])
        env.reset()
        env.step(0)                             # seed at level 5
        total = 0.0
        for _ in range(6):                      # 7,5,7,5,7,5 cycles
            _, r, done, _ = env.step(0)
            total += r
            if done:
                break
        self.assertAlmostEqual(total, 0.0, places=6)

    def test_calibrate_handles_gymnasium_5tuple(self):
        # stds=None forces the calibration rollout; a gymnasium env
        # (5-tuple step) must not crash it (the 4-tuple-unpack bug
        # this pins — step() handled both APIs, _calibrate didn't)
        env = DriveWrapper(
            Gym5TupleEnv([5, 6, 7, 8]),
            channels={"level": lambda o, i: i["level"],
                      "count": lambda o, i: i["count"]},
            maintain={"level": (5.5, 8.0)}, frontier=["count"],
            holds=(6, 50), sample_action=lambda: 0, calibrate_steps=12)
        env.reset()
        _, r0, _, info = env.step(0)
        self.assertEqual(r0, 0.0)
        self.assertIn("drive_events", info)

    def test_unit_free_eps_small_scale_channel(self):
        # a [0,1]-scale channel must still commit restores (the
        # Crafter-unit-eps bug this pins): battery drains 0.8 -> 0.4,
        # maintain (0.5, 0.8); with std=0.2 the eps is 0.07, so the
        # restore MUST commit once battery < 0.5
        levels = [0.8, 0.7, 0.6, 0.45, 0.4, 0.4, 0.4, 0.4]
        env = DriveWrapper(
            ScriptedEnv(levels),
            channels={"battery": lambda o, i: i["level"]},
            maintain={"battery": (0.5, 0.8)},
            stds={"battery": 0.2},
            holds=(6, 50))
        env.reset()
        for _ in range(7):
            _, _, done, _ = env.step(0)
            if done:
                break
        commits = [tr for tr in env.trace if tr[1] == "commit"
                   and tr[3] == "battery"]
        self.assertGreaterEqual(len(commits), 1)


if __name__ == "__main__":
    unittest.main()
