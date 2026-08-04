"""DriveWrapper structural tests: the B-harness laws on a scripted env."""

import unittest

from iga.wrapper import DriveWrapper


class ScriptedEnv:
    """Deterministic env: 'level' follows a script; 'count' steps up on
    action 1. Episode ends when the script does."""

    def __init__(self, levels):
        self.levels = list(levels)
        self.t = 0
        self.count = 0.0

    def reset(self):
        self.t = 0
        self.count = 0.0
        return 0

    def step(self, a):
        self.t += 1
        if a == 1:
            self.count += 1.0
        done = self.t >= len(self.levels)
        info = {"level": self.levels[min(self.t, len(self.levels) - 1)],
                "count": self.count}
        return 0, 0.0, done, info


def wrap(levels):
    return DriveWrapper(
        ScriptedEnv(levels),
        channels={"level": lambda o, i: i.get("level", levels[0]),
                  "count": lambda o, i: i.get("count", 0.0)},
        maintain={"level": (5.5, 8.0)},
        frontier=["count"],
        stds={"level": 1.0, "count": 1.0},
        holds=(6, 50), frontier_bonus=1.0)


class TestDriveWrapper(unittest.TestCase):
    def test_telescoping_and_bonus_once(self):
        env = wrap([5, 6, 7, 8, 8, 8, 8, 8])
        env.reset()
        total = 0.0
        for a in (0, 1, 0, 0, 1, 0, 0, 0):
            _, r, done, _ = env.step(a)
            total += r
            if done:
                break
        audit = env.audit()
        self.assertTrue(audit["telescoping_exact"])
        self.assertTrue(audit["mint_bound"])
        # level restore paid (8-5)=3 (w=1); count>=1 climb paid
        # w_stock*(1-0) + bonus 1; count>=2 climb the same; all exact.
        arrivals = [h for h in env.ledger if h["closed_by"] == "arrive"]
        self.assertGreaterEqual(len(arrivals), 2)

    def test_no_pay_across_reset(self):
        env = wrap([5, 5, 5])
        env.reset()
        rs = []
        for a in (0, 0, 0):
            _, r, done, _ = env.step(a)
            rs.append(r)
        # terminal step pays nothing (settle at last-seen state)
        self.assertEqual(rs[-1], 0.0)
        for h in env.ledger:
            self.assertEqual(h["closed_by"], "death")

    def test_oscillation_nets_zero(self):
        env = wrap([5, 7, 5, 7, 5, 7, 5])
        env.reset()
        total = 0.0
        for _ in range(6):
            _, r, done, _ = env.step(0)
            total += r
            if done:
                break
        # 5->7->5 cycles: level-hold pay telescopes to ~0 net
        self.assertAlmostEqual(total, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
