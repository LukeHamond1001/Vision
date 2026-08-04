"""DriveWrapper: the frozen drive layer over ANY environment.

    from iga.wrapper import DriveWrapper

    env = DriveWrapper(
        my_env,
        channels={"battery": lambda o, i: i["battery"],
                  "temp":    lambda o, i: i["temp"],
                  "boxes":   lambda o, i: i["boxes_sorted"]},
        maintain={"battery": (0.3, 0.8),     # restore when < lo, target hi
                  "temp":    (-0.2, 0.05)},  # keep-low: see note below
        frontier=["boxes"],                  # one-shot 'more than before'
    )
    obs = env.reset()
    obs, drive_reward, done, info = env.step(action)
    env.trace       # the live goal agenda (readable text)
    env.audit()     # telescoping-exactness check over the ledger

What it gives your agent: registers (wants held as measurable targets),
a prospective proposer (maintain what has a healthy range; seek the
frontier of anything countable, once each), and potential-based
progress pay that provably cannot be farmed (telescoping claims —
audited to exactness; run env.audit() on your own rollouts).

Keep-low channels: the machinery is "at least t" ramps. To keep a
quantity LOW, wire the negated value: channels={"cool": lambda o, i:
-i["temp"]}, maintain={"cool": (-0.2, -0.05)}.

Freeze discipline (the whole point): stds are calibrated ONCE, before
training, from a random rollout (or passed explicitly) and then frozen.
The drive layer holds no trainable parameters and never adapts under
the learner — that is what makes it unfarmable in practice, not just
in the theorem.
"""

from __future__ import annotations

import torch

from .goal_machine import GMConfig, GoalMachine


class DriveWrapper:
    def __init__(self, env, channels, maintain=None, frontier=None,
                 stds=None, holds=(60, 180), arrive_eps_frac=(0.35, 0.25),
                 frontier_bonus=1.0, horizon_stock=2.0,
                 calibrate_steps=2000, sample_action=None, danger=None):
        """channels: {name: extractor(obs, info) -> float} (ordered).
        maintain: {name: (restore_lo, target_hi)} — the vitals band.
        frontier: [names] — the stock band (one-shot 'more than before').
        stds: {name: float} to skip calibration (freeze-before-use is
        YOUR responsibility then). arrive_eps_frac: per-band arrival
        tolerance as a FRACTION of each channel's calibrated std —
        unit-free, so [0,1] batteries and 0-9 gauges both work
        untouched. danger: {name: floor} — channels the prospective
        veto watches (imagined states below floor are never committed
        to)."""
        maintain = maintain or {}
        frontier = list(frontier or [])
        assert channels, "at least one channel"
        if isinstance(channels, (list, tuple)):
            channels = {k: (lambda o, i, k=k: float(i[k])) for k in channels}
        for k in list(maintain) + frontier:
            assert k in channels, f"'{k}' not in channels"
        names = [k for k in channels if k in maintain] + \
                [k for k in channels if k in frontier]
        self._extract = [channels[k] for k in names]
        self.names = tuple(names)
        vit_idx = tuple(i for i, k in enumerate(names) if k in maintain)
        stk_idx = tuple(i for i, k in enumerate(names) if k in frontier)
        restore = {i: maintain[names[i]] for i in vit_idx}
        self.env = env
        self._danger = {names.index(k): v for k, v in (danger or {}).items()}

        if stds is None:
            stds = self._calibrate(calibrate_steps, sample_action)
        self._stds = tuple(max(float(stds[k]), 1e-6) for k in names)

        def fp(m):
            if not vit_idx:
                return 0.0
            tot = 0.0
            for i in vit_idx:
                lo, hi = restore[i]
                tot += max(0.0, min(1.0, (float(m[i]) - lo) / max(hi - lo, 1e-9)))
            return tot / len(vit_idx)

        def fn(m):
            bad = 0.0
            for i, floor in self._danger.items():
                bad += max(0.0, floor - float(m[i]))
            return bad

        band_of = lambda i: 0 if i in vit_idx else 1
        eps_pc = tuple(arrive_eps_frac[band_of(i)] * self._stds[i]
                       for i in range(len(self.names)))
        self.cfg = GMConfig(
            channels=self.names, vitals=vit_idx, stocks=stk_idx,
            goal_vitals=vit_idx, restore=restore,
            f_pos_fn=fp, f_neg_fn=fn,
            stds=self._stds, holds=holds,
            arrive_eps_per_channel=eps_pc,
            frontier_bonus=frontier_bonus, horizon_stock=horizon_stock,
            w_bands=(1.0, holds[1] / holds[0]))
        self.machine = GoalMachine(self.cfg)
        self._last_obs = None
        self._pending_seed = True

    # ------------------------------------------------------------ internals
    def _measure(self, obs, info) -> torch.Tensor:
        return torch.tensor([float(f(obs, info)) for f in self._extract])

    def _calibrate(self, steps: int, sample_action):
        """Random rollout to freeze per-channel stds. Consumes the env
        pre-training (standard); prints what it froze."""
        sample = sample_action or (lambda: self.env.action_space.sample())
        self.env.reset()
        vals = [[] for _ in self.names]
        for _ in range(steps):
            out = self.env.step(sample())
            if len(out) == 5:                             # gymnasium compat
                obs, _, term, trunc, info = out
                done = bool(term or trunc)
            else:
                obs, _, done, info = out
            m = self._measure(obs, info)
            for i in range(len(self.names)):
                vals[i].append(float(m[i]))
            if done:
                self.env.reset()
        stds = {k: (torch.tensor(v).std().item() or 1.0)
                for k, v in zip(self.names, vals)}
        print(f"[DriveWrapper] calibrated+frozen stds over {steps} random "
              f"steps: " + ", ".join(f"{k}={v:.4g}" for k, v in stds.items()),
              flush=True)
        return stds

    # ------------------------------------------------------------ env API
    def reset(self):
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out   # gymnasium compat
        self._last_obs = obs
        # No measurement here: extractors typically read `info`, which
        # does not exist at reset. The machine seeds from the FIRST
        # step's measurement (which pays nothing) — the same boundary
        # rule the campaign harnesses use.
        self._pending_seed = True
        return obs

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:                                 # gymnasium compat
            obs, native_r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            obs, native_r, done, info = out
        m = self._measure(obs, info)
        if self._pending_seed:
            self.machine.reset(m)
            self._pending_seed = False
            r, ev = 0.0, {"arrivals": [], "timeouts": [], "bonus": 0.0}
        elif done:
            # v1.2 boundary rule: the dying transition pays nothing;
            # claims settle at the last-seen state. Call reset() next.
            r, ev = self.machine.step(self.machine.m.clone(), done=True)
        else:
            r, ev = self.machine.step(m)
        info = dict(info)
        info["native_reward"] = native_r
        info["drive_events"] = ev
        self._last_obs = obs
        return obs, r, done, info

    # ------------------------------------------------------------ surfaces
    @property
    def trace(self):
        return self.machine.trace

    @property
    def ledger(self):
        return self.machine.ledger

    def audit(self, tol: float = 1e-5) -> dict:
        """The B-harness exactness check on YOUR rollouts: every closed
        hold must have paid exactly w*(phi_commit - phi_close), and the
        running pay must never exceed w*phi_commit (no minting)."""
        led = self.machine.ledger
        exact = all(abs(h["paid"] - h["w"] * (h["phi_commit"] - h["phi_close"]))
                    < tol for h in led)
        bound = all(h["max_prefix"] <= h["w"] * h["phi_commit"] + tol
                    for h in led)
        out = {"holds": len(led), "telescoping_exact": exact,
               "mint_bound": bound}
        print(f"[DriveWrapper.audit] {out}", flush=True)
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


def _demo():
    """$0 receipt: wrap the BatteryAnt robot, run random torques, show
    the agenda + the exactness audit."""
    import numpy as np
    from .battery_env import BatteryAnt

    class _Gym:
        # tiny adapter: BatteryAnt already returns (obs, r, done, info)
        def __init__(self):
            self.e = BatteryAnt(seed=3)
            self.rng = np.random.default_rng(0)

        def reset(self):
            return self.e.reset()

        def step(self, a):
            return self.e.step(a)

        def sample(self):
            return self.rng.uniform(-1, 1, self.e.act_dim)

    genv = _Gym()
    env = DriveWrapper(
        genv,
        channels={"battery": lambda o, i: i.get("battery", 1.0),
                  "cool": lambda o, i: -i.get("temp", 0.0),
                  "docked": lambda o, i: float(i.get("docked_steps", 0))},
        maintain={"battery": (0.30, 0.80), "cool": (-0.10, -0.02)},
        frontier=["docked"],
        sample_action=genv.sample,
        calibrate_steps=1500)
    obs = env.reset()
    for t in range(3000):
        obs, r, done, info = env.step(genv.sample())
        if done:
            obs = env.reset()
    print("[demo] trace tail:")
    for ev in env.trace[-8:]:
        print("   ", ev)
    env.audit()


if __name__ == "__main__":
    _demo()
