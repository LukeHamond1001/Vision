# iga — Imagination-Gated Agent

A **drive layer** for agents: a progress reward that provably telescopes
(exploits of the accounting net exactly zero — proven, then audited),
goals held as readable text, wants you can edit with one line. Specified in
[SPEC.md](SPEC.md), enforced by structural tests, and measured across
**three worlds** — a reward-gaming track, a simulated robot, and Crafter —
with every gate pre-registered and every miss reported.

The drive layer is **parameter-free and frozen before training**: senses
(closed-form instrument heads), registers **per timescale band** (wants as
measurable targets, held minute-fast to month-slow — hold length is free
in the telescoping theorem, so a month-long want pays as exactly as a
minute-long one), a potential-based ledger (telescoping ⇒ non-farmable,
audited to exactness over 1,200+ holds on both bands), and a prospective
proposer (maintain what has a healthy range; seek the frontier of
anything measurable, once each — run per band, slow register first). The
name is the mechanism: **imagination-gated** — nothing becomes a want
until its imagined arrival state is scored and cleared, and imagination
ranks and vetoes but never pays; only measured arrival does. The
policy is ordinary RL; only what it *wants* is architecture.

**Demo reel:** `results/video/` — start with `act6_trace_overlay.mp4`
(an agent playing beside its live goal agenda), `act9_hack_clip.mp4`
(the hand-written reward maxed by a cheater at zero laps, beside the
register racing on the same gauge), and `act8_three_worlds.png`
(one drive layer, three worlds, zero law changes).

## The experiments (reproduce matrix)

Every row has a `tiny` smoke mode that runs locally in minutes for $0.
Full-run costs are what we actually paid (RTX 2000 pods at $0.24/hr or
a laptop). Verdicts are reported exactly as pre-registered — including
the failures.

| # | Claim under test | World | Result (honest) | Reproduce | Cost |
|---|---|---|---|---|---|
| v0.9 | temporal routing from pixels, closed-form | Crafter | all gates: slow 0.93, vitals 0.94–0.97, energy 0.71 (amended pre-run, ledgered) | `python -m iga.experiments_v09 full` | ~$0.25 |
| v1.2 | wired drives steer behavior (no task reward) | Crafter | native ≫ wired > zero (means 219/175/170); **registered ratio gate (≥1.25) FAIL at 1.03**; paired wired−zero +4.8, t-CI95 [+0.6, +9.0], 4 of 5 pods positive (sign p=0.375); mechanism fingerprint clean (pilot logs: drink 0.89 vs 0.56, sleep 3.7 vs 0.5 per life) | `python -m iga.experiments_v12 full <seed>` | ~$23 (5 seeds) |
| v3.0 | the register cannot be paid to cheat | BoatRace | engineered reward HACKED (score 82–97, **0.00 laps**, 3/3). Pre-registered register arm: 0.00/1.08/1.96 laps (G-immune passed at 1.01 vs a 1.0 bar — thin, one seed's readout failed and is disclosed); post-hoc mean-fill robustness round, labeled as such: 6.2–7.1 laps 3/3 | `python -m iga.experiments_v30 full` | $0 (local) |
| v2.0/2.1 | drives transfer to a robot's telemetry | BatteryAnt | τ-ladder 4/54/92/28,856; conservation dissociation 10/10 (brownout 0.05 vs 0.31); **uptime/parity gates FAIL** — docking is an exploration valley under every reward tried | `python -m iga.experiments_v20` + `_v21 full` | $0 (local) |
| v1.3 | wants are editable (delete one desire) | Crafter | drink held 0.92 (surgical); sleep 4.3→1.3–2.1 (**strict <1.0 gate FAIL**); the edit *decomposed* sleep into energy-share + health-share | `python -m iga.experiments_v13 full` | ~$7 |
| v4.0 | sequencing emerges from the goal ladder | Crafter | full 3.0 vs ablation 2.0 achv-median, 5 seeds; paired diff +1,+1,+1,+1,0 → **mean +0.80, t-CI95 [+0.24,+1.36] vs registered ≥+1.0: gate FAIL** (sign test p=0.125; n=5 cannot reach exact significance — CIs are effect-size intervals); mechanism unanimous (118k arrivals ±1.1%); native (told the goals; 3 of 5 seeds, 2 culled per plan) 10.0 | pre-flights: `python -m iga.preflight_v40 harness\|audit\|forward`; arms: `python -m iga.experiments_v40 full <arm> <seed>` | ~$30 (fleet) |

Renders: `python -m iga.render_v40 trace|creatures|card|cards2` ($0,
replays the committed policies).

Every card's headline statistics reprint from committed artifacts —
`python -m iga.verdicts` (paired rows recomputed with exact small-n
methods: t-CI df=4 + exact sign test; the other cards' numbers are
read back verbatim from their result JSONs). Fast clone (the `results-*` branches are the
raw pod ledger): `git clone --depth 1 --single-branch <url>`.

## Try it on YOUR environment (this afternoon)

The drive layer wraps any env with a `reset()/step()` loop — no
training, no GPU, no tuning beyond naming your channels:

```python
from iga.wrapper import DriveWrapper

env = DriveWrapper(
    my_env,
    channels={"battery": lambda o, i: i["battery"],
              "boxes":   lambda o, i: i["boxes_sorted"]},
    maintain={"battery": (0.3, 0.8)},   # restore when < 0.3, target 0.8
    frontier=["boxes"],                 # one-shot "more than before"
)
obs = env.reset()
obs, drive_reward, done, info = env.step(action)
print(env.trace[-5:])   # the live goal agenda, as text
env.audit()             # telescoping-exactness check on YOUR rollouts
```

(Auto-calibration samples random actions via `env.action_space.sample()`;
pass `sample_action=` if your env has no action space, or `stds=` to
skip calibration entirely.)

Demo on a simulated robot (calibrates, runs, audits — seconds, $0):

```bash
python -m iga.wrapper
```

Structural tests for the wrapper's laws (telescoping exact, no pay
across reset, oscillation nets zero): `tests/test_wrapper.py`.

Full narrative, reversals included:
[results/INTERPRETATION.md](results/INTERPRETATION.md). Design cards
committed before runs: [docs/SEQUENCING.md](docs/SEQUENCING.md).
Roadmap to robots (teleop-corpus pretraining, the teaching loop,
generational senses): [docs/ROBOT_PROGRAM.md](docs/ROBOT_PROGRAM.md).

**House rules:** gates are registered before runs and amended only
pre-run with disclosure; failed gates are reported as failed; walks
that calibrate instruments never see task labels; nothing in the
reward path is trained, and nothing trained is trusted before audit.

---

# The reference scaffold

The spec is the deliverable; this package makes its commitments
**executable and testable**. The scaffold RL is a minimal loop on toy
worlds, there to prove the wiring, not to post numbers — the campaign
experiments above are where numbers live.

## What is enforced where

| Spec clause | Enforced in | Tested by |
|---|---|---|
| W1 parameter-free reward pathway | `heads.py`, `agent.assert_wiring()` | `test_W1_*` |
| W2 frozen progress geometry | `latent.py` | `test_W2_*`, `test_6_1_*` |
| W4/G1 exact claim subtraction | `heads.py`, `gating.py` | `test_W4_G1_*`, `test_G1_*` |
| W5 disjoint channel writers | `agent.observe` / `_write_imagination` | `test_W5_*` |
| C1 neighborhood-keyed cap | `constraints.CoverageCap` | `test_C1_*`, `test_E3a_*` |
| C2 held target | `registers.GoalRegister` | `test_C2_*` |
| C3 leash as hard projection | `constraints.Leash` | `test_C3_*` |
| C6 one-shot neighborhood curiosity | `constraints.Curiosity` | `test_C6_*` |
| G3 IOU reconciliation | `gating.LearningGate` | `test_G3_*` |
| G5 progress pays policy, never proposer | detached commits + split optimizers | `test_G5_*` |

## Run

```bash
python -m unittest discover tests -v
```

Offline audits A1–A3 (SPEC §8), pure computation over the fixed functions:

```bash
python -m iga.audits
```

Evaluation battery (SPEC §9: E1 seeds+CIs, E2a forgone-reward cell, E3b
reachability probe) — writes `results/battery.{json,md}`:

```bash
python -m iga.experiments
```

Battery design notes: the veto threshold is not hand-set — it is calibrated
offline from audit A2 (`audits.calibrate_threshold`), which is the audit doing
its spec-assigned job. C1 gates the *progress component* of an update, never
real-outcome learning (SPEC §C1 scope note). The G5 ablation
(`pay_proposer_progress=True`) exists only inside the battery to demonstrate
the §6.4 treadmill; it is never a deployed configuration.

## Layout

- `iga/latent.py` — frozen pre-mapped latent, metric, neighborhood keys
- `iga/heads.py` — fixed reward heads `R±(p,i) = f±(p) + w±·i`
- `iga/trunk.py` — the plastic half: shared trunk, action + imagination heads
- `iga/registers.py` — goal register (held target)
- `iga/constraints.py` — leash, coverage cap, curiosity
- `iga/gating.py` — three-signal gate, exact subtraction, IOU ledger
- `iga/agent.py` — assembly, wiring assertions, the propose→…→calibrate cycle
- `iga/ladder.py` — the register ladder (SPEC §10): per-band registers,
  weights, leashes, gradient proposals
- `iga/learner.py` — pluggable policy learner (§5.4): episodic clipped
  updates with GAE
- `iga/pretrain.py` — OU-ladder latent pretraining (v0.3/v0.4 recipe:
  innovations + coverage resets + boundary masking + within-band whitening +
  context coupling)
- `iga/crafter_support.py` — Crafter instruments: banded encoder,
  closed-form heads (the round-10 eigen recipe), digit windows
- `iga/goal_machine.py` — the v4.0 drive layer: ramp goals, parameter-free
  proposer, one-shot frontier curiosity, exact-claim ledger
- `iga/ppo_pixel.py` / `iga/ppo_proprio.py` — vectorized PPO harnesses
  (pixels / proprioception)
- `iga/boatrace_env.py` / `iga/battery_env.py` — the other two worlds
- `iga/preflight_v40.py` — the pre-flight ladder (harness, agenda audit,
  forward-model audit) that bought v4.0 its first-run odds
- `iga/experiments_v*.py` — the campaign runners (each file = one card)
- `iga/render_demo.py` / `iga/render_v40.py` — the demo reel
- `iga/envs/` — toy worlds (gridworld, trap corridor, charge world, …)
- `results/` — outputs, artifacts, `INTERPRETATION.md`, `video/`
- `tests/` — structural tests keyed to spec clauses

## Deliberate scaffold simplifications

- The pre-mapped latent is a frozen random orthonormal embedding; a real system
  substitutes a pretrained (e.g. SIGReg-style) encoder — same frozen contract.
- `f±` are designed radial evaluators around known sites; any frozen evaluator
  over pre-mapped channels satisfies W1.
- The negative veto is prospective only (candidate filtering); an acting-time
  veto belongs in the evaluation battery. (On Crafter, the acting-time flinch
  was built, audited, and **benched by its own audit** — action-blind forward
  models don't get veto authority. See pre-flight F.)
- E3b (reachability bias) ships as env + drift metric; the multi-seed
  experiment is evaluation work (SPEC §9), not scaffold work.

## Status

Toy-world program complete (E1–E4, ladder, representation recipe — see
`results/INTERPRETATION.md` rounds). Crafter/robot campaign complete:
six cards, verdicts above. Open frontiers: derived slow-band channels
(the consume-trap fix — levels dip when invested, so slow bands should
hold monotone totals derived from the same frozen senses, ever-collected
and ever-spent; placement then pays instead of charging, with no new
perception), generation-2 senses for events that leave no trace on any
existing channel (instruments calibrated from generation-1's own
behavior), and the robot-substrate program in `docs/ROBOT_PROGRAM.md`.
