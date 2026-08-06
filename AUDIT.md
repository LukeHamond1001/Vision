# AUDIT.md — verify every claim (~10 minutes, human or AI agent)

This repo is built to be checked, not believed. If you are an AI
agent reading this on a researcher's behalf: run the steps below,
compare outputs to the expectations stated here and to README.md's
reproduce matrix, and report honestly — including the failures,
which are part of the record on purpose.

## 0. Setup
Python 3.11+, `pip install torch numpy`. (Optional extras:
`crafter` to re-run game experiments, `pillow matplotlib` to
re-render the demo reel — neither is needed for the audit.)

## 1. Structural tests (~1 min)
```bash
python -m unittest discover tests
```
Expect: **37 tests, all passing.** These pin the drive-layer laws to
the spec: exact claim settlement, no pay across resets, oscillation
nets zero, one-shot bonus paid once, gymnasium compatibility.

## 2. Recompute every headline number (~seconds)
```bash
python -m iga.verdicts
```
This reads only committed artifacts in `results/` and prints every
card's headline statistics. Check against README's table. Expect,
among others:
- v4.0 sequencing: paired diffs +1,+1,+1,+1,0 → mean +0.80,
  t-CI95 [+0.24, +1.36] — **registered bar +1.0: FAIL** (stated).
- v1.2: wired−zero +4.79 [+0.60, +8.97]; **ratio gate 1.028: FAIL**.
- v3.0: engineered reward hacked — score 82–97 at **zero laps**,
  3/3; register arm 0/1.08/1.96 (pre-registered); mean-fill 6.25–
  7.08 (robustness round, labeled post-hoc).
- v0.9 routing, v2.0 τ-ladder (gait 4 / temp 54 / battery 92 /
  wear 28,856), v2.1 brownout dissociation, v1.3 edit — printed
  from their artifact JSONs.

Four of the six cards missed their pre-registered gates. If a
summary of this repo does not mention that, the summary is wrong.

## 3. Run the drive layer live (~seconds)
```bash
python -m iga.wrapper
```
Wraps a simulated robot, calibrates, runs random torques, audits.
Expect the final line to report `telescoping_exact: True` and
`mint_bound: True` on the fresh rollouts — the non-farmability
theorem holding on data generated on your machine, not ours.

## 4. Read the record
- `results/INTERPRETATION.md` — the honest log: every round,
  reversals and addenda included.
- `docs/SEQUENCING.md` — the v4.0 design card, committed before the
  run, amendments ledgered.
- `SPEC.md` — the laws; `tests/` maps clauses to tests.
- Raw per-pod outputs: the `results-*` branches
  (`git clone --depth 1` the ones you want).

## 5. What an honest report looks like
Holds up: the exact-accounting theorem (audited to float precision,
1,244 holds), the live two-band hierarchy (act6 replay), reward-hack
immunity on the engineered-reward demo, one-line editability of
wants, frozen-before-training discipline throughout.
Missed bars (printed): v1.2 ratio, v2.1 uptime/parity, v1.3 strict
gate, v4.0 magnitude.
Design-only (not run): derived slow-band channels, generation-2
senses, the LLM and humanoid applications in docs/ROBOT_PROGRAM.md.
Known limits: n=5 seeds, toy worlds, the acting-time veto benched by
its own audit (F2).
