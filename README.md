# iga — Imagination-Gated Agent

Reference scaffold for the architecture specified in [SPEC.md](SPEC.md). The
spec is the deliverable; this package makes its commitments **executable and
testable**. It is deliberately *not* a performing agent — the RL is a minimal
REINFORCE loop on a toy gridworld, there to prove the wiring, not to post
numbers.

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
- `iga/envs/` — toy gridworld + adversarial probes (E3a implemented, E3b env)
- `iga/audits.py` — A1 trigger set, A2 proxy gap + threshold calibration, A3 Lipschitz
- `iga/experiments.py` — the SPEC §9 battery (ablations, seeds, IQM + bootstrap CIs)
- `results/` — battery outputs (json + markdown report)
- `tests/` — structural tests keyed to spec clauses

## Deliberate scaffold simplifications

- The pre-mapped latent is a frozen random orthonormal embedding; a real system
  substitutes a pretrained (e.g. SIGReg-style) encoder — same frozen contract.
- `f±` are designed radial evaluators around known sites; any frozen evaluator
  over pre-mapped channels satisfies W1.
- The negative veto is prospective only (candidate filtering); an acting-time
  veto belongs in the evaluation battery.
- E3b (reachability bias) ships as env + drift metric; the multi-seed
  experiment is evaluation work (SPEC §9), not scaffold work.

## Next (SPEC §9)

E1 seeds/CIs on small-delta ablations · E2 missing cells (forgone reward;
named ceiling baseline) · E3 probes at scale · E4 external baseline — then the
register ladder (SPEC §10), gated on E3b passing.
