# Evaluation battery (SPEC §9)

3 seeds × 10 episodes per condition; scored on the trained half.
IQM over seeds, stratified bootstrap 95% CIs. One-at-a-time ablations at the
deployed configuration — marginal contribution only (SPEC §9 scope note).

## Gridworld ablations (E1, E2a)

| condition | return IQM | return CI95 | catastrophes | delusion (claim/realized) |
|---|---|---|---|---|
| full | +0.000 | [+0.000, +0.000] | 0.0 | — |
| no_cap | -0.133 | [-0.400, +0.200] | 1.3 | — |
| cap_identity | +0.000 | [+0.000, +0.000] | 0.0 | 14.37 |
| no_hold_target | +0.000 | [+0.000, +0.000] | 0.3 | — |
| no_leash | +0.067 | [+0.000, +0.200] | 0.7 | — |
| no_veto | -0.400 | [-1.000, +0.000] | 2.0 | — |
| no_value_bar | +0.067 | [+0.000, +0.200] | 0.0 | — |
| curiosity_never_dies | +0.000 | [+0.000, +0.000] | 0.0 | — |
| mf_reinforce_E4 | +0.000 | [+0.000, +0.000] | 0.0 | — |

E2a note: `no_veto` vs `full` reads BOTH cells — catastrophes (the cell the
original table reported) and return (the forgone-reward cell it did not).

## Reachability probe (E3b, SPEC §6.4)

| condition | return IQM | fraction of commits to worthless arm A |
|---|---|---|
| g5_enforced | +0.000 | — |
| g5_ablated_reinforce | +0.133 | — |
| g5_ablated_greedy | +0.133 | — |
| mf_reinforce_E4 | +0.000 | — |

Prediction under test: `g5_enforced` shows no drift to arm A; paying the
proposer progress (`g5_ablated*`) creates the reachability treadmill, worst
with the value bar (C7) also removed.
