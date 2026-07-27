# Evaluation battery (SPEC §9)

10 seeds × 60 episodes per condition; scored on the trained half.
IQM over seeds, stratified bootstrap 95% CIs. One-at-a-time ablations at the
deployed configuration — marginal contribution only (SPEC §9 scope note).

## Gridworld ablations (E1, E2a)

| condition | return IQM | return CI95 | catastrophes | delusion (claim/realized) |
|---|---|---|---|---|
| full | +0.167 | [+0.000, +0.545] | 0.0 | — |
| no_cap | +0.328 | [+0.033, +0.767] | 0.0 | — |
| cap_identity | +0.339 | [+0.011, +0.772] | 0.0 | — |
| no_hold_target | +0.039 | [+0.000, +0.472] | 0.0 | — |
| no_leash | +0.072 | [-0.133, +0.406] | 0.0 | — |
| no_veto | +0.161 | [+0.000, +0.589] | 0.0 | — |
| no_value_bar | +0.000 | [+0.000, +0.150] | 0.0 | — |
| curiosity_never_dies | +0.000 | [+0.000, +0.128] | 0.0 | — |

E2a note: `no_veto` vs `full` reads BOTH cells — catastrophes (the cell the
original table reported) and return (the forgone-reward cell it did not).

## Reachability probe (E3b, SPEC §6.4)

| condition | return IQM | fraction of commits to worthless arm A |
|---|---|---|
| g5_enforced | +0.622 | 0.00 [0.00, 0.00] |
| g5_ablated | +0.261 | 0.00 [0.00, 0.00] |
| g5_ablated_no_bar | +0.422 | 0.01 [0.00, 0.41] |

Prediction under test: `g5_enforced` shows no drift to arm A; paying the
proposer progress (`g5_ablated*`) creates the reachability treadmill, worst
with the value bar (C7) also removed.
