# Battery interpretation — run of 2026-07-27 (10 seeds × 60 episodes)

Companion to `battery.md`/`battery.json` (regenerated each run; this file is
stable commentary on the run above).

## Headline reads

1. **The E1 protocol is doing its job, which mostly means refusing to rank.**
   With IQM + bootstrap CIs at 10 seeds, nearly all gridworld return rows have
   overlapping intervals (`full` = +0.167 [0.0, 0.545]). The spec's own
   position — small-delta ablations cannot be ranked without seeds and CIs —
   applies to this scaffold's learner with force: directions below are
   *suggestive*, not established. Anyone tempted to quote single-run scalars
   from a battery like this now has the counterexample in-repo: the quick run
   (3 seeds) showed `full` at +0.133 and the full run shows how wide the truth
   is.

2. **Two rules are clearly load-bearing even at this noise level.**
   `no_value_bar` (+0.000 [0.0, 0.15]) and `curiosity_never_dies`
   (+0.000 [0.0, 0.128]) pin at zero with tight upper bounds: without C7 the
   agent commits to nearby low-value targets and never seeks the prize;
   with a farmable novelty bonus it dithers for bonus instead of progressing.
   These are the two cleanest results in the battery.

3. **Directionally consistent, pending seeds:** `no_hold_target` (+0.039) and
   `no_leash` (+0.072, the only row with negative CI mass) underperform
   `full`, as C2/C3 predict.

4. **C1's price is visible, and its benefit regime is absent by construction.**
   `no_cap` (+0.328) and `cap_identity` (+0.339) both *outperform* `full`
   (+0.167) on return here. This is the honest cost of gating the progress
   component in a single-target dense environment that contains no
   imagination-monoculture pressure — the failure C1 exists to prevent cannot
   occur in this grid, so the battery measures only the rule's cost. The
   mechanism-level benefit is covered by probe E3a (ε-ball attack: 64 distinct
   identities, neighborhood cap admits 2), not by this return table. A future
   battery env should include Dyna-style imagined updates so C1's benefit and
   cost appear in the same row.

5. **E2a is uninformative in this layout — battery-design gap.** Catastrophes
   are 0.0 in *every* row including `no_veto`: the hazard sits off the
   start→reward path, so trust-vs-verify is never exercised. The calibrated
   veto also costs ≈nothing (+0.161 vs +0.167) — good, but weak evidence. Fix
   for the next run: place the hazard between start and reward so both
   confusion-matrix cells are live.

6. **CORRECTION — the E3b treadmill drift did not replicate at scale.**
   The quick run's arm-A fraction of 0.45 under `g5_ablated_no_bar` was
   small-sample noise (3 seeds; near-start commits classify ~arbitrarily
   between equidistant arms). At 10×60: `g5_enforced` 0.00, `g5_ablated` 0.00,
   `g5_ablated_no_bar` 0.01. What *did* replicate is the cost of breaking G5:
   return drops from +0.622 (enforced) to +0.261/+0.422 (ablated) — paying the
   proposer progress degrades target quality even without visible drift.
   Why no drift: the single-sample REINFORCE on the proposer gets ~150
   gradient steps against σ=0.2 proposal noise, and claim-ranked selection
   still favors arm B while the bar is off. The §6.4 mechanism is not
   falsified — the probe is underpowered for it. Strengthening options, in
   order of preference: (a) longer training / higher proposer lr in the
   ablated arm only, (b) move arm A nearer than arm B so reachability bias has
   a gradient to exploit, (c) rank candidates by claim+progress-estimate in
   the ablated arm (the fully broken selection rule §6.4 actually describes).

## Status vs SPEC §9

- E1: protocol implemented and validated (its refusal to rank is the
  validation). ✔ mechanics / ✘ conclusive rankings at this budget.
- E2a: cell implemented; environment must be redesigned to exercise it. ✘
- E3a: passes as a structural test (in `tests/`). ✔
- E3b: cost-of-breaking-G5 shown; drift undemonstrated — probe underpowered. ◐
- E4 (external baseline): not yet run.
