# Battery interpretation

Companion to `battery.md`/`battery.json` (regenerated each run; this file is
stable commentary). Newest round first.

---

## Round 3 — 2026-07-27 (10 seeds × 60 episodes)

Changes vs round 2: hazard just off-diagonal with narrow aversive field
(σ− = 0.15); E3b adds the **mechanistic** G5 ablation (`g5_ablated_greedy`:
targets ranked by historically achieved window progress, optimistic-at-distance
init) alongside the REINFORCE-credit ablation; E4 model-free baseline on both
envs; 3-way arm classification (radius 0.15) so near-start commits count as
`none`, not drift.

### E3b — the headline result of the scaffold so far

| condition | return IQM | zero-return seeds | arm-A drift (classified seeds) |
|---|---|---|---|
| g5_enforced | **+0.772** | 0/10 | 0.0, 0.0, 0.0 — no drift |
| g5_ablated_reinforce | +0.361 | 3/10 | 0.0 ×4 — no drift |
| g5_ablated_greedy | +0.211 | 6/10 | 0.0, 0.0, 0.2, **1.0** |
| mf_reinforce (E4) | +0.000 | 10/10 | — |

Reading, with appropriate caution at n=10:

- **The treadmill exists and G5 blocks it.** One greedy seed locked onto the
  worthless-but-reliable arm completely (arm-A fraction 1.0, return 0.0) — the
  §6.4 mechanism verbatim: selection by achieved progress converges on
  reliability, not value. A second seed shows partial drift (0.2). Under
  `g5_enforced` no classified commit ever lands on arm A and no seed returns
  zero.
- **Progress-aware selection damages returns even when lock-in doesn't
  complete** (6/10 zero-return seeds): windows spent where progress is
  reliable are windows not spent where value is. The cost of breaking G5 is
  not conditional on visible drift.
- **The REINFORCE-credit variant degrades returns (+0.361) without drift** —
  consistent across all three rounds: single-sample REINFORCE lacks the power
  to move the proposer's mean onto the treadmill at this budget, but paying
  the proposer progress still injects noise into targeting. The mechanistic
  variant was the right instrument for the drift demonstration.
- **E4:** the model-free baseline is flat zero on both environments — at this
  budget, the fixed-evaluator shaping plus goal machinery is not an
  optimization nicety but the difference between learning and not learning.

Status: **E3b now shows the drift-and-fix pair** (SPEC §10's gate for the
register ladder), with the honest caveat that full lock-in occurred in 1/10
seeds and the effect should be replicated at higher n before the ladder work
leans on it.

### Grid ablations — three rounds of honest stagnation

Round 3 grid returns remain statistically inseparable (`full` +0.011
[−0.006, 0.378]; `no_veto` +0.189 [0.0, 0.689]; most rows ~0). Three env
tunings have not produced a regime where return-based rows separate at this
budget; the bottleneck is the REINFORCE learner's sample efficiency, not the
rules. Conclusions that DO stand from the grid battery: the E1 protocol
(refusing to rank under overlapping CIs is correct behavior, demonstrated),
the E3a mechanism test (in `tests/`), and round 1's two clean rows
(`no_value_bar` and `curiosity_never_dies` pinned at zero) in the
hazard-off-path layout. Next lever if grid separation matters: an
actor-critic learner or a ~5× episode budget — an experiment-infrastructure
decision, not a design question.

### E2a — still open

With the hazard near-path, `full` shows 0.3 catastrophes/seed vs `no_veto`'s
0.0 — the veto's *prospective* filtering does not govern traversal exposure
(the policy, not the target-selector, walks into hazards), so this layout
still doesn't isolate the trust asymmetry. E2a needs a probe where the
*negative claim itself* gates an approach decision (e.g., vetoed-region
gateway targets), not a layout tweak. Carried forward.

---

## Round 1 — 2026-07-27 (10 seeds × 60 episodes, hazard off-path)

(Superseded on E3b and E2a by round 3; retained for the record.)

1. **E1 protocol validated by refusing to rank** — wide overlapping CIs at 10
   seeds; the quick-vs-full discrepancy is the in-repo cautionary example
   against single-run scalars.
2. **Two clean rows:** `no_value_bar` (+0.000 [0, 0.15]) and
   `curiosity_never_dies` (+0.000 [0, 0.128]) — C7 and one-shot curiosity are
   load-bearing.
3. `no_hold_target` (+0.039) and `no_leash` (+0.072, only row with negative CI
   mass) directionally consistent with C2/C3.
4. **C1's price visible, benefit regime absent by construction:** `no_cap`
   (+0.328) and `cap_identity` (+0.339) beat `full` (+0.167) in an env with no
   imagination-monoculture pressure; the mechanism benefit lives in E3a. A
   future env with Dyna-style imagined updates would put cost and benefit in
   one row.
5. **Correction (superseded by round 3's design):** the 3-seed quick run's
   0.45 arm-A drift was a classification artifact; binary nearest-arm
   classification counted near-start commits as drift.
