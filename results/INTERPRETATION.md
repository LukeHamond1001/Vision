# Battery interpretation

Companion to `battery.md`/`battery.json` (regenerated each run; this file is
stable commentary). Newest round first.

---

## v0.4 round 1 — 2026-07-28: the deliberate objective works; hand-design
## still edges it at toy scale

Objective changes (from the round-2 mechanism identification): within-band
whitening only (full whitening penalized the proven ingredient) + a context
term targeting the fast band's relative charge-response (ratio form —
normalization-invariant; the absolute form under-shot 3× through scale
normalization and was caught in the routing report before any run).

`v04_objective.json` (12 × 150 each): random_now 0.757 [0.743, 0.774];
random_leakin 0.797 [0.788, 0.806]; **pretrained_v04 0.781 [0.764, 0.797]**.

- The deliberate objective RECOVERS the representation win (clear of the
  anchor's IQM, matches the accidental round-1 latent's 0.787) — the
  coupling can be learned on purpose. The recipe is now complete and
  reproducible: OU-ladder innovations + coverage resets + boundary masking
  + within-band whitening + ratio-form context coupling.
- It does NOT beat hand injection (0.781 vs 0.797, overlapping-at-the-edge
  CIs, IQM below). Consistent with round 2: the leak is the whole story at
  this scale, and learned routing's residual imperfection (fast-band c-corr
  0.93 vs the injection's clean structure) only adds metric noise.
- **Standing recommendation:** at toy scale, hand-design the banded latent
  (blocks + explicit context projections). The learned objective's value
  claim lives where hand-design is impossible — richer observations, unknown
  slow variables — which is the honest scope for the next scale-up, not
  something this testbed can settle.

Still open (unchanged by v0.3–v0.4): the charge→door last mile — returns
0.000 in every configuration ever run; goal sequencing / termination handoff
remains the control-side program.

---

## v0.3 round 2 — 2026-07-28: mechanism identified, both directions

Leak ablation (`v03_leak_ablation.json`, 12 × 150 each, pre-registered):

| condition | max_c IQM [CI95] | reading |
|---|---|---|
| random_now | 0.757 [0.743, 0.774] | anchor (replicates round 1) |
| random_leakin | **0.797 [0.788, 0.806]** | **leak SUFFICIENT** — synthetic 0.57-amp charge projection into the fast band reproduces and exceeds the full win on the untrained latent |
| pretrained_now | 0.787 [0.780, 0.795] | anchor (replicates round 1) |
| pretrained_leakfree | **0.744 [0.728, 0.757]** | **leak NECESSARY** — c-blind fast band collapses the win to the random anchor |

The v0.3 representation win reduces entirely to one interpretable
ingredient: a low-amplitude copy of the slow variable in the fast band's
coordinates. The learned geometry contributes nothing beyond delivering it
(leakfree ≈ random; hand-injected leak ≥ learned leak). Consequences:

1. **Clean band separation was the wrong routing target.** L1's slice
   structure and every safety property stand unchanged (writers, leashes,
   holds, metrics are slice-level); what falls is the aesthetic that band k
   should carry ONLY its timescale's variables. The winning representation
   entangles slow context into fast distances — making fast targets, fast
   progress, and prospective evaluation slow-aware with no controller
   change. SPEC L1 amended: **cross-band value coupling belongs in the
   metric, not the selector.**
2. **The OU-ladder objective succeeded despite itself**: covariance
   whitening penalizes exactly the cross-band covariance that constitutes
   the leak — the ingredient survived training rather than being produced
   by it. A v0.4 objective should make the slow→fast context projection a
   deliberate term. The slow band's purity (position-blind) held in every
   winning configuration and should be kept.
3. Still standing: returns 0.000 in every condition ever run — the
   charge→door handoff (goal sequencing) remains the open control-side
   item, untouched by representation changes.

---

## v0.3 round 1 — 2026-07-28: the pretrained latent wins, not how predicted

OU-ladder pretraining (the session-opening proposal, weak/sketched form:
per-band innovation losses at prescribed timescales + covariance whitening,
linear encoder, frozen). Two pretraining pathologies found and fixed by
measurement en route: coverage collapse (random policies never charge;
whitening amplified c-noise 12× into the wrong band → coverage resets) and
reset discontinuities poisoning the slow prior (→ episode-boundary masking).

**Prediction 1 — band discovery: CONFIRMED.** With prescribed timescales and
nothing else, charge routed into the slow band (corr 1.000, position-blind:
slow-band position step-scale 0.009). The multi-timescale thesis's routing
claim holds in miniature.

**Prediction 2 — emergent scale: mechanism REFUTED by measurement.**
Whitening equalizes (amplification ∝ 1/std) and coverage-uniform data gives
charge and position similar stds — the representation does not supply the
×3 slow-band weighting round 8 hand-tuned. Recorded tension: routing needs
coverage-uniform marginals; amplification needs natural small-variance
marginals; a linear whitened encoder cannot serve both from one dataset.
(Escape routes: per-band variance targets ∝ τ instead of isotropic
whitening; nonlinear encoders.)

**Headline (`v03_representation.json`, 12 × 150, ladder_short, round-9
mechanics):**

| condition | max_c IQM [CI95] |
|---|---|
| random latent + hand weights (1,3) | 0.761 [0.742, 0.777] |
| random latent, no weights | 0.750 [0.748, 0.758] |
| **pretrained latent, no weights** | **0.782 [0.765, 0.799]** |

The pretrained latent beats the random latent without any hand weighting
(disjoint CIs) and edges the hand-weighted configuration — the first
representation-side win — while the predicted mechanism is absent. Leading
hypothesis for the actual mechanism: the CROSS-BAND LEAK (fast band carries
charge at amplification 0.57, corr 0.836), which makes fast-band targets and
fast-band progress charge-aware — a soft, learned coupling of the bands that
the clean block-diagonal random latent cannot express. Status: hypothesis,
not claim. **Named discriminating experiment (next session): leak-ablated
pretrained latent (orthogonalize the fast band against c post-hoc) vs as-is
— if the win vanishes, the leak is the ingredient and "clean band
separation" was the wrong target all along; if it survives, the geometry
conditioning itself helps.**

Also standing: returns 0.000 in all conditions — the last mile (charge→door
handoff) remains v0.3's open control-side item; hand weights matter little
under round-9 mechanics at this budget.

---

## Round 9 — 2026-07-27: three spec-level fixes, one hard lesson

Instrumenting the ChargeWorld last mile (returns zero despite charge
mastery) produced a diagnosis chain, three spec-level corrections, and a
regression that reframes the headline result. In order:

1. **Selection, not policy, blocked the last mile** (commit hook: 4–8% of
   high-charge fast commits pointed doorward; seeds with chance doorward
   commits completed).
2. **The linear-ranking degeneracy (G1 correction)**: ranking by w·g orders
   every candidate pool by projection onto ONE fixed global direction —
   state-independent, unimodal, incapable of phase structure. Selection,
   prospective veto, and value bar moved to prospective evaluation: f±
   applied to the imagined target (the flinch's move, one hop earlier; kills
   the A2 proxy gap for selection).
3. **Composite-leash interference (C3 per band)**: a held slow target above
   achieved support made every fast composite off-manifold; projection spent
   the budget on the c-dimension and dragged position slices padward
   (doorward commits ~1%). Per-band projection — another bands-only
   capability — fixed it.
4. **G5 currency 1 had never been implemented**: proposers were trained for
   calibration accuracy only, never ambition. Implemented as specified
   (REINFORCE on proposal log-prob × realized value at arrival). Doorward
   selection rose to 17% best-seed; still no cold completions.
5. **Warm-start curriculum failed**: 30% warm episodes DILUTED phase-1
   learning (cold max_c fell to 0.69 for both agents); cold completions
   zero. Curriculum-dilution is real; the idea is retired.
6. **Regression check (`e5c_round9_regression.json`)**: under round-9
   mechanics, the round-8 separation is GONE — flat 0.744 [0.730, 0.763] vs
   ladder_short 0.758 [0.741, 0.770], and both below round-8 levels.

### The lesson: the broken ranking was an accidental global compass

Linear claim-ranking pointed toward the fitted value mass from anywhere on
the map. It was provably incapable of phase structure — the round-9 critique
stands — but it supplied global direction that honest local evaluation (f±
beyond bump reach ≈ 0, ranking ≈ noise) does not. Removing it hurt both
agents' charging and erased the differential high-charge region where
per-band shaping paid. **Status of the headline claim, restated honestly:**
the banded-vs-flat separation is demonstrated at n=30 under round-8
mechanics (`e5c_replication_n30.json`) and is not reproduced under round-9
mechanics. It is real but mechanics-conditioned; the repo keeps the
spec-true mechanics rather than the metric-flattering broken ones.

### v0.3's sharpened question

Where should GLOBAL direction legitimately come from? Candidates, in rough
order of appeal: (a) the proposer's learned ambition (currency 1, with more
exposure); (b) evaluator design that tiles space (multi-scale bumps =
designed global guidance — but this shades into hand-crafting solutions);
(c) frontier-directed proposals from the curiosity table (novelty gradient
as compass); (d) **a pretrained latent in which local value evaluation is
globally informative — smoother, longer-range structure in f±'s domain.
This is the SIGReg thread again**: the representation, not the controller,
may be where global direction has to live. Recommended v0.3 order: (a)
cheaply (longer currency-1 exposure), then (d) as the real program.

---

## Round 8 — 2026-07-27: charge world — first banded-vs-flat separation

### Why this task (and why the zone worlds never could discriminate)

The zone worlds' slow variable is discrete: slow-band progress toward a held
target is a STEP FUNCTION, paying only at the flip instant — the ladder's
signature mechanism (dense progress toward a held slow goal) structurally
cannot fire there. ChargeWorld's slow variable is continuous with
sustain/decay dynamics (charging pad raises c, everywhere else decays it;
door pays only at c ≥ 0.8), so a held c-target converts into a per-step pull.
A spec correction fell out of the same analysis (SPEC L3): linear claims mean
a held slow slice offsets every fast candidate equally — composite claims can
NEVER re-rank within a level; context coupling flows only through the
trunk/policy and per-band progress.

### Three pre-registered iterations, each instructive

1. **Discovery gradient** (8a): a pad bump at c=0.5 is invisible at c=0
   (value ~0.01) while shiny sites pay 0.3/step realized — training REMOVED
   pad contact (max_c 0.00 everywhere). Second demonstration (after E2a's
   trap shape) that the pre-mapped evaluator caps not just safety but
   discoverability. Fix: tile the charge path (bumps at c=0.1 and 0.6).
2. **Drowned signal** (8b): all three conditions then charged to max_c ≈0.78
   and plateaued exactly where evaluator guidance dies — slow-band progress
   (0.02/step) was inaudible against realized gradients ~10× larger. The
   ladder's mechanism existed and was too quiet.
3. **Per-band progress weights** (8c, L2 corollary): weighting band progress
   by hold-length ratio — a capability ONLY banded architectures have (flat
   cannot weight what it cannot separate).

### The result (`e5c_charge.json`, 12 seeds × 80 ep)

| condition | max_c IQM [CI95] | per-seed pattern |
|---|---|---|
| flat | 0.76 [0.728, 0.771] | uniform 0.66–0.82, never crosses |
| ladder (τ_slow=40) | 0.83 [0.595, 0.857] | 10/12 at 0.78–0.88; **2 lock-in failures (0.16, 0.0)** |
| ladder_short (τ_slow=12) | **0.80 [0.777, 0.823]** | 12/12 clean, no failures |

- **First behavioral separation between banded and flat**: ladder_short's CI
  clears flat's ([0.777, 0.823] vs [0.728, 0.771]); both banded variants
  cross the door threshold flat never reaches.
- **Pre-registered reading applies**: ladder_short ≈ ladder ⇒ the active
  ingredient is PER-BAND PROGRESS SHAPING, not commitment persistence. The
  bands' contribution is expressing a signal flat cannot express — the slow
  pull that keeps paying after evaluator guidance dies (the 0.78 → 0.86
  stretch).
- **Persistence is two-sided**: long holds added 2/12 catastrophic lock-ins
  (a wrong slow target held for 40 steps repeatedly walls off exploration)
  without median benefit — C2's anticipated trade, now measured. At this
  scale, short holds + per-band weights is the winning configuration.
- **Returns still 0.000 in all conditions**: the charge→door last mile (leave
  the pad attractor with c ≥ ~0.86, traverse a realized valley) is unlearned
  at 80 episodes. Extended run (150 ep) in flight; if completion appears,
  replicate at 30 seeds.

### Extended run (`e5c_charge_150ep.json`, 12 × 150) — separation strengthens;
### last mile is structural

- max_c: flat 0.78 [0.765, 0.789] vs ladder 0.83 [0.821, 0.845] vs
  ladder_short 0.84 [0.830, 0.847] — banded CIs fully disjoint from flat by a
  wide margin, and the ladder's 80-ep lock-in seeds recovered with training
  (its CI tightened from [0.595, 0.857] to [0.821, 0.845]). The separation
  replicates and grows with budget.
- **Replication at 30 seeds** (`e5c_replication_n30.json`, 30 × 150):
  flat max_c **0.776 [0.769, 0.787]** vs ladder_short **0.839
  [0.833, 0.846]** — tight, widely disjoint intervals. The banded-vs-flat
  separation is confirmed at the same evidential grade as E3b. This is the
  ladder's first replicated behavioral result: banded per-band progress
  shaping sustains a slow variable ~0.06 past the flat architecture's
  plateau, precisely across the region where evaluator guidance dies.
- Returns 0.000 at 150 ep ⇒ the last mile is NOT a budget problem. Diagnosis:
  a chicken-and-egg created by the leash — targets are admissible only within
  0.15 of VISITED support, and high-charge support exists only at the pad, so
  the door trip must be walked before it can be targeted; and nothing in the
  architecture proposes "now go" once "charge" is achieved. The slow register
  can only re-propose c-targets; there is no goal-sequencing / termination-
  handoff mechanism. That is the v0.3 design item: an achievement-conditioned
  slow proposal (when a slow target settles, the next slow-level proposal
  should be conditioned on the achieved state — the options/termination layer
  the architecture currently lacks). Not a patch to rush: it touches C2's
  held-target discipline and must preserve per-window telescoping.

---

## Round 7 — 2026-07-27: E2a completed; ladder honesty update

### E2a: the trust asymmetry, isolated at last (`e2a_trust.json`)

Trap-corridor probe, 10 seeds × 60 episodes, all CIs disjoint where it counts:

| cell | return IQM [CI95] | catastrophes/seed [CI95] |
|---|---|---|
| full (trap live, trust on) | **+0.039** [0.011, 0.061] | **0.0** [0.0, 0.0] |
| no_veto (trap live, trust off) | −0.217 [−0.300, −0.117] | **7.8** [5.0, 10.2] |
| paranoia (trap absent, alarm fires) | +0.028 [0.011, 0.089] | 0.0 [0.0, 0.0] |

Trusting an accurate negative head without verification: complete protection
(0 catastrophes in 600 episodes) at a paranoia price statistically
indistinguishable from zero (full vs paranoia returns overlap). Without
trust: ~8 deaths per seed and negative return. The confusion-matrix row that
began as a single "11 → 0" cell in the original design table is now a
three-cell result with intervals.

Two failures en route were themselves findings:
1. **Prospective target-vetoing alone is insufficient** (11.2 vs 14.8
   catastrophes, overlapping): a learning policy's route is not fully
   governed by its target chain. This motivated the C4 **acting-time
   flinch** — one-step lookahead in the frozen latent, evaluated by the
   fixed `f−` itself (not the linear proxy), acted on without verification.
   Parameter-free end to end; a learned one-step model on this path would
   make the flinch tamperable (now in SPEC §C4).
2. **Evaluator-shape misspecification is exactly as dangerous as §7 says**:
   a rectangular trap under a radial `f−` left lethal corners where the
   innate aversion is silent — the flinch fired correctly everywhere the
   evaluator could see and the agent died where it couldn't. Accidental,
   and kept: it is the cleanest demonstration in the repo that the
   pre-mapped evaluator's fidelity is the safety ceiling (W1's accepted
   price made visible).

### E5b: honesty update on the ladder (`e5b_threezone.json`)

- E5-easy at n=12: flat +0.377 [0.26, 0.49] vs ladder +0.260 [0.08, 0.43] —
  round 6's ladder-ahead reading was noise; if anything flat leads
  (overlapping CIs). **No ladder advantage is demonstrated on any current
  task.**
- ThreeZone (two slow transitions): both agents at zero return, gate rates
  0.22 vs 0.24 — unsolved by both; deeper timescale separation exceeds what
  the current learner + exploration reach.
- Standing conclusion: the ladder costs nothing measurable and inherits all
  safety properties per level, but its *behavioral* case is unproven. The
  discriminating experiment needs either curriculum/exploration work on
  compound tasks, or a task where the slow variable is not reachable by
  fast-chaining alone. Design question, recorded as the top v0.3 item.

### E-item scoreboard after round 7

E1 ✔ protocol · E2a ✔✔ complete with disjoint CIs · E3a ✔ structural ·
E3b ✔✔ two learners · E4 ✔ baseline null · E5 ◐ live, no ladder advantage
shown, discrimination task open.

---

## Round 6 — 2026-07-27: GAE learner unblocks the compound tasks

### The learner saga (three iterations, one lesson)

v1 (one-update-per-episode a2c) cut gradient throughput ~80× → reach
regressed 12×. v2 (PPO-lite on undiscounted returns-to-go) restored
throughput but broke credit assignment — with a cold critic, per-episode
advantage normalization of γ=1 returns-to-go makes advantage a function of
step position, not action quality → reach stayed at +0.04. v3 (GAE γ=0.9,
λ=0.8, terminal V=0) fixes credit assignment; targeted 3-seed checks (4 min)
found the working setting before any hour-scale run. The lesson worth
keeping: the reward stream's γ=1 is a SPEC commitment; the learner's γ is a
bias-variance dial between myopic credit (dense-shaping regimes) and
propagation (compound tasks) — γ=0.9 serves both. Wiring commitments and all
21 structural tests were untouched through all three learner swaps: §5.4's
learner-independence claim is now demonstrated, not asserted.

### E5 is live: both agents now solve the compound task

E5-easy: flat +0.390 [0.06, 0.56], ladder +0.450 [0.25, 0.54]; flip rates
0.94 / 0.91. E5-default: +0.050 vs +0.078, flip rates 0.41 / 0.48. The
ladder is directionally ahead with a much better lower bound on the easy
variant, but CIs overlap at n=8: **no ladder>flat separation claim yet** —
the honest statement is "both learn; ladder ≥ flat; needs more seeds and a
task with deeper timescale separation to discriminate." That the ladder is
not WORSE while carrying stricter constraints is itself informative.

### The treadmill reproduces under a second learner — and harder

`g5_ablated_greedy` under GAE: **arm-A drift 0.67 of classified commits**
(vs 0.10 IQM under REINFORCE at 10 seeds; 3/30 full lock-ins in the 30-seed
replication). Enforced and reinforce-credit conditions: 0.00, again. The
§6.4 phenomenon is learner-robust, which upgrades the E3b result from "an
artifact of one learner" to a property of progress-consulting selection.
Return separation between reach conditions washed out at this budget under
GAE (+0.12–0.16 all cells) — the drift metric, not return, is now the
discriminating instrument there.

### Grid: the guardrails' currency is catastrophes, not return

Returns remain inseparable (all CIs straddle ~0), but the catastrophe column
now has structure: `full` 0.2 per seed; `no_cap` **4.2**; `no_leash` 1.8.
Under a determined (GAE) learner, uncapped progress-shaping drives committed
corridors straight through the hazard — C1 and C3 are measurably protective
in exactly the currency guardrails should be measured in. Two rows stay
honestly open: `no_veto` ≈ `full` (0.2 vs 0.2 — the prospective veto still
does not govern traversal exposure; E2a needs a probe where the negative
claim gates an approach decision), and E4 stays flat zero (the machinery,
not the optimizer, is what learns at this scale).

### Status vs SPEC §9/§10 after round 6

- E1 protocol: operating as designed (rank only where CIs separate). ✔
- E2a: veto cell still unisolated — carried forward with a concrete design. ✘
- E3a: structural test. ✔  E3b: confirmed under two learners. ✔✔
- E4: baseline flat under all learners. ✔
- E5: live; directional; discrimination needs seeds + deeper-timescale task. ◐

---

## Round 4 — 2026-07-27: E5 ladder comparison (v0.2)

`e5_ladder.json` (10 seeds × 60 ep) and `e5_ladder_easy.json` (8 seeds ×
150 ep, nearer gate, larger sites): **flat and ladder agents both at zero
return and ~zero zone-flip rate in all four cells.** Neither agent reliably
reaches even the first-stage gate.

Conclusion, now supported by three independent lines (grid battery stagnation
across rounds 1–3, E5 default, E5 eased): **the scaffold's REINFORCE learner
is the binding constraint for any compound-task behavioral comparison.** The
reach environment learned well (up to +0.89 IQM) because a single
claim-guided target chain with dense evaluator shaping suffices there; the
two-zone task requires reaching a weakly-shaped intermediate gate and then
re-targeting, which this learner cannot bootstrap at any tested budget.

Consequences:
- The E5 *behavioral* claim (ladder > flat on multi-timescale tasks) is
  **untested, not falsified**. It is blocked on learner infrastructure — an
  actor-critic policy learner (or equivalent sample-efficiency upgrade) is
  the prerequisite, and the same upgrade unblocks the grid battery's return
  rows. This is the single highest-leverage piece of infrastructure work in
  the repo.
- The v0.2 *structural* deliverables stand regardless: per-band telescoping,
  hold discipline at boundaries, composite slice/claim superposition, and
  per-proposer G5 are all enforced and tested (tests/test_ladder.py, 20/20
  suite green). The ladder's safety argument never depended on E5 —
  it inherits per level from the v0.1 theorems plus the E3b-confirmed
  treadmill fix.

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

### E3b replication at 30 seeds (`e3b_replication.json`) — gate met

| condition | return IQM [CI95] | zero-return seeds | classified drift |
|---|---|---|---|
| g5_enforced | **+0.746** [0.525, 0.896] | 4/30 | 0.0 × 15 seeds |
| g5_ablated_reinforce | +0.615 [0.342, 0.833] | 6/30 | 0.0 × 7 |
| g5_ablated_greedy | +0.242 [0.050, 0.500] | 13/30 | **1.0, 1.0, 1.0, 0.83**, 0.12, 0.0 × 5 |

- **Confirmed with disjoint CIs:** greedy (progress-consulting) selection vs
  enforced separate cleanly on return, and full treadmill lock-in replicates
  (3/30 seeds at arm-A fraction 1.0, one at 0.83, one partial). The enforced
  config shows zero drift in all 15 classified seeds.
- **Honest revision of the round-3 reading:** at n=30 the REINFORCE-credit
  variant's degradation largely washes out (CI overlaps enforced) — the
  round-3 gap (+0.36 vs +0.77 at n=10) was substantially small-n noise. The
  harmful pathway is progress-consulting *selection*, not weak gradient credit
  to the proposer; G5's operative content is about what the selector may read.
- SPEC §10's precondition for the register ladder is met: the treadmill is
  demonstrated, and its fix (G5 + claim-ranked selection) is demonstrated,
  with confidence intervals.

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
