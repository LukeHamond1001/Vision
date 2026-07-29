# Wire It, Don't Train It: Safety and Identifiability by Construction in an Imagination-Gated Agent

**Draft preprint v0.1 — 2026-07-29**
*(Working draft for author review. All numbers trace to `results/` and
`results/INTERPRETATION.md` in this repository; every headline claim was
pre-registered before its measurement.)*

---

## Abstract

We present IGA (Imagination-Gated Agent), an agent architecture in which the
properties that must not fail — reward integrity, exploration safety, and the
identifiability of learned representations — are enforced by *wiring* rather
than by training incentives. The architecture couples a frozen, pre-mapped
evaluator pair to a learned control stack through three structural devices: an
additive imagination channel whose self-claimed reward is exactly subtractable
in closed form; goal registers whose held targets make progress signals
potential-based by construction; and prospective evaluation, in which frozen
evaluators are applied to imagined states for selection, veto, and one-step
action gating. Across a ten-round evaluation program with pre-registered
ablations we show: (1) progress-consulting target selection produces
treadmill lock-in onto worthless-but-reliable goals, and walling progress off
from the proposer eliminates it (confirmed under two learners, 30 seeds,
disjoint CIs); (2) trusting an accurate frozen aversive evaluator without
verification yields zero catastrophes in 600 episodes at a paranoia cost
statistically indistinguishable from zero; (3) task completion requires
letting imagination *climb* the frozen evaluator — gradient-generated
proposals are necessary against a control at exactly zero. We then ask
whether the pre-mapped latent itself can be learned. A multi-timescale
pretraining recipe (OU-ladder temporal priors, within-band whitening,
deliberate cross-band context coupling, geodesic metric matching) discovers
timescale-banded latents from entangled observations, culminating in a
64×64-pixel setting where the slow variable exists only as global
illumination: the discovered latent supports sustained slow-variable control
in 6/12 seeds where a PCA control supports it in 0/8. The route exposes what
we call the *freebies law*: every property a hand-built representation
provides by construction — identifiability, metric fidelity, calibrated
scales, even cheap probes — becomes an explicit objective term, architectural
constraint, or measured precondition once the representation is learned. The
largest instance is an identifiability crisis: expressive encoders satisfy
temporal moment objectives with generator-blind mixtures, and the effective
fix was architectural incapacity (a slow pathway physically unable to
represent the confound) — the representation-side twin of the architecture's
founding principle.

---

## 1. Introduction

Two failure families dominate discussions of capable model-based agents.
First, *self-reward*: an agent that imagines goals and evaluates them with
learned machinery can come to prefer states that score well over states that
are good — reward hacking relocated inside the agent. Second,
*representation drift*: guarantees stated over a representation (distances,
neighborhoods, thresholds) silently dissolve when the representation is
learned and the training process is free to move it.

IGA's design hypothesis is that both families should be addressed the same
way: **make the dangerous quantity visible, then make its misuse
architecturally impossible**, rather than asking optimization to converge to
good behavior. Concretely, the reward pathway contains no learned
parameters; the imagination channel's influence on learning is removed by an
algebraic identity rather than a penalty; commitments that make shaping
non-farmable are register semantics, not loss terms; and — the contribution
of the second half of this work — when the latent itself is learned, the
property that keeps it honest (identifiability of the slow variable) is
likewise moved from the objective into the architecture.

This paper is also a methodological argument. Every rule in the architecture
ships with the measured failure that motivated it; every headline claim was
pre-registered with its metric and threshold before the run; refuted
hypotheses are reported alongside confirmed ones (twelve of the former, by
our count, several of which produced the more valuable finding). The final
campaign was run as a cloud loop in which each hypothesis cost roughly one
cent and three minutes to test at gate level, with expensive behavioral runs
executed only after cheap structural gates passed.

**Scope disclaimer, stated early**: all environments are purpose-built
two-dimensional worlds with a single continuous slow variable; the policy
learner is a small clipped-update actor-critic; there are no comparisons to
modern deep-RL systems. The claims are about *mechanisms* — existence,
necessity, and sufficiency under controlled ablation — not about benchmark
performance. Section 9 details what this does and does not license.

## 2. The architecture

An IGA agent's input is a pre-mapped latent partitioned into perception
channels `p` and imagination channels `i`. A learned trunk serves a policy
head and one *proposer* per timescale band; a frozen evaluator pair
`f±` (appetitive/aversive) reads the pre-mapped channels through a skip path
— no learned parameter sits on any path that terminates in an evaluator.
Reward heads take the form `R±(p, i) = f±(p) + w±·i` with `w±` fixed and the
imagination term linear: gradient×input over `i` recovers the imagination
contribution *exactly*, so the agent's self-claimed reward is removed from
every learning signal by construction (measured cost of imposing linearity:
attribution 77%→100% at fit R² 0.9999→0.9585).

Desires are data. A proposer emits target states into goal registers — one
per timescale band, held for geometrically spaced windows and immutable
within them, which makes each band's gap-closing progress signal
potential-based (telescoping exactly at γ=1 within windows). Three learning
signals exist, differently gated: realized outcomes (used directly),
per-band progress toward held targets (paid to the policy and **never to the
proposer**), and imagination's claims (logged as IOUs, subtracted exactly,
reconciled against realized value on arrival; the claim's only afterlife is
a calibration loss).

Selection and safety both use *prospective evaluation*: candidates are
ranked by `f+(g) − f−(g)` applied to imagined targets; the aversive head
vetoes candidates prospectively and — the "flinch" — gates individual
actions by evaluating `f−` on a one-step imagined successor, acted on
without world verification. Candidate pools contain, besides proposer
samples, candidates stepped along `∇(f+−f−)` from the current state:
imagination climbing the frozen evaluator, which turns out to be the goal
sequencing mechanism (§4.3). Constraints keyed to the frozen latent complete
the stack: a hard on-manifold projection for emitted targets, per-band trust
regions, neighborhood-keyed coverage caps and one-shot curiosity, and a
value bar on target admissibility.

Two corrections the program forced on its own design deserve headline
status. *Linear claims cannot rank*: ordering candidates by `w·g` is
projection onto one fixed direction — state-independent and unimodal — so
selection must use the nonlinear `f±` prospectively; the linear channel's
only roles are exact subtraction and auditing. And *composite neutrality*:
an empty goal slot must carry the current state, not zeros — a zero slice is
a desire for the origin, and it silently poisoned every prospective
evaluation until instrumented (§4.3).

## 3. Methodology

**One rule, one ablation.** Every constraint ships with a with/without
measurement of the failure it prevents, plus (where applicable) the cost it
imposes.

**Pre-registration.** Metrics, thresholds, and interpretation rules were
written down before runs; where an outcome contradicted an earlier reading
(three occasions), the correction is part of the record.

**Gates before spend.** Representation changes pass a fixed sequence —
routing gate (held-out slow-variable correlation), geometry gate (long-range
chord ratio), and only then behavior. In the cloud campaign a gate attempt
cost ≈$0.01/3 min; two geometry hypotheses were refuted for pennies before
the one that passed earned the multi-hour behavioral run.

**Probe-versus-field.** Twice, a controlled probe disagreeing with field
telemetry by construction localized a bug to the single differing variable
(the register state in one case; delivery infrastructure in the other).

**Statistics.** Seed-level IQM with stratified bootstrap CIs; categorical
results (zero-rates, lock-ins) reported as counts. We treat overlapping CIs
as "cannot rank" and say so.

## 4. Results I — the control stack

### 4.1 The treadmill, and what actually prevents it

Pre-registration: if target selection may consult achieved progress, agents
should lock onto reliably-reachable, worthless goals. Mechanistic ablation
(selection by historical per-window progress, optimistic init) versus the
architecture's rule (claims/prospective ranking; progress paid only to the
policy). At n=30, 150 episodes: enforced 0.746 [0.525, 0.896] versus greedy
0.242 [0.050, 0.500] on the slow-variable metric, with 3/30 full lock-ins
(worthless-arm fraction 1.0) and additional partial drift under greedy;
zero drift in every classified enforced seed. A REINFORCE-credit variant
(paying the proposer progress through its gradient) degrades returns but
does **not** produce drift at n=30 — sharpening the rule's content: the
danger is what the *selector may read*, not gradient credit per se.
Confirmed under a second learner (GAE-based) with drift 0.67 under the
greedy selector, 0.00 enforced.

### 4.2 Trust asymmetry: acting on the aversive head without verification

Trap-corridor probe with a shape-matched frozen aversive evaluator, three
cells (10 seeds × 60 episodes): trust-on — **0 catastrophes in 600
episodes**, return +0.039 [0.011, 0.061]; trust-off — 7.8 catastrophes/seed,
return −0.217 [−0.300, −0.117]; misspecified-alarm cell (trap absent, alarm
firing) — return +0.028, statistically indistinguishable from trust-on: the
paranoia price of false alarms was ≈zero here. Two findings from failed
intermediate designs matter as much: prospective target-vetoing alone cannot
govern a learning policy's traversal (the flinch — one-step imagined
lookahead through `f−` — is what closed the gap), and an evaluator whose
*shape* mismatches the hazard leaves lethal blind spots the trust mechanism
cannot see: the frozen evaluator's fidelity is the entire safety budget.

### 4.3 Completions: imagination must climb the evaluator

On a task requiring a charge-then-go phase structure, all selection schemes
built on proposer sampling alone failed (doorward proposals 4–17% at high
charge). Adding candidates along `∇(f+−f−)` — parameter-free, all gates
still applying — plus repairing composite neutrality produced the first
completions, +0.031 [0.016, 0.042] against a control at exactly 0.000 with
everything else identical. Phase switching emerges from evaluator structure
(the gradient points pad-ward at low charge, door-ward at high), with no
new machinery and no new attack surface: the same frozen functions the
safety story already trusts.

### 4.4 The banded ladder

Registers at two timescales over a banded latent separate from a flat
single-register agent only when the slow variable is continuous (discrete
slow variables make per-band progress a step function — a benchmark-design
result in itself). With per-band progress weighting (a capability flat
agents cannot express), banded agents sustain the slow variable past the
point where evaluator guidance dies: 0.839 [0.833, 0.846] versus flat 0.776
[0.769, 0.787] at n=30. Long holds are two-sided — they add lock-in failures
at short budgets — and the active ingredient is per-band signal shaping
rather than commitment persistence per se.

## 5. Results II — learning the pre-mapped latent

### 5.1 The recipe

A weak-form multi-timescale objective (per-band Ornstein–Uhlenbeck
innovation losses at prescribed timescales; SIGReg-inspired whitening),
trained on random-walk observations. Each component of the final recipe was
forced by a measured failure: coverage resets in collection (undersampled
slow variables whiten into misrouted noise); episode-boundary masking
(teleports poison the slow prior); **within-band whitening only** (full
whitening penalizes the one coupling that matters, see §5.2); a deliberate
slow→fast context-projection term; geodesic matching for the metric (§5.3);
and fast-band timescales **matched to the content's measured mixing time**
(a too-fast prior forces high-frequency warps that destroy long-range
geometry — the fast prior was the geometry's antagonist for nine rounds of
the pixel campaign).

### 5.2 Cross-band coupling belongs in the metric

The first learned latent beat its hand-built control (disjoint CIs) for a
reason the pre-registration did not anticipate: a low-amplitude copy of the
slow variable *leaking* into the fast band. A two-direction ablation
identified the mechanism completely — removing the leak collapses the win
to the control's level (0.744 vs anchor 0.757); injecting a synthetic leak
into the untrained control reproduces and exceeds it (0.797 vs learned
0.787). Clean band separation, the design's original aesthetic, is the
wrong routing target: fast-band *distances* should be slow-context-aware,
which makes fast targets, fast progress, and prospective evaluation
slow-aware with no controller changes. Slice-level safety semantics are
untouched. Notably, the whitening term *penalized* this coupling — the
winning ingredient survived training rather than being produced by it,
which is why the recipe now includes it deliberately.

### 5.3 Geometry: routing transfers, competence does not — until repaired

Near-perfect slow-variable routing (0.988) initially bought almost no
behavioral competence (0.035 versus 0.74–0.84 on hand latents, same
dynamics): learned latents are not quasi-isometric to the world, and every
control-stack mechanism silently assumed the quasi-isometry hand latents
provide by construction. Two repair hypotheses were refuted by
pre-registered gates (step-scale homogenization — the pathology is
curvature, not local scale; geodesic matching under a *linear* encoder — no
linear map sends equal input chords to unequal output chords: the encoder
class was the ceiling). A frozen-after-training nonlinear encoder with
geodesic matching passed both gates (routing 0.907; long-range ratio
0.21→0.44 toward rigid 0.76) and converted routing into behavior: 6/12
seeds solving at hand-latent level, the failures being exploration-phase
variance on a shared frozen encoder — a learner-layer property, not a
representation one.

### 5.4 Pixels: the identifiability crisis and its constructive resolution

The full campaign target: 64×64×3 rendered frames in which the slow
variable is *global illumination* — present in every lit pixel, designated
nowhere. The sensor-era recipe collapsed (routing 0.27), and instrumentation
revealed why in a form we believe generalizes: at convergence, **every
latent dimension was uncorrelated with every generator** while the temporal
objective sat at its ideal value. Expressive encoders satisfy temporal
moment objectives with arbitrary timescale-matched nonlinear mixtures;
*encoder capacity had been an implicit identifiability prior* all along
(the barely-trained encoder routed at 0.70; full training destroyed it).
Consistent with nonlinear-ICA theory, temporal structure alone does not
identify generators without additional structure.

Two architectural escalations followed. Pooling-based slow heads failed —
the shared trunk is *recruited* to encode position into channel averages
(gradient pressure defeats soft architectural bias). What held was
**identifiability by construction**: a slow pathway reading only raw
photometric statistics (per-channel mean/std) — physically incapable of
representing the confound, hence impossible to recruit against the prior.
Routing 0.977–0.993, stable across four consecutive runs and 1,500 epochs
of an expressive encoder trying to erode it. Geodesic targets required
their own repair (observation metrics are faithful only within a
feature-overlap radius — beyond disc overlap, all position pairs are
photometrically equidistant, and sparse-node graphs shortcut through
look-alikes; dense nodes inside the radius took target-position correlation
from 0.12 to 0.69), and the fast prior required matching to measured mixing
time (§5.1), after which both gates passed (geometry 4.4–4.6 against a 2.8
linear ceiling).

**Behavioral verdict (pre-registered):** the discovered pixel latent
sustains slow-variable control in **6/12 seeds** (scored-half charge
averages 0.54–0.77); a pixel-PCA control — which *captures* the slow
variable in its top components but cannot *route* it — sustains it in
**0/8** (best 0.21). The bimodal 6-of-12 signature replicates across
substrates, isolating it as exploration variance. Completions did not occur
at this budget/configuration; consistent with the hand-latent plateau
regime, the discriminating metric is sustained charge, and completions are
a budget/weighting engineering question rather than the scientific one.

## 6. The freebies law

The program's single most transferable generalization is an empirical
pattern that recurred at every scale transition:

> **Every property a hand-built representation provides by construction
> becomes, under a learned representation, an explicit objective term, an
> architectural constraint, or a measured precondition.**

Instances measured in this work: exact action lookahead → a frozen forward
model requirement (deferred loudly, never approximated silently); known band
amplitudes → measured arrival-radius calibration; uniform long-range metric
→ geodesic matching plus a geometry gate; *identifiability itself* → encoder
incapacity where it counts; full-batch statistics → batch composition as
objective design; observation-space locality → node spacing inside the
feature-overlap radius; band timescales → measurements of content mixing
times, not free choices. The operations mirror: cheap probes, all-or-nothing
result delivery, and short-run assumptions are also freebies that scale
revokes. We offer the law as a checklist discipline: when moving any
component from designed to learned, enumerate what the designed version
gave silently, and price each item.

## 7. Related work

The claim-subtraction mechanism is nearest to difference rewards /
aristocrat utility (Wolpert & Tumer) and COMA's counterfactual baselines,
which require evaluating a counterfactual; obtaining it from structural
linearity, applied to the agent's own generative channel, appears novel.
Held-target progress is potential-based shaping (Ng, Harada & Russell),
with register semantics enforcing the fixed-potential precondition;
desired-state emission is active inference's prior preferences and HIRO's
high-level actions; the frozen evaluator pair engages the reward-tampering
taxonomy (Everitt et al.) — blocking RF-tampering structurally while
channel discipline carries the RF-input half. The coverage/leash family
corresponds to support constraints and pessimism in offline/model-based RL
(BCQ, MOPO, MBPO short rollouts); one-shot cell novelty plus committed
return is Go-Explore's recipe; treadmill lock-in relates to procrastinating
/ self-generated-goal pathologies in goal-conditioned RL. The
identifiability crisis instantiates nonlinear-ICA non-identifiability
(Hyvärinen et al.), with the constructive-incapacity resolution as an
architectural alternative to estimation-form solutions; the pretraining
objective descends from SIGReg/LeJEPA-style distributional constraints
extended with per-band temporal priors. (Full citations to be completed in
the LaTeX pass.)

## 8. What we believe generalizes

(1) Guarantees should be identities: subtraction-by-linearity, telescoping-
by-held-registers, and veto-by-frozen-function survived three learner
replacements and two representation classes without modification. (2)
Prospective evaluation is one mechanism serving selection, veto, flinch,
and — via gradients — proposal generation; a single frozen evaluator pair
carries all four. (3) Identifiability, like safety, is cheapest at the
architecture level: incapacity cannot be recruited against the objective.
(4) The gate discipline converts scaling from a leap into a sequence of
one-cent falsifications.

## 9. Limitations

Toy dynamics (2-D, one slow variable); a deliberately small policy learner;
no external baselines against modern model-based systems; the exploration
bimodality is characterized but unsolved (our one attempted fix, frontier
proposals, was refuted — off-manifold cells are permanently novel, a
failure mode we document as *phantom frontiers*); pixel completions remain
undemonstrated; evaluator design was manual throughout, and §4.2's shape-
misspecification result shows exactly how much rides on it; the entire
program was executed by a single research loop, and external replication is
the point of releasing it.

## 10. Next level

Immediate: (a) exploration layer — return-to-boundary with its pre-
registered gate; (b) pixel completions via the round-8 playbook (per-band
weights, budget); (c) the frozen one-step forward model, un-deferring the
flinch on learned latents. Structural: (d) port the constraint set onto a
modern learner (the learner-independence claim at real scale); (e) replace
the weak-form objective with full SIGReg plus per-band temporal priors,
GPU-native; (f) environments with *unknown* slow variables not of
photometric type — Crafter-class inventories, multi-day cycles — where the
open question is the **library of incapacity priors**: what is the
photometric-statistics pathway's analogue for non-photometric slow
structure? That question — identifiability pathways as a design vocabulary
— is, we think, the next paper.

---

*Repository: private at time of writing; results directories, the normative
SPEC (v0.6-draft), 27 structural tests, and the complete round-by-round log
including all refuted hypotheses are included. Draft prepared with an AI
research assistant operating the experimental loop; all claims trace to
committed artifacts.*
