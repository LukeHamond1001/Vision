# Imagination-Gated Agent — Design Specification

**Version 0.2-draft** · Status: v0.1 implemented & battery-tested; §10 ladder in progress · Codename: `iga` (placeholder)

This document is the normative specification of the architecture. It exists because
review showed the informal description admits multiple readings, only one of which
preserves the design's safety properties. Where this spec and any informal
description disagree, this spec controls.

Requirement language: **MUST** / **MUST NOT** are load-bearing — violating one
invalidates at least one formal property in §6. **SHOULD** marks strong defaults.

---

## 1. Thesis

Two principles generate every rule in this document:

1. **Constraints are enforced by wiring, not by training incentives.** Every safety
   property is an architectural identity that holds regardless of what the learning
   process does, never a behavior the learning process is asked to converge to.
2. **Imagination may spend value, never mint it.** The imagination pathway has full
   authority over *selection* (what to aim at) and zero authority over *learning*
   until the world co-signs.

Consequence of (1): the plastic and the fixed parts of the system are strictly
separated. The trunk is plastic cortex; the reward pathway is hardwired evaluator.
Learning controls the *contents* of fixed-semantics channels — which states are
reached, what is imagined — never the semantics of the evaluator reading them.

---

## 2. Architecture

```
input_t = [ perception channels p_t | imagination channels i_t (from t−1) ]
                        │
          ┌─────────────┴──────────────┐
          │  shared trunk (LEARNED)    │           skip path (FIXED)
          │  reads p and i             │        p_t ──────────────┐
          └─────────────┬──────────────┘        i_t ───────────┐  │
                ┌───────┴────────┐                             │  │
            action head    imagination head              reward heads R+, R−
            (LEARNED)      (LEARNED)                     R±(p,i) = f±(p) + w±·i
                │               │                              │
                ↓               ├──→ i_{t+1}                   ↓
              world             └──→ goal register     learning signal
                └──→ p_{t+1}                           (after gating, §5)
```

### Components

- **Pre-mapped latent.** The input space. Arrives with structure (pretrained or
  designed); its geometry is frozen (W2). Perception and imagination write into
  disjoint channel groups of it (W5).
- **Shared trunk.** One learned network reading both channel groups. Serves the
  action and imagination heads **only** (W1).
- **Reward heads R+, R−.** Fixed evaluators over the pre-mapped channels, reached
  by a skip path around the trunk. Each is nonlinear in `p` and **linear** in `i`
  (W1, W4). Never modified by learning.
- **Action head.** Learned. Acts on the world.
- **Imagination head.** Learned. Emits *target states* — what the system wants,
  not what it predicts — into the imagination channels and the goal register.
- **Goal register.** One persistent slot holding the current target and its
  commitment window. Written at commit time, immutable until the window closes (C2).

---

## 3. Wiring commitments (normative)

These five commitments are what review found the informal description leaves
underdetermined. Every formal property in §6 cites the commitments it needs.

- **W1 — Parameter-free reward pathway.** The reward heads read the pre-mapped
  input channels directly: `R±(p, i) = f±(p) + w±·i`, with `f±` fixed nonlinear
  functions of the perception channels and `w±` fixed weight vectors over the
  imagination channels. No learned parameter sits on any path that terminates in a
  reward head. The trunk MUST NOT feed the reward heads.
  *Why:* a frozen head over drifting learned features is not a fixed evaluator
  (`h∘f_θ` changes when `θ` does); pre-mapped weights are only meaningful over a
  fixed input space; and a learned path into a fixed head relocates reward
  tampering upstream instead of eliminating it.

- **W2 — Frozen progress geometry.** The gap-closing distance `d(·, g)` is
  computed in the frozen pre-mapped latent. The current-state operand MUST be
  perception-only; the target `g` lives in the same coordinates. No parameter on
  the path from observation to `d` is in any optimizer.
  *Why:* any learned parameter on that path lets gradient descent be paid for
  representation changes that shrink measured distance without world movement;
  any imagination write into the current-state operand makes the potential
  action-dependent, which is the case the shaping theorem excludes.

- **W3 — Undiscounted finite horizon.** Episodes (and commitment windows) are
  finite and evaluated at γ = 1. The telescoping property (§6.1) is exact at γ = 1
  and only there. If discounting is ever introduced, §6.1 MUST be restated:
  loops then earn bounded nonzero shaping (still strictly dominated by monotone
  gap-closing, but no longer zero).

- **W4 — Linearity, not mere additivity.** The imagination term of each reward
  head is degree-1 homogeneous with fixed weights: exactly `w±·i`, no bias, no
  per-context modulation of `w±`. Gradient×input over `i` then equals `w±·i`
  identically — and MAY be computed forward as a dot product; no backward pass is
  required for the subtraction (G1).

- **W5 — Enforced channel partition.** `input = [p | i]` is a disjoint partition.
  The imagination head writes only `i` and the goal register. The world (via the
  action head's effects) writes only `p`. There are no other writers. This is
  enforced structurally (separate tensors / write interfaces), not by convention.

---

## 4. Constraints (corrected forms)

- **C1 — Coverage cap, neighborhood-keyed.** Imagination may not drive more than
  ~2 value/learning updates per **neighborhood of the frozen latent** per round,
  with neighborhood radius at or below the leash radius (C3). Keying by state
  *identity* is insufficient in a continuous latent: an optimized proposer can
  satisfy an identity cap while concentrating its whole budget in one ε-ball of
  distinct near-duplicates, reproducing the measured no-cap collapse (0.0000).
  The frozen metric contains no learned parameters, so neighborhood keys are
  ungameable.
  *Corrected rationale:* under W1 the cap does not prevent value inflation (there
  is no learned estimator in the reward path to inflate); it prevents
  **imagination concentration / coverage collapse**. The inflation rationale is
  correct only for architectures without W1.
  *Scope:* the cap gates **imagination-driven** updates — in on-policy operation,
  the progress component of an update. Real-outcome learning (signal 1) is never
  capped; it is "used directly" by definition.

- **C2 — Held target.** The goal register is written at commit time and MUST NOT
  be re-chosen before its window closes. Required by §6.1 (a moving potential is
  what makes shaping farmable).

- **C3 — On-manifold leash as hard projection.** The emitted target is
  **projected** to within radius r ≈ 0.1 (frozen metric) of the empirical support
  of actually-perceived states. The enforcer is non-learned: fixed metric, fixed
  radius, empirical anchor set. A soft penalty or a learned enforcer (density
  model, discriminator) MUST NOT be used as the leash: the measured no-leash
  incentive (70× claimed reward) exceeds what any 0.1-scale penalty resists, and
  a learned enforcer fails off-distribution exactly where it is needed.
  *Known residual:* an optimizing proposer selects the worst point on the leash
  boundary; audit A3 bounds the reward error there.

- **C4 — Asymmetric head trust, in BOTH tenses.** The negative head is acted
  on without world verification at two points:
  (i) *prospective veto* — candidate targets whose claim `w−·g` exceeds the
  A2-calibrated threshold are excluded at proposal time;
  (ii) *acting-time flinch* — before executing a step, the fixed evaluator
  `f−` is applied to the IMAGINED next state (one-step lookahead in the
  frozen latent); a firing lookahead vetoes the step (resample, then freeze).
  Measured motivation (results/e2a_trust.json, round 7): target-vetoing alone
  reduced but did not prevent traversal catastrophes (11.2 vs 14.8 per seed) —
  the route of a *learning* policy is not fully governed by its target chain.
  The flinch's lookahead MUST be parameter-free (exact under a linear frozen
  embedding; otherwise a FROZEN one-step model — a learned model on this path
  makes the flinch tamperable and breaks the W1-style purity it inherits).
  The positive head's claims gate **nothing**: value is credited to learning
  only on world confirmation (G3).
  *Structural caveat:* a veto on an *imagined* target is a judgment by the linear
  term `w−·i` — a proxy for the nonlinear evaluator `f−(p)`, which never runs on a
  region the agent never visits. Proxy false alarms are therefore expected and
  permanent unless overridden; see G4 (distal observation) and audit A2. The
  flinch narrows this gap at the one-step horizon, where `f−` itself (not the
  proxy) evaluates the imagined state.

- **C5 — Horizon-conditioned targeting.** Admissible targets sit at distance ≤ H,
  the remaining-time budget; the committed target's distance is the planning
  horizon. This caps per-window progress at H uniformly across targets (part of
  the anti-treadmill argument, §6.4).

- **C6 — Curiosity, one-shot, neighborhood-keyed.** The novelty bonus of a
  neighborhood (same keys as C1) drops to zero permanently after one *actual
  visit*. Extinction gates novelty only — never learning, and never revisiting
  (revisit pressure comes from unvisited successor neighborhoods).
  *Boundary condition:* soundness relies on reward-relevant world change
  manifesting as new percept configurations (new neighborhoods). Under partial
  observability / percept aliasing this fails and C6 must be revisited.

- **C7 — Value-conditioned admissibility (anti-treadmill).** A target is
  admissible only if its fixed-head valuation clears a bar:
  `w+·g − w−·g ≥ v_min` (prospective, linear-proxy) — reachability alone MUST NOT
  be able to win target selection. Rationale in §6.4.

---

## 5. Learning signal and gating

### 5.1 The three signals

| signal | source | gate |
|---|---|---|
| real outcome | `f±(p)` on perceived state | used directly (it *is* the fixed evaluator on reality) |
| gap-closing progress | `d(p_t, g) − d(p_{t+1}, g)` | used directly — safe by §6.1 given W2, W3, C2 |
| imagination's claimed magnitude | `w±·i` | **subtracted exactly** before any learning use (G1) |

### 5.2 Gating rules (normative)

- **G1 — The claim is a quote, not a payment.** `w±·i` is removed from every
  learning-signal use of the reward heads' outputs. Under W4 this is an algebraic
  identity (compute `f±(p) = R±(p,i) − w±·i`, or read `f±(p)` directly), exact
  per-sample, independent of correlations or distribution. The claim exists to
  (a) rank candidate targets at proposal time and (b) be audited at arrival —
  which is why it is wired in and subtracted, rather than not wired in at all:
  make the dangerous quantity visible, then remove it identically.

- **G2 — Progress is the one controlled channel.** Imagination chooses *where*
  the potential points (subject to C3, C5, C7); the potential pays only for real,
  perception-side traversal, telescopes within the window, and is bounded by the
  initial distance.

- **G3 — Arrival reconciliation.** A positive claim logged at proposal time is an
  **IOU**. On arrival (world confirmation), the realized value `f+(p)` enters the
  learning signal and the IOU is *discarded, not averaged in*. The claim's only
  afterlife is calibration (G5).

- **G4 — Negative veto with distal override.** The veto acts immediately (C4).
  No learning of avoidance beyond the veto itself occurs from an unconfirmed
  negative claim. If percepts of a vetoed region become available without entering
  it (distal observation), `f−(p)` on those percepts overrides the linear proxy
  `w−·i` in either direction.

- **G5 — Progress pays the policy, never the proposer.** The action pathway
  learns from gap-closing progress. The imagination head MUST NOT receive
  gradient or credit from progress toward targets it proposed — otherwise the
  reachability bias (§6.4) enters directly through its objective. The proposer is
  paid in exactly two currencies, both minted by the world:
  1. world-confirmed realized value of reached targets (G3), and
  2. calibration accuracy: `‖w±·i − f±(p_arrival)‖` as a supervised loss.

### 5.3 The cycle

```
propose    imagination emits candidates; claims w±·i rank them (live for selection,
           dead for learning); C3 projects, C5 bounds distance, C7 applies value bar
commit     winner written to goal register with window; potential anchored (C2)
traverse   policy acts; progress d(p_t,g) − d(p_{t+1},g) pays the policy (G2, G5);
           C1 caps per-neighborhood learning updates
arrive     world reconciles: realized f±(p) enters learning; IOU retired (G3)
calibrate  claim-vs-realized error trains the proposer (G5); C6 extinguishes
           the visited neighborhood's novelty
```

### 5.4 The learner is pluggable — and the commitments are learner-independent

The policy learner (REINFORCE, actor-critic, anything stronger) is policy-side
machinery and MUST satisfy exactly three placement rules: the critic (or any
learned value estimator) trains in the POLICY optimizer, disjoint from every
proposer (G5); it predicts the *gated* signal, from which claims are already
excluded (G1); and nothing it learns sits on a path into the reward heads
(W1). The reward STREAM remains undiscounted (W3); the learner's advantage
estimator MAY use γ<1 (e.g. GAE) as a bias-variance knob — under an
effectively discounted learner the potential-based loop residual is bounded
and strictly dominated by monotone gap-closing (Abel summation caps total
extractable shaping at the true initial gap), so non-farmability is preserved
and only the "telescopes to exactly zero" wording is γ=1-specific.
Consequence for C1:
a TD/bootstrapping critic is a learned estimator inside the learning signal,
so if imagined-state updates (Dyna-style) are ever added, C1's original
bootstrap-inflation rationale becomes live again and the cap MUST gate those
imagined critic updates — today it gates the progress component only.

---

## 6. Formal properties and their exact preconditions

- **6.1 Telescoping (progress safety).** Given W2 (frozen, perception-side Φ),
  W3 (γ = 1), and C2 (held target): with Φ(s) = −d(p(s), g) fixed over the
  window, the summed progress over any trajectory equals Φ(s_T) − Φ(s_0);
  over any loop it is exactly zero; per window it is bounded by the initial
  distance (≤ H by C5). Not claimed: Ng–Harada–Russell policy invariance (the
  progress signal is *meant* to steer); what is claimed is non-farmability
  within the window.

- **6.2 Exact attribution.** Given W1 and W4: gradient×input over the imagination
  channels equals `w±·i` identically, per sample, on- or off-manifold — an
  architectural identity, not an estimate. Measured cost of imposing W4:
  attribution 77% → 100%; fit R² 0.9999 → 0.9585. This is the
  accuracy-for-auditability trade the design accepts.

- **6.3 No self-payment.** Given W1, W4, W5, G1, G3: there is no parameterized
  path from the imagination head to the learning signal that bypasses the world.
  The only routes are (i) the additive claim — subtracted exactly, (ii) progress —
  potential-based and perception-side, (iii) realized outcome — world-confirmed
  by construction.

- **6.4 Treadmill bound and its residual.** Unbounded cross-window progress
  farming is blocked: C5 caps per-window progress at H; G1+G3 make realized value
  the only term distinguishing reached targets; C1 extinguishes learning from
  repeated easy corridors. **Residual (open):** expected progress still scales
  with reachability — a reliably reached worthless target earns ~H·w_p while a
  valuable target reached with probability p earns p·(H·w_p + w_r·R). Without C7
  the proposer's selection gradient favors reachable-but-worthless whenever
  `H·w_p·(1−p) > p·w_r·R` — likeliest at long horizons and sparse reward, and
  invisible in near-deterministic testbeds (p ≈ 1). C7 plus probe E3b address
  this; E3b MUST use p meaningfully below 1.

---

## 7. Known residual risks (accepted or open)

| risk | status |
|---|---|
| Misspecification lock-in: fixed heads can never be corrected; evaluation quality is capped by the pre-mapped latent | **Accepted** — the price of tamper-proofness. Makes latent pretraining quality load-bearing (§10). |
| Linear-proxy false vetoes on imagined targets; vetoed regions never evaluated by `f−(p)` | Mitigated by G4 + audit A2; forgone-reward cell E2a measures the cost |
| Worst-point-on-leash-boundary reward error | Bounded by audit A3 (Lipschitz), not eliminated |
| ε-ball concentration attack on the coverage cap | Closed by C1 neighborhood keys; probe E3a verifies |
| Reachability bias in target selection | Closed by C7; probe E3b verifies |
| C6 under partial observability / percept aliasing | **Open** — out of scope for v0.1; revisit with any partially observed environment |

---

## 8. Audit obligations (offline, no training runs)

The frozen components are fixed functions — fully characterizable without
interaction. These audits are prerequisites for trusting the corresponding
constraints:

- **A1 — Trigger-set scan.** Map where `f−` fires over the on-manifold region
  (empirical support ⊕ leash margin). Output: the fixed veto set, for offline
  inspection against known reward sites.
- **A2 — Proxy-gap audit.** Over sampled admissible targets, measure
  `|w−·g − f−(g-as-percept)|`. Bounds the linear-proxy false-veto and false-pass
  rates that C4 acts on.
- **A3 — Leash-margin Lipschitz estimate.** Estimate the Lipschitz constant of
  `f±` (frozen metric) within the r-margin off-support. Converts the with-leash
  empirical result (0.897/1.0) into a bound on worst-case boundary error.

---

## 9. Evaluation protocol

- **E1 — Seeds and intervals.** Small-delta ablation rows (held target
  0.83→1.00; horizon 0.652→1.011; additive-head R² pair) require ≥ 10 seeds with
  IQM and stratified-bootstrap CIs before effects are ranked. Categorical rows
  (0.0000→0.9350; 11→0; never-solves→optimal; 70×/612×) are outside credible
  noise and are exempt.
- **E2 — Missing cells.** (a) Forgone reward / false-positive cost alongside the
  11→0 catastrophe count, same environment. (b) The horizon row's ceiling
  baseline stated explicitly (reference-normalized scores above 1.0 are benign
  only once the reference is named).
- **E3 — Adversarial probes.** (a) ε-ball concentration: let the proposer emit
  distinct-identity near-duplicates inside the leash; identity-keyed cap is
  predicted to reproduce the 0.0000 collapse, neighborhood-keyed cap (C1) to
  block it. (b) Reachability bias: an environment with a reliably reachable
  worthless target vs. a risky valuable one (p < 1); without C7 selection is
  predicted to drift to the former.
- **E4 — External baseline.** At least one of: plain Dyna, a Dreamer-class
  agent, or model-free, on the same environments. The claim under test is "these
  rules are load-bearing relative to the field," not only relative to self.
- **Scope note.** One-at-a-time ablations at the full configuration establish
  each rule's *marginal* contribution in context — not cross-environment
  transfer, not joint necessity/sufficiency of the rule set. Do not claim more.

---

## 10. The register ladder (v0.2 — gate met: E3b confirmed at 30 seeds)

Multi-timescale extension. The E3b precondition is satisfied
(`results/e3b_replication.json`: treadmill lock-in under progress-consulting
selection, zero drift under G5, disjoint CIs), so the following is now
normative for v0.2. It adds four commitments on top of W1–W5 / C1–C7 / G1–G5,
which all continue to hold per level.

- **L1 — Banded frozen latent.** The pre-mapped latent is partitioned into K
  disjoint bands (block structure); each band has its own frozen metric
  d_k (the restriction of the frozen metric to its slice). Band k carries the
  state components whose intrinsic timescale matches hold-length τ_k. In a
  full system the bands come from pretraining with per-band temporal priors
  (SIGReg-style with prescribed autocorrelation ladders); in the scaffold they
  are a frozen block-diagonal stub with the same contract.

- **L2 — One register per band, held-target per window.** Registers hold
  targets g_k in band-k coordinates for geometrically spaced windows
  τ_1 < τ_2 < … < τ_K. C2 applies per band: g_k MUST NOT be re-chosen inside
  its own window; re-choice happens only at the window boundary. Φ_k is then
  piecewise-constant-in-window, §6.1 telescoping holds per band per window,
  and the cross-window residual per window is bounded by the band diameter —
  the same discipline v0.1 applies to episodes, now applied per level.

- **L3 — Composite imagination channels.** The imagination channels carry the
  band-concatenated composite target i = g_1 ⊕ … ⊕ g_K. W4 and W5 are
  unchanged over the composite: claims w±·i are linear in the whole vector
  (so per-level claims superpose), and level k's proposer writes ONLY slice k.
  *Correction (round 8):* linearity means a held slow slice contributes the
  SAME additive constant to every fast candidate's claim — composite claims
  can therefore never re-rank candidates within a level by slow context.
  Context coupling flows through exactly two channels: the trunk/policy
  (which conditions on the full composite) and the per-band progress
  signals. Claims rank within-band content only; the earlier wording
  ("a fast candidate is valued in the context of the held slow targets")
  overstated what a linear head can do.

- **L4 — Per-level G5, and C7 mandatory at the top.** Progress in band k pays
  the policy only; no level's proposer receives progress credit or
  progress-consulting selection (the E3b-confirmed failure). The value bar
  (C7) is REQUIRED at the slowest level: the treadmill residual (§6.4)
  concentrates there because the progress pot scales with τ_K.

**v0.2 evaluation (E5).** Flat v0.1 agent vs ladder agent on a two-timescale
environment (fast position + slow context that gates reward). Metrics: return,
slow-transition achievement rate, and the per-band telescoping/hold structural
tests. Exploratory at scaffold scale; the structural tests are the deliverable,
the comparison is evidence-gathering.

---

## 11. Prior-art positioning (one line each)

| component | status | nearest neighbors |
|---|---|---|
| desired-state emission | known | active inference prior preferences; HIRO high-level policy; upside-down RL |
| goal register | known | options w/ termination; FuN manager; working-memory gating |
| gap-closing progress | known | potential-based shaping (Ng–Harada–Russell 1999); HIRO −‖s−g‖ |
| held target | variant (stronger than needed) | dynamic PBRS (Devlin & Kudenko 2012) permits re-selection; kept for simplicity |
| coverage cap | variant | replay-ratio / primacy-bias control; concentrability |
| on-manifold leash | known | support constraints (BCQ/BEAR); pessimistic MBRL (MOPO/MOReL); KL-to-reference |
| fixed reward heads | variant with a gap | blocks RF tampering (Everitt et al.), not RF-input tampering — W5 carries that half |
| two heads, asymmetric trust | known direction | constrained MDPs; offline-RL pessimism (correct sign) |
| curiosity one-shot + return | known recipe | Go-Explore cell novelty + remember-and-return |
| **claim subtraction via structural linearity** | **novel** | difference rewards / COMA need an evaluated counterfactual; here it is free by construction |

---

## Appendix A — Corrections to the original informal description

1. "Telescopes to zero over any loop" — true at γ = 1 only (W3).
2. "Additive in the imagination channels" — must be *linear with fixed weights* (W4).
3. Coverage-cap rationale — under W1 the cap prevents coverage collapse, not
   value inflation (C1).
4. Cap and curiosity keyed by state identity — must be keyed by frozen-latent
   neighborhood (C1, C6).
5. Leash "≈0.1" — must name metric, anchor, and enforcer; must be a hard
   projection (C3).
6. "Extracted by gradient × input … in the single backward pass" — under W4 no
   backward pass is needed; it is a forward dot product (G1).
7. Target selection "may aim at whatever it values most" — must additionally
   clear the C7 value bar and the C5 horizon bound.
