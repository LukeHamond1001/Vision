# The from-scratch robot program
*(north-star roadmap; written 2026-08-03 while the v4.0 fleet was landing.
Every phase names the receipt that already demonstrated it in miniature.)*

**Premises:** own the whole stack (no borrowed VLA; pretrained perceptual
latents allowed), a growing narrated-teleop corpus, no simulator — the
world is the gym. The drive-layer laws are constants of the program:
frozen before each life, pay only measured progress, audits gate all
authority, humans legislate / reality grades.

**The kernel (given, never learned, never grows):**
maintain what has a healthy range; seek the frontier of anything
measurable, once each; in-stream progress pay every step of every life.
Everything else — senses, skills, goals, language — grows around this.

---

## R0 — Foundations
*(receipts: v0.5–v0.9 banded pretraining; pre-flight A/B instrument
discipline; pre-flight F audit)*

- **Data protocol**: teleop logging manifest — synchronized video,
  audio, ALL telemetry, actions, and OPERATOR NARRATION ("picking up
  the red cup"). Narration starts on day one; it is nearly free and
  funds R3 entirely.
- **Substrate v1**: perceptual latents (pretrained ok) feeding banded
  multi-timescale structure + an action-conditioned predictor trained
  on the teleop corpus.
- **Instrument auto-sweep**: calibrate closed-form heads for EVERY
  numeric channel in the logs against its own telemetry; emit the
  audit table (held-out corr/MAE, coverage exclusions).
- **Gates**: predictor holdout; F1 (danger ranking) + F2 (action
  discrimination) for flinch authority; instrument table reviewed by a
  human — the values step: which audited channels become drives.

## R1 — Drive layer on the robot
*(receipts: v2.1 conservation; v4.0 fleet; B-harness zero-phantom)*

- Maintain lane live (battery, thermal, contact force) with leashes
  and caps from day one; flinch armed iff F2 passed.
- **Teaching loop v1 — the core product mechanic:**
  1. human demonstrates (1–2 teleop episodes) → BC-initializes the
     skill AND mints a register target from the demo's measured
     endpoint deltas ("this is what done looks like");
  2. clicker ratifies or vetoes ("good" = keep target; "no" = exclude
     endpoint, sharpen the arrival conjunction) — feedback edits the
     GOAL ROSTER between lives, never the reward stream (the
     timescale firewall: no in-stream pay for eliciting approval);
  3. proposer schedules practice (one-shot novelty), ledger pays only
     measured progress, arrival verified by instruments.
- **Gate**: N tasks acquired from ≤5 demos each; zero phantom
  arrivals in field logs; force/no-harm bounds never breached.

## R2 — The flywheel and generational scaling
*(receipts: Crafter gen-1→gen-2 loop; the saturation metrics)*

- All practice is recorded; corpus = teleop + self-collected.
- Saturation is MEASURED, not felt: predictor loss plateau,
  instrument-audit plateau, demos-to-competence plateau.
- On saturation: scale substrate params, retrain on the full corpus,
  re-run the sweep (sense roster grows), re-audit flinch, re-anchor
  existing registers to the new latent, freeze, continue.
- Drive-layer code never changes across generations. Only the brain
  and the senses grow.

## R3 — Language grounding (overlaps R1–R2)
*(receipt: narration protocol from R0)*

- Speech is audio; predictive pretraining on narrated demos learns
  speech↔scene↔action correlation — grounded understanding, self-built.
- Utterance → register-target keying: spoken commands select goals
  from the demo-minted roster; "instruction satisfied" is one more
  audited, unfarmable channel.
- Honest ceiling: robot-grade COMMAND grounding (hundreds–thousands of
  utterances, compositional over known registers) — not open
  conversation. Broader language = add text corpora to substrate
  pretraining later, still self-built.

## R4 — Learning to learn
- Zero-shot goal adoption: new tasks expressible as new targets over
  existing skills — no gradient steps (count them).
- Few-shot skill acquisition: BC-init + world model make demos cheaper
  every generation.
- In-context sensorimotor adaptation: emerges with substrate scale
  (the same mechanism as LLM in-context learning, over sensorimotor
  streams).
- **THE metric of the whole program: demos-needed-per-new-task,
  plotted per generation. Meta-learning = that curve falling.**

---

## Safety spine (constants, all phases)
- Success grounded in WORLD-STATE, never human reaction: no facial /
  physiognomic channels — no ground truth exists to audit them, and
  involuntary human signals are what a manipulator farms. Voluntary
  channels only (speech, clicker), and even those never pay in-stream.
- Arrival conditions are conjunctions (count AND force AND damage
  quiet) — kills crammed-plates Goodhart.
- Capability growth is quantized and refusable: senses freeze in only
  through audits, between lives, at a gate a human holds.
- The want roster is legible text at all times; anyone can read what
  the robot wants and object BEFORE it runs.
