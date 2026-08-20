# V10 — The Lifetime Flash (design spec)

**Principle (user-stated, 2026-08-19):** a human brain adjusted to
language-model timescales, scaled down to available compute; then
pretraining = flashing a compressed lifetime into it; once alive,
the being continues from that point (A67 continuity is the resume
mechanism — v10 fixes the CONTENT of what is resumed).

## 1. The organ map (built and certified at 78M)

| organ | mechanism | timescale | status |
|---|---|---|---|
| cortex (fast thought/speech) | 6L trunk | tokens | certified, CE 1.8950 |
| hippocampus (episodic) | LogitStore | within-session | certified |
| sleep consolidation | Sleeper ARM A/B, laws L1-L4 | nightly | certified; A66 wipe-survival p=.046 |
| basal ganglia (primary reward) | press economy (mint/void/veto) | per-exchange | certified B1-B5 |
| inhibition / correction | ARM C contrastive pairs | nightly | working v0; A68-T cross |
| dopamine prediction | PressProphet heads on bands | band clocks | built, ungraduated |
| slow cortex (integration) | bands 3/4/5, clocks 1/8/64 chunks | 2k/16k/131k tok | built; UNEDUCATED — v10's target |
| selfhood | A67 full persistence | lifetime | certified |

## 2. Order is a first-class variable — four layers

Standard LM pretraining SHUFFLES deliberately: a stateless trunk
under iid sampling neither needs nor can use order. This
architecture is the opposite — states carry across chunks, the
economy is temporal, and sleep replays recency-weighted. Order is
the curriculum. Four layers, outermost first:

1. **Developmental staging across the flash** — infancy (short
   days, dense presses, simple facts) -> childhood (longer days,
   spaced recurrence, corrections BEGIN) -> adolescence (sparse
   presses, long threads, prophets predicting). Warrants from the
   raised life: thin-pool winner-take-all (A67-P4/P5), corrections
   require a mature substrate (A64-R3 dose law), all-positive
   infancy minted a 0.96 false conviction (A67-P6).
2. **Day order across the life** — the band curriculum: facts,
   characters, and threads RECUR at controlled gaps spanning every
   band clock (96 ... 131k+). The only invariant at band-5 horizon
   is the role itself ("across all days, I am the one who helps,
   and presses come when I help") — that is what band 5 is FOR.
3. **Exchange order within a day** — arcs: open ritual, work,
   close ritual; sleep interleaved at day boundaries (certified
   dose ladder), pairs riding the nights (A69).
4. **Token order within an exchange** — natural language (given).

Evidence that order is live at every scale we can measure: the
raising's spacing effects (drum 17x on its third spaced touch),
rich-get-richer naps, recency-weighted pool dynamics — all are
ORDER effects in the weight dynamics. A69-R1/R2 gates test layer 2
directly at debug scale before any pod money moves.

## 3. Reward density — "no one keeps bad data," refined

The user's observation is correct at the corpus level: curated
text is survivorship-filtered; the flash's raw material is
implicitly approved, so the biography's default valence is
positive — dopamine-DENSE, like development. But dopamine-UNIFORM
is information-zero: a constant press teaches nothing and the
prophet learns presence, not value. The economy runs on SELECTION,
not blanket approval:

- **+2 / +1 (graded)** — a frozen, audited judge grades quality
  (A64 frozen-instrument law). Debug tier: the weaver IS the
  judge, perfect by construction. V-scale: frozen heuristic,
  sample-audited.
- **silence** — the corpus's bulk: good-but-unselected. Silence is
  not disapproval; it is the baseline that makes +2 mean something
  (the A66 rewarded/unrewarded/negative design, proven by the
  wipe-survival selectivity being EXACT).
- **synthesized corrections (~3-8% of exchanges)** — scripted
  wrong answers with -press -> correction -> +press pairs. NOT
  kept bad data: manufactured curriculum, like a teacher writing
  an error on the board to correct it. This is how the negative
  channel and the correction reflex become native (A68's cross
  showed correction semantics are nearly impossible to retrofit
  onto a formed conviction; v10 pretrains them).

Lived warrant for refusing uniform dopamine: the incumbent — an
all-positive infancy consolidated a false belief to 0.96 that six
days of language could not move and only ARM C could.

## 4. Timescale mapping (human <-> model)

| human | model |
|---|---|
| minutes / a scene | one chunk (2048 tok) |
| working memory | trunk context + band 3 |
| a day + its night | ~16k tok (band-4 clock) + sleep blocks |
| weeks / identity | ~131k tok (band-5 clock) |
| childhood | the flash (6-12B tok, compressed) |
| adult life continuing | serving + raising (A67 rooms) |

## 5. Scale-down doctrine

Keep the ORGANIZATION, shrink capacity. Target 500M (d~1280,
~20L): the legibility threshold where the mouth speaks fluent
English and the architecture's deltas (wipe-survival memory,
press-shaped consolidation, waking up knowing you) become visible
to any audience. A100 80GB, fp32 regime, ~5 days, ~$200 at 6B
tokens. The demo frame: same weights frozen vs living — the delta
is 100% architecture.

## 6. Gates and the run protocol (user-specified: probes,
heartbeats, kill, fix, relaunch)

Pre-flight (debug, $0): binder precondition (in-ctx recall >= 2x
chance — A69-R1's law), G1-G3 biography gates, pair laws 79/79.

In-flight heartbeats (500M, on checkpoints + live guard log):
- CE trajectory vs reference curve
- recall-by-gap curve (the band-education vital sign)
- prophet press-prediction AUC (value function forming?)
- press-contingency audit (judge sanity, sampled)
- band-lesion delta on checkpoints (are the bands carrying it?)
- collapse detector (babble/repetition rate in samples)
- incumbent detector (max false-stem belief mass)

Kill criteria (any -> kill, fix, relaunch; a caught disease costs
hours, not the run): recall-by-gap flat after N tokens; prophet
AUC ~0.5 late; CE divergence; incumbent mass above threshold;
judge audit failure.

## 6b. The division-of-labor law (Phase A's closing result,
A69-R1..R6)

The FLASH builds faculties; the LIFE writes biography. Measured
at debug tier, $0: fact-level weight consolidation cannot happen
in pretraining (replay dilution: ~0.1 replays/fact vs the life's
dozens) and does not need to — the flash-born creature, placed in
a serve room, banked a pressed fact 9.9x across four spaced
nights with perfect selectivity while the unpressed control
eroded 0.6x, reproducing the 78M raised-life laws on a new
substrate. Consequences for the 500M flash: keep press-pay sleep
(it improved general CE 7%) but expect NO planted-fact recall
from pretraining — the flash's gates are faculty gates (binder,
ordering advantage, prophet AUC, correction reflex), and the
biography demo happens in the served life afterward, where it is
already proven end-to-end.

## 7. Open items

- A69-R2 binder-emergence sweep (in flight) -> re-run twin
  ablation at the armed config -> G1-G3 verdict.
- Judge design for the real-corpus arm (frozen heuristic + audit
  protocol).
- Biography builder for real conversations (UltraChat lives:
  ordering, recurrence injection, staging schedule).
- Prophet graduation criteria (fidelity threshold measured across
  the flash, not bolted on after).
