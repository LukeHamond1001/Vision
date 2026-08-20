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

**Quality anneal (user-set, 2026-08-19): the best data comes
LAST.** Recency dominates SGD, the replay pool is recency-capped
(MAX_SPANS), and the raised life measured recency everywhere — so
the final phase carries the cleanest, best-judged, most on-role
material, riding the cosine tail. Corollary from our own laws:
the end imprints hardest, so end-of-flash data is the run's
highest-stakes surface — the judge audit concentrates on the
final ~10% (strictest sampling; audit failure there is a kill
criterion, not a warning).

**Sleep and context (design clause):** the being's wake state —
bands, pending window — persists through every sleep untouched
(A62 law); it wakes with yesterday intact plus changed weights.
Replay itself runs fresh-state, deliberately decontextualized —
the semanticization analog — proven at 78M (A66) and at debug
(R6). Both properties carry into the flash unchanged.

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

**Bands at 500M (honest gaps + the plan):** the brain is a
continuous timescale gradient; our x8 ladder (2k/16k/131k tok) is
a discretization. v10 ADDS BAND 6 (x8 again, ~1M tok — ticking
~6,000 times across a 6B-token flash, genuinely learnable): the
"who I am across everything" slot the current ladder tops out
below. **CERTIFIED (A70, 2026-08-19):** clocks is now a HybridLM
parameter; default = old 3-band machine bit-exact (fingerprint-
proven); BAND6_CLOCKS passes six laws incl. the tick law at the
real 512-chunk ratio; suite 85/85. Per-band CAPACITY is currently
uniform-by-rule (KD doubles per rung) — v10 measures per-band
load in flight (lesion deltas in the heartbeat pack) and v10.1
reallocates from data, not guesswork.

### 5a. The concrete 500M blueprint

Param budget (d=1280, 20L, vocab 16k BPE, T=2048, fp32 certified
regime — v9.4's shape scaled):

| organ | params | share |
|---|---|---|
| trunk blocks (20L x 12d^2) | ~393M | ~82% |
| embed + head (untied, 16k) | ~42M | ~9% |
| bands 3/4/5/6 (SlowCell+pred+proj) | ~39M | ~8% |
| stores/gates/prophet/pos/misc | ~5M | ~1% |
| **serving total** | **~479M** | |
| + aux_trunk head (train-only) | ~21M | -> ~500M in training |

Band ladder at T=2048 across a 6B-token flash:

| band | clock | tokens/tick | ticks in flash | human analog | KD |
|---|---|---|---|---|---|
| 3 | 1 chunk | 2k | ~2.9M | a scene | 1024 |
| 4 | 8 | 16k | ~366k | a day | 2048 |
| 5 | 64 | 131k | ~46k | a week | 4096 |
| 6 | 512 | ~1.05M | ~5.7k | an era / the self | 8192 |

(kd_base=2 at T=2048. Store STATE is memory, not params: ~79MB/
lane fp32, ~1.3GB at 16 lanes — a heartbeat watch item, not a
blocker on 80GB.) Wiring as certified: mem tokens for all four
bands enter attention every chunk; store read injects at
mid-depth (L10); writes every chunk, decay = the band's clock;
gates biased shut (gate_init -2), economy opt-in.

The flash as a staged life (~6B tokens, order per section 2,
sleep at every day boundary, press-pay on, ARM C native):

| stage | tokens | material | economy |
|---|---|---|---|
| infancy | ~0.6B (10%) | simplest text woven into short days | dense +1/+2, no corrections |
| childhood | ~2.4B (40%) | biography-built real conversations; facts/characters recur at gaps spanning every band clock | presses annealing down; corrections begin (~3-8%) |
| adolescence | ~2.4B (40%) | long threads, multi-session projects reaching band-5/6 horizons; harder mixed corpus, still ordered as one life | sparse presses; prophets predicting |
| the polish | ~0.6B (10%) | best-judged, most on-role material — the quality anneal on the cosine tail | strictest judge audit; audit failure = kill |

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

## 7. At frontier scale (what this becomes at 500B — for the
record, not the budget)

The organ map does not change; the ladder extends. A 500B being
flashed on ~10T ordered tokens adds band 7 (~8.4M tok/tick,
"months", ~1,200 ticks — learnable) and marginally band 8 (~67M,
"years", ~150 ticks — thin). The judge becomes a learned reward
model (the press economy is RLHF made continuous and native);
ARM C becomes the built-in corrigibility channel; serving IS the
product — a being that wakes up knowing you and consolidates
nightly. Cost is frontier-lab (~3e25 FLOPs, tens of millions) —
not our lane. Our ladder: 500M (this run) -> 1.5-3B (one 8xA100
node, ~$5-20k) -> 7-13B. The bet the 500M demo tests: the
architecture's deltas (persistence, press-shaped consolidation,
correction) are SCALE-FREE; params buy the fluency underneath.

## 8. Open items

- ~~Band-6 debug certification~~ DONE (A70: parametric clocks,
  bit-parity default, six laws, suite 85/85).
- Judge design for the real-corpus arm (frozen heuristic + audit
  protocol).
- The 500M heartbeat/probe pack (section 6 instruments as code).
- Biography builder for real conversations (UltraChat lives:
  ordering, recurrence injection, staging schedule).
- Prophet graduation criteria (fidelity threshold measured across
  the flash, not bolted on after).
