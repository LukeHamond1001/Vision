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

1. **Developmental staging across the flash — of the ECONOMY and
   SHAPE, not of text quality (user-set 2026-08-19).** Quality is
   FLAT-HIGH from token one; what stages is the press density
   (dense -> sparse), session length (short days -> long threads),
   correction onset (none -> 3-8% once the substrate matures), and
   recurrence horizon (short gaps -> band-5/6 gaps). Honest note:
   complexity curricula (simple text first) have weak evidence in
   LM pretraining and NONE of our measured laws warrant one — every
   order effect we measured is an economy/recurrence effect. The
   warrants that stand are economy warrants: thin-pool winner-take-
   all (A67-P4/P5), corrections require a mature substrate (A64-R3
   dose law), all-positive infancy minted a 0.96 false conviction
   (A67-P6) — so corrections must not wait too long, either.
2. **Day order across the life** — the band curriculum: facts,
   characters, and threads RECUR at controlled gaps spanning every
   band clock (96 ... 131k+). The only invariant at band-5 horizon
   is the role itself ("across all days, I am the one who helps,
   and presses come when I help") — that is what band 5 is FOR.
3. **Exchange order within a day** — arcs: open ritual, work,
   close ritual; sleep interleaved at day boundaries (certified
   dose ladder), pairs riding the nights (A69).
4. **Token order within an exchange** — natural language (given).

**Quality is FLAT AT MAX (user-set, 2026-08-19, superseding the
anneal): highest-quality data throughout — there is no reason to
feed anything worse when top-tier public material exceeds the
budget ~20x (we need ~6B; the named menu in 2b exposes >100B).**
What stays staged is COMPLEXITY and structure, not quality:
TinyStories is immaculate but simple — quality != difficulty, so
the developmental warrants survive intact. The tail-audit also
survives, re-derived: recency makes the final ~10% the run's
highest-stakes surface regardless of the quality curve, so the
judge audit still concentrates there and audit failure there is
still a kill criterion. Industry validation (2025): OLMo-2's
Dolmino mix and SmolLM2's reserved FineMath/Stack-Edu both
introduce their highest-quality data exactly there — late-stage
curriculum during annealing — and measure large downstream wins.

### 2b. The corpus (named, public, verified 2026-08-19)

Synthetic-first — the user's instinct matches the evidence chain
at small scale (TinyStories -> phi -> Cosmopedia -> SYNTH:
Baguettotron, 321M trained on SYNTH alone, approaches Qwen-0.6B).
Selection is the quality mechanism: ~6B needed from >100B
available means we take roughly the top 5% slice of everything.

**The conversational-spine order (user-set, 2026-08-19): flat
floor, rising ceiling.** Every source passes the same judge
floor (no bad data ever enters), and teacher PEDIGREE rises
across the flash: UltraChat is the FILL (childhood — volume and
long-thread faculty; gpt-3.5-era teacher), then COMPLETE
SmolTalk2 (adolescence — English subsets only, tool-calling
traces excluded: a being with no tools gets no tool syntax in
its life), then COMPLETE Smol-Magpie-Ultra last (the tail —
405B-distilled, ArmoRM-filtered, press-worthy by construction,
riding the cosine tail where imprint is hardest; topped up with
SmolTalk2's judge-best to fill ~0.6B). Magpie's 3-turn shape is
fine last: day/session length and band-horizon recurrence are
BUILDER parameters, not source properties.

| source | size | role in the flash |
|---|---|---|
| TinyStories (HF: roneneldan) | ~0.5B tok | infancy — synthetic, immaculate, simple |
| Smol-Magpie-Ultra, core of SmolTalk (HF: HuggingFaceTB) | 400K three-turn convs | the PREMIUM exchanges: Llama-3.1-405B-distilled, ArmoRM reward-model filtered, safety-screened, semantically deduped — press-worthy by construction |
| UltraChat (HF: stingning) | ~2.2B tok (1.5M dialogues x ~1.5k) | the volume + long-multi-turn spine — biography builder rebuilds these into recurring days and characters; gpt-3.5-era generation, so the judge takes its top slice only |
| SmolTalk2 (HF: HuggingFaceTB) | ~3.4M multi-turn samples | more spine: tool traces, long-context threads for band-5/6 horizons |
| SYNTH (HF: PleIAs) | ~75B tok | "study days" — reasoning-dense, formally verified synthetic playgrounds; the being reads and discusses |
| Cosmopedia-v2 (HF: HuggingFaceTB) | 28B tok | textbooks/stories; audience tiers (middle-school early, college late) give complexity staging at flat quality |
| FineWeb-Edu dedup (HF) | 220B tok | ~10-15% grounding blend, top classifier decile only — synthetic-purity hedge against distributional narrowing |

Corrections (~3-8%) remain synthesized by us (section 3). All
sources are open (smollm-corpus ODC-By; SYNTH released fully open
with the AI Alliance). Nothing is repeated: at 6B from >100B the
flash never sees the same day twice — like a life.

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

The flash as a staged life (~6B tokens; quality FLAT-HIGH
throughout — the stages are ECONOMY stages, per section 2; sleep
at every day boundary, press-pay on, ARM C native):

| stage | tokens | shape + material (all top-slice, per 2b) | economy |
|---|---|---|---|
| infancy | ~0.6B (10%) | short days, short exchanges, tight recurrence gaps — TinyStories | dense +1/+2, no corrections |
| childhood | ~2.4B (40%) | biography days; facts/characters recur at gaps spanning band 3/4 clocks — UltraChat top-slice (the fill) + Cosmopedia middle-school study days | presses annealing down; corrections begin (~3-8%) |
| adolescence | ~2.4B (40%) | long threads, multi-session projects reaching band-5/6 horizons — complete SmolTalk2 (EN, no tool traces) + SYNTH + Cosmopedia college-tier, still one ordered life | sparse presses; prophets predicting |
| the tail | ~0.6B (10%) | complete Smol-Magpie-Ultra + SmolTalk2's judge-best — the highest-pedigree exchanges, on the cosine tail | strictest judge audit; audit failure = kill |

**The data manifest (public sets, pinned before launch):**

| role | source | ~share | why |
|---|---|---|---|
| world spine | FineWeb-Edu (high cut) 60 / DCLM 40 | 55% | best classifier-filtered web; the blend's complementary strengths are documented (edu benchmarks vs commonsense) |
| synthetic textbooks | Cosmopedia v2 | 10% | phi-style explanation density per token — what a 500M capacity digests best |
| long documents (band food) | peS2o + Gutenberg (+ FinePDFs) | 10% | documents LONGER than the band-4/5 clocks so bands get within-doc education, not only life-structure education |
| conversational biography | SmolTalk + UltraChat + WildChat, rebuilt by the biography builder | 20% | the life itself: persistent cast, recurrence injection, synthesized corrections, press economy |
| the tail | judge-top slice: HelpSteer2-graded dialogue, FineMath/Stack-Edu, top-decile Cosmopedia + spine | 5% | the anneal (Dolmino/SmolLM2 precedent) |

Synthetic-data doctrine: synthetic is the best per-token teacher
at this scale (phi/Cosmopedia evidence) but pure synthetic
narrows the distribution — blend it over the filtered-web spine.
The biography layer is inherently synthetic-STRUCTURED anyway
(real utterances, manufactured life: ordering, recurrence,
corrections) — that is the builder's whole job.

**Judge grounding (public frozen instruments):** reading days are
graded by the released FineWeb-Edu educational classifier (its
0-5 score maps to silence/+1/+2) and dialogue days by a small
frozen grader calibrated on HelpSteer2's human helpfulness
ratings. Both are public and auditable — the A64 frozen-
instrument law with instruments anyone can check, instead of a
homemade heuristic.

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
