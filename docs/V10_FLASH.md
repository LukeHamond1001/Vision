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

## 1b. Fidelity audit — brain vs this architecture (2026-08-19)

In ORGANIZATION: every major memory/reward system has a working
counterpart, several with the brain's own phenomena reproduced
(CLS wipe-survival, spacing curves, unrehearsed erosion, reward-
tagged decontextualized replay, a developmental pathology). In
PROPORTION and fine detail: known divergences, named here.

**Band count is principled by COVERAGE, not neuroanatomy** — the
cortex is a continuous log-spaced timescale gradient, so any
geometric ladder discretizes it; "right" = (a) no gap between
attention's reach and the first rung (band 3 = 1 chunk: exact),
(b) sane ratio (x8), (c) top rung slow enough for identity yet
ticking often enough to learn (band 6: ~9.5k weight-educating
ticks across the 10B flash; ~1-1.6k per life/lane). The count is
a function of lifetime length — an engineering scaling rule, not
a developmental claim; a longer flash earns band 7.

Honest divergences, in priority order — EACH NOW HAS A GATED
CANDIDATE IN THE v10 ORGAN PROGRAM (section 1c; user-set "go all
out", 2026-08-19):
1. PROPORTIONS — bands ~8% of params vs the brain's ~half-of-
   cortex association areas. -> A71 (slow-band capacity), gated;
   in-flight lesion probes still measure real load for v10.1.
2. Band states are single d-vectors, not maps — A71 half-closes
   (wider slow vectors + KD); true map-like states are the
   1.5-3B rung's job.
3. No instant aversive path (amygdala) -> A72 (hot-press salience
   tag), gated.
4. NREM only -> A73 (splice recombination) + A77 (dream on a
   leash), gated.
5. One neuromodulator -> A74 (surprise-weighted replay), gated.

Two FREE correspondences already present (noticed 2026-08-19):
cosine LR decay = declining plasticity with age — the flash has
critical periods (early data lands plastic, late data stiff);
store decay-with-turnover ~ neurogenesis slot recycling.
Adolescent synaptic pruning: OUT, not queued (user: "only stuff
that works") — revisited only if a measurement asks. Verdict:
after the organ program, no biological gap with an evidence hook
remains for a language being at 500M (completeness is BY
FUNCTION, not anatomy).

## 1c. The v10 organ program (A71-A77) — all debug-gated, $0;
winners ship, nulls stay out with evidence

- **A71 slow-band capacity**: per-band widths (bands 5/6 at
  1.5-2x d) + KD boost, targeting ~15-18% of params in bands.
  Gate: must beat MATCHED-PARAM uniform on the hybrid substrate
  (slowheavy lost once at v5.0, do-not-repeat #17); else the
  capacity goes to KD. Band-7 machinery certified config-only.
- **A72 amygdala tag**: a hot -press (v <= -2) tags its span; the
  correction pair is GUARANTEED that night's replay (skips the
  lottery). Instant capture; weight change stays in certified
  nightly ARM C — no wake-time negative gradients (A67-P7/P8).
- **A73 splice recombination** (label: SWS schema abstraction,
  not REM): sleep replays real spans recombined across episodes —
  no self-generated tokens. Gate: beats verbatim replay on
  cross-episode integration probes.
- **A74 surprise-weighted replay** (label: prediction-error-gated
  consolidation, not ACh): unpaid-span lottery weighted by
  harvest-time CE. Gate: recall-by-gap/CE beats recency-only;
  watch A67-P4/P5 pool dynamics.
- **A75 tied embeddings + 24-32k vocab** vs 16k untied at matched
  params (vocab scaling laws put 16k below optimum; tying must
  pass the STORE laws — logit-store values live in embedding
  space). RoPE: named v10.1 candidate (mem-token position
  geometry to be re-decided), not gated now.
- **A76 sleep homeostasis** (synaptic downscaling): tiny decoupled
  weight decay during sleep steps only, on sleep's certified
  weight surface. Evidence hook: the incumbent + rich-get-richer
  are no-renormalization diseases. Gate: mini-incumbent at debug —
  conviction saturation damped while taught recall and CE hold.
- **A77 dream on a leash** (true REM, math verified): generation
  SEEDED by a real memory span, N candidates, frozen-judge +
  fact-consistency selection, tiny-lr, <=1% of daily tokens — the
  provably-bounded ACCUMULATE regime (Gerstgrasser 2024), not the
  collapse (replace) regime; entropy contraction monitored with a
  pre-registered kill. STRICT double gate: must beat plain replay
  AND A73 on integration, zero entropy contraction, zero CE cost;
  may honestly defer to the served life ("dreams need a bigger
  brain"). Warning ledgered: generation samples the STRONGEST
  belief (A67-P6) — unguarded dreaming REHEARSES convictions; the
  fact-consistency filter is load-bearing, and the anti-conviction
  organs remain A76 + ARM C.

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
budget ~10x (we need ~10B; the named menu in 2b exposes >100B).**
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
Baguettotron, 321M trained on SYNTH alone, approaches Qwen-0.6B —
honest footnote: on 200B tokens and 80 layers, so it supports
synthetic-first, not 10B-is-plenty; our demo is delta-not-absolute
by design). Selection is the quality mechanism: ~10B needed from
>100B available means we take roughly the top 10% of everything.

**THE manifest (user-final 2026-08-19 — supersedes ALL earlier
share tables: the spine IS the corpus):**

| role | sources |
|---|---|
| infancy | the SHORTEST, SIMPLEST exchanges of the spine, sorted to the front (conversational infancy — no TinyStories) |
| childhood | UltraChat top-slice — the FILL, sized to whatever the spine needs |
| adolescence | complete SmolTalk2 (EN subsets, tool traces excluded) |
| the tail | complete Smol-Magpie-Ultra + SmolTalk2's judge-best top-up |

Everything else — TinyStories, Cosmopedia, SYNTH study days,
FineWeb-Edu grounding, peS2o/Gutenberg long documents, DCLM,
WildChat — is OUT (user directive: the life is conversations;
the role invariant gets the densest possible signal). Ledgered
consequences, eyes open: (a) budget = what ONE EPOCH of the spine
yields, ~5-7B tokens measured at download (cost ~$200-300; if the
user still wants 10B, <=2-epoch repetition is the named fallback);
(b) knowledge breadth narrows — acceptable: the demo is
delta-not-absolute and the claim is the architecture, not trivia;
(c) within-doc long-range food is gone — the bands feed ENTIRELY
on life structure (recurrence at band-clock gaps), which is the
architecture's own claim anyway (A69-R4), now undiluted.

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
SmolTalk2's judge-best to fill ~1B). Magpie's 3-turn shape is
fine last: day/session length and band-horizon recurrence are
BUILDER parameters, not source properties.

| source | size | role in the flash (LEAST -> BEST, like learning) |
|---|---|---|
| UltraChat (HF: stingning) | ~2.2B tok (1.5M dialogues x ~1.5k) | FIRST — infancy's simplest slices + the childhood fill; volume and long-multi-turn faculty; gpt-3.5-era teacher, judge top-slice only |
| SmolTalk2 (HF: HuggingFaceTB) | ~3.4M multi-turn samples | SECOND — complete (EN subsets, tool traces excluded); adolescence's long threads for band-5/6 horizons |
| Smol-Magpie-Ultra, core of SmolTalk (HF: HuggingFaceTB) | 400K three-turn convs | LAST — the tail: Llama-3.1-405B-distilled, ArmoRM reward-model filtered, press-worthy by construction, on the cosine tail where imprint is hardest |

Corrections (~3-8%) remain synthesized by us (section 3). All
sources are open (smollm-corpus/smoltalk ODC-By/Apache; UltraChat
MIT). Nothing is repeated within one epoch — the flash never sees
the same day twice, like a life. (If the user opts into a bigger
budget than one epoch yields, <=2-epoch repetition is near-fresh
per Muennighoff's data-constrained scaling — an explicit opt-in,
not a default.)

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
  judge, perfect by construction. V-scale: the PUBLIC instruments
  of 2b's judge-grounding clause (FineWeb-Edu classifier scores;
  HelpSteer2-calibrated frozen grader), fixture-locked and copied
  into every shard manifest — auditable by anyone.
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
| a 13-year upbringing | the flash (10B tok ~ 50x a 13-year linguistic diet — SGD's exchange rate) |
| adult life continuing | serving + raising (A67 rooms) |

## 5. Scale-down doctrine

Keep the ORGANIZATION, shrink capacity. Target 500M (d~1280,
~20L): the legibility threshold where the mouth speaks fluent
English and the architecture's deltas (wipe-survival memory,
press-shaped consolidation, waking up knowing you) become visible
to any audience. A100 80GB, fp32+TF32 regime, 10B tokens one
epoch, ~6-9 days, cost band $300-450 — settled by the paid
real-shard smoke BEFORE commit (A54d), never assumed. lr ~4e-5
(width-lr law A45) with ~1-2k-step WARMUP then cosine on the
global step; lam recomputed from measured holds/step (A60f
pairing); lanes smoke-determined; sleep dose ladder (sleepless
infancy, A64-R3). The demo frame: same weights frozen vs living —
delta-not-absolute BY DESIGN (a 10B-token mouth will not out-talk
4T-token industrial peers; the architecture's deltas are the
claim). Demo protocol pre-registered before launch: speech-gated,
within-run contrasts only, n>=20/class.

**Bands at 500M (honest gaps + the plan):** the brain is a
continuous timescale gradient; our x8 ladder (2k/16k/131k tok) is
a discretization. v10 ADDS BAND 6 (x8 again, ~1M tok — ~9.5k
weight-educating ticks across the 10B flash): the
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

Band ladder at T=2048 across the 10B-token flash (ticks =
weight-educating tick events summed over lanes; each LIFE/lane
sees ~1/lanes of these):

| band | clock | tokens/tick | ticks in flash | human analog | KD |
|---|---|---|---|---|---|
| 3 | 1 chunk | 2k | ~4.9M | a scene | 1024 |
| 4 | 8 | 16k | ~610k | a day | 2048 |
| 5 | 64 | 131k | ~76k | a week | 4096 |
| 6 | 512 | ~1.05M | ~9.5k | an era / the self | 8192 |

(kd_base=2 at T=2048. Store STATE is memory, not params: ~79MB/
lane fp32, ~1.3GB at 16 lanes — a heartbeat watch item, not a
blocker on 80GB.) Wiring as certified: mem tokens for all four
bands enter attention every chunk; store read injects at
mid-depth (L10); writes every chunk, decay = the band's clock;
gates biased shut (gate_init -2), economy opt-in.

The flash as a staged life (~10B tokens; quality FLAT-HIGH
throughout — the stages are ECONOMY stages, per section 2. SLEEP
DOSE LADDER, corrected per A64-R3's fresh-trunk collapse law:
infancy is SLEEPLESS (or homeopathic, every>=64); sleep ramps in
at the childhood boundary on a pre-registered ladder; press-pay
on and ARM C native once sleeping. One complete staged life per
LANE — lanes consume the shard in parallel, so every lane lives
infancy->tail across training time):

| stage | share | shape + material (all conversations, per 2b) | economy |
|---|---|---|---|
| infancy | 10% | short days, the spine's shortest/simplest exchanges, tight recurrence gaps | dense +1/+2, no corrections, NO SLEEP |
| childhood | 40% | biography days; facts/characters recur at gaps spanning band 3/4 clocks — UltraChat top-slice (the fill) | presses annealing down; corrections begin (~3-8%); sleep ramps in |
| adolescence | 40% | long threads, multi-session projects reaching band-5/6 horizons — complete SmolTalk2 (EN, no tool traces), one ordered life | sparse presses; prophets predicting |
| the tail | 10% | complete Smol-Magpie-Ultra + SmolTalk2's judge-best — the highest-pedigree exchanges, on the cosine tail | strictest judge audit; audit failure = kill |

(Token budget: one epoch of the spine, measured at download —
~5-7B expected; the 10B figure stands only if the user opts into
<=2-epoch repetition after seeing the measured number.)

Human mapping (user framing): ~0-2 / 2-8 / 8-12 / 12-13 years —
one 13-year role-played upbringing at ~30-50x the linguistic
volume.

**Judge grounding (public frozen instruments):** every exchange is
graded by the small frozen grader calibrated on HelpSteer2's
public human helpfulness ratings (iga/lm_judge, coefficients
frozen in-file, fixture-locked, re-derivable by anyone via its
calibrate CLI). Upstream public quality columns are honored where
a source ships them (Magpie-Ultra's ArmoRM scores); the
FineWeb-Edu classifier mapping stays in the module as dormant
machinery for any future document source. The A64 frozen-
instrument law with instruments anyone can check.

## 6. Gates and the run protocol (user-specified: probes,
heartbeats, kill, fix, relaunch)

Pre-flight (debug, $0): binder precondition (in-ctx recall >= 2x
chance — A69-R1's law), G1-G4 life gates on PREPARED real-corpus
mini-shards (G2 control = shuffle-sessions-KEEP-WORLD, per
A69-R4's own confound note — never cross_days=False), the organ
gates A71-A77 (section 1c), THE QUAD (hybrid-vs-transformer x
ordered-vs-shuffled — answering A50's cancelled twin at debug
scale) + a pre-registered written rebuttal of the
nightly-SFT-on-pressed-spans skeptic baseline, and the pair laws
suite (85/85 at A70).

In-flight heartbeats (500M, on checkpoints + live guard log):
- CE trajectory vs reference curve
- recall-by-gap curve (the band-education vital sign)
- prophet press-prediction AUC (value function forming?)
- press-contingency audit (judge sanity, sampled)
- band-lesion delta on checkpoints (are the bands carrying it?)
- collapse detector (babble/repetition rate in samples)
- incumbent detector (max false-stem belief mass)

Additional heartbeat rows (2026-08-19): lm-eval-harness mini-row
(HellaSwag/ARC-e/PIQA — external fluency legibility);
correction-collateral guard (matched uncorrected controls — the
A68-T S3 lesson); late-run health watch (v9.4 peaked at 266k of
488k); tail-imprint check (does tail material imprint under the
decayed LR — the best-data-last x LR-decay tension, measured);
pairs-per-night volume (untested-axis watch).

THE GROWTH CHART (soft milestones on the same probes — warn, not
kill; the pediatrician's chart beside the pathologist's): end of
infancy = fluent simple sentences (collapse clean, CE < ref); end
of childhood = binder armed + corrections landing (pair margins
moving); adolescence = prophet AUC rising, band-4/5 lesion deltas
positive; tail = recall-by-gap advantage at 131k+.

Kill criteria (any -> kill, fix, relaunch; a caught disease costs
hours, not the run; constants pre-registered in code BEFORE token
one): recall-by-gap flat after N tokens; prophet AUC ~0.5 late;
CE divergence; incumbent mass above threshold; judge audit
failure (tail mismatch >1% = grading/plumbing bug = kill).
Run-safety carried from the ledger: banking (peval_best persists),
atomic saves, resume-aware boot + false-start guard, NaN/stall
watchdog, holdout-only kill metrics (A60e template), --chunk
always passed, --lanes never omitted, drive-ledger pruning built
before the ~800k-step run (A54e F6).

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
already proven end-to-end. The two apt human analogies (added
2026-08-19): INFANTILE AMNESIA — faculties survive childhood,
episodes don't — and SCHEMA-ACCELERATED CONSOLIDATION (Tse et al.
2007: with a schema in place, new facts consolidate in days
instead of weeks — exactly R6's flash-born creature banking 9.9x
in a served life).

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

## 8. Open items (the approved pre-flight build order)

- ~~Band-6 debug certification~~ DONE (A70: parametric clocks,
  bit-parity default, six laws, suite 85/85).
- ~~Spec reconciliation~~ DONE (this commit: one manifest, public
  judge, sleep dose ladder, 10B numbers, organ program 1c).
- Judge module (PUBLIC instruments per 2b) + fixture + calibrate.
- Press plumbing (attribute= hygiene, specials kwarg) + parity
  fingerprint.
- Biography builder iga/lm_data_life.py (one life per lane).
- Organ gates A71-A77 (section 1c).
- Life gates G1-G4 + THE QUAD + skeptic rebuttal.
- The 500M heartbeat/probe pack (section 6 instruments as code) +
  drive-ledger pruning.
- Judge freeze -> full 10B build -> pod_v10.sh + paid smoke ->
  USER GO -> launch.
- Biography builder for real conversations (UltraChat lives:
  ordering, recurrence injection, staging schedule).
- Prophet graduation criteria (fidelity threshold measured across
  the flash, not bolted on after).
