# v5.0 — The language ladder (design card, committed before any run)

*(Registered 2026-08-05, before any code or run. Amendments only pre-run,
ledgered here, per house rules. This card extends the campaign to a new
substrate: language modeling with NO attention — the band ladder as the
only sequence mixer, the drive layer as the only auxiliary objective.)*

## Claim under test

A multi-timescale band ladder — recurrent state at geometrically slower
clocks, with the frozen drive layer proposing and paying targets over
audited channels — can hold document-scale meaning that a same-compute
transformer structurally cannot hold past its context window, and the
drive layer's self-scheduled curriculum beats uniform training at
matched compute.

Two sub-claims (A2: L1 is this round's registered claim; L2 runs but
is observational this round — its causal gate is deferred to the next
card, since no bands-without-drive arm is trained):
- **L1 (holding):** the ladder carries long-range structure with no
  attention anywhere; long-range performance does not cliff at any
  window boundary, because there is no window.
- **L2 (growing up):** the drive layer, holding targets over the
  model's own measured competences and scheduling data by learning
  progress, improves on uniform training at matched compute.

## The model (one arm — A2)

One trained model per size: the **full architecture** — the band
ladder (each band ~8× slower than the one below, each embedding the
window of the band below and predicting the next window; the
v0.5–v0.9 recipe: context coupling, within-band whitening, boundary
masking) plus the drive layer. No trained baseline, no bands-only
arm. The control is structural: with attention absent, any long-range
behavior can only flow through band state, and the **lesion test**
exercises that control at evaluation time. Reference points: public
checkpoints of similar size (e.g. Pythia-70M/160M), downloaded not
trained, labeled non-matched (different data and compute) — landmarks,
not baselines.

Sizes: ~10M, ~30M, ~90M params. **One seed per size** — replication
is carried by the size axis: no claim counts unless its direction
holds at all three sizes. This is standard scaling-study methodology;
LM pretraining is low-variance relative to RL.

## Band geometry (content bands)

Clock ratio ~8× per band. Band 1 ticks per token (state spans words);
band 2 ~8 tokens (phrase); band 3 ~64 (paragraph/topic); band 4 ~512
(section); band 5 ~4k (document intent); band 6 ~32k (book scale,
active only on long documents). **All sizes carry all six content
bands (A3) — state widths scale with model size — plus the continuous
competence band.**

**The conveyor law (A1, completed by A6): the life is the training
run, and the stream is one unbroken dialogue.** No scene brackets, no
masking, nothing ever reset — the model lives once and reads on. The
referent law survives without brackets as a **horizon**: every hold
carries a time budget set by its band's clock (scaffold default 4
ticks — a placeholder; before registered runs, horizons are frozen
from measured ask-back statistics on the calibration split, per the
data-set-not-hand-set discipline in docs/ROBOT_PROGRAM.md); at the
horizon it settles on the readings that arrived during the hold, or
expires at exactly zero. No hold outlives its horizon; the telescoping
audit applies per hold; closed loops still net zero.

**The competence band is continuous.** One slowest register bank runs
on the training-run clock, holding targets over the model's measured
competences (below), paid at hold ends on measured improvement. Its
referents — the model's own competences — genuinely persist, so its
holds may span the whole run. This band never resets: it is the run
growing up.

## Data (public, free, mixed; ratios frozen before runs)

All-synthetic (A5) — every cue, boundary, and label known by
construction; the reward word is placed by verified outcome, never by
parsing:

- **UltraChat** — synthetic human↔assistant dialogue; the turn-taking
  substrate.
- **OpenHermes / Magpie-class** — synthetic instruction-chat
  episodes.
- **Cosmopedia** (sampled) — synthetic long-form prose; mid-band
  food.
- **TinyStories** — clean simple English; the fluency floor that
  makes the 10M go/no-go readable.
- **Public synthetic agent trajectories** (AgentInstruct,
  ToolBench-class tool-use episodes) — goal → actions → outcome
  scenes.
- **Procedural weaver (ours; the precision core, and A6's sole v0
  source)** — one endless person↔agent conversation per lane: facts
  planted in human turns and asked back at controlled gaps (up to
  24k tokens — band 6's food), find-the-object tasks verified by a
  world sim; under A7 the depicted agent is always right and every
  exchange is thanked (failure rates parameterized for real-data
  rounds).

**Scope (A5):** training and headline evaluation are synthetic
held-out (generator-verified ground truth). Real-data headlines are
deferred to the next card; this round's claims say "on synthetic
data" wherever they are quoted.

### Stream markers (A6 — two special tokens, nothing else)

- `<eot_human>` / `<eot_model>` — the only special tokens in the
  stream. The page shows pure conversation, like regular LLM data.
- The earned mark is natural human speech — "thanks . good job ." —
  and under A7 every exchange earns it: v0 data depicts only correct
  answers and successful tasks, so uniform thanks is truthful. The
  pipeline records it as an invisible `earned` event; a thanks MINTS
  the channel it followed and never pays. (With a uniform label the
  minting gate fires trivially — its selective function is dormant by
  construction in v0 and reactivates on real data, where the failure
  branches, kept in code behind rate parameters, return.)
- Probes (planted-fact ask-backs) are invisible events; under A7
  every depicted answer is true, so every ask-back is annotated.
- The model never earns by emitting anything: reward exists only at
  training time, computed by the frozen ledger from pipeline events.

Token budgets ~chinchilla: 10M→~200M tok, 30M→~600M, 90M→~2B.
Three splits under the instrument discipline: train / calibration
(probes fitted and audited here; never sees eval) / held-out eval.

**The label law: labels select, the world pays.** Quality scores and
accepted/thanked endings gate which endpoints may mint arrival
templates. No label is ever a reward. Reward is paid only by the
frozen ledger on measured arrival. A uniform label (appended to
everything) carries zero information and is banned by construction.

## Channels (all data-grounded; all pass a held-out admission audit
before bearing reward, or they do not become wants)

The anti-wireheading condition, transposed: a channel must be graded
against the part of the stream the model has not seen (the model
cannot edit the future of the text) — never against a free reading of
its own latent.

- **recall-at-distance** — when an entity returns after a gap, does
  the slow-band state identify it against distractors (truth from the
  text itself); binned by gap length.
- **next-window fidelity per band** — band-k's prediction of the next
  window embedding, graded when the window arrives.
- **topic persistence** — band-3/4 state vs the realized topic of the
  following window.

Admission: closed-form probes calibrated on the calibration split,
held-out audited; the audit table is committed with the runs.

## Drive layer

- Registers per content band hold targets over admitted channels
  (maintain: keep each band's competence in its healthy range;
  frontier: push each past its record, once per level). φ = shortfall;
  pay = φ_prev − φ_now at the band's clock; telescoping ⇒ closed loops
  net zero; pay lands at hold ends; audited exact from training logs.
- The continuous competence band holds improvement targets across the
  run.
- **Scheduler:** the proposer allocates the next training bucket
  (source × quality × length) by measured learning progress on the
  competence channels. Buckets that stop paying stop being chosen.
- Imagination gate: candidate targets are screened by predicted
  reachability/health before commitment; the gate ranks and vetoes,
  never pays.
- Nothing in the pay path is trained. Ledger code is shared with the
  agent campaign.

## Evaluation

- **Recall-at-distance curves** on held-out synthetic scenes
  (planted-dependency; generator-verified ground truth): performance
  vs gap length, overlaid with the reference checkpoints'
  context-window boundaries. The registered picture: references
  cliff at their windows; the ladder's curve does not cliff, because
  it has none.
- **Window dial (references only, eval-time, $0):** public
  checkpoints re-evaluated at shrinking windows (2048/512/128) to
  render the cliff the ladder is claimed not to have.
- **Lesion (the control):** slow bands knocked out at eval — live,
  repeatable; beyond-scale recall must collapse while local
  perplexity survives. With no attention present, whatever the lesion
  kills was band-carried.
- **The talking eval (demo protocol):** plant a name or fact early in
  a held-out document or dialogue; continue thousands of tokens of
  unrelated stream; probe whether it returns. Run with the register
  panel open — every band's held state printed live from audited
  probes — and repeat with band 5 lesioned on camera. Persistence,
  not eloquence, is what is being read. The go/no-go read at 10M
  distinguishes three outcomes: gibberish (engineering failure —
  fix and retry, no verdict on the claim), fluent-but-memoryless
  (the bet failing — the honest negative), fluent-and-holding (L1
  existence — with no attention present, what is held is
  band-held).
- **Short-range perplexity** vs reference checkpoints (descriptive,
  non-matched; reported, not gated).
- Ledger audit: telescoping exactness recomputed from committed logs.

## Gates (pre-registered; exact constants may be amended only pre-run,
with the amendment ledgered here)

- **G-context (existence):** at every size, entity recall at gaps
  ≥4k tokens beats the distractor-chance floor by a margin fixed
  pre-run (after the probe admission audit, before any training).
  With no attention in the model, passing this gate is attributable
  to band state.
- **G-flat:** the ladder's recall-at-distance curve from 512-token
  gaps to 8k-token gaps degrades by less than half the drop the
  reference checkpoints show crossing their own window edge.
- **G-trend:** the ladder's beyond-4k recall is monotone
  non-decreasing across 10M→30M→90M.
- **G-lesion:** slow-band lesion reduces beyond-4k recall by ≥50%
  while local perplexity moves <10%.
- **(Deferred, A2):** the L2/drive-layer causal gate — requires a
  bands-without-drive arm; registered for the next card. Scheduler
  decisions are logged this round and reported descriptively.
- Misses are printed beside passes, as always.

## Compute and cost (A3)

Debug on RunPod RTX 2000 at $0.24/hr (the campaign's workhorse);
registered runs on a rented RTX 4090 (~$0.35–0.70/hr): 10M in under
an hour, 30M in ~2–4 h, 90M in ~20–40 h. Kaggle free tier for
spillover debugging. TRC application submitted for the next card's
scale, not blocking this one. Target cost: $15–40 total.

## Amendment ledger

- **A1** (2026-08-05, pre-run, pre-code): the original episode framing
  ("a document is a life; bands and registers reset at boundaries;
  documents are never concatenated") is superseded by the conveyor law
  above: one continuous stream, boundary-masked fast state, holds
  scoped to their referents. The accounting is unchanged — every hold
  still settles within one scene and the audit is identical — but the
  model lives once, for the whole run.
- **A2** (2026-08-05, pre-run, pre-code): the three trained arms
  (baseline / bands / full) are reduced to one — the full
  architecture only. Rationale: with attention absent, any long-range
  capability is attributable to band state, so the lesion test is the
  control and trained baselines are replaced by downloaded reference
  checkpoints (labeled non-matched). Cost, priced openly: the
  drive-layer causal claim (L2) loses its gate this round and is
  deferred; this round registers existence (L1) only. Gates
  re-registered accordingly above; no runs existed under the old
  gates.
- **A3** (2026-08-05, pre-run, pre-code): six content bands at every
  size (widths scale, clocks fixed); data goes dialogue-primary
  (WildChat, OASST, HH-RLHF added; books and web retained for long
  scenes) with turn-end markers `<eot_human>`/`<eot_model>` and
  turn-scoped holds; compute plan set to RTX 2000 debug → RTX 4090
  registered runs.
- **A4** (2026-08-05, pre-run, pre-code): synthetic tier added
  (TinyStories, Cosmopedia, UltraChat, and a planted-dependency
  generator) with the mix policy: synthetic-heavy at 10M annealing to
  real-heavy at 90M; headline numbers quoted only on real held-out
  data. The talking eval's three-outcome read (gibberish /
  fluent-but-memoryless / fluent-and-holding) registered as the
  go/no-go interpretation. Success bar restated: L1 is existence —
  some form of the attention function held in band state — not
  superiority over any baseline.
- **A5** (2026-08-05, pre-run, pre-code): all-synthetic supersedes
  the A3/A4 mix. Rationale: every boundary cue and earned-mark is
  known by construction, `<ok>` is placed by generator-verified
  outcome (so the label varies truthfully), fluency at tiny scale is
  derisked, and the pipeline sheds all real-data parsing. Cost,
  printed: this round's claims are scoped "on synthetic data";
  real-data headlines are the next card's work. Real corpora
  (WildChat, OASST, HH-RLHF, PG-19, FineWeb-Edu, StackExchange)
  remain listed in A3/A4 as the rung-2 roster.
- **A6** (2026-08-05, assembled pre-run): no scene tokens. The stream
  is one endless person↔agent dialogue; the only special tokens are
  `<eot_human>`/`<eot_model>`; earned marks are natural speech plus
  invisible pipeline events ("thanks" mints the channel it followed);
  scene masking is removed (nothing unrelated exists to firewall —
  the conversation is genuinely continuous), and hold settlement is
  horizon-scoped (~4 ticks of the hold's band; expiry pays exactly
  zero, readings count only if strictly inside the hold). Scheduler
  wired as bin-weights biasing the weaver's plant gaps. Code and law
  tests updated; suite 45/45.
- **A7** (2026-08-05, assembled pre-run): all-good data — every
  depicted answer correct, every task successful, every exchange
  thanked. Truthful because nothing failing is depicted; the earned
  label therefore carries no selection information in v0 (printed),
  minting fires trivially, and what varies and pays remains the
  model's measured readings. Failure branches kept in code behind
  rate parameters for real-data rounds. Horizon constants flagged as
  scaffold defaults to be frozen from calibration-split ask-back
  statistics before registered runs. Suite 46/46.
- **A8** (2026-08-05, assembled pre-run): the training corpus is
  UltraChat — real synthetic long back-and-forth — prepared onto the
  conveyor by `iga/lm_data_ultrachat.py`: every human turn ends
  `<eot_human>`, every agent turn `<eot_model>`, every conversation
  closes with the human's "thanks . good job ." turn, all
  concatenated into one unbroken stream; sparse instrument convos
  (planted facts, ask-backs at gap targets to 24k) threaded between
  conversations as the measuring layer, probe positions
  token-verified at build time. Tokenizer: ByteLevelBPE (16k
  registered; 8k on the smoke shard). The closed-vocab weaver
  remains as the debug conveyor. Long conversations carry natural
  cross-turn dependencies (1–3k tokens), so mid bands get real work
  beyond the instruments. Proposer dedup added: one register per
  channel per lane. Suite 46/46; real-data smoke passing (CE falls,
  ledger exact, probes aligned, gaps measured to 7k+ on a
  400-conversation shard).
- **A9** (2026-08-05, assembled pre-run): dense talk + calibration.
  (1) Talk mode: `BandLM(talk="dense")` projects every band's held
  state into band 1's input at every token — reads only; write
  clocks untouched, so holding is unaffected by construction. The
  debug tier A/Bs dense vs tick; the winner is frozen before
  registered runs. (2) `iga/lm_calibrate.py`: horizons = 2 x p90 of
  measured ask-back gaps per bin (floor 4 ticks); chance floors per
  bin from a random-init model on the calibration split (the
  G-context gate's empirical floor); fidelity floors = random-init
  mean + 2 sigma. Drive consumes the constants file. First artifact
  (weaver calib split) committed at results/lm_constants.json — and
  it already corrected the scaffold: random-init band-1 fidelity
  floors at 0.25 where hand-set floors sat at 0.05-0.15, so the
  hand-set imagination veto would have trusted an ignorant band.
  Registered-run precondition: re-run calibration on the real-data
  calibration shard and freeze. Suite 48/48.
- **A10** (2026-08-05, assembled pre-run): width shaping — per-band
  state widths (`shape_widths`: uniform | slowheavy, where slow bands
  get 1.5-2x room; they carry the most compressed meaning). The debug
  A/B (`iga/lm_ab.py`) covers talk x shape at matched params (base
  width searched per cell, ~3% tolerance) and reports CE + short-gap
  recall + throughput; winners frozen in this card before registered
  runs. Pod bootstrap committed (`scripts/pod_debug.sh`): shard prep
  (train + disjoint calibration shard), the A/B, real-data
  calibration — one paste on the RTX 2000. Suite 49/49.
- **A11** (2026-08-06, pre-run): the three-size ladder collapses to
  ONE registered run. One pod session decides everything — the
  talk x shape A/B, a lanes/compile throughput sweep on the winner,
  and real-data calibration — then every choice (talk, shape, lanes,
  compile, model size, constants) is frozen here, and a single
  registered run executes at the largest size that fits ~12 hours at
  the MEASURED throughput. Cost, printed: with one size and one seed
  there is no replication axis at all — G-trend is dropped, and this
  round is a pure n=1 existence demonstration (G-context, G-flat,
  G-lesion at the single size), with the lesion as the only control.
  Incident ledger from the first pod attempt, also on the record:
  a roster-exhaustion infinite loop in the instrument generator
  (64 name-object combos, never freed; froze the pod and the local
  build at ~1,000 conversations — fixed: pairs free on ask, deep-run
  regression test added) and HF unauthenticated streaming throttle
  (fixed: bulk-download the raw file; iter_convos prefers local).
  ~$0.80 spent on the lessons.
- **A12** (2026-08-06, FROZEN from the measured debug session —
  pod mpavr4z85, A4000, ~25 min, ~$0.07): the registered run's
  configuration. **talk=tick** (dense lost by ~0.15-0.2 nats CE in
  both shapes at 400 steps — early slow-band noise injected into
  every token; dense's hypothetical late-run advantage goes to the
  rung-2 list, and the A9 dense wiring is benched by its own audit).
  **shape=uniform** (beat slowheavy in both talk modes). **lanes=32**
  (throughput scaled linearly 8.4k -> 32.1k tok/s; 128 lanes OOM on
  16GB from the [B,T,V] logits tensor). **No torch.compile** (only
  tested at the OOM'd size; unproven code does not ride a 12-hour
  run); TF32 matmul + expandable-segments allocator on. Constants
  from results/lm_constants_real.json (fidelity floors measured on
  the real calibration shard; horizons mostly at 4-tick defaults —
  probe sparsity in the calibration window, thinness printed).
  Size: **d=256 (~16M params), chunk 512, ~600M fresh tokens
  (~37k steps), est. 5-8 h, ~$1.30.** The memorization trap is a
  law of the run: recall probes measure held state only if planted
  facts never repeat across epochs, so the train shard must be
  ~500k unique conversations (parallel prep) and the eval/calib
  shards come from a disjoint raw file. No mid-run resume: if the
  pod dies, the run reruns from zero rather than splicing a ledger.

- **A13** (2026-08-06, post-run-1): run 1 executed the full frozen
  config (36,500 steps, 12.5 h, CE 3.49→2.95, 20,896 holds settled,
  fidelity records on all six bands, measured 13.8k tok/s) and is
  **VOID for claiming**: (1) the eval shard was prepared with a
  freshly trained tokenizer, so the held-out tables and talk sample
  were computed in a vocabulary that is not the model's — the 0.000
  readings are mistranslation, not measurement; (2) the checkpoint
  push failed silently and blocked all later pushes, and the pod
  removed itself with run.pt aboard — the artifact is lost. What
  survives, printed as telemetry only (not a registered verdict):
  training-stream recall on never-repeated facts reached records
  0.151 (2k-16k gaps) and 0.106 (16k+), EMA ~0.07-0.08 against the
  measured 0.0001 chance floor — held-state signal ~700x above
  chance in an attention-free model. Fixes for the rerun, all
  committed: prepare() reuses the train tokenizer for eval/calib
  shards; probes whose plant falls outside a lane's eval segment are
  marked unanswerable and skipped; the checkpoint is pushed FIRST
  after training, with verification against the remote SHA, retries,
  and a 6-hour pod-hold rescue window on failure. The rerun repeats
  the identical frozen A12 configuration.

- **A14** (2026-08-07, post-run-2): run 2 trained clean and REPLICATED
  run 1's telemetry (records recall:b2 0.135 / recall:b3 0.122, EMA
  ~0.076/0.062; fidelity on all six bands, fid:5 to 0.357; 33,376
  holds; 15.4k tok/s) — and is also VOID for claiming: the ~67MB
  single checkpoint push failed again (now established as systematic),
  and the failure branch's `git reset --hard` restored the working
  tree, DELETING run.pt from the pod's disk six hours before the eval
  needed it; the eval crashed on the missing file and the wrapper
  reported success anyway. Honesty notes printed here rather than
  discovered later: (1) the registered chance floor (0.0001,
  random-init) is weak — a format-only model that learned "a color
  goes here" scores ~0.08 with zero memory, and the training EMAs sit
  near that value while only the records clear it; the run-3 verdict
  will be read against BOTH floors, and the lesion delta (format
  survives, memory dies) is the load-bearing discriminator. (2) Model
  init was unseeded in runs 1-2; A14 pins torch.manual_seed. Wrapper
  v4 for run 3: eval runs FIRST after training; checkpoint travels as
  25MB pieces with per-piece verified pushes; failure path uses
  reset --mixed (keeps files) + external fp16 mirror; rolling ckpt
  snapshots push to a side branch every 2h DURING training from an
  isolated repo dir (mid-run death can no longer lose the artifact);
  hb reports eval/train exit codes truthfully; eval widened to 160
  chunks for larger n (pre-run, ledgered).

- **A15** (2026-08-07, THE VERDICT — run 3, valid end to end): the
  infrastructure was flawless (heartbeats, rolling snapshots, eval
  first, all checkpoint pieces verified on remote, self-removed;
  6.2 h training at 26.7k tok/s, CE 2.93 — best of campaign, ~$1.40).
  The science verdict, printed in full: **L1 NOT DEMONSTRATED at
  16.7M under this training signal.** Registered cold tables (n=16):
  full 0.072/top1 12%, lesioned 0.040/6% — above the weak registered
  floor, AT the honest color-prior floor (~0.083). The local battery
  on the final weights settled every open question: (1) lesion-CE
  control FAILED memory-specificity — warm CE rose +27.2% under
  lesion (bar: <10%), so the recall drop is general breakage, not
  removed memory; (2) warm-protocol recall is prior-level
  (0.069/9%, n=23); (3) binding margins are zero-to-negative
  (-0.010 vs all colors, -0.003 vs recency set). The replicated
  training-stream signal (EMA ~0.076, records 0.13-0.15) is now
  fully explained as instrument FORMAT plus a drifting color prior
  (uniform-over-12 = 0.083; favored-cluster = 1/8 pending set) —
  not recall. Root cause, stated in the architecture's own terms:
  the channel paid on p(answer), prior-tracking was the cheapest
  path to that pay, and the machine learned exactly what the ledger
  rewarded and nothing more — the v3.0 mis-specification lesson,
  recursively demonstrated on our own instrument. What HELD: bands
  carry real predictive context (held-out CE 5.46 cold -> 4.64 warm,
  ~0.8 nats of band-borne context; lesion breaks it), all six band
  predictors learn (fid:5 to 0.36), the exact-accounting ledger was
  perfect across all three runs, and telemetry replicated 3x.
- **v5.1 (design, not run):** make binding the only paid path —
  (1) binding-margin channel: pay p(correct) minus max p(other
  pending colors), closed-form, parameter-free; prior- and
  recency-tracking become worth exactly zero; (2) graduated gap
  curriculum: restore short-gap instruments so binding is learned
  first where it is nearly free, then frontiered outward — the
  proposer's own philosophy applied to the training data; (3)
  densify instruments ~5x. One run at the frozen A12 config,
  ~$2.50.

- **A16** (2026-08-07, pre-run, registered): **v5.1-lite — the
  binding-curve probe.** Purpose: measure binding vs distance under a
  pay rule that makes non-binding worth zero. Changes from A12, all
  committed: (1) binding-margin channel — probe reading =
  max(0, p(answer) - max p(distractors)), distractor colors recorded
  per ask (pending + recent pool, min 3); (2) graduated instruments —
  short self-contained units at ~48/200/800-token gaps (plant, filler
  chatter, ask inside one slot) plus long facts at 3.2k/12.8k;
  density instrument_every=1 (smoke shard: 570 probes/800 convos,
  every curve bin populated, alignment token-exact); (3) debug scale:
  d=128 (7.4M params), 10k steps, ~164M tokens, one epoch of 150k
  dense conversations; ~2h, ~$0.50. Scope: mechanism probe — claims
  at debug scale on synthetic-instrumented real chat; no calibration
  pass (scaffold constants; ledgered). Registered read: the deliverable
  is the binding-vs-distance CURVE on the held-out dense shard
  (bins <64 / 64-256 / 256-1k / 1k-4k / 4k+), warm protocol, with
  lesion and CE control. Interpretation table, committed in advance:
  curve alive and extending -> scale (rung 2); solid short but dying
  at fast-band reach -> the query-blind readout is convicted and the
  rung-2 design is the clocked associative matrix (multiplicative
  query-memory readout); flat even at gap<64 -> deeper objective or
  capacity issue, debug before any scale-up.

- **A17** (2026-08-07, pre-run, registered — user intent clarified):
  **v5.2 — the Crafter machine on an LLM.** The claim was never
  "language without attention"; it is the drive layer that worked on
  Crafter, working on a language model. Substrate: a standard
  decoder-only transformer, regular attention intact, unmodified —
  playing exactly PPO's role from the agent campaign. On top, the
  full drive layer: frozen instruments, registers per timescale,
  maintain+frontier proposer, telescoping ledger, thanks-minting,
  binding-margin channel, graduated instruments, scheduler. One
  ledgered subtraction: the imagination gate's forward model was the
  band predictors; the transformer carries none, so the gate holds no
  authority (F2: absent instruments cannot veto) — fid-maintain and
  fid vetoes are skipped. One arm, existence claim: an LLM whose
  training is governed by an unfarmable goal ledger — margin-verified
  binding, self-scheduled practice, wants readable as text, exact
  accounting. Physics printed in advance: the transformer's memory
  ends at its 512-token window, so the binding curve is expected to
  show native binding inside the window and a wall at its edge — the
  wall is a measurement of standard training's boundary, not a
  failure, and it is where the band ladder would later earn its place
  as extension. d=128-class, 10k steps, one epoch of the dense shard;
  transformer parallelism makes this ~15-30 min, ~$0.15. The v5.1-lite
  band run (A16) proceeds in parallel as substrate science.

- **A18** (2026-08-07, twin verdicts): v5.1-lite (bands) and v5.2
  (transformer + drive) both trained clean at 164M tokens with the
  margin channel and graduated instruments (255k holds settled on
  lite; all pieces landed on both; ~$0.90 combined). Both read FLAT
  at the color-prior floor at every gap — including gaps INSIDE the
  transformer's attention window with the fact in plain view
  (v5.2: 0.085/8% at 0-256; lite: 0.077/5%; n=250 each) — and
  lesion deltas are negligible. The paired reading: this is not an
  architecture verdict; it is an emergence-threshold verdict. Neither
  substrate reaches in-context binding at 164M tokens / few-M params
  — consistent with the known induction-emergence regime for small
  transformers (~1B+ tokens). The A16 interpretation table's
  "flat everywhere" branch fires, amended by the control: the deeper
  issue is training quantity, shared by both substrates. Next:
  v5.3 past the threshold.

- **A19** (2026-08-07, pre-run, registered): **v5.3 — the complete
  architecture as an LLM, past the emergence threshold.** Substrate
  (`iga/lm_hybrid.py`): a standard transformer over each 512 window
  (language + in-window lookup) with slow band latents at clocks
  512/4k/32k persisting across the whole run, updated from pooled
  hidden states at their ticks, each carrying a predictor — so the
  forward model exists and THE IMAGINATION GATE IS LIVE (smoke: 30
  vetoes, fid:3 measuring, maintain holds on all slow bands, frontier
  register open). The slow latents are injected back as MEMORY TOKENS
  the transformer attends over — query-conditioned readout of the
  latent ladder (the multiplicative lookup the pure-band machine
  lacked). Full drive layer: margin-paid binding, graduated
  instruments, minting, scheduler, exact ledger. Carry-band remap for
  frontier proposals: bins 0-3 -> bands 3/3/4/5. Run: the ENTIRE
  UltraChat corpus, ~1.5M conversations / ~1.75B unique tokens, one
  epoch (no fact repeats), d=128, 107k steps, ~6 h, ~$1.40. Read
  against the A18 twins as same-architecture-class controls at 164M:
  the question is whether binding emerges past the threshold under
  the complete machine, at which bins, and whether the lesion
  (memory tokens + band reads zeroed) now carves it out
  specifically. Single arm, per the program's standing rule.

- **A20** (2026-08-08, runs 1-2 VOID, infrastructure): both v5.3
  launches died the same death — prep held the whole corpus as a
  Python list (~60GB of PyObjects at 1.7B ids) and the host OOM
  killer ended it silently (run 1 at 1.02B ids, run 2 at 1.70B; no
  traceback, log truncates mid-line), training then crashed on the
  missing tokens.bin in 3 seconds. Two wrapper lies compounded it:
  the prep heartbeat announced "built" unconditionally over the
  corpse, and run 1's checkpoint phase heartbeat "FULLY VERIFIED on
  remote" vacuously over ZERO pieces (the piece loop never ran).
  Run 1 was initially misdiagnosed as a dead GPU from eval's CUDA
  warning — its train_tail.log shows the same FileNotFoundError.
  ~$0.27 combined. Fixes, tested: TokenSink (disk-backed stream,
  spill-equivalence pinned by test — byte-identical shard at any
  spill; suite 53/53); wrapper v5.1 — prep beats every 5 min,
  success heartbeat requires rc=0 AND tokens.bin on disk, missing
  checkpoint FAILS the verify phase, CUDA canary at boot and
  pre-train (real device allocation; nvidia-smi passes while cuInit
  wedges), boot heartbeat carries host RAM, phase-complete carries
  train rc + ckpt state. Trails archived:
  results-v53-run1-prepoom, results-v53-run2-prepoom.

- **A21** (2026-08-08, run 3 VOID, infrastructure): the streaming
  prep worked first try (2.216B tokens / 4.43GB tokens.bin on disk,
  RAM flat, 15 min — past both prior death points) and every
  preflight passed — but training CRAWLED instead of crashing: GPU
  0% with one CPU core pegged. Cause: UltraConveyor.chunk()'s
  per-chunk linear scan over ALL lane events — O(30ms) at v5.0's
  150k events, ~1 s/step at the full corpus's ~4M, a ~30-hour /
  ~$7 run instead of ~6 h / ~$1.40. Fix: events are written sorted,
  so the in-window slice is two np.searchsorted calls; equivalence
  with the linear scan pinned by test on a real shard (54/54).
  Trainer step prints now flush (the 8KB stdout buffer made the
  first training heartbeats blind). ~$0.13. Run 4 is the same
  registered configuration, next host.

- **A22** (2026-08-08, run 4 CLEAN, THE VERDICT): end-to-end
  delivery — 107k steps at 142k tok/s (3h55m, ~$0.90; the A21 fix
  is why), rc=0, eval ran, checkpoint pieces verified truthfully,
  7,009,152 holds settled with the ledger EXACT and scoped;
  6,002,016 of 13,397,632 proposals vetoed (the gate refusing
  frontier targets on unpredictable carry bands — F2 live at
  scale). CE 9.87 -> ~2.6. Held-out eval (n=250):
  **in-context binding EMERGED** — gap 0-256: p(ans) 0.568, top1
  61% vs the 0.083/8.5% color-prior floor both A18 twins sat at.
  The A18 emergence-threshold reading is CONFIRMED: same
  architecture class, ~10x tokens, binding appears — the first
  demonstrated binding of the campaign, and the channel that paid
  for it was the margin channel (prior-tracking worth zero), so
  what was paid is what is real. Cross-window (256-2048:
  0.073/0%; 2048-16384: 0.079/12%, n=91 — 12% vs 8.3% chance is
  p~0.15, not a claim): NOT demonstrated at run's end. But the
  drive records show it EXISTED: recall:b1 record 0.155,
  recall:b2 record 0.142 mid-run — real margin over distractors —
  then collapse to ~0.003 by the end, while fid:4 ended NEGATIVE
  and fid:5 at zero: the carry medium was never predictable to
  its own forward model, the gate vetoed its frontiers, and the
  transient circuit lost to in-window specialization. Lesion
  (0-256: 0.568 -> 0.047): general breakage, not band-borne
  memory — band 3 updates only at 512-token chunk boundaries, so
  a fact planted <256 tokens back has never entered ANY band;
  zeroing memory tokens is an off-manifold input shift (the A15
  lesson; the lesion instrument remains confounded for in-window
  claims). Cosmetic: perpetual zero-pay fid:1/2 maintains (the
  hybrid has no such organs; proposer should mask absent bands).
  Autopsy material exists: rolling snapshots on results-v53-ckpt
  bracket the b1/b2 rise and fall. Artifacts: results-v53-run4
  (checkpoint pieces, eval shard, full train.log, eval tables).
  Next levers, in order: (1) the transient — why cross-window
  binding formed and died (snapshot autopsy); (2) selective band
  writes (pooled-mean wash-out is the prime suspect); (3) fid
  weight / stabilizing slow bands so the gate can open their
  frontiers.

- **A23** (2026-08-08, THE AUTOPSY — four checkpoints: steps
  30.5k/61.5k/93.5k/107k vs the run-4 eval shard;
  results/v53_autopsy.json, results/v53_finebins.txt):
  (1) **The transient was real cross-window carry.** The eval
  curriculum has ZERO probes at gaps 256-511 (units target ~200
  then ~800), so bin b1 is purely cross-window — the training
  margin records (b1 0.155 before 30.5k; b2 spiking to 0.142
  between 30.5k-61.5k, long after the color prior formed, over
  recency-matched distractors) cannot be an in-window slice or a
  prior artifact. Genuine band-borne binding existed twice and
  died both times. No weight snapshot brackets a living episode
  (hourly cadence, transients shorter); no held-out witness
  exists. (2) **It died unpaid, by the gate's own correct hand.**
  fid:4/5 sat below floor from the start (fid:4 ema NEGATIVE all
  run — pooled-mean writes of fast-evolving hiddens are
  unpredictable to the band's forward model), so every b1/b2
  frontier proposal was vetoed (~65/step, 6.0M total): F2
  enforced exactly as designed, and therefore the carry circuit
  got zero pay support at the moments it lived. Instrument
  mismatch, not law violation: a band can carry a retrievable
  fact while being globally unpredictable — global next-window
  fidelity is the wrong reachability instrument for a carry
  target. (3) **The lesion story reverses A22's reading:** at
  30.5k the lesion barely moves in-window binding (0.722 ->
  0.685 — pure-attention circuit); by 61.5k it collapses it
  (0.618 -> 0.146); by the end 0.568 -> 0.047. The model
  progressively REROUTED binding through the memory tokens —
  real pathway integration (asymmetry check: lesion hits binding
  bins 12x, floor bins 2.5x — specific, not off-manifold). The
  intended wiring integrated; the latents just never held stable
  facts, so integration bought fragility without range. (4)
  **In-window binding peaked early and decayed all run** at
  every fine distance (0-64: 0.868 -> 0.700; 64-128: 0.824 ->
  0.652; 128-256: 0.587 -> 0.450) while train CE improved:
  margin pay at lam=0.1 loses to the LM objective late; binding
  is also steeply distance-graded well inside the window (the
  2-block transformer's induction is short-ranged). Lessons ->
  levers, evidence-ranked: (L1) selective/gated band writes so
  slow bands become predictable -> fid clears floor -> the gate
  can OPEN cross-window frontiers and pay the transient when it
  appears (the veto ledger is direct evidence pay was withheld
  exactly where support was needed); (L2) drive channel EMAs
  printed every train.log line — the b1 transient lived and died
  entirely between snapshots; never again unobserved; (L3)
  pay-weight vs CE calibration late in training (in-window decay
  under a live margin channel is a scheduler/weight question).

- **A24** (2026-08-08, pre-run, registered): **v5.4 — pay the
  carry.** The autopsy's levers, built: (L1) selective
  closed-by-default band writes — each band attention-pools the
  window with its own learned query (chooses, does not average);
  SlowCell gated delta-write with the update gate biased shut at
  init (sigmoid(-2)~0.12) so the medium drifts slowly and its
  forward model has something learnable; write-sparsity cost
  WRITE_W=0.01 on open gates. (L2) drive channel EMAs on every
  train.log line + full trace (ckpt.trace.jsonl, 50-step
  resolution, pushed in heartbeats) — no transient unobserved
  again. (L3) lam 0.1 -> 0.25 against late-run CE competition.
  Plus: absent-band mask (hybrid has no bands 1/2; no ghost
  maintains), and 384-gap units close the 256-511 probe hole so
  the window edge is measurable. Suite 57/57; 200-step smoke:
  fid:4 POSITIVE (0.114-0.128 vs negative all of run 4), fid:3
  0.392, recall:b1 FRONTIER HOLD OPEN (vetoed 6.0M/6.0M times in
  run 4), ledger exact. Registered read for the run (same corpus
  discipline, d=128, 107k steps, wrapper v5.1 renamed v54): the
  primary question is whether PAID cross-window carry sustains —
  confirm = fid:4/5 above floor for sustained stretches AND b1/b2
  frontier holds open/settling with pay AND held-out 512-1024
  above the 0.083 floor at end (then the lesion at those bins is
  the carve-out); if fid clears floor and pay flows but carry
  still dies, the gate was not the binding constraint and the
  write/readout design is still short (next: dedicated write
  head / associative matrix); if fid stays below floor, the gated
  write failed to stabilize the medium (next: predictor targets
  own next state, not next read). Secondary reads: in-window >=
  v5.3 (0.568/61%) and its decay slope under lam 0.25; the new
  384 bin's first window-edge curve.

- **A25** (2026-08-08, v5.4 run CLEAN, verdict: IGNITION CAUGHT
  MID-BIRTH): end-to-end delivery again (107k steps, 3h50m at
  ~110k tok/s, ~$0.90; trace 2,141 rows at 50-step resolution —
  the L2 instrument). **The L1 mechanism is confirmed at scale:**
  fid:4 rose 0.25 -> 0.83 (negative all of run 4) and the veto
  counter FROZE at 613,056 by step ~20k — zero vetoes for the
  final 87k steps (run 4: 6.0M at a constant rate to the end).
  The medium stayed predictable, frontiers stayed open, carry
  was paid all run. **In-window binding ignited at step ~95k as
  a sharp phase transition** — b0 flat at 0.03-0.07 for 94k
  steps (genuinely flat: no precursor flickers at 50-step
  resolution), then 0.27 (96k) -> 0.59 (98k) -> 0.67 (100k),
  stable ~0.63 after. First fully-observed emergence of the
  campaign; the run-4-vs-5 onset delta (<30k vs ~95k) reads as
  onset variance under changed dynamics, and ignition WITH the
  near-constant memory tokens present demotes the
  attention-sink hypothesis. **Post-ignition, carry answered:**
  b1 spiking 0.10-0.14 (repeatedly; pre-ignition ~0.04),
  correlated with b0 life — forming under paid support, exactly
  the A24 design intent. Eval fired ~10k steps after birth:
  held-out 0-256 at 0.170/25% (2-3x floor, far below run 4's
  mature 0.568/61%), cross-window at floor, lesion delta ~zero
  (young circuit attention-pure — matches run 4's early phase
  before memory-token integration). fid:5 never cleared floor
  (0.061; too few 32k-clock ticks in one run). Primary A24 read
  lands in the "incomplete, not negative" branch: the run ended
  ~10k steps after its own ignition. Artifacts: results-v54
  (pieces, eval shard, full trace, train.log).

- **A26** (2026-08-08, pre-run, registered): **v5.5 — the
  continuation.** Resume the just-ignited circuit rather than
  re-roll a 94k-step ignition lottery: load the v5.4 final
  checkpoint (model + optimizer + drive EMAs/records/minted/
  vetoes; trainer --resume/--offset, suite-pinned), rebuild the
  identical shard (v5.4's own tokenizer fetched from its branch
  — same ids by construction; prep deterministic at seed 0),
  and ride each lane's UNSEEN tail: offset_frac 0.7909347 (the
  107k x 512 consumed mark), 28,000 steps to just before wrap —
  the one-epoch no-repeat law holds. Known scars, accepted and
  ledgered: open holds and band states are not checkpointed
  (holds re-propose within a sweep; slow states rebuild within
  ~64 chunks), and the tail's first ~12.8k tokens per lane carry
  probes whose plants were consumed pre-continuation with cold
  bands (reads ~0, drags EMAs briefly, self-corrects). ~70 min,
  ~$0.30. Registered read: PRIMARY — does b1 (512-1024, now
  populated by the 384/800 units) consolidate under continued
  pay and show above the 0.083 held-out floor at end, with the
  lesion at that bin as the carve-out; SECONDARY — b0
  consolidation toward run-4 maturity (0.57/61%) and memory-token
  integration beginning (lesion delta at 0-256 growing); if b1
  stays at floor with b0 mature and gates open, the write/readout
  path is genuinely short and the next lever is the dedicated
  write head / associative matrix, designed on a full autopsy.

- **A27** (2026-08-08, v5.5 continuation CLEAN, verdict: THE
  VECTOR IS THE BOTTLENECK): resume worked — b0 at 0.57 on the
  first heartbeat, step numbering/trace continuous, fid:4
  rebuilt 0.50 -> 0.82 through the ledgered cold-band scar
  (~10k steps), ledger exact across the boundary; 133k tok/s,
  ~$0.25. OFFSET CORRECTION (miss, printed): the 0.7909347
  offset was computed from run 3b's pre-384 shard size; true
  consumed mark on the v5.4 shard (2,231,154,918 tokens) is
  0.785732 — the continuation started ~362k tokens/lane deep,
  SKIPPING ~11.6M unseen tokens (2.4% of tail). Safe direction
  (nothing repeated), bounded scar. **In-window matured:
  held-out 0.505 / 64% top1** (from 0.170/25% at 107k;
  secondary read MET — top1 above run 4's 61%), still fully
  attention-pure (lesion delta ZERO — the model near-ignores
  the memory tokens, consistent with a store carrying almost
  nothing). **Cross-window: PRIMARY READ NOT MET** — held-out
  b1 0.078/10% (n=52), b2 0.074/7%: at floor. But the trace
  shows carry was NOT a transient this time: b1 held 0.04-0.10
  and b2 0.01-0.095 for the ENTIRE continuation — persistent,
  paid, PARTIAL carry (~10-15% of probes by margin mass),
  below held-out detection. The elimination is now complete:
  medium predictable, gate open (vetoes froze at 908k by step
  ~118k after the cold rebuild), pay active, in-window mature —
  and a single d-vector per band still cannot reliably hold
  multi-fact spans (bands 3/4 face ~2-15 concurrent facts).
  Q1 of the closure plan is ANSWERED: per-band associative
  storage is required. Next (A28, to be registered): the
  matrix A/B — fast-weights store per slow band (outer-product
  writes, query reads) + dedicated write head, head-to-head vs
  v5.5 at identical budget, the last open architecture
  question. Artifacts: results-v55 (pieces, trace, tables).

## Status

Assembled scene-free (A6), no registered runs. Weaver (`iga/lm_gen.py`),
conveyor (`iga/lm_conveyor.py`), six-band model (`iga/lm_bands.py`),
drive layer (`iga/lm_drive.py`), trainer (`iga/lm_train.py`), eval
harness (`iga/lm_eval.py`); UltraChat prep + conveyor
(`iga/lm_data_ultrachat.py`); calibration harness
(`iga/lm_calibrate.py`); law tests in `tests/test_lm_ladder.py`
(suite: 48/48). End-to-end smokes pass on both conveyors (weaver and
a real UltraChat shard): CE falls, ledger audits exact,
thanks-mints/expiry-zero/closed-loop-zero pinned by tests, panel
readable, lesion + talk harness run, probe positions token-verified.
First calibration artifact committed (weaver split). v0 engineering
notes (honest) in module docstrings: slow-band predictor gradients
flow only within a chunk; competence band = records + scheduler
bin-weights. Remaining before any registered run: (1) debug-tier A/B
of talk=dense vs tick, winner frozen; (2) calibration re-run on the
real-data calibration shard, constants frozen. Debug next on RTX
2000; registered runs on a 4090.
