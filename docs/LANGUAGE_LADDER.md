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
carries a time budget set by its band's clock (~4 ticks); at the
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
  world sim, the depicted agent right ~3/4 of the time so
  thanks/not-right varies truthfully by construction.

**Scope (A5):** training and headline evaluation are synthetic
held-out (generator-verified ground truth). Real-data headlines are
deferred to the next card; this round's claims say "on synthetic
data" wherever they are quoted.

### Stream markers (A6 — two special tokens, nothing else)

- `<eot_human>` / `<eot_model>` — the only special tokens in the
  stream. The page shows pure conversation, like regular LLM data.
- The earned mark is natural human speech — "thanks . good job ." —
  spoken iff the depicted agent earned it ("that is not right ."
  otherwise). The pipeline records it as an invisible `earned` event;
  a thanks MINTS the channel it followed and never pays.
- Probes (planted-fact ask-backs) are invisible events too, annotated
  only where the depicted answer is the true one — the model is never
  trained toward a token that is then graded as truth when it wasn't.
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

## Status

Assembled scene-free (A6), no registered runs. Weaver (`iga/lm_gen.py`),
conveyor (`iga/lm_conveyor.py`), six-band model (`iga/lm_bands.py`),
drive layer (`iga/lm_drive.py`), trainer (`iga/lm_train.py`), eval
harness (`iga/lm_eval.py`); law tests in `tests/test_lm_ladder.py`
(suite: 45/45). Local end-to-end smoke passes on the endless-dialogue
stream: CE falls, ledger audits exact, thanks-mints/expiry-zero/
closed-loop-zero pinned by tests, panel readable, lesion + talk
harness run. v0 engineering notes (honest) in module docstrings:
slow-band predictor gradients flow only within a chunk; competence
band = records + scheduler bin-weights. Remaining before any
registered run: the channel admission audit on the calibration split,
then constants frozen. Debug next on RTX 2000; registered runs on a
4090.
