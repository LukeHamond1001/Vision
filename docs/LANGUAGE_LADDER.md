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
active only on long documents). Bands at 10M: 4; 30M: 5; 90M: 6.

**The conveyor law (A1): the life is the training run; a document is
a scene.** All data rides one continuous stream, in the order the
scheduler chooses; the model is never reset. Fast-band hidden state is
boundary-masked between scenes (carrying one scene's state into an
unrelated one is noise, not memory), slow bands persist, and the
ledger's law is scoped rather than episodic: **no hold outlives its
referent.** A hold opened on a scene's entities settles by that
scene's end — arrival or expiry, paid or zero, never across it. The
telescoping audit applies per hold; closed loops still net zero.

**The competence band is continuous.** One slowest register bank runs
on the training-run clock, holding targets over the model's measured
competences (below), paid at hold ends on measured improvement. Its
referents — the model's own competences — genuinely persist, so its
holds may span the whole run. This band never resets: it is the run
growing up.

## Data (public, free, mixed; ratios frozen before runs)

- **FineWeb-Edu** (sampled) — general text; ships a per-document
  quality score used as a selection label.
- **PG-19** — full public-domain books; the long-document anchor that
  gives slow bands real work.
- **StackExchange dumps** — threads as episodes; the accepted-answer
  bit and in-the-wild "thanks" are natural human ratification labels.

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

- **Recall-at-distance curves** on held-out books: performance vs gap
  length, overlaid with the reference checkpoints' context-window
  boundaries. The registered picture: references cliff at their
  windows; the ladder's curve does not cliff, because it has none.
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
  not eloquence, is what is being read.
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

## Compute and cost

Kaggle free tier (30 GPU-h + 20 TPU-h/week) covers 10M and 30M.
90M on TPU Research Cloud (application submitted) or ~$50–100 rented
GPU fallback. Target cost: $0–100.

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

## Status

Design. No code, no runs. Pipeline and scaffold next; first 10M
curves targeted ~10 days from card commit; full package 4–6 weeks.
