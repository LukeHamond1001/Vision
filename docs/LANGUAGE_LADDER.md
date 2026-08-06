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

Two sub-claims, separable by the arms below:
- **L1 (holding):** the ladder carries long-range structure with no
  attention anywhere; long-range performance does not cliff at any
  window boundary, because there is no window.
- **L2 (growing up):** the drive layer, holding targets over the
  model's own measured competences and scheduling data by learning
  progress, improves on uniform training at matched compute.

## Arms (all matched parameter count and matched training compute)

1. **baseline** — standard decoder-only transformer (the reference the
   field trusts).
2. **bands** — the clock ladder alone: per-band recurrent state, each
   band ~8× slower than the one below; each band embeds the window of
   the band below and predicts the next window (the v0.5–v0.9 recipe:
   context coupling, within-band whitening, boundary masking). No
   registers, no reward. Isolates the backbone.
3. **full** — bands + the drive layer: probes, registers, proposer,
   ledger, and the learning-progress data scheduler. The complete
   architecture. Arm 3 vs arm 2 isolates the drive layer; arm 3 vs
   arm 1 is the headline.

Sizes: ~10M, ~30M, ~90M params. **One seed per (size, arm) cell** —
replication is carried by the size axis: no claim counts unless its
direction holds at all three sizes. This is standard scaling-study
methodology; LM pretraining is low-variance relative to RL.

## Band geometry (content bands)

Clock ratio ~8× per band. Band 1 ticks per token (state spans words);
band 2 ~8 tokens (phrase); band 3 ~64 (paragraph/topic); band 4 ~512
(section); band 5 ~4k (document intent); band 6 ~32k (book scale,
active only on long documents). Bands at 10M: 4; 30M: 5; 90M: 6.

**The episode law carries over: a document is a life.** Content bands
and registers reset at document boundaries; no pay across a reset;
documents are never concatenated into one stream. (Same law as the
agent campaign; the telescoping audit applies per document.)

**The competence band is continuous.** One slowest register bank runs
on the training-run clock, holding targets over the model's measured
competences (below), paid at hold ends on measured improvement. This
band never resets: it is the run growing up.

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

## Drive layer (arm 3 only)

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
  length, overlaid with the baseline's context-window boundary. The
  registered picture: baseline cliffs at its window; ladder does not.
- **Window dial:** baseline re-evaluated at shrinking windows
  (2048/512/128) vs the ladder arm (which has none).
- **Lesion:** slow bands knocked out at eval; long-range recall must
  collapse (mechanism check) while local perplexity survives.
- **Short-range cost:** within-window perplexity, ladder vs baseline.
- **Live register panel:** the act6 demo transposed — mid-book state
  of every band, printed as text from audited probes.
- Ledger audit: telescoping exactness recomputed from committed logs.

## Gates (pre-registered; exact constants may be amended only pre-run,
with the amendment ledgered here)

- **G-hold:** at every size, ladder recall-at-distance beyond the
  baseline's window exceeds the baseline's, and the ladder's curve
  from near to far degrades by less than half the baseline's
  within-to-edge degradation. Direction must hold at all 3 sizes.
- **G-trend:** the ladder-vs-baseline long-range advantage is
  monotone non-decreasing across 10M→30M→90M.
- **G-drive:** arm 3 beats arm 2 on the pre-registered eval suite at
  matched compute at ≥2 of 3 sizes.
- **G-cost:** arm 3 within-window perplexity within 15% of baseline
  at 90M.
- **G-lesion:** slow-band lesion reduces beyond-window recall by
  ≥50% while within-window perplexity moves <10%.
- Misses are printed beside passes, as always.

## Compute and cost

Kaggle free tier (30 GPU-h + 20 TPU-h/week) covers 10M and 30M.
90M on TPU Research Cloud (application submitted) or ~$50–100 rented
GPU fallback. Target cost: $0–100.

## Status

Design. No code, no runs. Pipeline and scaffold next; first 10M
curves targeted ~10 days from card commit; full package 4–6 weeks.
