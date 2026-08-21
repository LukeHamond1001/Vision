# The centerpiece: one being, two removals

Registered 2026-08-21, before the V10.1 flash's first token. This is the
plan the repository is organized around: one 500M model flashed with a
lifetime ([ARCHITECTURE.md](ARCHITECTURE.md)), and two demonstrations
made by REMOVING an organ from that same model on the same day of the
same life — never by training a second model. Every headline is a
within-run contrast; every bar below is fixed before the weights exist.

## The claim

> A being that has lived a life carries two kinds of memory besides its
> language: a slow thread (the timescale bands) and an exact contextual
> memory (the stores). Switch the bands off and it lives in the moment.
> Switch the stores off with the bands on and it keeps the thread but
> loses the facts — it juggles, and fails.

The contrasts are read-path switches in the forward pass
(`model.mem_off`, `model.store_read_off`, `model.lesioned`); with all
off the model is bit-exact to the certified one. Nothing is retrained
for a demo. Note the organ boundary: every band owns a state (its slow
summary, read by the cortex as a memory token) AND a store (exact
contextual memory). `lesioned` removes both; the demos separate them.

## The model (V10.1)

HybridLM d=1280 / 20L / T=2048 / 16k vocab, bands 3–6 at clocks
{1, 8, 64, 512} chunks (horizons 2k → 1M tokens), per-band logit-keyed
stores, press economy, ARM C sleep with A76 homeostasis; bf16 trunk with
fp32 organs; 5.1B tokens as 8 staged lives with the episodic cast;
~2.3 days on one H100. Full spec: [V10_1_RERUN_PLAN.md](V10_1_RERUN_PLAN.md).
Status is the last entry of the ledger ([LANGUAGE_LADDER.md](LANGUAGE_LADDER.md))
and the `results-v10` branch.

Gate before money: the 78M mini-flash on the mule (4 lives × 50M, the
real driver and battery) must show the binder arming on unseen lives —
in-ctx recall visibly rising above 20% chance through childhood, or
≥ .40 by the end of the mini life. Flat = no launch.

## Act 1 — the being as lived (the base)

Everything is measured against this: the served model on day one, the
end-of-life band states and stores seeded from the flash, the certified
serve loop (economy, presses, nightly sleep). Probes are score-only
forwards on state copies; the committed stream is never perturbed.

## Demo 1 — bands removed (the slow thread off, stores on)

`mem_off = True`: the four memory tokens are zeroed; the stores are
still read. The being keeps its exact contextual memory but loses the
slow thread — the summaries at 2k / 16k / 131k / 1M tokens.

Pre-registered expectation (the direction the organ program predicts;
a miss is reported as a miss):

| probe | base | thread off |
|---|---|---|
| thread continuity (who / what the day has been about, judge-scored and name-counted) | held | falls |
| recall at long gaps (b4, b5, b6 bins) | above chance | falls by what the band STATES carry beyond their stores (the battery's `lesion_thread` row is the in-flight estimate) |
| in-ctx / short recall | — | holds within the base's own CI (attention and the stores own the chunk) |
| CE on held-out lives | — | rises by the `lesion_thread` delta |

The reel moment: the being is asked what the day has been about — the
base tells the story; thread off, it answers from the sentence in front
of it. (The prophet heads read the band states directly, so press
anticipation is unchanged by any read-path switch; it is a diagnostic of
the states, not a demo reading.)

## Demo 2 — contextual memory removed, bands active: juggles but fails

`store_read_off = True`: the band states and their memory tokens stay
live; no store is read.

| probe | base | stores off |
|---|---|---|
| exact recall of planted facts (in-ctx, short, b3) | above chance | falls toward chance — the identities live in the stores |
| continuity of the thread (topic, counterparty, the day's shape — scored by the frozen judge on the reply) | held | held (the bands still carry the summary) |
| CE | — | rises by the battery's `lesion_store` delta |

The reel moment: asked the same fact, the being knows it was told and
what the conversation was about, and reaches for the wrong name.

## The fourth reading — both off (`lesioned = {3,4,5,6}`)

Tokens and stores gone: the cortex alone with its 2048-token chunk, the
in-the-moment extreme. Read beside the two demos on the same probe set;
it bounds what the two organs carry together, and `both − store` is
the band states' share when no store is there to lean on.

## Laws (inherited from [DEMO_PROTOCOL.md](DEMO_PROTOCOL.md))

1. **Speech-gated.** Every headline number is greedy-decoded speech.
   Logit readouts are diagnostics.
2. **Within-run contrasts only.** Same checkpoint, same committed state,
   organ on vs organ off, on the SAME probe set. No cross-run, no
   cross-model claims. Seed variance is unmeasured; single-run
   attribution stays banned for anything but these paired contrasts.
3. **n ≥ 20 per class**, paired, one-sided sign test p < 0.05 per
   headline bin; effect sizes with CIs; underpowered cells are anecdotes
   and are labeled so.
4. **Nothing adjusted after launch** except by a ledgered amendment that
   names what changed and why.

## Go / no-go, read in flight

The demos only show what the organs contribute if the organs are
load-bearing. The battery reports `lesion_thread` (memory tokens off,
stores on — Demo 1's organ), `lesion_store` (Demo 2's), `lesion_bands_all`
(both) and the per-band lesions every second beat, CE and recall-by-bin
against the same base, on the H100 run (the mule's mini-flash carries the
older battery: per-band CE lesions only). Decision rule:

- By the end of childhood (~35% of the life): a band lesion moves at
  least one long-gap bin or CE beyond the row-to-row noise → the demos
  are on track.
- Flat through adolescence → the demos are reported as a NULL (the
  organ was not load-bearing at this scale and diet), and the gated band
  boosts (A71 widths, `band_lr_mult` 3×, the conveyor arm) get their own
  mini-flash before any further money. A null is published, not hidden.

## Build items (all $0, during the flash)

| item | where | law |
|---|---|---|
| **BUILT 2026-08-21** — `lesion none|bands|store|both` in the serve room (`ServeSession.lesion`, `lesion_scope`; room command `lesion`): bands = `mem_off` (thread off, stores on), store = `store_read_off`, both = `lesioned` all; applies to replies and score-only probes, never to a commit, a sleep block or a save — the committed life stays the certified forward's, so the contrast is "same state, organ off" | `iga/lm_hybrid.py` (`mem_off`), `iga/lm_serve.py`, `scripts/live_room.py` | `tests/test_lm_serve_lesion.py` L1–L5: 'none' bit-exact; commits identical under any switch; the three switches are three different readings; flags never outlive a forward |
| **BUILT 2026-08-21** — `scripts/demo_lesions.py`: seeded probe set (n per gap bin: in-ctx / short = 1 chunk / b4 = 8 chunks / b5 = 64), plants with neutral dialogue between plant and ask, then every fact read under the four conditions (none / bands / store / both) on the SAME committed state — greedy speech on a copy (never appended) + p_true; paired one-sided sign test base > removed per bin; thread-continuity read; evidence JSON, frozen `probe_set.json`, transcripts | `scripts/demo_lesions.py` | `tests/test_demo_lesions.py` D1–D5: seeded/frozen/disjoint set; gap invariant per bin; reads paired and silent; exact sign test; summary shape. Dry run on the 78M raised life before day one |
| reel capture: the two moments above plus the base, as text transcripts first, video second | `results/evidence/v10_1_demo/` | nothing edited; full transcripts committed |
| day-one growth chart: the flash's lesion rows plotted per bin over the life | `results/v10_1_flash/` | — |

## Timeline

| when (UTC) | what |
|---|---|
| 2026-08-21 evening | mini-flash gate rows (hourly); full corpus build on the mule; ship code posted |
| on PASS | H100 launch (`RUN_TAG=v10_1`, bf16); fp32-vs-bf16 smoke rows verified; flash runs ~2.3 days with the battery every 6000 steps |
| in flight | serve-room switch + `demo_lesions.py` built and tested on the 78M raised life; probe set frozen |
| flash lands | end-of-life state banked; day-one serve; Act 1 base, Demo 1, Demo 2, then the living-vs-frozen protocol |
| after | evidence committed; ledger entry; README headline updated from the committed JSON — and if a demo is a null, the README says so |

## What is not claimed

Absolute capability; consciousness or sentience language; comparisons
to other models; any number whose class, metric and bar do not appear in
this file or in DEMO_PROTOCOL.md.
