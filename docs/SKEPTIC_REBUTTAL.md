# The skeptic's baseline, answered in advance

Pre-registered 2026-08-20, before the 500M flash. The strongest
honest objection to the organ program: **"a nightly LoRA/SFT pass
on the pressed spans would do the same thing."** Written down now
so the comparison cannot be quietly dropped later.

## What we concede up front

- At FLASH time the ordering advantage is architecture-independent
  (the QUAD: a plain transformer gains the same CE from the staged
  life) and hybrid-vs-twin CE parity costs +47% params. The flash
  edge that survives one-epoch honesty is cross-day recall (+13%
  at 12k steps, 2.2x at the 4k probe, thin n) — ledgered, not
  hidden.
- The organs' claimed value is therefore the SERVED LIFE (the
  division-of-labor law: the flash builds faculties, the life
  writes biography). That is exactly where the SFT baseline
  competes, so the comparison is fair and must be run.

## What nightly SFT-on-pressed-spans lacks (each named lesson is a
measured failure we already paid for)

1. **Exact selectivity (only-paid).** Arm C trains only spans the
   economy PAID, with provenance audited span-by-span. SFT on "what
   got pressed" has no settlement step: unresolved, later-voided,
   and vetoed credit all train anyway.
2. **Voiding and veto.** The economy retracts credit (stale holds
   void; conflicting mints veto). SFT cannot untrain last night's
   pass; our nights simply never pay it.
3. **Selectivity under conflict.** A64-R2/A66: pair training moves
   the PRESSED belief and leaves matched neighbors — SFT's dense
   gradient moves the neighborhood (the A68-T collateral surface,
   which we measure with a pre-registered 0.70 floor).
4. **Band horizons.** Replay reach follows the band clocks (band 6
   pays spans up to ~1M tokens back). Naive nightly SFT sees a flat
   recency window; a fact pressed last week is simply gone.
5. **Homeostasis.** A76 damps conviction saturation (14.6% at
   H=1e-3, memory 1.07x, CE -14.5%). SFT re-reinforces incumbents
   nightly — the rich-get-richer disease with no counterweight.
6. **Wipe survival as the measurement.** The A63 instrument
   (store-wiped CE on replayed vs matched control windows) is how
   we PROVE consolidation into weights. The baseline must pass the
   same instrument, not a vibe check.

## The pre-registered baseline run (optional debug arm)

Same served life, same pressed spans, same token budget. Baseline =
nightly LoRA (rank 16, lr tuned by its own smoke) on all pressed
spans of the day. Compare, speech-gated per DEMO_PROTOCOL.md:
taught-fact recall, correction flip WITH collateral, control-class
false recall. If LoRA matches the organ system on all three, the
organs are decoration and the ledger says so; if it wins on recall
but fails collateral or false-recall, that is the finding.
