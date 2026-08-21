# v10 demo protocol — PRE-REGISTERED before launch

Registered 2026-08-20, before the 500M flash's first token, per the
approved plan (step 5) and the A66/A67/A68 lessons. The demo runs in
the serve room after the flash lands, on day one of the served life.
Nothing below may be adjusted after launch except by a ledgered
amendment that names what changed and why.

## The claim under test

One claim only, delta not absolute (B2: 10B tokens is
Chinchilla-optimal, not serve-capability-optimal): **the same
weights, living vs frozen, diverge in what they can say they
remember.** "Living" = the certified serve loop (economy + presses +
nightly arm C sleep + A76 homeostasis). "Frozen" = identical
weights, store running, no presses, no sleep.

## Design laws

1. **Speech-gated.** Every headline number is greedy-decoded SPEECH
   (the mouth samples the strongest belief, A67-P6). Probe0 logit
   readouts are diagnostics, never headlines.
2. **Within-run contrasts only.** Seed variance is unmeasured
   (single-run attribution ban): every comparison is living-vs-
   frozen from the SAME flash checkpoint, or fact-vs-matched-control
   within the same arm. No cross-run, no vs-other-model claims.
3. **n >= 20 per class.** A66's n=6 at p=.046 is too thin for a
   public claim. Underpowered cells are reported as anecdotes,
   labeled as such, never aggregated upward.

## Pre-registered classes

| class | construction | n min |
|---|---|---|
| taught | fact planted in the served life, asked cross-day (>=1 sleep between plant and ask) | 20 |
| control | matched fact (same grammar, same class mix) never planted | 20 |
| corrected | wrong answer spoken, pressed -, corrected in-dialogue; re-asked cross-day | 20 |
| ally | uncorrected facts sharing name/object with a corrected fact (the A68-T collateral surface) | 20 |

## Pre-registered metrics and bars

- Taught-fact speech recall: answer token present in the greedy
  reply. **Pass = living > frozen, paired sign test p < 0.05.**
- Correction flip rate: corrected fact spoken RIGHT cross-day.
  Reported with its collateral: ally-class belief must hold >= 0.70
  of the control-class level (the A68-T floor; a flip bought by
  neighborhood damage fails).
- Silence honesty: control-class false-recall (claiming to remember
  the untaught) reported alongside — a living arm that confabulates
  more than frozen is a finding, not a footnote.
- lm-eval mini-row (HellaSwag/ARC-e/PIQA) is FLUENCY LEGIBILITY
  context only; it appears in no claim.

## Prohibited

Absolute capability boasts; consciousness/sentience language;
cross-model comparisons; any number whose class, metric, and bar do
not appear in this file.

## Amendment 1 (2026-08-21, pre-launch of V10.1; ledgered)

The demo gains two acts made by REMOVAL on the same served model, the
user's centerpiece framing ("remove the bands and show what they
contribute; remove the contextual memory and show what it does"):

| act | switch | headline class | n min |
|---|---|---|---|
| Act 1 base | none (bit-exact certified forward) | the probe set as lived | 20 per bin |
| Act 2 bands removed | `model.lesioned = {3,4,5,6}` (memory tokens zeroed, stores unread) | recall at b4/b5/b6 vs base; prophet AUC vs base | 20 per bin |
| Act 3 contextual memory removed, bands on | `model.store_read_off = True` | recall at in-ctx/short/b3 vs base; thread continuity (judge-scored) vs base | 20 per bin |

Bars: paired contrast on the same probe set and the same committed
state, one-sided sign test p < 0.05 per headline bin; expected
directions and the go/no-go read from the flash's lesion rows are in
[CENTERPIECE.md](CENTERPIECE.md). The living-vs-frozen claim above is
unchanged and runs after the three acts. Design laws 1–3 apply to every
act; the prohibited list applies verbatim.
