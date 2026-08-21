# The being and how it is trained

What runs on the pod, in one page each: the organism (`iga/lm_hybrid.py`
and its organs), the lifetime it is flashed with (`iga/lm_data_life.py`,
`scripts/v10_driver.py`), and what each organ contributes when it is
removed — which is the whole point of the centerpiece demos
([CENTERPIECE.md](CENTERPIECE.md)). Numbers below are read from the code
at the V10.1 shape (d=1280, 20 layers, T=2048, 16k vocab), not quoted.

## 1. The organism

```
tokens ──► embed (+abs pos) ──► 20 transformer blocks ──► head ──► next token
                                    ▲            │
            4 memory tokens ────────┘            │ mid-depth (layer 10)
            (one per band)                       ▼
                                       store read (per band, logit-keyed,
                                       gated) ──► added to the residual
   band states h[3..6]  ◄── SlowCell gated delta-write on each band's clock
   stores   M[3..6]     ◄── written every chunk: key = preceding-context
                            mix, value = the token's identity
```

| organ | mechanism | what it holds | params |
|---|---|---|---|
| cortex | 20 pre-norm blocks, 8 heads, GELU MLP, learned absolute positions; attention sees one T=2048 chunk plus the 4 memory tokens | the language, the in-the-moment thought | 393.6M (75%) + 2.6M pos |
| timescale bands 3–6 | one `SlowCell` per band; band k updates its state every `clock[k]` chunks with a gated delta-write (gate init −2: closed until the model opens it); a predictor head per band scores write fidelity (the `fid:k` channel); each state is projected to one memory token the cortex attends | the slow thread — summaries at 2k / 16k / 131k / 1M-token horizons | cells 26.2M + pred 6.6M + mem_proj 6.6M = 39.3M (7.5%) |
| contextual memory (the stores) | one associative matrix per band, written every chunk (key = a learned mix of the preceding context, value = the token's own identity), read at layer 10 through a learned gate (`read_drop` 0.5 in training so the trunk never leans on it) | exact recall of what was said — names, objects, the fact planted 40k tokens ago | write/read apparatus 26.3M (5%) |
| vocabulary | untied 16k embed + head; an auxiliary head pays the trunk blocks 0.2× CE so they earn their own gradient beside the memory path | — | 63.0M (12%; 42M served) |

Total 524.7M built, ~504M served (the auxiliary head is training-only).
The band clocks are `BAND6_CLOCKS = {3: 1, 4: 8, 5: 64, 6: 512}` chunks;
horizons 2,048 / 16,384 / 131,072 / 1,048,576 tokens. Band 6 ticks ~600
times per 5.1B-token life: the slowest thing in the organism that still
learns.

**Three read-path switches in the forward pass** (every band owns a
state — read by the cortex as a memory token — and a store):

- `model.mem_off = True` — the slow thread off: every band's memory
  token is zeroed; the stores are still read. "Bands removed" (Demo 1).
- `model.store_read_off = True` — the stores are not read; band states
  and memory tokens stay live. "Contextual memory removed, bands
  active" (Demo 2).
- `model.lesioned = {3,4,5,6}` — the amputation: tokens AND stores of
  those bands gone. The cortex alone with its chunk (the fourth reading;
  per band, the battery's `lesion_b{k}` rows).

With all off the forward is bit-exact to the certified model (law L2;
`tests/`). The prophet heads read the band states directly and are
unaffected by any of them.

## 2. The drive: presses, economy, sleep

The being is not trained on text alone. Its stream carries **press
tokens** `<+1> <+2> <-1> <-2>` — the counterparty's graded buttons — and
the training loop runs the drive economy beside the language loss:

- **Holds and mints.** A press opens a hold on the span it judges; the
  ledger pays (mints) or retracts (voids/vetoes) by the economy's exact
  accounting (`iga/lm_drive.py`). Pay weight `lam` is set from the paid
  smoke's measured holds/step so `lam × holds ≈ 0.25` (the A60f pairing).
- **Prophet heads.** Each band carries a press-prediction head: the slow
  thread learning to anticipate the primary reward at its horizon.
- **Sleep** (`iga/lm_sleep.py`). Nightly, the sleeper replays the spans
  the economy PAID (never what was merely seen), runs ARM C contrastive
  pairs for corrections (wrong turn vs corrected turn, only-paid), and
  applies A76 homeostasis (decoupled decay on the slow weights,
  H = 1e-3) so convictions do not saturate. Dose ladder: infancy
  sleepless (A64-R3: a fresh trunk under sleep collapses), then 1 sleep
  step per 16 wake steps. Laws L1–L4 make the sleeper's trunk-alone
  student pass and its replay exact and audited.

## 3. The lifetime (the flash)

Pretraining = flashing one compressed life per lane. Eight lanes, eight
lives, each life ~640M tokens of the same spine in the same order, read
once (the late stages twice):

| stage | share | corpus | presses | sleep |
|---|---|---|---|---|
| infancy | .08 | the simplest UltraChat slice | dense, positive-only, no corrections | none |
| childhood | .27 | UltraChat | dense → sparse; corrections 3–8% | 1:16 |
| adolescence | .38 | SmolTalk2 (EN, no tools; LongAlign last) | sparse | 1:16 |
| tail | .27 | Smol-Magpie-Ultra | sparse; strict tail audit | 1:16 |

The builder (`iga/lm_data_life.py`) assembles each life as days: open
ritual → exchanges → close ritual. Every exchange is graded by a frozen
judge (`iga/lm_judge.py`, version pinned in the manifest) into a press
class; below-floor exchanges are dropped (a flat quality floor, a rising
ceiling). Into the stream it weaves the **episodic cast**: facts drawn
from a 273×165 name/object vocabulary, planted on a token cadence, asked
2–6 times at gaps from the stage's menu (up to the 131k and 1M band
horizons in adolescence/tail), then retired so the weights cannot hoard
them. Eval lives draw their own novel facts — recall on those is what
the battery scores. (The v10 run's cast was 24 persistent facts per life;
the model memorized them and the binder never armed. That is why V10.1
exists: [V10_1_RERUN_PLAN.md](V10_1_RERUN_PLAN.md).)

The driver (`scripts/v10_driver.py`) runs the life in 6000-step
segments (warm restarts carry band states, checkpoints every 500 steps,
a banked best-holdout checkpoint), sets the sleep dose per stage, and
fires the battery every 6000 steps.

**Precision.** fp32 master weights and AdamW; bf16 autocast on the trunk
blocks only; band states, stores, losses and the economy's readings stay
fp32 (a 1M-token horizon accumulates in steps too small for bf16). lr
4e-5, warmup 2000, cosine on the global step; lanes 8; ~2.3 days on one
H100 at the measured bf16 rate.

## 4. The battery (what the pod measures, every 6000 steps)

`scripts/heartbeat_v10.py` on the unseen eval lives, fresh state, fixed
warm-up walk:

- CE and **recall by gap bin** — in-ctx, short, b3, b4, b5, b6 (the gap
  between a fact's plant and its ask, binned at the band horizons).
  In-ctx on unseen lives is the binder: chance is 20%.
- Collapse (greedy and sampled distinct-3gram, entropy), cast incumbent
  mass and confidently-wrong prevalence, tail press audit against the
  frozen judge, prophet AUC.
- **Lesions, every second beat:** each band alone, `lesion_thread`
  (tokens off, stores on), `lesion_store`, `lesion_bands_all` (both), and
  `lesion_base` — CE delta and recall-by-bin under each removal against
  the same base. These rows are the growth curve of
  the two demos: they say, in flight, what the bands and the stores are
  contributing and at which horizons.

Kill criteria are pre-registered in the code (non-finite loss, tail audit
mismatch, dead instruments stop the run; judgment calls are WARN lines
for the manual kill-fix-relaunch protocol). Every amendment is ledgered
in [LANGUAGE_LADDER.md](LANGUAGE_LADDER.md) before it applies.

## 5. After the flash: the served life

The flash builds faculties; the life writes biography (the
division-of-labor law, A69). The banked end-of-life state (band states
and stores) seeds the serve room (`iga/lm_serve.py`,
`scripts/serve_v94s.py`): one continuous life, the human presses the
buttons, nightly sleep consolidates what was paid. The demos run there,
speech-gated, on day one — [DEMO_PROTOCOL.md](DEMO_PROTOCOL.md).
