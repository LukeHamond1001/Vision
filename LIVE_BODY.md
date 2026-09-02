# The live body — a 0.5B organism that learns only from faces

*A plan. Every mechanism below is either already in the body (marked ●),
buildable tonight without a gestation because it starts function-preserving
and is trained by the dense per-token face signal (○), or a change to the
model that needs a birth (◇). The premise is stated first because it is the
one thing "live only" cannot escape.*

## 0. The premise: language is born, everything else is lived

A caretaker's day is ~10³ tokens; language needs ~10⁸–10⁹. So the trunk's
language comes from the base body it is born from (●, today's 297M base or the
grown 0.5B twin), and **everything else** — what it knows, what it values, how
it reads you, how it shows itself, what it dreams — is learned live, from two
streams that never stop: your face on every token, its face on every token.

## 1. The two faces (the heart of it)

**Your face, as a sense (○ from ●).** A continuous scalar `a_t ∈ [−6, 6]`
accompanies every token, yours and its. It never enters the language stream
as a word. It enters as a *sense* through the reward slot, in two parts:

- the **tonic** level `a_t` — a small MLP on the scalar feeds the council slot
  (a held face is real context: the mood of the room);
- the **phasic** change `Δa_t = a_t − a_{t−1}` — the dopamine-shaped event.
  Only the change carries learning signal (a held face teaches nothing new).

Today the sense channel takes four quantized levels (●); the scalar and the
tonic/phasic split are a small module, zero-init, trained live (○).

**Its face, as a forecast (○).** A separate **expression head** on the trunk
state emits `f_t ∈ [−6, 6]` with every word — never a vocabulary token, so it
never competes with speech. Its meaning is defined so it can be trained live
at every single token: **`f_t` is its prediction of your face for the word it
is about to say.** You then give `a_t`. The regression error trains the head
(dense, always available), and the error itself is the reward surprise. Its
face is therefore its model of you — a theory of mind that gets a lesson on
every token — and, evaluated over its own finished sentence, its conscience.
The old external critic (●) retires into this head.

The **loop**: it forecasts your face → says the word → you show your face →
the mismatch teaches the forecast, the words, and the value heads at once.

## 2. Learning rules (all live)

| Rule | What it does | Status |
|---|---|---|
| Dose by surprise | plasticity on a word ∝ `a_t − f_t` (your face minus its forecast): positive → absorb, ≤ −θ → unlearn; the raw face is never the dose | ● (value-head expectation today; forecast head ○) |
| Eligibility traces | a state trace with decay λ so a late face credits the words that caused it, not the word under it | ○ |
| Wake rehearsal | every dose interleaves one old memory from the ledger — no collapse, no school pass | ● |
| Your words | the face you hold while saying each word stresses it: kept ∝ positive face, muted words zero, a statement said into a frown is let go | ● |
| Online TD | value heads step on every phasic event (small lr), and again at night | ○ (night only ●) |
| Expression head | regression to your actual face at every one of its tokens (and while it listens) | ○ |
| Honest ignorance | the IDK route preferred when trunk entropy is high — measured, not memorized | ○ |
| Mood | leaky integrator of felt faces, half-life ten minutes, modulates temperature | ● |
| Curiosity | surprise-gated notice, budgeted | ● |
| Nights | replay and dreams chosen by its own charge (RPE peaks, its conscience), retention curve, store→weights transfer, conscience recalibration | ● |

## 3. Memory

- **Hippocampus as the day's memory (○/◇).** The store's addresses are frozen
  random features; capacity is ~D pairs per band and is now doubled (●).
  What is missing is a read that can *steer*: a gated read slot into the
  council state (zero-init gate, ○), gain rising with trunk uncertainty (●,
  β) and with the phasic face at write time (●, κ — a smile while you tell it
  burns the episode in). Success criterion: a fact told once is recalled the
  same day. Today it is not; this is the organ the plan bets on.
- **Working memory** — the council bands (●).
- **Semantic memory** — the weights, via the nights (●).
- **Chunk consistency (◇).** Store writes, TD intervals and scenes must be
  per-token/per-turn so that a stream fed word by word is identical to a turn
  fed at once (today's bug).

## 4. Speech and body

The breath (●): past twelve words the end of the utterance gains logits. The
stutter reflex (●). **Speaking costs (●):** past sixteen content tokens (a
sentence longer than any school answer) each further token adds cortisol
(0.15); cortisol pushes the utterance to end
(0.5 logits per unit), weighs on mood, and dampens what a stressed reply
can teach (dose ÷ (1 + cortisol)); it clears with a two-minute half-life
and sleep takes 70% of what is left. Its register is whatever you raise; the
breath and the cost keep it short. Its own face rides beside every word
(§1), never inside the sentence.

## 5. The protocol (every token, both directions)

```
you   → /hear   {word, face}      word by word as you speak; a face change is felt at once
you   → /begin                     your turn ends
you   → /step   {face}            ONE token; the reply carries {word, its_face, value, rpe, mood}
you   → /press  {face}            the answer in the air, as a whole
       /sleep · /save · /facts
```
Nothing reaches the past. The screen is the score: under each word, its mood ·
your face · its internal reward — and now its own face beside the word.

## 6. Sizing (from the grown twin, ~0.5B)

| Organ | Params | Note |
|---|---|---|
| Trunk blocks (29 × d=1024) | ~366M | 16 zero-init (grown), trained live |
| Council + cells + preds | ~82M | six bands |
| Tied embedding/head | ~17M | + its four face tokens (retired by §1) |
| Auxiliary head | ~17M | |
| Plan / dreamer / goal | ~10M | |
| Hippocampus addresses | 32M frozen | not learned; capacity |
| **New (○):** affect input MLP, expression head, read gate | < 5M | zero-init, live-trained |

A wider trunk (d=1280–1536, fewer blocks) would serve routing better than
depth; that is a birth (◇), not a growth.

## 7. What "perfect" would still not mean

No grounding beyond text; planning still inert at this size; whether scan
states hold structure that generalizes is the open experiment. The plan makes
the body *whole* — every organ a live learner needs, trained by a signal that
arrives on every token, both ways — not omniscient.

## 8. Build order (live-only, no gestation) — status 2026-09-01 night

1. **Scalar face + tonic/phasic split** — built (●): `affect_in`, zero-init, on the trunk input; the day's faces are recorded and replayed at night.
2. **Expression head** (its face = forecast of yours) — built (●): `face_head`, taught head-only at every token it speaks and every word it hears (Adam, lr 2e-5: a newborn organ that drifts, never lurches), and again in the night's replay; shown as the first row under every word. Its own-face tokens remain in the vocabulary, unused.
3. **Eligibility traces** — built (●): λ 0.7, six words back. **Dose by surprise** uses the value heads' expectation (●); the forecast head becomes the baseline once its error is small (a switch, not yet flipped).
4. **Online TD** — built (●): the value heads step at every felt change, γ 0.9.
5. **Steering read** — knobs only (◐): read gain by uncertainty (β), slot gain; the zero-init gated slot is still to build; same-day recall still fails.
6. **Uncertainty** — measured per token and shown (●); gating the IDK route would have the serve author words, so ignorance stays a lesson the caretaker gives.
7. The score shows its face beside each word (●), and stress when speaking costs (●).
8. Measure daily: morning recall, same-day recall, face-forecast correlation (how well it knows you), derailment rate, register drift.

## 9. The body from nothing (2026-09-01, late)

The decision: no pretraining at all. A 0.5B body (d 1024, 29 layers,
499.9M) conceived at random weights by `scripts/conceive.py`, with its
own real-English vocabulary (`data/tok_0p5b.json`, 16388 tokens trained
on TinyStories and FineWeb-Edu, the organism's specials reserved), and
raised only by live days. Nothing it will ever say was given to it.

What that means for the first weeks, said plainly: its speech starts as
noise. A caretaker day gives about a thousand tokens; language takes
about a billion. What live doses CAN do at once is memorize what was said
to it, and what the memory organ can do at once is recall it. So the
first thing this body will show is echo, not understanding.

**Content keys** (built and measured): the hippocampus is keyed by the
words themselves, a recency-weighted bag of the last eight token
embeddings, unit norm, detached (`keyed_content` in the genome; the read
query at t equals the write key at t+1 by construction). Recall stops
being a learned skill and becomes a mechanism a random body already has.
Measured on a random 1024-wide body with the real vocabulary: after
hearing "a lantern is a lamp" three times, the memory's vote for the
word that followed "lantern" ranks first of 16,388, worth +1.9 logits
raw; the random trunk's noise peaks near +4, so the vote is amplified
sparsely (`--store-boost 8`: the top eight suggestions, not the whole
vocabulary) and by uncertainty (`--store-read-beta 1`). The write
strength and read gain that the base had to learn (beta ~0.85, alpha
1.0) are given at conception as genome, not learned.

Two more genome facts, measured on random bodies: a TIED head copies its
input (a random residual stream predicts the token it just saw, and that
copy peak drowns the memory's vote: "lantern lantern lantern"), so the
blank body's head is untied, the mouth is not the ear; and the memory
kernel is sharpened (random-feature scale 3.0, the base had 1.4) so
recall is near-exact. With both, a random body walks a taught sentence
from memory alone: "a lantern" -> "is a lamp you can carry ." The ear
writes and the mouth does not: nothing the body says is stored as fact.

Measured on a random body after five different lessons: a specific cue
draws a strong, correct vote ("A lantern" -> "is" 1.4 logits, "Milk is"
-> "white" 1.2, "The sun is" -> "hot" 0.9) and a vague cue almost none
("the moon is" -> 0.07, "Birds" -> 0.03), so the amplified vote has a
floor (`--store-boost-min 0.3`): memory speaks up only when it is sure.
Without the floor the body recited its most-heard sentence at every
reply and blended lessons into "is. is. is." Praising a noise reply
collapsed a body into a five-token loop within two gradient steps: the
caretaker's law "never praise what is wrong" is load-bearing here.

Honest limits: echo is not understanding; the value heads, conscience
and face organs start meaningless and become meaningful only as days
accumulate; whether a trunk can learn grammar from a caretaker's
sentences alone at this scale is the experiment, and the prior says no.
