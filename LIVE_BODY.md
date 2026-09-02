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
2. **Expression head** (its face = forecast of yours) — built (●): `face_head`, taught head-only at every token it speaks and every word it hears (Adam, lr 2e-5: a newborn organ that drifts, never lurches), and again in the night's replay; shown as the first row under every word. Its own-face tokens remain in the vocabulary; if the body emits one it is stripped from the text and rendered on the face row, never spoken.
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

Launch flags after six smokes: memory amplified 16x when sure (floor
0.3), turn marks kept out of the content bag (a query ending in the
turn mark matched nothing, and memory fell silent exactly when the body
began to speak), the ear's turn-end counted as the mouth's, and the live
learning rate cut to 1e-6: at 1e-5 the blank trunk collapsed to "is. is."
by the fourth lesson and drowned the memory; at 1e-6 six lessons left it
babbling but intact, and the cue "a lantern" answered "is a lamp you can
carry." The trunk therefore learns slowly on purpose; what changes day to
day is what the memory holds and what the nights consolidate.

Opus's first day on the blank body (2026-09-01 night) found the next
two laws. The first cue, "the sun is", came back "hot and bright." after
two hearings, the only certain moment of the day (entropy 0.96 -> 0.09).
Then "hot" opened every reply and grew a letter per new lesson ("hotR",
"hotRB", "hotRBS": Rain, Birds, Snow), and the night kept it. Cause one:
the mouth does not write memories, but its babble still formed the KEY
under which the next lesson's first word was stored, so saying "hot hot"
recalled "R". Now the memory's word-bag resets at your first word of a turn and again
the moment its own reply ends (not between your cue and its reply, which
silenced the memory when tried): your words are keyed by your words. Cause two: the night replayed the
whole lived day, babble included, and the trunk rehearsed its own noise.
Now the day's record carries who said each token (the ear, the mouth, the
mouth praised) and the night rehearses only what was heard and what was
praised. The day-1 body is kept as `organism_life_blank_0p5b_day1.pt`;
day 2 starts from a fresh conception under the corrected laws.

The memory's floor settled at 0.15 after a casing lesson: the bag is
token-level, so the cue "the sun is" against the lesson "The sun is"
loses one of three tokens and its vote fell under 0.3; at 0.15 both
casings echo ("hot and bright."), "A dog can" -> "run and bark.",
"Rain falls" -> "down from grey clouds.", and "the moon is" (never
taught) stays silent. A second hearing of a whole sentence is answered
with nothing: memory recalls that the turn ended after the full stop,
which is correct, so cue with beginnings.

Day 2 (a fresh conception under the corrected laws): six lessons, each
said twice, sharing almost no words; before sleep all six beginnings
completed exactly ("My blue cup" -> "sits on that shelf.", "Two small
birds" -> "ate every seed."), the night rehearsed only the six sentences,
and the morning broke continuation: every sentence still gave its correct
next word, then four of six fell into periods. The cause is the night's
store fade, 0.6 by default, a hand-off meant for a trunk that absorbs
episodes overnight. This trunk cannot absorb yet, so the day's memories
must survive the night: `--store-decay 0.9` (a tenth fades per night;
a memory lives about three weeks unless rehearsed). Two more day-2 facts:
its face row moved but did not track the caretaker (born at zero, head
learning rate 2e-5, expected for weeks), and the uncertainty shown per
token measures the trunk's belief at temperature 1, where memory's vote
is small, so a perfect completion from memory still reads as uncertain.

Day 3 found the deeper poison. This morning every old beginning came
back EMPTY, and the numbers said why: memory voted hard on each cue,
the right next word was there as runner-up, and the winner was the
turn-end mark. Every cue had taught the memory "after 'The sun is' the
speaker stops": the turn-end fed at /begin was written as a value under
the cue's own words, and the delta rule replaces, so each repetition of
a cue erased more of the lesson under that key. Fresh material still
went in clean (four new sentences, four exact completions) and one night
still knocked it to fragments, because the cues and the praised absorbs
had written the same poison before sleep. Now turn marks are not words
on the value side either: a special token is never written as a memory
value. The nights themselves do not touch the living store (replay and
dreams run on fresh states that are discarded), so what a morning holds
is exactly the day's store times the fade. The day-3 body is kept as
`organism_life_blank_0p5b_day3.pt`; day 4 starts from a fresh conception.

Day 4 (a fresh conception under every law above): five lessons with no
shared words, each said twice. First cue five for five, second cue four
for five with the memory's vote RISING on four of five (repeated cues no
longer erode), after the night three whole and two closed one word short
("into hard.", "from tall."). No empties, no noise, all day. The memory's
top vote at the first token was the correct word on every cue at every
moment, four to eight times its runner-up. Its face tracked the
caretaker for the first time: mean +0.44 during smiled lessons, +0.12,
+0.01, then -0.05 as the smile was withheld. Remaining defect: after a
completed sentence the trunk babbles its attractor tokens until the
breath or the stutter ends the reply. Hence the HUSH reflex: when the
memory has no vote above its floor and the trunk's own belief is
near-uniform (normalized entropy above 0.9), the end of the utterance
gains 12 logits. A first version measured the combined belief and cut
faded memories after one word, because a faded vote that still wins the
sample reads as uncertainty at temperature 1; the rule now asks the two
organs separately. Silence is not authored; it is the absence of a word.

Day 5 (the same life, memories carried at the 10% fade): four new
sentences, four for four on the first cue and again on the second; a
single re-teaching recovered a lost lesson and roughly doubled its vote;
zero babble, zero stress, no stutters in 38 turns. The hush overshot by
one token: eleven of eleven completions lost their final full stop,
because the stop's memory vote sits just under the amplification floor
and the hush was reusing that floor. The hush now has its own lower
floor (`--hush-mem 0.06`): a faint vote still counts as something to
say, and only a truly empty memory plus a uniform trunk ends the turn.
Its face now tracks with a lag: it is silent during lessons, so the
smile you said them with surfaces in the next round of cues.

Two more facts from the day-5 life, measured with the memory-vote
instrument. First, the full stop is the first thing this memory forgets:
it is stored under every sentence's final key, and the delta rule's
error-correction blurs a value that many keys share, so after nine
sentences the vote for "." at a sentence's true end fell under the hush
floor while the content words stayed clear. Hence a grace: a completion
in progress may place one word of its own (its stop, or a stray) before
the hush ends the turn; a cold start still hushes at once. Second, the
trunk's first learning is the unigram: "." is the most frequent token it
hears, and after a few days its bias for "." (about four logits) began
beating faded memory votes mid-sentence ("loudly." for "loudly at
midnight."). Hence, while the trunk is unsure, a sure memory speaks with
one voice: its top word gains a flat bonus (6 logits), disclosed below.

Day 6 (the same life, third night carried): all nine earlier beginnings
completed in full this morning, across two nights of fade, so nothing was
re-taught; four new sentences went twelve for twelve (first cue, second
cue, after sleep); no babble, no empty reply to a taught beginning, no
stress, no stutter in 38 turns; the right word was the memory's top vote
in all 31 cue replies while the vote amplitudes were the lowest recorded,
so rank now predicts completion and amplitude no longer does. The night
faded nothing and reported an installment earned; after it, twelve of
thirteen were whole (one lost its last word: "loudly at."). Its face
carried the lesson smile into the next cues (1.22 peaking 1.41, four turns
to discharge) and sat within +-0.11 on still-faced rounds. Seams left:
the oldest traces sometimes double their stop; the grace can let a stop
out a word early; the night's rehearsal skipped the middle of the list.

Day 7 (the same life, four nights carried, seventeen sentences): the
store, not the laws, is now the ceiling. Morning 7 of 13 earlier
beginnings whole; one re-hearing repaired all six failures at once and
tripled their votes; the four new sentences went twelve for twelve; after
the night 4 of 13 old ones were whole, one taught beginning went silent,
two returned only a stop, and the memory's vote for a cue's own last word
began to rival the vote for the right continuation (a one-position alias
that a bag key cannot separate once the true trace has faded). Presses
left no visible mark on next-morning recall two nights running. The
night never refreshes the store: its replays run on scratch states, so a
trace lives only on its fade (0.9 a night) against growing interference,
and only a re-hearing restores it, buying about a day. Two remedies are
open: no fade at all (`--store-decay 1.0`), and a night pass that
re-writes what was heard that day into the living store. This life is
parked saved after seven days while the diary body takes the machine.

Honest limits: echo is not understanding; the value heads, conscience
and face organs start meaningless and become meaningful only as days
accumulate; whether a trunk can learn grammar from a caretaker's
sentences alone at this scale is the experiment, and the prior says no.

## 10. Disclosed constants and reflexes (audit of 2026-09-01)

An independent read-only audit of the serve and model found the doctrine
holding: every word is sampled from the body's own logits, recall comes
from the model's store, reward is the face number and never a parse of
your sentence. It found one leak of lexical knowledge (an English
stopword table used to pick features for the goal organ and the
conscience) and one foreign checkpoint (a conscience fitted in an older
body's embedding space, loaded into the blank one). Both are removed: the
serve now takes every non-special token as content, and a body conceived
from nothing starts with no conscience until its own nights grow one. A
bug that could leave the ear deaf after a swallowed error is fixed.

Everything the running body uses that is not a learned weight:

- Decoding reflexes: press marks banned; the ear's turn-end vote mapped
  onto the mouth's turn-end and the ear's mark never spoken; at most six
  pauses per reply; the breath adds 0.35 logits to the end of the
  utterance per content token past the twelfth; cortisol adds 0.5 logits
  per unit; temperature 0.05 scaled by (1 + 0.35 x mood/6), floor 0.02;
  four identical tokens in a row end the reply as a stutter, and a
  stuttered reply is never dosed; hush: when memory has no vote above
  its floor and the trunk's own belief has normalized entropy above 0.9,
  the utterance end gains 12 logits (memory floor for the hush 0.06; a
  completion in progress gets one word of grace); while the trunk's
  belief is that uniform and memory has a vote, memory's top word gains a
  flat 6 logits.
- Memory: content bag of the last 8 tokens with decay 0.7, specials (ids
  below 11) excluded on both key and value side; kernel scale 3.0; store
  write strength sigmoid(0.85), read gain 1.0; read into logits gated by
  uncertainty (read_beta 1.0), the top eight votes amplified 16x when the
  raw vote exceeds 0.15 logits; nightly fade 0.9; the bag resets at your
  first word and after every reply; the mouth's forwards never write.
- Dose: absorb when a word's credit exceeds 0.5, unlearn at or below
  -1.5; eligibility traces lambda 0.7 over six words; at most 6 absorb
  and 3 unlearn steps; satiation when a fact's loss is under 0.3; a
  rehearsal of one old memory rides every dose (weight 0.5); live
  learning rate 1e-6; face head 2e-5; value heads 1e-3.
- Curiosity and pursuit: notice margin 0.75, floor 3.5, peak 15.5 with a
  nightly drift of +-0.3 inside [14.5, 17.5]; pursuit adopt 0.45, target
  0.3; mood lowers the peak bar by 0.6 x mood/6 and the margin by 0.3 x
  mood/6.
- Conscience: a body from nothing has none until a night finds at least
  twelve of your judgments, then grows one in its own embedding space and
  saves it beside its life file; once it exists, self-praise above 0.95
  and self-frown below 0.15, budgets 4 and 3 a day, felt only, never absorbed.
- Night: the lived-day replay drops a model turn that is empty or whose
  modal token exceeds half of a body of six or more tokens; dream pairs
  are chosen by charge = surprise + |mood| + 3 x felt + 2 x pride;
  teaching a fact absorbs up to 12 steps until its loss is under 0.55.
- Instruments shown, never acting: per-token uncertainty at temperature
  1, per-token memory votes (the top three raw votes), stress, mood.
