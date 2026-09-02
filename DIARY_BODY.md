# The diary body

Two writers, one page, one symbol at a time. The user's idea (2026-09-02):
like the diary in the Chamber of Secrets, both hands write into the same
stream on a shared clock; there are no turns, no end-of-utterance marks,
no breath, no hush. Silence is a symbol the body must choose, and your
face on its noise and on its quiet is how it learns when to be still.

## What it is

- **Alphabet:** printable ASCII plus newline, 96 symbols, and the
  organism's specials (`data/tok_char.json`, 107 ids; `<pad>` is silence).
- **One stream, two hands.** Each tick has two positions: your symbol (or
  silence) as speaker 0, then its symbol (or silence) as speaker 1. A
  fixed small speaker vector joins each embedding, so a blank body tells
  the two hands apart from birth. Simultaneity lives at the tick; the
  trunk still reads one sequence. Two symbols are never blended into one
  position (that is noise, not a conversation).
- **The ear writes, the mouth does not,** now inside the model by the
  speaker channel: only speaker 0's symbols become memory values.
- **Three hands in the bag.** The first GPU run showed the newborn mouth
  babbling between your letters as you typed, and its junk letters became
  part of the keys your sentences were stored under. So the mouth has two
  hands: a letter it writes from memory (speaker 2) joins the thought and
  advances the echo; a letter it writes from noise (speaker 1) is in the
  stream but not in the bag.
- **Memory over letters:** the same content-keyed store with a running
  bag (decay 0.92 per symbol) that fades in silence (x0.95 per silent
  position, two positions to an idle tick) and is cleared by a new line;
  kernel sharpness 4.5. Measured on a random 2-layer body, letter by
  letter through the two-hand stream: "The s" -> "un is hot and bright.",
  "Cold w" -> "ater freezes into hard ice.", repeated cues intact, its own
  echoed letters never poison the page. An untaught cue babbles, because
  nothing stops it but your face.
- **Faces per tick,** as in the word body; credit by eligibility traces;
  doses become rolling instead of per-utterance.
- **Same nights, same cortisol** (a cost per emitted symbol), same
  doctrine, same instruments (its face, mood, reward, stress, uncertainty,
  memory votes).

## The body

`data/organism_diary_0p5b.pt`: conceived from nothing, d 1024 x 29
layers, untied head, content keys 40/0.92, kernel 4.5, three hands
(speakers 3), silence decay 0.95:

```bash
python3 scripts/conceive.py data/organism_diary_0p5b.pt data/tok_char.json \
    --d 1024 --n-layers 29 --content-keys --kc-w 40 --kc-decay 0.92 --kernel 4.5 \
    --speakers 3 --sil-decay 0.95
```

## What to expect

Letters are five times slower than words and a blank trunk learns
letters before words, so the trunk lags even further behind than in the
word body. The memory carries the diary behavior from the first day:
what you write, it can write back from a few letters. The first thing
your face has to teach is silence; until then it fills every quiet tick
with noise.

## Day 1 (Opus, 2026-09-02, 25 minutes, six sentences written twice)

Shadowing on 18 of 18 writings: on a second pass it rode the whole
sentence one letter behind with 30 memory-backed letters. Recall after a
cue failed every time, and the memory-vote instrument showed why: the
votes were right ("Snow c-o-vers" in order) while the hand wrote a
carrier letter. Twenty frowns did not teach quiet; they moved the babble
from one carrier to the next ("}", "i", "a", "s", "f", "n", "g", "b").
Mood sat at the floor all session. After the night it wrote "Rain S"
unprompted from memory and repeated it every 39 ticks.

Four causes, four fixes:

- **A thought faded too fast.** The bag lost 40% per silent tick, so a
  four-second pause erased the cue before the mouth could use it. Now 5%
  per silent tick: a thought lasts about half a minute of silence.
- **A frown lowered letters but never raised silence.** First answer: a
  QUIET LESSON that wrote silence as the target wherever it babbled. It
  worked (babble 1.0 to 0.0 in eight lessons) and was withdrawn the same
  day on the user's principle: silence is not something we teach. Now the
  only teacher is your face on what it actually did. Every tick is a
  choice, silence included; a choice whose credit rises above 0.5 is
  absorbed, a choice at or below -1.5 is unlearned (its probability pushed
  down to a floor of 3 nats, never replaced by a hand-written target). It
  finds quiet only where quiet paid, and where speaking cost it.
- **Stress and mood were miscalibrated.** Each symbol now adds twice the
  stress (a physiological brake toward silence under nonstop babble) and
  a fifth of the mood cost.
- **An empty page recalled beginnings.** The first symbol of a thought was
  written under an empty key, and an empty bag matched it. No context, no
  key: such positions are not stored. (This applies to the word body too.)

The page now shows its memory-backed letters in brown and its noise in
pale sand, so the two hands are visible.

Measured after the fixes (a fresh conception): eight face lessons took
the babble fraction from 1.0 to 0.0 in under a minute, and the trunk's
uncertainty fell to 0.0: it had learned silence completely. It overshot:
certain silence also silenced recall (a cue that had just echoed "Rain
falls" came back empty). Hence one more disclosed rule, a memory is a
reason to speak: when the memory has a vote above its floor, its top
symbol's logit is raised to at least the silence logit plus memory's
TRUST, a scalar your face moves (start 4, bounded 0..8: praise on a
memory-backed letter raises it by 0.1, a frown lowers it by 0.2), so a
trunk that has learned quiet does not silence what it remembers, and a
trunk that learns to speak well can have memory's voice shrink. The day-1 body is kept as
`organism_diary_0p5b_day1.pt`; day 2 starts from a fresh conception.

## The thought, measured (2026-09-02, CPU harness with the mouth writing)

Recall failed whenever the mouth wrote during the lessons, and the
memory-vote instrument found three reasons, each now a rule:

- **Noise must leave the thought untouched.** A noise letter was fading
  the running bag like a silent tick, so keys were written into a decayed
  context and cues never matched them. Now a noise letter neither adds to
  nor fades the thought; only silence fades it.
- **A memory letter joins the thought only when your hand is still.**
  While you write, its shadowing letters interleaved with yours inside
  the bag and corrupted the keys; now they stay out, and only a completion
  written while you are silent advances the thought.
- **A new line ends a thought.** With a slow fade (5% per silent tick) the
  previous sentence's residue outweighed a five-letter cue; with a fast
  fade a two-second typing pause erased the sentence. The symbol you
  already use to separate thoughts resolves it: a new line clears the bag.
  Start every sentence and every cue on a new line.
- **No parroting** was tried and withdrawn: banning the letter you just
  typed hid a memory alias but restricted what it may say. The only
  restriction on the mouth is stamina; the alias echo is now simply
  visible, and your face decides what becomes of it.

With all four, on a random body with the mouth babbling through three
sentences written twice: "My do" -> "g sleeps under a wooden table.",
"Green" -> " leaves move when wind blows hard.", "Rain " -> "falls on the
cold grey", and an untaught cue babbles until your face teaches quiet.

## Day 2 (Opus, on the old fade) and the child curriculum

Five new sentences written twice each: five of five came back in full
through the stop from a five-letter cue. Then the recall tests poisoned
what they measured: twenty cue fragments, each followed by the next cue's
leading newline, taught the memory "a short fragment, then the thought
ends", and later cues stopped after the fragment. The newline is the
thought-break symbol; it is now never a memory value, so a cue cannot
teach an ending. Quiet held the whole session (0.0 of idle ticks), with
no frowns given because it produced no noise.

Day 3 onward is raised like a child, not fed a list (the user's
direction). Children's first produced words are mostly relational and
social (hi, bye, no, more, up, all gone, uh oh) plus a few names of what
matters (mama, dog, ball, milk); "mommy" is produced by 93% and "ball" by
64% of children by sixteen months in the MacArthur-Bates CDI norms, and
the first grammar is pivot pairs built from a small set of relational
words ("more milk", "no milk", "milk all gone", "dog up", "my ball"). So
the caregiver: greets, says little, waits, imitates what the child
offers, expands it by one step, responds within seconds, never frowns at
babble, and teaches relations by recombining the same few words. Recall
tests are few, never back to back, and never followed by a newline. The
instrument that decides whether the trunk learns anything is "own": its
own top symbol with memory set aside; the day it becomes a letter after
"more " is the day the trunk knows something.

Sources: the MacArthur-Bates CDI (Fenson et al., https://mb-cdi.stanford.edu/documents/Fensonetal2000.pdf)
and its Wordbank update (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10264806/).

## The review (2026-09-02, independent, read-only, with CPU measurements)

Verdict: the mechanism is what it claims. Query/key alignment exact to the
bit; the empty-key guard fires; the speaker channel keeps the mouth out of
the thought; the kernel at 4.5 separates a true continuation from its
one-behind alias about 5:1; a blank body recalls six distinct sentences
letter-perfect from a five-letter cue; the dose's credit-to-position
mapping is exact. One architectural flaw and several bugs, all fixed:

- **The store faded per forward pass, not per symbol written.** The
  ladder was tuned for 64-token training chunks; a diary tick is two
  one-symbol forwards, so the slowest band forgot in about 128 seconds
  whether or not anyone wrote, and the day-1 recall tests failed on
  time, not capacity. Now a chunk fades each band by the share of
  symbols it actually wrote: half-life = the band's clock in written
  symbols (8, 64, 512, 4096, 32768 for bands 4-8; band 3 holds the last
  write). Measured after the fix: a sentence recalled letter-perfect
  after 1,240 idle ticks and fourteen further sentences. This also
  explains part of the word body's multi-day fade: its serve fed one
  token per forward too.
- **A noise newline wiped the thought.** The mouth's noise never ends a
  thought now.
- **The face event fired on the level, not the change.** Easing off a
  frown counted as a frown, easing off a smile as a smile. Now, as in the
  word body, only a face that grows or flips sign is an event.
- **The night trained a speaker-blind trunk from the wrong end of the
  day.** The two hands now ride the replay, and the most recent chunks
  are rehearsed first.
- Operational: a dose window of the last eight ticks (a long dose froze
  the clock), a bounded page and typing queue, a reset that clears the
  thought, memory's amplification never lifting a banned mark.

## Day 3, watched live: the run, and remembered quiet

Supervising Opus's third session (the child curriculum) showed the one
pathology the alias leaves: after "all gone" the mouth wrote
"goneeeeeeee". Nothing true follows the last letter of a thought (the
newline is never a value), so the memory's top vote is the one-behind
alias of the letter just written, the mouth writes it, the bag still
matches, and the loop feeds itself until stress stops it. Restricting the
mouth is out (only stamina restricts), so the answer is on the memory's
value side: THE EAR'S QUIET IS A MEMORY. The first silent tick after your
word is stored as what followed it, once per pause; the memory then
recalls quiet where you fell quiet, and the mouth stops from memory
instead of running on its alias. Measured on a CPU body: "ball " ->
"all gone" then silence; "more " -> "milk" then silence; "my " -> "dog
runs" then silence (without it: "all goneeiiiiiiii"). Written once per
pause on purpose: writing it on every silent tick over-weighted quiet and
cut "all gone" to "al".

## Status

- model: speaker channel, ear-writes by speaker, running bag with
  silence decay — built and measured on CPU (2026-09-02).
- serve: `scripts/diary.py` — the tick loop (two forwards per tick: your
  symbol as the ear, its symbol as the mouth), the page (your letters
  black, its memory letters brown, its noise pale; arrow keys move your
  face; Enter is a new line), doses on its actual choices by the word
  body's thresholds over the last eight ticks, the same nights, save and
  reset. Numbers: stress +0.03 per symbol written (half-life 120 s), mood
  -0.002 x stress per symbol and +0.5 x a felt face; a face registers on
  whole-unit crossings (int); characters outside the alphabet are dropped;
  the typing queue holds 600 symbols. Smoked on a tiny CPU body (2026-09-02): ticks, both
  hands on the page, faces felt, a night, a morning.
- first serve: the word body is parked after seven days; the diary body
  runs on port 8018:

```bash
python3 scripts/diary.py data/organism_diary_0p5b.pt data/tok_char.json --dev mps --port 8018 \
    --temp 0.05 --store-read-beta 1.0 --store-boost 16 --store-boost-min 0.15 \
    --live-lr 1e-5 --store-decay 0.9 --save data/organism_diary_0p5b.pt --diary-period 0.5
```

The live rate is ten times the word body's on purpose: the first thing
frowns must teach a newborn mouth is silence, which is the cheapest
attractor a trunk can fall into, so here the collapse the word body had
to avoid is the lesson.

The principle (the user's, 2026-09-02): the only restriction on the mouth
is stamina. What remains: the bookkeeping marks it cannot emit (plumbing,
not language); stress leaning it toward silence (0.5 logits per unit,
applied before the memory rule) and weighing on mood; the face lesson
every tick; a new line ends a thought. Not restrictions but memory's
voice, disclosed: a sure memory speaks with one voice while the trunk is
unsure (6 logits), and a memory is a reason to speak (memory's top symbol
raised to at least the silence logit plus 4, so neither learned quiet nor
stress cuts a recall short). The night replays the day as lived, silences
included (runs kept to one tick), so a quiet tick can be learned as
thinking time. Instruments per tick: its face, mood, stress, uncertainty,
memory's votes, and the trunk's own top symbol with memory set aside (the
measurement that will show the day the trunk itself begins to propose
letters). Reflexes it drops: the breath, the hush, the end-is-an-end
rule, the bag reset (silence fades the bag instead).
